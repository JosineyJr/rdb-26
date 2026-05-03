// Package search implements an Inverted File (IVF) index for extremely fast
// approximate nearest neighbor search.
//
// Brute-force scanning 3M vectors per request causes high CPU usage and leads
// to queuing delays (and HTTP timeouts) under load when limited to 0.45 CPU.
// IVF partitions the 3M vectors into clusters at startup. At query time, it
// only scans the `nprobe` closest clusters, reducing distance calculations
// by ~95% while maintaining near-perfect accuracy.
package search

import (
	"cmp"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sync"

	"github.com/JosineyJr/rdb-26/internal/dataset"
)

// Index holds the IVF structure.
type Index struct {
	vecs      [][14]int8
	labels    []bool
	origIDs   []uint32
	centroids [][14]int8
	clusters  []clusterBound // contiguous memory blocks for cache locality
	bboxMin   [][14]int8
	bboxMax   [][14]int8
	nprobe    int
	repair    bool
}

type clusterBound struct {
	Start int32
	End   int32
}

type centroidDist struct {
	id   int
	dist int32
}

type BuildOptions struct {
	Clusters   int
	NProbe     int
	Iterations int
	Seed       int64
}

func DefaultBuildOptions() BuildOptions {
	return BuildOptions{
		Clusters:   1024,
		NProbe:     8,
		Iterations: 5,
		Seed:       42,
	}
}

// Build constructs the IVF index by picking random centroids and assigning
// each vector to the nearest centroid. It also reorganizes the vectors in memory
// so that all vectors in a cluster are contiguous, maximizing CPU cache locality.
func Build(vecs [][14]int8, labels []bool) *Index {
	return BuildWithOptions(vecs, labels, DefaultBuildOptions())
}

func BuildWithOptions(vecs [][14]int8, labels []bool, opts BuildOptions) *Index {
	defaults := DefaultBuildOptions()
	if opts.Clusters <= 0 {
		opts.Clusters = defaults.Clusters
	}
	if opts.NProbe <= 0 {
		opts.NProbe = defaults.NProbe
	}
	if opts.Iterations <= 0 {
		opts.Iterations = defaults.Iterations
	}
	if opts.Seed == 0 {
		opts.Seed = defaults.Seed
	}

	numClusters := opts.Clusters
	nprobe := min(opts.NProbe, numClusters)

	if len(vecs) < numClusters {
		numClusters = len(vecs)
		nprobe = min(nprobe, numClusters)
	}

	idx := &Index{
		centroids: make([][14]int8, numClusters),
		clusters:  make([]clusterBound, numClusters),
		bboxMin:   make([][14]int8, numClusters),
		bboxMax:   make([][14]int8, numClusters),
		nprobe:    nprobe,
	}

	// 1. Pick random centroids to partition the space.
	rng := rand.New(rand.NewSource(opts.Seed))
	perm := rng.Perm(len(vecs))
	for i := 0; i < numClusters; i++ {
		idx.centroids[i] = vecs[perm[i]]
	}

	// 2. Spherical K-Means iterations (runs offline during Docker build).
	workers := max(runtime.NumCPU(), 1)
	stripe := (len(vecs) + workers - 1) / workers

	type localClusters [][]int32
	var results []localClusters

	for iter := range opts.Iterations {
		results = make([]localClusters, workers)
		var wg sync.WaitGroup

		for w := range workers {
			start := w * stripe
			end := min(start+stripe, len(vecs))
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(w, start, end int) {
				defer wg.Done()
				lc := make([][]int32, numClusters)
				for i := start; i < end; i++ {
					v := vecs[i]
					bestDist := int32(1<<31 - 1)
					bestC := -1

					for c := 0; c < numClusters; c++ {
						d := euclidSq8(v, idx.centroids[c])
						if d < bestDist {
							bestDist = d
							bestC = c
						}
					}
					lc[bestC] = append(lc[bestC], int32(i))
				}
				results[w] = lc
			}(w, start, end)
		}
		wg.Wait()

		if iter < opts.Iterations-1 { // Recalculate centroids
			for c := 0; c < numClusters; c++ {
				var sums [14]int64
				var count int64
				for w := range workers {
					if results[w] != nil {
						for _, vi := range results[w][c] {
							for d := range 14 {
								sums[d] += int64(vecs[vi][d])
							}
							count++
						}
					}
				}
				if count > 0 {
					var magSq int64
					var means [14]float64
					for d := range 14 {
						mean := float64(sums[d]) / float64(count)
						means[d] = mean
						magSq += int64(mean * mean)
					}

					// Spherical Re-normalization: scale centroid back to magnitude ~127
					// so it remains on the surface of the hyper-sphere.
					if magSq > 0 {
						mag := math.Sqrt(float64(magSq))
						scale := 127.0 / mag
						for d := range 14 {
							val := min(int(math.Round(means[d]*scale)), 127)
							if val < -128 {
								val = -128
							}
							idx.centroids[c][d] = int8(val)
						}
					}
				}
			}
		}
	}

	// 3. Reorder vecs and labels to be contiguous by cluster, sorted by distance to centroid.
	// This is "The Golden Optimization": vectors near the centroid are evaluated first,
	// dropping maxDist instantly and maximizing Early Abandon skips.
	orderedVecs := make([][14]int8, len(vecs))
	orderedLabels := make([]bool, len(labels))
	orderedIDs := make([]uint32, len(labels))

	var offset int32 = 0

	type vecWithDist struct {
		idx  int32
		dist int32
	}

	for c := 0; c < numClusters; c++ {
		startOffset := offset

		var clusterVecs []vecWithDist
		for w := range workers {
			if results[w] != nil {
				for _, vecIdx := range results[w][c] {
					clusterVecs = append(clusterVecs, vecWithDist{
						idx:  vecIdx,
						dist: euclidSq8(vecs[vecIdx], idx.centroids[c]),
					})
				}
			}
		}

		slices.SortFunc(clusterVecs, func(a, b vecWithDist) int {
			return cmp.Compare(a.dist, b.dist)
		})

		for _, vd := range clusterVecs {
			v := vecs[vd.idx]
			if offset == startOffset {
				idx.bboxMin[c] = v
				idx.bboxMax[c] = v
			} else {
				for d := range 14 {
					if v[d] < idx.bboxMin[c][d] {
						idx.bboxMin[c][d] = v[d]
					}
					if v[d] > idx.bboxMax[c][d] {
						idx.bboxMax[c][d] = v[d]
					}
				}
			}
			orderedVecs[offset] = v
			orderedLabels[offset] = labels[vd.idx]
			orderedIDs[offset] = uint32(vd.idx)
			offset++
		}
		idx.clusters[c] = clusterBound{Start: startOffset, End: offset}
	}

	idx.vecs = orderedVecs
	idx.labels = orderedLabels
	idx.origIDs = orderedIDs

	return idx
}

func (idx *Index) Ready() bool { return true }

func (idx *Index) SetRepair(enabled bool) {
	idx.repair = enabled
}

func (idx *Index) SetNProbe(n int) {
	if n <= 0 {
		return
	}
	idx.nprobe = min(n, len(idx.centroids))
}

// euclidSq8 computes squared Euclidean distance manually unrolled.
// This allows the compiler to perfectly inline and optimize without loops.
func euclidSq8(a, b [14]int8) int32 {
	d0 := int32(a[0]) - int32(b[0])
	d1 := int32(a[1]) - int32(b[1])
	d2 := int32(a[2]) - int32(b[2])
	d3 := int32(a[3]) - int32(b[3])
	d4 := int32(a[4]) - int32(b[4])
	d5 := int32(a[5]) - int32(b[5])
	d6 := int32(a[6]) - int32(b[6])
	d7 := int32(a[7]) - int32(b[7])
	d8 := int32(a[8]) - int32(b[8])
	d9 := int32(a[9]) - int32(b[9])
	d10 := int32(a[10]) - int32(b[10])
	d11 := int32(a[11]) - int32(b[11])
	d12 := int32(a[12]) - int32(b[12])
	d13 := int32(a[13]) - int32(b[13])

	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}

// top5 maintains the 5 nearest neighbors with zero heap interface overhead.
type top5 struct {
	entries [5]struct {
		dist  int32
		id    uint32
		label bool
	}
	count int
}

func betterCandidate(dist int32, id uint32, worstDist int32, worstID uint32) bool {
	return dist < worstDist || (dist == worstDist && id < worstID)
}

func (t *top5) push(d int32, id uint32, label bool) {
	if t.count < 5 {
		t.entries[t.count].dist = d
		t.entries[t.count].id = id
		t.entries[t.count].label = label
		t.count++
		// bubble up
		for i := t.count - 1; i > 0; i-- {
			if betterCandidate(t.entries[i].dist, t.entries[i].id, t.entries[i-1].dist, t.entries[i-1].id) {
				t.entries[i], t.entries[i-1] = t.entries[i-1], t.entries[i]
			} else {
				break
			}
		}
	} else if betterCandidate(d, id, t.entries[4].dist, t.entries[4].id) {
		t.entries[4].dist = d
		t.entries[4].id = id
		t.entries[4].label = label
		// bubble up
		for i := 4; i > 0; i-- {
			if betterCandidate(t.entries[i].dist, t.entries[i].id, t.entries[i-1].dist, t.entries[i-1].id) {
				t.entries[i], t.entries[i-1] = t.entries[i-1], t.entries[i]
			} else {
				break
			}
		}
	}
}

func (t *top5) worstDist() int32 {
	if t.count < 5 {
		return int32(0x7FFFFFFF)
	}
	return t.entries[4].dist
}

// FraudCount returns how many of the 5 nearest neighbours are frauds.
func (idx *Index) FraudCount(query [14]float32) (int, int) {
	if len(idx.vecs) == 0 {
		return 0, 0
	}

	qInt := dataset.QuantizeVec(query)

	nprobe := idx.nprobe
	if nprobe <= 0 || nprobe > len(idx.centroids) {
		nprobe = len(idx.centroids)
	}

	// 1. Keep only the closest nprobe centroids. Full sorting 1024 centroids on
	// every request costs more than this fixed-size insertion list.
	var cDists [32]centroidDist
	if nprobe > len(cDists) {
		nprobe = len(cDists)
	}
	topCount := 0
	for c := 0; c < len(idx.centroids); c++ {
		v := idx.centroids[c]
		d := euclidSq8(qInt, v)

		if topCount < nprobe {
			i := topCount
			topCount++
			for i > 0 && d < cDists[i-1].dist {
				cDists[i] = cDists[i-1]
				i--
			}
			cDists[i] = centroidDist{id: c, dist: d}
			continue
		}
		if d >= cDists[nprobe-1].dist {
			continue
		}
		i := nprobe - 1
		for i > 0 && d < cDists[i-1].dist {
			cDists[i] = cDists[i-1]
			i--
		}
		cDists[i] = centroidDist{id: c, dist: d}
	}

	// 3. Scan clusters with an unrolled int8 distance kernel and early abandon.
	var t5 top5
	maxDist := int32(0x7FFFFFFF)

	for i := 0; i < topCount; i++ {
		c := cDists[i].id
		idx.scanCluster(idx.clusters[c], qInt, &t5, &maxDist)
	}

	if idx.repair && len(idx.bboxMin) == len(idx.centroids) && len(idx.bboxMax) == len(idx.centroids) {
		for c := range idx.centroids {
			if probedCluster(c, cDists[:topCount]) {
				continue
			}
			bound := idx.clusters[c]
			if bound.End <= bound.Start {
				continue
			}
			if idx.bboxLowerBound(qInt, c) <= maxDist {
				idx.scanCluster(bound, qInt, &t5, &maxDist)
			}
		}
	}

	// 4. Return fraud count.
	frauds := 0
	for i := 0; i < t5.count; i++ {
		if t5.entries[i].label {
			frauds++
		}
	}
	return frauds, t5.count
}

func probedCluster(c int, probed []centroidDist) bool {
	for i := range probed {
		if probed[i].id == c {
			return true
		}
	}
	return false
}

func (idx *Index) scanCluster(bound clusterBound, q [14]int8, t5 *top5, maxDist *int32) {
	q0 := int32(q[0])
	q1 := int32(q[1])
	q2 := int32(q[2])
	q3 := int32(q[3])
	q4 := int32(q[4])
	q5 := int32(q[5])
	q6 := int32(q[6])
	q7 := int32(q[7])
	q8 := int32(q[8])
	q9 := int32(q[9])
	q10 := int32(q[10])
	q11 := int32(q[11])
	q12 := int32(q[12])
	q13 := int32(q[13])

	for j := bound.Start; j < bound.End; j++ {
		v := idx.vecs[j]

		x0 := q0 - int32(v[0])
		x1 := q1 - int32(v[1])
		x2 := q2 - int32(v[2])
		x3 := q3 - int32(v[3])
		d := x0*x0 + x1*x1 + x2*x2 + x3*x3
		if d >= *maxDist {
			continue
		}

		x4 := q4 - int32(v[4])
		x5 := q5 - int32(v[5])
		x6 := q6 - int32(v[6])
		x7 := q7 - int32(v[7])
		d += x4*x4 + x5*x5 + x6*x6 + x7*x7
		if d >= *maxDist {
			continue
		}

		x8 := q8 - int32(v[8])
		x9 := q9 - int32(v[9])
		x10 := q10 - int32(v[10])
		x11 := q11 - int32(v[11])
		d += x8*x8 + x9*x9 + x10*x10 + x11*x11
		if d >= *maxDist {
			continue
		}

		x12 := q12 - int32(v[12])
		x13 := q13 - int32(v[13])
		d += x12*x12 + x13*x13

		id := uint32(j)
		if int(j) < len(idx.origIDs) {
			id = idx.origIDs[j]
		}
		if betterCandidate(d, id, *maxDist, 0xFFFFFFFF) {
			t5.push(d, id, idx.labels[j])
			*maxDist = t5.worstDist()
		}
	}
}

func (idx *Index) bboxLowerBound(q [14]int8, c int) int32 {
	mins := idx.bboxMin[c]
	maxs := idx.bboxMax[c]
	var sum int32
	for d := range 14 {
		var diff int32
		if q[d] < mins[d] {
			diff = int32(mins[d]) - int32(q[d])
		} else if q[d] > maxs[d] {
			diff = int32(q[d]) - int32(maxs[d])
		}
		sum += diff * diff
	}
	return sum
}
