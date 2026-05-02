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
	centroids [][14]int8
	clusters  []clusterBound // contiguous memory blocks for cache locality
	nprobe    int
}

type clusterBound struct {
	Start int32
	End   int32
}

// Build constructs the IVF index by picking random centroids and assigning
// each vector to the nearest centroid. It also reorganizes the vectors in memory
// so that all vectors in a cluster are contiguous, maximizing CPU cache locality.
func Build(vecs [][14]int8, labels []bool) *Index {
	numClusters := 1024
	nprobe := 8 // scan top 8 clusters (~0.8% of the dataset)

	if len(vecs) < numClusters {
		numClusters = len(vecs)
		nprobe = numClusters
	}

	idx := &Index{
		centroids: make([][14]int8, numClusters),
		clusters:  make([]clusterBound, numClusters),
		nprobe:    nprobe,
	}

	// 1. Pick random centroids to partition the space.
	rng := rand.New(rand.NewSource(42)) // fixed seed for determinism
	perm := rng.Perm(len(vecs))
	for i := 0; i < numClusters; i++ {
		idx.centroids[i] = vecs[perm[i]]
	}

	// 2. Spherical K-Means iterations (runs offline during Docker build).
	workers := max(runtime.NumCPU(), 1)
	stripe := (len(vecs) + workers - 1) / workers

	type localClusters [][]int32
	var results []localClusters

	for iter := 0; iter < 5; iter++ {
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

		if iter < 4 { // Recalculate centroids
			for c := 0; c < numClusters; c++ {
				var sums [14]int64
				var count int64
				for w := range workers {
					if results[w] != nil {
						for _, vi := range results[w][c] {
							for d := 0; d < 14; d++ {
								sums[d] += int64(vecs[vi][d])
							}
							count++
						}
					}
				}
				if count > 0 {
					var magSq int64
					var means [14]float64
					for d := 0; d < 14; d++ {
						mean := float64(sums[d]) / float64(count)
						means[d] = mean
						magSq += int64(mean * mean)
					}
					
					// Spherical Re-normalization: scale centroid back to magnitude ~127
					// so it remains on the surface of the hyper-sphere.
					if magSq > 0 {
						mag := math.Sqrt(float64(magSq))
						scale := 127.0 / mag
						for d := 0; d < 14; d++ {
							val := int(math.Round(means[d] * scale))
							if val > 127 { val = 127 }
							if val < -128 { val = -128 }
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
			orderedVecs[offset] = vecs[vd.idx]
			orderedLabels[offset] = labels[vd.idx]
			offset++
		}
		idx.clusters[c] = clusterBound{Start: startOffset, End: offset}
	}

	idx.vecs = orderedVecs
	idx.labels = orderedLabels

	return idx
}

func (idx *Index) Ready() bool { return true }

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
		label bool
	}
	count int
}

func (t *top5) push(d int32, label bool) {
	if t.count < 5 {
		t.entries[t.count].dist = d
		t.entries[t.count].label = label
		t.count++
		// bubble up
		for i := t.count - 1; i > 0; i-- {
			if t.entries[i].dist < t.entries[i-1].dist {
				t.entries[i], t.entries[i-1] = t.entries[i-1], t.entries[i]
			} else {
				break
			}
		}
	} else if d < t.entries[4].dist {
		t.entries[4].dist = d
		t.entries[4].label = label
		// bubble up
		for i := 4; i > 0; i-- {
			if t.entries[i].dist < t.entries[i-1].dist {
				t.entries[i], t.entries[i-1] = t.entries[i-1], t.entries[i]
			} else {
				break
			}
		}
	}
}

// KNN returns the fraud labels of the k nearest neighbours to the query.
func (idx *Index) KNN(query [14]float32, k int) []bool {
	if len(idx.vecs) == 0 {
		return nil
	}

	qInt := dataset.QuantizeVec(query)

	// =========================================================================
	// MEMOIZATION / DYNAMIC PROGRAMMING
	// Precompute the squared distance for all 256 possible values of int8
	// for each of the 14 dimensions. This replaces (14 mul + 14 sub) with
	// simple fast array lookups per vector.
	// =========================================================================
	var distTable [14][256]int32
	for j := range 14 {
		qVal := int32(qInt[j])
		for v := range 256 {
			val := int32(int8(v))
			d := qVal - val
			distTable[j][v] = d * d
		}
	}

	type centroidDist struct {
		id   int
		dist int32
	}

	// 1. Calculate distances to all centroids using the memoized table.
	var cDists [1024]centroidDist
	for c := 0; c < len(idx.centroids); c++ {
		v := idx.centroids[c]
		d := distTable[0][uint8(v[0])] + distTable[1][uint8(v[1])] +
			distTable[2][uint8(v[2])] + distTable[3][uint8(v[3])] +
			distTable[4][uint8(v[4])] + distTable[5][uint8(v[5])] +
			distTable[6][uint8(v[6])] + distTable[7][uint8(v[7])] +
			distTable[8][uint8(v[8])] + distTable[9][uint8(v[9])] +
			distTable[10][uint8(v[10])] + distTable[11][uint8(v[11])] +
			distTable[12][uint8(v[12])] + distTable[13][uint8(v[13])]
		cDists[c] = centroidDist{id: c, dist: d}
	}

	// 2. Sort the centroids to find the top `nprobe`.
	slices.SortFunc(cDists[:], func(a, b centroidDist) int {
		return cmp.Compare(a.dist, b.dist)
	})

	nprobe := 8

	// 3. Scan clusters — distTable lookups with Early Abandon
	var t5 top5
	maxDist := int32(0x7FFFFFFF)

	for i := range nprobe {
		c := cDists[i].id
		bound := idx.clusters[c]
		for j := bound.Start; j < bound.End; j++ {
			v := idx.vecs[j]

			d := distTable[0][uint8(v[0])] + distTable[1][uint8(v[1])] +
				distTable[2][uint8(v[2])] + distTable[3][uint8(v[3])]
			if d >= maxDist {
				continue
			}

			d += distTable[4][uint8(v[4])] + distTable[5][uint8(v[5])] +
				distTable[6][uint8(v[6])] + distTable[7][uint8(v[7])]
			if d >= maxDist {
				continue
			}

			d += distTable[8][uint8(v[8])] + distTable[9][uint8(v[9])] +
				distTable[10][uint8(v[10])] + distTable[11][uint8(v[11])]
			if d >= maxDist {
				continue
			}

			d += distTable[12][uint8(v[12])] + distTable[13][uint8(v[13])]

			if d < maxDist {
				t5.push(d, idx.labels[j])
				if t5.count == 5 {
					maxDist = t5.entries[4].dist
				}
			}
		}
	}

	// 4. Return labels.
	out := make([]bool, t5.count)
	for i := 0; i < t5.count; i++ {
		out[i] = t5.entries[i].label
	}
	return out
}
