package main

import (
	"flag"
	"log"
	"time"

	"github.com/JosineyJr/rdb-26/internal/dataset"
	"github.com/JosineyJr/rdb-26/internal/search"
)

func main() {
	defaults := search.DefaultBuildOptions()
	referencesPath := flag.String("references", "resources/references.json.gz", "path to references.json.gz")
	outPath := flag.String("out", "resources/index.bin", "path for the generated binary IVF index")
	clusters := flag.Int("clusters", defaults.Clusters, "number of IVF clusters")
	nprobe := flag.Int("nprobe", defaults.NProbe, "number of closest clusters to scan per query")
	iterations := flag.Int("iterations", defaults.Iterations, "number of K-Means assignment iterations")
	seed := flag.Int64("seed", defaults.Seed, "deterministic random seed for centroid initialization")
	format := flag.String("format", "compact", "index format: compact or metadata")
	flag.Parse()

	log.Println("Loading reference vectors...")
	t0 := time.Now()
	vecs, labels, err := dataset.LoadReferences(*referencesPath)
	if err != nil {
		log.Fatalf("failed to load references: %v", err)
	}
	log.Printf("Loaded %d vectors in %v", len(vecs), time.Since(t0))

	opts := search.BuildOptions{
		Clusters:   *clusters,
		NProbe:     *nprobe,
		Iterations: *iterations,
		Seed:       *seed,
	}
	log.Printf("Building K-Means clusters with clusters=%d nprobe=%d iterations=%d seed=%d (this will use all available CPU cores)...",
		opts.Clusters, opts.NProbe, opts.Iterations, opts.Seed)
	t1 := time.Now()
	idx := search.BuildWithOptions(vecs, labels, opts)
	log.Printf("Built clusters in %v", time.Since(t1))

	log.Printf("Saving binary index to %s...", *outPath)
	t2 := time.Now()
	var saveErr error
	switch *format {
	case "metadata":
		saveErr = idx.SaveWithMetadata(*outPath)
	case "compact":
		saveErr = idx.Save(*outPath)
	default:
		log.Fatalf("unknown index format %q: use compact or metadata", *format)
	}
	if saveErr != nil {
		log.Fatalf("failed to save index: %v", saveErr)
	}
	log.Printf("Index saved successfully in %v", time.Since(t2))
}
