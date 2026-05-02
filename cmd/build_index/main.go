package main

import (
	"log"
	"time"

	"github.com/JosineyJr/rdb-26/internal/dataset"
	"github.com/JosineyJr/rdb-26/internal/search"
)

func main() {
	log.Println("Loading reference vectors...")
	t0 := time.Now()
	vecs, labels, err := dataset.LoadReferences("resources/references.json.gz")
	if err != nil {
		log.Fatalf("failed to load references: %v", err)
	}
	log.Printf("Loaded %d vectors in %v", len(vecs), time.Since(t0))

	log.Println("Building K-Means clusters (this will use all available CPU cores)...")
	t1 := time.Now()
	idx := search.Build(vecs, labels)
	log.Printf("Built clusters in %v", time.Since(t1))

	log.Println("Saving binary index to resources/index.bin...")
	t2 := time.Now()
	if err := idx.Save("resources/index.bin"); err != nil {
		log.Fatalf("failed to save index: %v", err)
	}
	log.Printf("Index saved successfully in %v", time.Since(t2))
}
