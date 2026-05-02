package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/JosineyJr/rdb-26/internal/api"
	"github.com/JosineyJr/rdb-26/internal/dataset"
	"github.com/JosineyJr/rdb-26/internal/search"
)

func main() {
	// Set GOMAXPROCS to 1 because each API container is limited to 0.45 CPU.
	// Using NumCPU() (e.g., 8) causes heavy context-switching and OS throttling.
	runtime.GOMAXPROCS(1)

	// ------------------------------------------------------------------
	// Resolve resource paths
	// ------------------------------------------------------------------
	// Default: look for a ./resources directory next to the binary.
	// Override with RESOURCES_DIR env var if needed.
	resourcesDir := os.Getenv("RESOURCES_DIR")
	if resourcesDir == "" {
		resourcesDir = "./resources"
	}

	refPath  := filepath.Join(resourcesDir, "references.json.gz")
	mccPath  := filepath.Join(resourcesDir, "mcc_risk.json")
	normPath := filepath.Join(resourcesDir, "normalization.json")

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// ------------------------------------------------------------------
	// Load static files (mcc_risk, normalization)
	// ------------------------------------------------------------------
	log.Println("Loading normalization constants...")
	norm, err := dataset.LoadNormalization(normPath)
	if err != nil {
		log.Fatalf("failed to load normalization: %v", err)
	}

	log.Println("Loading MCC risk table...")
	mccRisk, err := dataset.LoadMCCRisk(mccPath)
	if err != nil {
		log.Fatalf("failed to load mcc_risk: %v", err)
	}

	// ------------------------------------------------------------------
	// Start HTTP server early so /ready can respond 503 during startup.
	// ------------------------------------------------------------------
	srv := api.New(nil, mccRisk, norm) // index not ready yet

	mux := http.NewServeMux()
	srv.RegisterRoutes(mux)

	httpServer := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
	}

	go func() {
		log.Printf("HTTP server listening on :%s", port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("http server error: %v", err)
		}
	}()

	// ------------------------------------------------------------------
	// Load reference vectors (the big file — 3M records)
	// ------------------------------------------------------------------
	log.Println("Loading reference vectors (this may take a while)...")
	t0 := time.Now()
	vecs, labels, err := dataset.LoadReferences(refPath)
	if err != nil {
		log.Fatalf("failed to load references: %v", err)
	}
	log.Printf("Loaded %d reference vectors in %v", len(vecs), time.Since(t0))

	// ------------------------------------------------------------------
	// Build KNN index (brute-force — instantaneous, no construction needed)
	// ------------------------------------------------------------------
	index := search.Build(vecs, labels)

	// ------------------------------------------------------------------
	// Wire index into server and mark ready
	// ------------------------------------------------------------------
	srv.SetIndex(index)
	
	// CRITICAL: The main() function blocks forever on select{}.
	// If we don't nil these variables, the original 45MB of startup data
	// stays pinned in the stack frame forever, causing massive GC pressure
	// when Nginx sends requests!
	vecs = nil
	labels = nil
	runtime.GC()
	debug.FreeOSMemory()

	srv.SetReady()
	log.Printf("Server ready on :%s — total startup time %v", port, time.Since(t0))

	// Block forever
	select {}
}

// Ensure the binary exits cleanly on signal (optional, good practice).
func init() {
	fmt.Println("Rinha de Backend 2026 – Fraud Detection (Go)")
}
