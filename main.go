package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
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

	mccPath := filepath.Join(resourcesDir, "mcc_risk.json")
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

	go func() {
		log.Printf("HTTP server listening on :%s", port)
		httpServer := &http.Server{
			Addr:              ":" + port,
			Handler:           mux,
			ReadHeaderTimeout: 2 * time.Second,
		}
		if err := httpServer.ListenAndServe(); err != nil {
			log.Fatalf("http server error: %v", err)
		}
	}()

	// ------------------------------------------------------------------
	// ------------------------------------------------------------------
	// Load Pre-computed Binary Index (Instantaneous)
	// ------------------------------------------------------------------
	log.Println("Loading pre-computed binary index from resources/index.bin...")
	t0 := time.Now()
	index, err := search.Load("resources/index.bin")
	if err != nil {
		log.Fatalf("failed to load binary index: %v", err)
	}
	if repair, err := strconv.ParseBool(os.Getenv("IVF_REPAIR")); err == nil {
		index.SetRepair(repair)
	}
	if nprobe, err := strconv.Atoi(os.Getenv("IVF_NPROBE")); err == nil {
		index.SetNProbe(nprobe)
	}
	log.Printf("Index loaded successfully in %v", time.Since(t0))

	// ------------------------------------------------------------------
	// Wire index into server and mark ready
	// ------------------------------------------------------------------
	srv.SetIndex(index)

	srv.SetReady()
	log.Printf("Server ready on :%s — total startup time %v", port, time.Since(t0))

	// Block forever
	select {}
}

// Ensure the binary exits cleanly on signal (optional, good practice).
func init() {
	fmt.Println("Rinha de Backend 2026 – Fraud Detection (Go)")
}
