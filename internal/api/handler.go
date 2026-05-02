// Package api implements the HTTP handlers for the fraud detection API.
package api

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/JosineyJr/rdb-26/internal/dataset"
	"github.com/JosineyJr/rdb-26/internal/search"
	"github.com/JosineyJr/rdb-26/internal/vector"
)

// Server holds the shared state for all HTTP handlers.
type Server struct {
	index   *search.Index
	mccRisk map[string]float32
	norm    dataset.NormConstants
	ready   atomic.Bool
}

// New creates a new Server.
func New(index *search.Index, mccRisk map[string]float32, norm dataset.NormConstants) *Server {
	s := &Server{
		index:   index,
		mccRisk: mccRisk,
		norm:    norm,
	}
	return s
}

// SetIndex injects the HNSW index once it has been built.
func (s *Server) SetIndex(index *search.Index) {
	s.index = index
}

// SetReady marks the server as ready to receive traffic.
func (s *Server) SetReady() {
	s.ready.Store(true)
}

// RegisterRoutes attaches the handlers to the provided mux.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("POST /fraud-score", s.handleFraudScore)
}

// handleReady answers the health-check. Returns 200 once the index is built.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

// --- request / response types ---

type fraudRequest struct {
	LastTx      *lastTxBlock `json:"last_transaction"`
	ID          string       `json:"id"`
	Merchant    merchBlock   `json:"merchant"`
	Transaction txBlock      `json:"transaction"`
	Customer    custBlock    `json:"customer"`
	Terminal    termBlock    `json:"terminal"`
}

type txBlock struct {
	RequestedAt  string  `json:"requested_at"`
	Installments int     `json:"installments"`
	Amount       float32 `json:"amount"`
}

type custBlock struct {
	KnownMerchants []string `json:"known_merchants"`
	TxCount24h     int      `json:"tx_count_24h"`
	AvgAmount      float32  `json:"avg_amount"`
}

type merchBlock struct {
	ID        string  `json:"id"`
	MCC       string  `json:"mcc"`
	AvgAmount float32 `json:"avg_amount"`
}

type termBlock struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float32 `json:"km_from_home"`
}

type lastTxBlock struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float32 `json:"km_from_current"`
}

type fraudResponse struct {
	Approved   bool    `json:"approved"`
	FraudScore float32 `json:"fraud_score"`
}

// handleFraudScore is the core detection endpoint.
func (s *Server) handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
		return
	}

	// Decode request
	var req fraudRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	// Parse timestamps
	requestedAt, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		http.Error(w, "invalid requested_at", http.StatusBadRequest)
		return
	}

	// Build vector.Request
	vreq := vector.Request{
		Amount:         req.Transaction.Amount,
		Installments:   req.Transaction.Installments,
		RequestedAt:    requestedAt,
		AvgAmount:      req.Customer.AvgAmount,
		TxCount24h:     req.Customer.TxCount24h,
		KnownMerchants: req.Customer.KnownMerchants,
		MerchantID:     req.Merchant.ID,
		MCC:            req.Merchant.MCC,
		MerchantAvgAmt: req.Merchant.AvgAmount,
		IsOnline:       req.Terminal.IsOnline,
		CardPresent:    req.Terminal.CardPresent,
		KmFromHome:     req.Terminal.KmFromHome,
	}

	if req.LastTx != nil {
		ts, err := time.Parse(time.RFC3339, req.LastTx.Timestamp)
		if err == nil {
			vreq.LastTx = &vector.LastTransaction{
				Timestamp:     ts,
				KmFromCurrent: req.LastTx.KmFromCurrent,
			}
		}
		// If timestamp parse fails, treat as null (sentinel -1 for dims 5 & 6)
	}

	// Vectorize
	vec := vector.Vectorize(vreq, s.mccRisk, s.norm)

	// KNN search (k=5)
	neighbors := s.index.KNN(vec, 5)

	// Compute fraud_score
	var fraudCount float32
	for _, isFraud := range neighbors {
		if isFraud {
			fraudCount++
		}
	}
	fraudScore := fraudCount / float32(len(neighbors))
	if len(neighbors) == 0 {
		fraudScore = 0
	}

	resp := fraudResponse{
		Approved:   fraudScore < 0.6,
		FraudScore: fraudScore,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}
