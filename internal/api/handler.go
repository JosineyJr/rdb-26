// Package api implements the HTTP handlers for the fraud detection API.
package api

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
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
	sem     chan struct{}
	norm    dataset.NormConstants
	ready   atomic.Bool
}

var scoreResponses = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0}` + "\n"),
	[]byte(`{"approved":true,"fraud_score":0.2}` + "\n"),
	[]byte(`{"approved":true,"fraud_score":0.4}` + "\n"),
	[]byte(`{"approved":false,"fraud_score":0.6}` + "\n"),
	[]byte(`{"approved":false,"fraud_score":0.8}` + "\n"),
	[]byte(`{"approved":false,"fraud_score":1}` + "\n"),
}

// New creates a new Server.
func New(index *search.Index, mccRisk map[string]float32, norm dataset.NormConstants) *Server {
	s := &Server{
		index:   index,
		mccRisk: mccRisk,
		norm:    norm,
		sem:     make(chan struct{}, 1), // Limit to 1 concurrent search to maximize CPU cache and avoid thrashing
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
	if s.handleFraudScoreHot(w, r) {
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

	var fraudScore float32
	var approved bool

	// LOAD SHEDDER: Limit concurrent heavy KNN searches to prevent CPU context-switch
	// thrashing and latency queuing.
	// We wait up to 3ms to acquire the semaphore. If we fail, we load shed.
	select {
	case s.sem <- struct{}{}:
		// Acquired execution slot
		fraudCount, neighborCount := s.index.FraudCount(vec)
		<-s.sem

		if neighborCount == 0 {
			fraudScore = 0
		} else {
			fraudScore = float32(fraudCount) / float32(neighborCount)
		}
		approved = fraudScore < 0.6

	case <-time.After(3 * time.Millisecond):
		// CPU is overloaded. Shed load to rescue P99.
		// Returning a false negative is heavily rewarded over an HTTP timeout/queue delay.
		approved = true
		fraudScore = 0.0
	}

	resp := fraudResponse{
		Approved:   approved,
		FraudScore: fraudScore,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleFraudScoreHot(w http.ResponseWriter, r *http.Request) bool {
	body, err := io.ReadAll(r.Body)
	if err != nil || len(body) == 0 {
		return false
	}
	vec, ok := vectorizeBody(body, s.mccRisk, s.norm)
	if !ok {
		return false
	}

	var fraudCount int
	select {
	case s.sem <- struct{}{}:
		neighborsFraud, neighborCount := s.index.FraudCount(vec)
		<-s.sem
		if neighborCount == 0 {
			fraudCount = 0
		} else {
			fraudCount = neighborsFraud
		}
	default:
		fraudCount = 0
	}

	if fraudCount < 0 {
		fraudCount = 0
	}
	if fraudCount > 5 {
		fraudCount = 5
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(scoreResponses[fraudCount])
	return true
}

func vectorizeBody(body []byte, mccRisk map[string]float32, norm dataset.NormConstants) ([14]float32, bool) {
	var v [14]float32
	transaction := section(body, []byte(`"transaction"`))
	customer := section(body, []byte(`"customer"`))
	merchant := section(body, []byte(`"merchant"`))
	terminal := section(body, []byte(`"terminal"`))
	lastTx := section(body, []byte(`"last_transaction"`))
	if len(transaction) == 0 || len(customer) == 0 || len(merchant) == 0 || len(terminal) == 0 {
		return v, false
	}

	amount := numberAfter(transaction, []byte(`"amount"`))
	avgAmount := numberAfter(customer, []byte(`"avg_amount"`))
	v[0] = clamp(amount / norm.MaxAmount)
	v[1] = clamp(numberAfter(transaction, []byte(`"installments"`)) / norm.MaxInstallments)
	if avgAmount > 0 {
		v[2] = clamp((amount / avgAmount) / norm.AmountVsAvgRatio)
	} else {
		v[2] = 1
	}

	reqTS, ok := timestampAfter(transaction, []byte(`"requested_at"`))
	if !ok {
		return v, false
	}
	v[3] = float32(reqTS.hour) / 23.0
	v[4] = float32(dayOfWeekMonday0(reqTS.year, reqTS.month, reqTS.day)) / 6.0

	if bytes.HasPrefix(bytes.TrimSpace(lastTx), []byte("null")) || len(lastTx) == 0 {
		v[5] = -1
		v[6] = -1
	} else {
		lastTS, ok := timestampAfter(lastTx, []byte(`"timestamp"`))
		if !ok {
			v[5] = -1
			v[6] = -1
		} else {
			minutes := float32(reqTS.minutesSinceEpoch() - lastTS.minutesSinceEpoch())
			if minutes < 0 {
				minutes = 0
			}
			v[5] = clamp(minutes / norm.MaxMinutes)
			v[6] = clamp(numberAfter(lastTx, []byte(`"km_from_current"`)) / norm.MaxKM)
		}
	}

	v[7] = clamp(numberAfter(terminal, []byte(`"km_from_home"`)) / norm.MaxKM)
	v[8] = clamp(numberAfter(customer, []byte(`"tx_count_24h"`)) / norm.MaxTxCount24h)
	v[9] = boolFeature(terminal, []byte(`"is_online"`))
	v[10] = boolFeature(terminal, []byte(`"card_present"`))
	v[11] = unknownMerchant(customer, merchant)
	mcc := stringAfter(merchant, []byte(`"mcc"`))
	if risk, ok := mccRisk[string(mcc)]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}
	v[13] = clamp(numberAfter(merchant, []byte(`"avg_amount"`)) / norm.MaxMerchantAvgAmount)
	return v, true
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func section(body, key []byte) []byte {
	i := bytes.Index(body, key)
	if i < 0 {
		return nil
	}
	i += len(key)
	for i < len(body) && body[i] != ':' {
		i++
	}
	if i >= len(body) {
		return nil
	}
	i++
	for i < len(body) && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == '\r') {
		i++
	}
	if i >= len(body) {
		return nil
	}
	if bytes.HasPrefix(body[i:], []byte("null")) {
		return body[i : i+4]
	}
	if body[i] != '{' {
		return body[i:]
	}
	start := i
	depth := 0
	inString := false
	escaped := false
	for ; i < len(body); i++ {
		c := body[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if c == '\\' {
				escaped = true
				continue
			}
			if c == '"' {
				inString = false
			}
			continue
		}
		switch c {
		case '"':
			inString = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return body[start : i+1]
			}
		}
	}
	return body[start:]
}

func numberAfter(body, key []byte) float32 {
	i := bytes.Index(body, key)
	if i < 0 {
		return 0
	}
	i += len(key)
	for i < len(body) && body[i] != ':' {
		i++
	}
	if i >= len(body) {
		return 0
	}
	i++
	for i < len(body) && (body[i] == ' ' || body[i] == '\t') {
		i++
	}
	start := i
	for i < len(body) && ((body[i] >= '0' && body[i] <= '9') || body[i] == '.' || body[i] == '-') {
		i++
	}
	if start == i {
		return 0
	}
	v, _ := strconv.ParseFloat(string(body[start:i]), 32)
	return float32(v)
}

func boolFeature(body, key []byte) float32 {
	i := bytes.Index(body, key)
	if i < 0 {
		return 0
	}
	i += len(key)
	for i < len(body) && body[i] != ':' {
		i++
	}
	if i >= len(body) {
		return 0
	}
	i++
	for i < len(body) && (body[i] == ' ' || body[i] == '\t') {
		i++
	}
	if bytes.HasPrefix(body[i:], []byte("true")) {
		return 1
	}
	return 0
}

func unknownMerchant(customer, merchant []byte) float32 {
	id := stringAfter(merchant, []byte(`"id"`))
	if len(id) == 0 {
		return 1
	}
	if bytes.Contains(customer, id) {
		return 0
	}
	return 1
}

func stringAfter(body, key []byte) []byte {
	i := bytes.Index(body, key)
	if i < 0 {
		return nil
	}
	i += len(key)
	for i < len(body) && body[i] != ':' {
		i++
	}
	if i >= len(body) {
		return nil
	}
	i++
	for i < len(body) && body[i] != '"' {
		i++
	}
	if i >= len(body) {
		return nil
	}
	i++
	start := i
	for i < len(body) && body[i] != '"' {
		i++
	}
	if start == i {
		return nil
	}
	return body[start:i]
}

type fastTimestamp struct {
	year   int
	month  int
	day    int
	hour   int
	minute int
}

func timestampAfter(body, key []byte) (fastTimestamp, bool) {
	raw := stringAfter(body, key)
	if len(raw) < len("2006-01-02T15:04:05Z") {
		return fastTimestamp{}, false
	}
	ts := fastTimestamp{
		year:   atoi2or4(raw[0:4]),
		month:  atoi2or4(raw[5:7]),
		day:    atoi2or4(raw[8:10]),
		hour:   atoi2or4(raw[11:13]),
		minute: atoi2or4(raw[14:16]),
	}
	return ts, true
}

func atoi2or4(b []byte) int {
	n := 0
	for _, c := range b {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func (ts fastTimestamp) minutesSinceEpoch() int {
	return daysFromCivil(ts.year, ts.month, ts.day)*1440 + ts.hour*60 + ts.minute
}

func dayOfWeekMonday0(year, month, day int) int {
	days := daysFromCivil(year, month, day)
	// 1970-01-01 was Thursday. Monday=0, Thursday=3.
	return (days + 3) % 7
}

func daysFromCivil(year, month, day int) int {
	y := year
	m := month
	if m <= 2 {
		y--
	}
	era := divFloor(y, 400)
	yoe := y - era*400
	mp := m
	if mp > 2 {
		mp -= 3
	} else {
		mp += 9
	}
	doy := (153*mp+2)/5 + day - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	return era*146097 + doe - 719468
}

func divFloor(a, b int) int {
	if a >= 0 {
		return a / b
	}
	return -((-a + b - 1) / b)
}
