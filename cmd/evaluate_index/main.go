package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"time"

	"github.com/JosineyJr/rdb-26/internal/dataset"
	"github.com/JosineyJr/rdb-26/internal/search"
	"github.com/JosineyJr/rdb-26/internal/vector"
)

type testFile struct {
	Stats   datasetStats `json:"stats"`
	Entries []testEntry  `json:"entries"`
}

type datasetStats struct {
	Total         int     `json:"total"`
	FraudCount    int     `json:"fraud_count"`
	LegitCount    int     `json:"legit_count"`
	FraudRate     float64 `json:"fraud_rate"`
	LegitRate     float64 `json:"legit_rate"`
	EdgeCaseCount int     `json:"edge_case_count"`
	EdgeCaseRate  float64 `json:"edge_case_rate"`
}

type testEntry struct {
	Request          fraudRequest `json:"request"`
	ExpectedApproved bool         `json:"expected_approved"`
}

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

type result struct {
	IndexPath string       `json:"index_path"`
	Expected  datasetStats `json:"expected"`
	Latency   latencyStats `json:"latency"`
	Scoring   scoring      `json:"scoring"`
}

type latencyStats struct {
	TotalMS float64 `json:"total_ms"`
	P50MS   float64 `json:"p50_ms"`
	P95MS   float64 `json:"p95_ms"`
	P99MS   float64 `json:"p99_ms"`
	RPS     float64 `json:"rps"`
}

type scoring struct {
	Breakdown          breakdown `json:"breakdown"`
	FailureRate        string    `json:"failure_rate"`
	WeightedErrorsE    int       `json:"weighted_errors_E"`
	ErrorRateEpsilon   float64   `json:"error_rate_epsilon"`
	P99Score           scorePart `json:"p99_score"`
	DetectionScore     detPart   `json:"detection_score"`
	FinalScore         float64   `json:"final_score"`
	PassingSuccessGate bool      `json:"passing_success_gate"`
}

type breakdown struct {
	FalsePositiveDetections int `json:"false_positive_detections"`
	FalseNegativeDetections int `json:"false_negative_detections"`
	TruePositiveDetections  int `json:"true_positive_detections"`
	TrueNegativeDetections  int `json:"true_negative_detections"`
	HTTPErrors              int `json:"http_errors"`
}

type scorePart struct {
	Value        float64 `json:"value"`
	CutTriggered bool    `json:"cut_triggered"`
}

type detPart struct {
	Value           float64  `json:"value"`
	RateComponent   *float64 `json:"rate_component"`
	AbsolutePenalty *float64 `json:"absolute_penalty"`
	CutTriggered    bool     `json:"cut_triggered"`
}

func main() {
	indexPath := flag.String("index", "resources/index.bin", "path to binary IVF index")
	testDataPath := flag.String("test-data", "/Users/SPEBR3314/Documents/Laboratory/rinha/rinha-de-backend-2026/test/test-data.json", "path to preview test-data.json")
	mccPath := flag.String("mcc", "resources/mcc_risk.json", "path to mcc_risk.json")
	normPath := flag.String("normalization", "resources/normalization.json", "path to normalization.json")
	minScore := flag.Float64("min-score", 2424.14, "minimum final score for the success gate")
	maxFailureRate := flag.Float64("max-failure-rate", 0.01, "maximum raw failure rate for the success gate")
	repair := flag.Bool("repair", false, "enable bounding-box repair after the initial nprobe cluster scan")
	nprobe := flag.Int("nprobe", 0, "override the index nprobe for tuning")
	flag.Parse()

	norm, err := dataset.LoadNormalization(*normPath)
	if err != nil {
		log.Fatalf("load normalization: %v", err)
	}
	mccRisk, err := dataset.LoadMCCRisk(*mccPath)
	if err != nil {
		log.Fatalf("load mcc risk: %v", err)
	}
	idx, err := search.Load(*indexPath)
	if err != nil {
		log.Fatalf("load index: %v", err)
	}
	idx.SetRepair(*repair)
	idx.SetNProbe(*nprobe)

	dataBytes, err := os.ReadFile(*testDataPath)
	if err != nil {
		log.Fatalf("read test data: %v", err)
	}
	var tf testFile
	if err := json.Unmarshal(dataBytes, &tf); err != nil {
		log.Fatalf("decode test data: %v", err)
	}

	res := evaluate(*indexPath, tf, idx, mccRisk, norm, *minScore, *maxFailureRate)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(res); err != nil {
		log.Fatalf("encode result: %v", err)
	}
}

func evaluate(indexPath string, tf testFile, idx *search.Index, mccRisk map[string]float32, norm dataset.NormConstants, minScore, maxFailureRate float64) result {
	var b breakdown
	durations := make([]float64, 0, len(tf.Entries))

	started := time.Now()
	for i := range tf.Entries {
		entry := tf.Entries[i]
		t0 := time.Now()
		approved := classify(entry.Request, idx, mccRisk, norm)
		durations = append(durations, float64(time.Since(t0).Microseconds())/1000.0)

		if approved == entry.ExpectedApproved {
			if approved {
				b.TrueNegativeDetections++
			} else {
				b.TruePositiveDetections++
			}
		} else if approved {
			b.FalseNegativeDetections++
		} else {
			b.FalsePositiveDetections++
		}
	}
	totalElapsed := time.Since(started).Seconds()
	sort.Float64s(durations)

	n := b.TruePositiveDetections + b.TrueNegativeDetections + b.FalsePositiveDetections + b.FalseNegativeDetections + b.HTTPErrors
	e := b.FalsePositiveDetections + 3*b.FalseNegativeDetections + 5*b.HTTPErrors
	failures := b.FalsePositiveDetections + b.FalseNegativeDetections + b.HTTPErrors
	failureRate := 0.0
	epsilon := 0.0
	if n > 0 {
		failureRate = float64(failures) / float64(n)
		epsilon = float64(e) / float64(n)
	}

	p99 := percentile(durations, 0.99)
	p99Part := scoreP99(p99)
	det := scoreDetection(e, failureRate, epsilon)
	final := p99Part.Value + det.Value

	return result{
		IndexPath: indexPath,
		Expected:  tf.Stats,
		Latency: latencyStats{
			TotalMS: round2(totalElapsed * 1000),
			P50MS:   round3(percentile(durations, 0.50)),
			P95MS:   round3(percentile(durations, 0.95)),
			P99MS:   round3(p99),
			RPS:     round2(float64(n) / totalElapsed),
		},
		Scoring: scoring{
			Breakdown:          b,
			FailureRate:        fmt.Sprintf("%.2f%%", failureRate*100),
			WeightedErrorsE:    e,
			ErrorRateEpsilon:   round6(epsilon),
			P99Score:           p99Part,
			DetectionScore:     det,
			FinalScore:         round2(final),
			PassingSuccessGate: b.HTTPErrors == 0 && failureRate <= maxFailureRate && final > minScore,
		},
	}
}

func classify(req fraudRequest, idx *search.Index, mccRisk map[string]float32, norm dataset.NormConstants) bool {
	requestedAt, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		return true
	}

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
	}

	vec := vector.Vectorize(vreq, mccRisk, norm)
	fraudCount, neighborCount := idx.FraudCount(vec)
	if neighborCount == 0 {
		return true
	}
	fraudScore := float32(fraudCount) / float32(neighborCount)
	return fraudScore < 0.6
}

func scoreP99(p99 float64) scorePart {
	const (
		k        = 1000.0
		tMaxMS   = 1000.0
		p99MinMS = 1.0
		p99MaxMS = 2000.0
	)

	if p99 <= 0 {
		return scorePart{Value: 0}
	}
	if p99 > p99MaxMS {
		return scorePart{Value: -3000, CutTriggered: true}
	}
	return scorePart{Value: round2(k * math.Log10(tMaxMS/math.Max(p99, p99MinMS)))}
}

func scoreDetection(e int, failureRate, epsilon float64) detPart {
	const (
		k          = 1000.0
		epsilonMin = 0.001
		beta       = 300.0
		cutoff     = 0.15
	)

	if failureRate > cutoff {
		return detPart{Value: -3000, CutTriggered: true}
	}

	rateComponent := round2(k * math.Log10(1/math.Max(epsilon, epsilonMin)))
	absolutePenalty := round2(-beta * math.Log10(1+float64(e)))
	return detPart{
		Value:           round2(rateComponent + absolutePenalty),
		RateComponent:   &rateComponent,
		AbsolutePenalty: &absolutePenalty,
	}
}

func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Floor(q * float64(len(sorted))))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

func round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func round6(v float64) float64 {
	return math.Round(v*1_000_000) / 1_000_000
}
