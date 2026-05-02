// Package dataset handles loading and parsing of the reference data files.
package dataset

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// NormConstants holds the constants used in the 14-dimension normalization formulas.
type NormConstants struct {
	MaxAmount            float32 `json:"max_amount"`
	MaxInstallments      float32 `json:"max_installments"`
	AmountVsAvgRatio     float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float32 `json:"max_minutes"`
	MaxKM                float32 `json:"max_km"`
	MaxTxCount24h        float32 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float32 `json:"max_merchant_avg_amount"`
}

// refRecord is one entry in references.json.gz as decoded from JSON.
type refRecord struct {
	Label  string      `json:"label"`
	Vector [14]float32 `json:"vector"`
}

// LoadNormalization reads normalization.json into NormConstants.
func LoadNormalization(path string) (NormConstants, error) {
	f, err := os.Open(path)
	if err != nil {
		return NormConstants{}, fmt.Errorf("open normalization: %w", err)
	}
	defer f.Close()

	var c NormConstants
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return NormConstants{}, fmt.Errorf("decode normalization: %w", err)
	}
	return c, nil
}

// LoadMCCRisk reads mcc_risk.json into a map[mcc]→risk_score.
func LoadMCCRisk(path string) (map[string]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mcc_risk: %w", err)
	}
	defer f.Close()

	var raw map[string]float32
	if err := json.NewDecoder(f).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode mcc_risk: %w", err)
	}
	return raw, nil
}

// QuantizeFloat converts a float32 value in [-1, 1] to int8 using a ×127 scale.
//
// The sentinel value -1.0 (used for missing last_transaction) maps to -127,
// which is clearly distinct from normal normalised values in [0, 127].
// Relative ordering of Euclidean distances is preserved, so KNN results are
// identical to float32 KNN.
func QuantizeFloat(f float32) int8 {
	v := f * 127.0
	if v >= 127.0 {
		return 127
	}
	if v <= -127.0 {
		return -127
	}
	return int8(v)
}

// QuantizeVec converts a 14-D float32 vector to int8.
func QuantizeVec(v [14]float32) [14]int8 {
	var q [14]int8
	for i, f := range v {
		q[i] = QuantizeFloat(f)
	}
	return q
}

// LoadReferences reads references.json.gz and returns the vectors (quantized to
// int8) and their fraud labels.
//
// Memory usage: 3,000,000 × 14 × 1 byte = 42 MB — vs 168 MB for float32.
// labels[i] == true means "fraud", false means "legit".
func LoadReferences(path string) ([][14]int8, []bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open references: %w", err)
	}
	defer f.Close()

	gr, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gr.Close()

	// Stream-decode to avoid materialising the full JSON in memory.
	dec := json.NewDecoder(io.Reader(gr))

	// Opening bracket
	if _, err := dec.Token(); err != nil {
		return nil, nil, fmt.Errorf("read opening bracket: %w", err)
	}

	vectors := make([][14]int8, 0, 3_000_000)
	labels := make([]bool, 0, 3_000_000)

	var rec refRecord
	for dec.More() {
		rec = refRecord{}
		if err := dec.Decode(&rec); err != nil {
			return nil, nil, fmt.Errorf("decode record: %w", err)
		}
		vectors = append(vectors, QuantizeVec(rec.Vector))
		labels = append(labels, rec.Label == "fraud")
	}

	return vectors, labels, nil
}
