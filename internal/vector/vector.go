// Package vector implements the 14-dimensional vectorization of a transaction payload.
package vector

import (
	"simd/archsimd"
	"slices"
	"time"

	"github.com/JosineyJr/rdb-26/internal/dataset"
)

// clamp restricts v to [0, 1].
func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// LastTransaction represents the optional previous transaction.
type LastTransaction struct {
	Timestamp     time.Time
	KmFromCurrent float32
}

// Request holds all inputs needed to build the 14-D vector.
type Request struct {
	RequestedAt time.Time

	// last transaction (nil if none)
	LastTx *LastTransaction

	// merchant
	MerchantID     string
	MCC            string
	KnownMerchants []string

	Installments int
	TxCount24h   int
	// transaction
	Amount float32

	// customer
	AvgAmount      float32
	MerchantAvgAmt float32

	KmFromHome float32

	// terminal
	IsOnline    bool
	CardPresent bool
}

// Vectorize converts a transaction request into the canonical 14-D float32 vector
// following the normalization rules from DETECTION_RULES.md.
func Vectorize(req Request, mccRisk map[string]float32, norm dataset.NormConstants) [14]float32 {
	var v [14]float32

	// [0] amount
	v[0] = clamp(req.Amount / norm.MaxAmount)

	// [1] installments
	v[1] = clamp(float32(req.Installments) / norm.MaxInstallments)

	// [2] amount_vs_avg  = clamp((amount / avg_amount) / amount_vs_avg_ratio)
	if req.AvgAmount > 0 {
		v[2] = clamp((req.Amount / req.AvgAmount) / norm.AmountVsAvgRatio)
	} else {
		v[2] = 1.0 // treat division by zero as max risk
	}

	// [3] hour_of_day: 0-23 UTC / 23
	hour := float32(req.RequestedAt.UTC().Hour())
	v[3] = hour / 23.0

	// [4] day_of_week: mon=0 … sun=6  / 6
	// Go's time.Weekday: Sunday=0, Monday=1, ..., Saturday=6
	// We need: Monday=0 ... Sunday=6
	wd := req.RequestedAt.UTC().Weekday() // 0=Sun..6=Sat
	dow := (int(wd) + 6) % 7              // shift so Mon=0, Sun=6
	v[4] = float32(dow) / 6.0

	// [5] minutes_since_last_tx  or -1
	// [6] km_from_last_tx        or -1
	if req.LastTx == nil {
		v[5] = -1
		v[6] = -1
	} else {
		diff := req.RequestedAt.Sub(req.LastTx.Timestamp)
		minutes := float32(diff.Minutes())
		if minutes < 0 {
			minutes = 0
		}
		v[5] = clamp(minutes / norm.MaxMinutes)
		v[6] = clamp(req.LastTx.KmFromCurrent / norm.MaxKM)
	}

	// [7] km_from_home
	v[7] = clamp(req.KmFromHome / norm.MaxKM)

	// [8] tx_count_24h
	v[8] = clamp(float32(req.TxCount24h) / norm.MaxTxCount24h)

	// [9] is_online
	if req.IsOnline {
		v[9] = 1
	}

	// [10] card_present
	if req.CardPresent {
		v[10] = 1
	}

	// [11] unknown_merchant: 1 if merchant NOT in known list (inverted: 1 = unknown)
	// slices.Contains returns true when found (= known), so we negate.
	if !slices.Contains(req.KnownMerchants, req.MerchantID) {
		v[11] = 1
	}

	// [12] mcc_risk
	risk, ok := mccRisk[req.MCC]
	if !ok {
		risk = 0.5 // default
	}
	v[12] = risk

	// [13] merchant_avg_amount
	v[13] = clamp(req.MerchantAvgAmt / norm.MaxMerchantAvgAmount)

	return v
}

// DotProduct computes the dot product of two vectors using AVX-512 SIMD.
func DotProduct(a, b []float32) float32 {
	var sum float32
	i := 0
	length := len(a)

	if length >= 16 {
		var sum16 archsimd.Float32x16
		loopEnd := length - length%16

		for ; i < loopEnd; i += 16 {
			va := archsimd.LoadFloat32x16Slice(a[i : i+16])
			vb := archsimd.LoadFloat32x16Slice(b[i : i+16])
			sum16 = sum16.Add(va.Mul(vb))
		}

		sum8 := sum16.GetLo().Add(sum16.GetHi())
		sum4 := sum8.GetLo().Add(sum8.GetHi())
		sum += sum4.GetElem(0) + sum4.GetElem(1) + sum4.GetElem(2) + sum4.GetElem(3)
	}

	for ; i < length; i++ {
		sum += a[i] * b[i]
	}

	return sum
}

// L2DistanceSquared computes the squared Euclidean distance using AVX-512 SIMD.
func L2DistanceSquared(a, b []float32) float32 {
	var sum float32
	i := 0
	length := len(a)

	if length >= 16 {
		var sum16 archsimd.Float32x16
		loopEnd := length - length%16

		for ; i < loopEnd; i += 16 {
			va := archsimd.LoadFloat32x16Slice(a[i : i+16])
			vb := archsimd.LoadFloat32x16Slice(b[i : i+16])
			diff := va.Sub(vb)
			sum16 = sum16.Add(diff.Mul(diff))
		}

		sum8 := sum16.GetLo().Add(sum16.GetHi())
		sum4 := sum8.GetLo().Add(sum8.GetHi())
		sum += sum4.GetElem(0) + sum4.GetElem(1) + sum4.GetElem(2) + sum4.GetElem(3)
	}

	for ; i < length; i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum
}
