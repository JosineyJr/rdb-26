package search

import (
	"encoding/binary"
	"fmt"
	"os"
)

// Magic header to ensure we're reading the right file.
var magicHeader = []byte("RDB26IVF")

// Save writes the built index to a raw binary file for instant loading.
func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 1. Write Header
	if _, err := f.Write(magicHeader); err != nil {
		return err
	}

	// 2. Write metadata
	numC := int32(len(idx.centroids))
	numV := int32(len(idx.vecs))
	nprobe := int32(idx.nprobe)

	meta := []int32{numC, numV, nprobe}
	if err := binary.Write(f, binary.LittleEndian, meta); err != nil {
		return err
	}

	// 3. Write centroids
	if err := binary.Write(f, binary.LittleEndian, idx.centroids); err != nil {
		return err
	}

	// 4. Write clusterBounds
	if err := binary.Write(f, binary.LittleEndian, idx.clusters); err != nil {
		return err
	}

	// 5. Write vecs
	if err := binary.Write(f, binary.LittleEndian, idx.vecs); err != nil {
		return err
	}

	// 6. Write labels
	if err := binary.Write(f, binary.LittleEndian, idx.labels); err != nil {
		return err
	}

	return nil
}

// Load reads the pre-computed binary index directly into memory arrays.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// 1. Read and verify header
	header := make([]byte, len(magicHeader))
	if _, err := f.Read(header); err != nil {
		return nil, err
	}
	if string(header) != string(magicHeader) {
		return nil, fmt.Errorf("invalid magic header")
	}

	// 2. Read metadata
	var meta [3]int32
	if err := binary.Read(f, binary.LittleEndian, &meta); err != nil {
		return nil, err
	}
	numC := meta[0]
	numV := meta[1]
	nprobe := meta[2]

	idx := &Index{
		centroids: make([][14]int8, numC),
		clusters:  make([]clusterBound, numC),
		vecs:      make([][14]int8, numV),
		labels:    make([]bool, numV),
		nprobe:    int(nprobe),
	}

	// 3. Read centroids
	if err := binary.Read(f, binary.LittleEndian, idx.centroids); err != nil {
		return nil, err
	}

	// 4. Read clusterBounds
	if err := binary.Read(f, binary.LittleEndian, idx.clusters); err != nil {
		return nil, err
	}

	// 5. Read vecs
	if err := binary.Read(f, binary.LittleEndian, idx.vecs); err != nil {
		return nil, err
	}

	// 6. Read labels
	if err := binary.Read(f, binary.LittleEndian, idx.labels); err != nil {
		return nil, err
	}

	return idx, nil
}
