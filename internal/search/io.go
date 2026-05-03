package search

import (
	"encoding/binary"
	"fmt"
	"os"
)

// Magic headers to ensure we're reading the right file.
var magicHeaderV1 = []byte("RDB26IVF")
var magicHeaderV2 = []byte("RDB26IV2")

// Save writes the built index to a raw binary file for instant loading.
func (idx *Index) Save(path string) error {
	return idx.saveCompact(path)
}

func (idx *Index) SaveWithMetadata(path string) error {
	return idx.saveWithMetadata(path)
}

func (idx *Index) saveCompact(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(magicHeaderV1); err != nil {
		return err
	}

	numC := int32(len(idx.centroids))
	numV := int32(len(idx.vecs))
	nprobe := int32(idx.nprobe)

	meta := []int32{numC, numV, nprobe}
	if err := binary.Write(f, binary.LittleEndian, meta); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.centroids); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.clusters); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.vecs); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.labels); err != nil {
		return err
	}

	return nil
}

func (idx *Index) saveWithMetadata(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 1. Write Header
	if _, err := f.Write(magicHeaderV2); err != nil {
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

	// 5. Write bounding boxes
	if err := binary.Write(f, binary.LittleEndian, idx.bboxMin); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, idx.bboxMax); err != nil {
		return err
	}

	// 6. Write vecs
	if err := binary.Write(f, binary.LittleEndian, idx.vecs); err != nil {
		return err
	}

	// 7. Write labels
	if err := binary.Write(f, binary.LittleEndian, idx.labels); err != nil {
		return err
	}

	// 8. Write original reference ids
	if err := binary.Write(f, binary.LittleEndian, idx.origIDs); err != nil {
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
	header := make([]byte, len(magicHeaderV1))
	if _, err := f.Read(header); err != nil {
		return nil, err
	}
	version := string(header)
	if version != string(magicHeaderV1) && version != string(magicHeaderV2) {
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
	if version == string(magicHeaderV2) {
		idx.bboxMin = make([][14]int8, numC)
		idx.bboxMax = make([][14]int8, numC)
		idx.origIDs = make([]uint32, numV)
	}

	// 3. Read centroids
	if err := binary.Read(f, binary.LittleEndian, idx.centroids); err != nil {
		return nil, err
	}

	// 4. Read clusterBounds
	if err := binary.Read(f, binary.LittleEndian, idx.clusters); err != nil {
		return nil, err
	}

	if version == string(magicHeaderV2) {
		// 5. Read bounding boxes
		if err := binary.Read(f, binary.LittleEndian, idx.bboxMin); err != nil {
			return nil, err
		}
		if err := binary.Read(f, binary.LittleEndian, idx.bboxMax); err != nil {
			return nil, err
		}
	}

	// 6. Read vecs
	if err := binary.Read(f, binary.LittleEndian, idx.vecs); err != nil {
		return nil, err
	}

	// 7. Read labels
	if err := binary.Read(f, binary.LittleEndian, idx.labels); err != nil {
		return nil, err
	}

	if version == string(magicHeaderV2) {
		// 8. Read original reference ids
		if err := binary.Read(f, binary.LittleEndian, idx.origIDs); err != nil {
			return nil, err
		}
	}

	return idx, nil
}
