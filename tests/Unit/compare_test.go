package unit_test

import (
	"errors"
	"math"
	"testing"

	face "github.com/hyz-is/arandu-face"
	onnx "github.com/hyz-is/arandu-onnx"
)

var sface = onnx.ModelInfo{ID: face.ModelSFace, Version: "2021dec-ba91a3b91d00", SHA256: "0ba9fbfa", Status: onnx.StatusDevOnly}

func embedding(model onnx.ModelInfo, v ...float32) face.Embedding {
	return face.Embedding{Vector: v, Dim: len(v), Model: model, Normalization: face.NormalizationL2}
}

func TestCompareMeasuresCosineAndDistance(t *testing.T) {
	t.Parallel()

	ref := face.SFaceReferenceThresholds()
	for _, tc := range []struct {
		name       string
		a, b       face.Embedding
		cosine, l2 float64
	}{
		{"identical", embedding(sface, 1, 0, 0), embedding(sface, 1, 0, 0), 1, 0},
		{"orthogonal", embedding(sface, 1, 0, 0), embedding(sface, 0, 1, 0), 0, math.Sqrt2},
		{"opposite", embedding(sface, 0, 0, 1), embedding(sface, 0, 0, -1), -1, 2},
		// Not unit length: Compare measures directions. (3,4)/5 · (4,3)/5 = 24/25.
		{"unnormalized", embedding(sface, 3, 4), embedding(sface, 4, 3), 0.96, math.Sqrt(2 - 2*0.96)},
	} {
		got, err := face.Compare(tc.a, tc.b, ref)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if !near(got.Cosine, tc.cosine, 1e-12) || !near(got.L2, tc.l2, 1e-12) {
			t.Errorf("%s: cosine %v L2 %v, want %v and %v", tc.name, got.Cosine, got.L2, tc.cosine, tc.l2)
		}
		if got.Thresholds != ref || got.Model != sface {
			t.Errorf("%s: the comparison carries %+v and %+v", tc.name, got.Thresholds, got.Model)
		}
	}
}

func TestCompareRefusesWhatItCannotRelate(t *testing.T) {
	t.Parallel()

	ref := face.SFaceReferenceThresholds()
	other := sface
	other.SHA256 = "another file"
	if _, err := face.Compare(embedding(sface, 1, 0), embedding(other, 1, 0), ref); !errors.Is(err, face.ErrModelMismatch) {
		t.Errorf("embeddings of two files = %v, want ErrModelMismatch", err)
	}
	stranger := onnx.ModelInfo{ID: "some_other_model", Version: "1", SHA256: "x"}
	if _, err := face.Compare(embedding(stranger, 1, 0), embedding(stranger, 1, 0), ref); !errors.Is(err, face.ErrModelMismatch) {
		t.Errorf("SFace thresholds on another model = %v, want ErrModelMismatch", err)
	}
	if _, err := face.Compare(embedding(sface, 0, 0), embedding(sface, 1, 0), ref); !errors.Is(err, face.ErrInvalidEmbedding) {
		t.Errorf("a zero vector = %v, want ErrInvalidEmbedding", err)
	}
	if _, err := face.Compare(embedding(sface, 1, 0), embedding(sface, 1, 0, 0), ref); !errors.Is(err, face.ErrInvalidEmbedding) {
		t.Errorf("two lengths = %v, want ErrInvalidEmbedding", err)
	}
	nan := embedding(sface, float32(math.NaN()), 1)
	if _, err := face.Compare(nan, embedding(sface, 1, 0), ref); !errors.Is(err, face.ErrInvalidEmbedding) {
		t.Errorf("a NaN = %v, want ErrInvalidEmbedding", err)
	}
	lying := embedding(sface, 1, 0)
	lying.Dim = 128
	if _, err := face.Compare(lying, lying, ref); !errors.Is(err, face.ErrInvalidEmbedding) {
		t.Errorf("a Dim that is not the length = %v, want ErrInvalidEmbedding", err)
	}
	unnamed := ref
	unnamed.Version = ""
	if _, err := face.Compare(embedding(sface, 1, 0), embedding(sface, 1, 0), unnamed); err == nil {
		t.Error("thresholds without a version were accepted")
	}
}

func TestTheSFaceReferenceThresholds(t *testing.T) {
	t.Parallel()

	ref := face.SFaceReferenceThresholds()
	if ref.Version != "sface-opencv-2021dec" || ref.Model != face.ModelSFace || ref.CosineReference != 0.363 || ref.L2Reference != 1.128 {
		t.Errorf("reference thresholds = %+v", ref)
	}
	if err := ref.Validate(); err != nil {
		t.Error(err)
	}
}

func TestL2Normalize(t *testing.T) {
	t.Parallel()

	v, norm, err := face.L2Normalize([]float32{3, 4})
	if err != nil || norm != 5 || !near(float64(v[0]), 0.6, 1e-7) || !near(float64(v[1]), 0.8, 1e-7) {
		t.Errorf("L2Normalize(3, 4) = %v, %v, %v", v, norm, err)
	}
	if _, _, err := face.L2Normalize([]float32{0, 0}); !errors.Is(err, face.ErrInvalidEmbedding) {
		t.Errorf("a zero vector = %v", err)
	}
}
