package face

import (
	"context"
	"errors"
	"fmt"
	"image"
	"math"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// NormalizationL2 is the normalization Embed applies: the vector divided by
// its Euclidean norm.
const NormalizationL2 = "l2"

// SFaceDim is the length of an SFace embedding.
const SFaceDim = 128

// The errors Compare reports. Each is wrapped with the detail of the case.
var (
	// ErrModelMismatch is two embeddings made by different models, or by
	// different files of one model, which no distance relates.
	ErrModelMismatch = errors.New("face: embeddings from different models")
	// ErrInvalidEmbedding is an embedding that cannot be measured: no values,
	// a value that is not finite, a zero vector, or a length other than Dim.
	ErrInvalidEmbedding = errors.New("face: invalid embedding")
)

// Embedding is one face's vector.
type Embedding struct {
	// Vector is the model's output divided by its Euclidean norm.
	Vector []float32
	// Dim is the vector's length.
	Dim int
	// Model identifies the model, file and runtime that produced it.
	Model onnx.ModelInfo
	// Normalization is how Vector was normalized: NormalizationL2.
	Normalization string
	// RawNorm is the Euclidean norm of the model's output before
	// normalization.
	RawNorm float64
}

// ThresholdSet is a versioned set of reference values for comparing
// embeddings of one model. It is a parameter returned beside a measurement,
// never a decision.
type ThresholdSet struct {
	// Version names this set of values. A change to any value is a new
	// version.
	Version string
	// Model is the identifier of the model the values were stated for.
	Model string
	// CosineReference is the cosine similarity the source states as its
	// reference: at or above it, the source would call two faces the same.
	CosineReference float64
	// L2Reference is the Euclidean distance the source states as its
	// reference: at or below it, the source would call two faces the same.
	L2Reference float64
	// Source is where the values come from.
	Source string
}

// SFaceReferenceThresholds returns the reference values opencv_zoo publishes
// for SFace 2021dec: cosine 0.363 and L2 1.128. They are not calibrated on
// any population this package knows of.
func SFaceReferenceThresholds() ThresholdSet {
	return ThresholdSet{
		Version:         "sface-opencv-2021dec",
		Model:           ModelSFace,
		CosineReference: 0.363,
		L2Reference:     1.128,
		Source:          "opencv_zoo face_recognition_sface (FaceRecognizerSF reference thresholds)",
	}
}

// Validate reports a threshold set that cannot be returned as a reference.
func (t ThresholdSet) Validate() error {
	if t.Version == "" || t.Model == "" {
		return fmt.Errorf("face: a ThresholdSet needs a Version and a Model: a threshold nobody can name is one nobody can audit")
	}
	if math.IsNaN(t.CosineReference) || t.CosineReference < -1 || t.CosineReference > 1 {
		return fmt.Errorf("face: ThresholdSet.CosineReference is %v, and has to be between -1 and 1", t.CosineReference)
	}
	if math.IsNaN(t.L2Reference) || t.L2Reference < 0 || t.L2Reference > 2 {
		return fmt.Errorf("face: ThresholdSet.L2Reference is %v, and has to be between 0 and 2", t.L2Reference)
	}
	return nil
}

// Comparison is the measured distance between two embeddings, with the
// reference values that apply to them. It holds no verdict.
type Comparison struct {
	// Cosine is the cosine similarity of the two vectors, -1..1.
	Cosine float64
	// L2 is the Euclidean distance between the two unit vectors, 0..2.
	L2 float64
	// Model is the model both embeddings came from.
	Model onnx.ModelInfo
	// Thresholds are the reference values for that model, with their
	// version.
	Thresholds ThresholdSet
}

// Compare measures two embeddings of the same model: their cosine similarity
// and the Euclidean distance between their unit vectors. It refuses
// embeddings from different models or files (ErrModelMismatch), a threshold
// set stated for another model, and vectors it cannot measure
// (ErrInvalidEmbedding).
func Compare(a, b Embedding, thresholds ThresholdSet) (Comparison, error) {
	if a.Model.ID != b.Model.ID || a.Model.Version != b.Model.Version || a.Model.SHA256 != b.Model.SHA256 {
		return Comparison{}, fmt.Errorf("%w: %s %s (%s) and %s %s (%s)", ErrModelMismatch,
			a.Model.ID, a.Model.Version, a.Model.SHA256, b.Model.ID, b.Model.Version, b.Model.SHA256)
	}
	if err := thresholds.Validate(); err != nil {
		return Comparison{}, err
	}
	if thresholds.Model != a.Model.ID {
		return Comparison{}, fmt.Errorf("%w: the thresholds %s are stated for %s and the embeddings come from %s",
			ErrModelMismatch, thresholds.Version, thresholds.Model, a.Model.ID)
	}
	na, err := unitVector(a)
	if err != nil {
		return Comparison{}, err
	}
	nb, err := unitVector(b)
	if err != nil {
		return Comparison{}, err
	}
	if len(na) != len(nb) {
		return Comparison{}, fmt.Errorf("%w: lengths %d and %d", ErrInvalidEmbedding, len(na), len(nb))
	}
	dot, sq := 0.0, 0.0
	for i := range na {
		dot += na[i] * nb[i]
		d := na[i] - nb[i]
		sq += d * d
	}
	return Comparison{
		Cosine:     math.Max(-1, math.Min(1, dot)),
		L2:         math.Sqrt(sq),
		Model:      a.Model,
		Thresholds: thresholds,
	}, nil
}

// unitVector is the embedding's vector divided by its norm, in float64.
func unitVector(e Embedding) ([]float64, error) {
	if len(e.Vector) == 0 || len(e.Vector) != e.Dim {
		return nil, fmt.Errorf("%w: %d values and Dim %d", ErrInvalidEmbedding, len(e.Vector), e.Dim)
	}
	out := make([]float64, len(e.Vector))
	norm := 0.0
	for i, v := range e.Vector {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, fmt.Errorf("%w: value %d is %v", ErrInvalidEmbedding, i, f)
		}
		out[i] = f
		norm += f * f
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return nil, fmt.Errorf("%w: a zero vector has no direction", ErrInvalidEmbedding)
	}
	for i := range out {
		out[i] /= norm
	}
	return out, nil
}

// L2Normalize returns the vector divided by its Euclidean norm, and that
// norm. A zero or non-finite vector is refused.
func L2Normalize(v []float32) ([]float32, float64, error) {
	norm := 0.0
	for i, x := range v {
		f := float64(x)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return nil, 0, fmt.Errorf("%w: value %d is %v", ErrInvalidEmbedding, i, f)
		}
		norm += f * f
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return nil, 0, fmt.Errorf("%w: a zero vector has no direction", ErrInvalidEmbedding)
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out, norm, nil
}

// Embed turns a 112×112 aligned crop, as Align makes it, into an L2-normalized
// SFace embedding. The crop is given to the model as RGB 0..255; SFace
// normalizes inside its graph. It needs Config.EnableEmbedding.
func (m *Module) Embed(ctx context.Context, aligned image.Image) (Embedding, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	session, err := m.session(m.embedder, "Embed", "EnableEmbedding")
	if err != nil {
		return Embedding{}, err
	}
	if b := aligned.Bounds(); b.Dx() != AlignedSize || b.Dy() != AlignedSize {
		return Embedding{}, fmt.Errorf("face: Embed takes a %dx%d aligned crop, and was given %dx%d", AlignedSize, AlignedSize, b.Dx(), b.Dy())
	}
	result, err := session.Run(ctx, map[string]onnx.Tensor{"data": RawPlanes(aligned, false)})
	if err != nil {
		return Embedding{}, err
	}
	raw := result.Outputs["fc1"].Float32
	if len(raw) != SFaceDim {
		return Embedding{}, fmt.Errorf("face: %s returned %d values, want %d", result.Model.ID, len(raw), SFaceDim)
	}
	vector, norm, err := L2Normalize(raw)
	if err != nil {
		return Embedding{}, err
	}
	return Embedding{Vector: vector, Dim: len(vector), Model: result.Model, Normalization: NormalizationL2, RawNorm: norm}, nil
}

// Compare is the package's Compare with the configured thresholds.
func (m *Module) Compare(a, b Embedding) (Comparison, error) {
	return Compare(a, b, m.cfg.Thresholds)
}

// Quality is the package's Quality with the configured thresholds.
func (m *Module) Quality(img image.Image, face Face) (QualityReport, error) {
	return Quality(img, face, m.cfg.Quality)
}
