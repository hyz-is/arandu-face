package feature_test

import (
	"context"
	"image"
	"math"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

// These tests run SFace on synthetic crops. They hold the shape of the
// embedding -- its length, its norm, its determinism, its provenance -- and
// not its accuracy, which only consented real data can measure.

func norm(v []float32) float64 {
	s := 0.0
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func TestEmbedIsDeterministicAndUnitLength(t *testing.T) {
	m := bootModule(t, face.Config{EnableEmbedding: true})
	ctx := context.Background()

	first, err := m.Embed(ctx, patternCrop(1))
	if err != nil {
		t.Fatal(err)
	}
	again, err := m.Embed(ctx, patternCrop(1))
	if err != nil {
		t.Fatal(err)
	}
	other, err := m.Embed(ctx, patternCrop(2))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range []face.Embedding{first, other} {
		if e.Dim != face.SFaceDim || len(e.Vector) != 128 || e.Normalization != face.NormalizationL2 {
			t.Errorf("embedding of dim %d, %d values, normalization %q", e.Dim, len(e.Vector), e.Normalization)
		}
		if n := norm(e.Vector); math.Abs(n-1) > 1e-5 {
			t.Errorf("the embedding's norm is %v, want 1", n)
		}
		if !(e.RawNorm > 0) || math.IsInf(e.RawNorm, 0) {
			t.Errorf("raw norm %v", e.RawNorm)
		}
	}
	for i := range first.Vector {
		if first.Vector[i] != again.Vector[i] {
			t.Fatalf("value %d differs between two runs on one crop: %v and %v", i, first.Vector[i], again.Vector[i])
		}
	}

	same, err := m.Compare(first, again)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(same.Cosine-1) > 1e-6 || same.L2 > 1e-3 {
		t.Errorf("one crop against itself: cosine %v L2 %v", same.Cosine, same.L2)
	}
	diff, err := m.Compare(first, other)
	if err != nil {
		t.Fatal(err)
	}
	if diff.Cosine > 1-1e-4 || diff.Thresholds.Version != "sface-opencv-2021dec" {
		t.Errorf("two crops: cosine %v thresholds %s", diff.Cosine, diff.Thresholds.Version)
	}

	if _, err := m.Embed(ctx, image.NewRGBA(image.Rect(0, 0, 100, 112))); err == nil {
		t.Error("Embed accepted a crop that is not 112×112")
	}
}

// TestOneDrawingEmbedsCloserToItselfThanToAPattern runs the whole chain on
// the cartoon drawn at two places and sizes. Both crops are aligned onto the
// same template, so their embeddings should be nearer each other than either
// is to a crop holding no face. It is a check of the alignment's wiring, not
// a measurement of recognition.
func TestOneDrawingEmbedsCloserToItselfThanToAPattern(t *testing.T) {
	m := bootModule(t, face.Config{EnableEmbedding: true})
	ctx := context.Background()
	embed := func(img image.Image) face.Embedding {
		t.Helper()
		faces, err := m.Detect(ctx, img)
		if err != nil || len(faces) != 1 {
			t.Fatalf("Detect = %d faces, %v", len(faces), err)
		}
		aligned, err := face.Align(img, faces[0].Landmarks)
		if err != nil {
			t.Fatal(err)
		}
		e, err := m.Embed(ctx, aligned.Image)
		if err != nil {
			t.Fatal(err)
		}
		return e
	}
	a := embed(cartoonFace(480, 560, 240, 250, 95))
	b := embed(cartoonFace(640, 480, 300, 230, 70))
	pattern, err := m.Embed(ctx, patternCrop(1))
	if err != nil {
		t.Fatal(err)
	}
	ab, err := m.Compare(a, b)
	if err != nil {
		t.Fatal(err)
	}
	ap, err := m.Compare(a, pattern)
	if err != nil {
		t.Fatal(err)
	}
	if ab.Cosine < 0.8 || ab.Cosine-ap.Cosine < 0.3 {
		t.Errorf("drawing to drawing cosine %.3f, drawing to pattern %.3f", ab.Cosine, ap.Cosine)
	}
}
