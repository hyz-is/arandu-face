package feature_test

import (
	"context"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

// The benchmarks time each stage on the 480×560 cartoon, one call at a time:
// go test -run '^$' -bench . ./tests/Feature/

func benchFace(b *testing.B, cfg face.Config) (*face.Module, face.Face) {
	b.Helper()
	m := bootModule(b, cfg)
	img := cartoonFace(480, 560, 240, 250, 95)
	faces, err := m.Detect(context.Background(), img)
	if err != nil || len(faces) != 1 {
		b.Fatalf("Detect = %d faces, %v", len(faces), err)
	}
	return m, faces[0]
}

func BenchmarkDetectYuNet(b *testing.B) {
	m, _ := benchFace(b, face.Config{})
	img := cartoonFace(480, 560, 240, 250, 95)
	for b.Loop() {
		if _, err := m.Detect(context.Background(), img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDetectBlazeFace(b *testing.B) {
	m, _ := benchFace(b, face.Config{Detector: face.DetectorBlazeFace})
	img := cartoonFace(480, 560, 240, 250, 95)
	for b.Loop() {
		if _, err := m.Detect(context.Background(), img); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAlign(b *testing.B) {
	_, f := benchFace(b, face.Config{})
	img := cartoonFace(480, 560, 240, 250, 95)
	for b.Loop() {
		if _, err := face.Align(img, f.Landmarks); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuality(b *testing.B) {
	m, f := benchFace(b, face.Config{})
	img := cartoonFace(480, 560, 240, 250, 95)
	for b.Loop() {
		if _, err := m.Quality(img, f); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEmbed(b *testing.B) {
	m, f := benchFace(b, face.Config{EnableEmbedding: true})
	aligned, err := face.Align(cartoonFace(480, 560, 240, 250, 95), f.Landmarks)
	if err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := m.Embed(context.Background(), aligned.Image); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEstimateAge(b *testing.B) {
	m, f := benchFace(b, face.Config{EnableAge: true})
	img := cartoonFace(480, 560, 240, 250, 95)
	for b.Loop() {
		if _, err := m.EstimateAge(context.Background(), img, f, nil); err != nil {
			b.Fatal(err)
		}
	}
}
