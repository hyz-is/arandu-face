package feature_test

import (
	"context"
	"math"
	"strings"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

// MiVOLO runs here on drawings and gradients, so what is held is the shape of
// its answer -- one finite number of years in a plausible range, with its
// provenance -- and never its value.

func TestEstimateAgeReturnsOneFiniteEstimate(t *testing.T) {
	m := bootModule(t, face.Config{EnableAge: true})
	ctx := context.Background()

	scene := noFaceScenes()["gradient"]
	box := face.Face{Box: face.Box{X: 200, Y: 100, Width: 160, Height: 200}}
	body := face.Box{X: 120, Y: 60, Width: 360, Height: 400}
	for _, tc := range []struct {
		name string
		body *face.Box
	}{{"face only", nil}, {"face and body", &body}} {
		got, err := m.EstimateAge(ctx, scene, box, tc.body)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if math.IsNaN(got.Years) || got.Years < 0 || got.Years > 120 {
			t.Errorf("%s: %v years", tc.name, got.Years)
		}
		if got.BodyUsed != (tc.body != nil) {
			t.Errorf("%s: BodyUsed %v", tc.name, got.BodyUsed)
		}
		if got.Model.ID != face.ModelMiVOLOAge || got.ReportedMAE != 3.65 || !strings.Contains(got.Uncertainty, "LAGENDA") {
			t.Errorf("%s: model %+v, MAE %v, uncertainty %q", tc.name, got.Model, got.ReportedMAE, got.Uncertainty)
		}
	}

	again, err := m.EstimateAge(ctx, scene, box, nil)
	if err != nil {
		t.Fatal(err)
	}
	first, _ := m.EstimateAge(ctx, scene, box, nil)
	if again.Years != first.Years {
		t.Errorf("two runs on one input: %v and %v", first.Years, again.Years)
	}

	if _, err := m.EstimateAge(ctx, scene, face.Face{Box: face.Box{X: 10, Y: 10}}, nil); err == nil {
		t.Error("a face box with no area was accepted")
	}
	if _, err := m.EstimateAge(ctx, scene, face.Face{Box: face.Box{X: 5000, Y: 5000, Width: 10, Height: 10}}, nil); err == nil {
		t.Error("a face box outside the image was accepted")
	}
}
