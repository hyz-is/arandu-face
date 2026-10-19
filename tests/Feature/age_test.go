package feature_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	face "github.com/hyz-is/arandu-face"
	onnx "github.com/hyz-is/arandu-onnx"
	"github.com/hyz-is/arandu-onnx/graph"
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
		if got.Model.ID != face.ModelMiVOLOAgeConvTranspose || got.ReportedMAE != 3.65 || !strings.Contains(got.Uncertainty, "LAGENDA") {
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

// loadVerifiedGraph reads a model file through its manifest's digest and
// decodes its graph.
func loadVerifiedGraph(t *testing.T, dir, id string) *graph.Model {
	t.Helper()
	manifest, err := face.LookupManifest(id)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, manifest.File))
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if got := hex.EncodeToString(sum[:]); got != manifest.SHA256 {
		t.Fatalf("%s: sha256 %s, the manifest records %s", id, got, manifest.SHA256)
	}
	model, err := graph.Load(data)
	if err != nil {
		t.Fatalf("decoding %s: %v", id, err)
	}
	return model
}

// TestTheDefaultAgeModelHasNoCol2Im holds why the rewritten file is the
// default: TensorRT and ONNX Runtime's CUDA provider have no Col2Im kernel, and
// a graph with one is split, with that node on the CPU. The original keeps its
// four, which is what the rewrite replaced.
func TestTheDefaultAgeModelHasNoCol2Im(t *testing.T) {
	dir := modelsDir(t)
	defaults := face.Config{ModelsDir: dir}.WithDefaults()

	rewritten := loadVerifiedGraph(t, dir, defaults.AgeModel).Graph.OpCounts()
	if rewritten["Col2Im"] != 0 {
		t.Errorf("the default age model %s has %d Col2Im nodes", defaults.AgeModel, rewritten["Col2Im"])
	}
	original := loadVerifiedGraph(t, dir, face.ModelMiVOLOAge).Graph.OpCounts()
	if original["Col2Im"] != 4 || original["ConvTranspose"] != 0 {
		t.Errorf("the original has %d Col2Im and %d ConvTranspose, want 4 and 0", original["Col2Im"], original["ConvTranspose"])
	}
	if rewritten["ConvTranspose"] != original["Col2Im"] {
		t.Errorf("the rewritten model has %d ConvTranspose for the original's %d Col2Im", rewritten["ConvTranspose"], original["Col2Im"])
	}
	for op, n := range original {
		switch op {
		case "Col2Im", "ConvTranspose", "Reshape":
			continue
		}
		if rewritten[op] != n {
			t.Errorf("%s: %d in the original, %d rewritten; only Col2Im, ConvTranspose and Reshape may change", op, n, rewritten[op])
		}
	}
}

// ageTolerance is the largest difference, in years, the rewritten model may
// show against the original on the CPU. The rewrite reorders no sum the CPU
// computes differently, and was measured at exactly 0; the bound leaves room
// for a runtime that schedules the ConvTranspose's accumulation differently.
const ageTolerance = 1e-4

// TestTheRewrittenAgeModelMatchesTheOriginalOnTheCPU runs both files on the
// ONNX Runtime CPU provider: on seeded synthetic tensors, which cover the
// whole input range rather than what a photograph looks like, and through
// EstimateAge on drawn scenes, with and without a body.
func TestTheRewrittenAgeModelMatchesTheOriginalOnTheCPU(t *testing.T) {
	dir := modelsDir(t)
	rt := newRuntime(t)
	ctx := context.Background()

	open := func(id string) *onnx.Session {
		manifest, err := face.LookupManifest(id)
		if err != nil {
			t.Fatal(err)
		}
		session, err := rt.Open(manifest, dir)
		if err != nil {
			t.Fatalf("opening %s: %v", id, err)
		}
		t.Cleanup(func() { _ = session.Close() })
		return session
	}
	original, rewritten := open(face.ModelMiVOLOAge), open(face.ModelMiVOLOAgeConvTranspose)

	worst := 0.0
	rng := rand.New(rand.NewPCG(7, 2026))
	for i := range 4 {
		data := make([]float32, 6*face.MiVOLOInputSize*face.MiVOLOInputSize)
		for j := range data {
			data[j] = float32(rng.Float64()*6 - 3)
		}
		input := map[string]onnx.Tensor{"faces_bodies": {Shape: []int64{1, 6, face.MiVOLOInputSize, face.MiVOLOInputSize}, Float32: data}}
		want, err := original.Run(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		got, err := rewritten.Run(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		a, b := want.Outputs["age"].Float32, got.Outputs["age"].Float32
		if len(a) != 1 || len(b) != 1 {
			t.Fatalf("input %d: outputs %v and %v, want one value each", i, a, b)
		}
		worst = math.Max(worst, math.Abs(float64(a[0])-float64(b[0])))
	}

	byModel := map[string]*face.Module{}
	for _, id := range []string{face.ModelMiVOLOAge, face.ModelMiVOLOAgeConvTranspose} {
		m, err := face.New(face.Config{ModelsDir: dir, EnableAge: true, AgeModel: id}, rt)
		if err != nil {
			t.Fatal(err)
		}
		if err := m.Boot(ctx); err != nil {
			t.Fatalf("booting with %s: %v", id, err)
		}
		t.Cleanup(func() { _ = m.Close(ctx) })
		byModel[id] = m
	}
	box := face.Face{Box: face.Box{X: 200, Y: 100, Width: 160, Height: 200}}
	body := face.Box{X: 120, Y: 60, Width: 360, Height: 400}
	for name, scene := range noFaceScenes() {
		for _, b := range []*face.Box{nil, &body} {
			want, err := byModel[face.ModelMiVOLOAge].EstimateAge(ctx, scene, box, b)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			got, err := byModel[face.ModelMiVOLOAgeConvTranspose].EstimateAge(ctx, scene, box, b)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if want.Model.ID != face.ModelMiVOLOAge || got.Model.ID != face.ModelMiVOLOAgeConvTranspose {
				t.Fatalf("the estimates came from %s and %s", want.Model.ID, got.Model.ID)
			}
			worst = math.Max(worst, math.Abs(want.Years-got.Years))
		}
	}
	if worst > ageTolerance {
		t.Errorf("max |original - rewritten| = %g years, more than %g", worst, ageTolerance)
	}
	t.Logf("max |original - rewritten| = %g years over 4 synthetic tensors and %d drawn scenes", worst, 2*len(noFaceScenes()))
}
