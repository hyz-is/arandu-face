package feature_test

import (
	"context"
	"errors"
	"image"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"

	face "github.com/hyz-is/arandu-face"
	onnx "github.com/hyz-is/arandu-onnx"
)

// These tests drive the module the way an application does: build it,
// register it, boot it, ask its health and close it.

// unbootedModule builds the module on a runtime that loads nothing.
func unbootedModule(t *testing.T, cfg face.Config) *face.Module {
	t.Helper()
	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatalf("building the runtime: %v", err)
	}
	m, err := face.New(cfg, rt)
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	return m
}

func TestTheModuleIsRegisteredAsAModule(t *testing.T) {
	t.Parallel()

	var module foundation.Module = unbootedModule(t, face.Config{ModelsDir: t.TempDir()})
	if module.Name() != "face" {
		t.Fatalf("Name() = %q, want face: route names and the boot log are keyed by it", module.Name())
	}
}

// TestTheModuleServesNoRoute holds the reason Routes is empty: faces are
// measured through the application's own controllers, and a route here would
// measure any face for anybody.
func TestTheModuleServesNoRoute(t *testing.T) {
	t.Parallel()

	m := unbootedModule(t, face.Config{ModelsDir: t.TempDir()})
	router := fhttp.NewRouter()
	m.Routes(router.ForModule(m.Name()))
	if routes := router.Routes(); len(routes) != 0 {
		t.Fatalf("the module registered %d routes, want none: %+v", len(routes), routes)
	}
}

func TestNewRefusesWhatCannotWork(t *testing.T) {
	t.Parallel()

	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := face.New(face.Config{ModelsDir: t.TempDir()}, nil); err == nil {
		t.Error("New accepted a nil runtime")
	}
	if _, err := face.New(face.Config{}, rt); err == nil || !strings.Contains(err.Error(), "ModelsDir") {
		t.Errorf("New without a models directory = %v, want an error naming ModelsDir", err)
	}
	if _, err := face.New(face.Config{ModelsDir: "x", Detector: "retina"}, rt); err == nil {
		t.Error("New accepted an unknown detector")
	}
}

// TestRefuseDevOnlyStopsTheBootBeforeAnyFileIsRead is the production switch:
// with it set, a configuration that needs a dev-only model does not boot, and
// it says which model. It needs neither ONNX Runtime nor the files, because
// the refusal comes first.
func TestRefuseDevOnlyStopsTheBootBeforeAnyFileIsRead(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	for _, tc := range []struct {
		cfg   face.Config
		model string
	}{
		{face.Config{EnableEmbedding: true}, face.ModelSFace},
		{face.Config{EnableAge: true}, face.ModelMiVOLOAge},
	} {
		tc.cfg.ModelsDir = t.TempDir()
		tc.cfg.RefuseDevOnly = true
		m := unbootedModule(t, tc.cfg)
		err := m.Boot(ctx)
		if !errors.Is(err, face.ErrDevOnlyRefused) || !strings.Contains(err.Error(), tc.model) {
			t.Errorf("Boot with %s and RefuseDevOnly = %v, want ErrDevOnlyRefused naming it", tc.model, err)
		}
		if err := m.Health(ctx); !errors.Is(err, face.ErrNotBooted) {
			t.Errorf("Health after the refusal = %v, want ErrNotBooted", err)
		}
	}
}

func TestRefuseDevOnlyBootsTheApprovedDetectors(t *testing.T) {
	for _, detector := range []string{face.DetectorYuNet, face.DetectorBlazeFace} {
		m := bootModule(t, face.Config{Detector: detector, RefuseDevOnly: true})
		models := m.Models()
		if len(models) != 1 || models[0].Status != onnx.StatusApproved {
			t.Errorf("%s with RefuseDevOnly opened %+v, want one approved model", detector, models)
		}
		if _, err := m.Embed(context.Background(), image.NewRGBA(image.Rect(0, 0, 112, 112))); !errors.Is(err, face.ErrNotEnabled) {
			t.Errorf("Embed without EnableEmbedding = %v, want ErrNotEnabled", err)
		}
		if _, err := m.EstimateAge(context.Background(), image.NewRGBA(image.Rect(0, 0, 10, 10)), face.Face{}, nil); !errors.Is(err, face.ErrNotEnabled) {
			t.Errorf("EstimateAge without EnableAge = %v, want ErrNotEnabled", err)
		}
	}
}

func TestTheModuleIsUnusableBeforeBootAndAfterClose(t *testing.T) {
	m := bootModule(t, face.Config{EnableEmbedding: true, EnableAge: true})
	ctx := context.Background()
	if err := m.Health(ctx); err != nil {
		t.Fatalf("Health after Boot = %v, want nil", err)
	}
	if got := len(m.Models()); got != 3 {
		t.Errorf("%d models open, want 3", got)
	}
	if err := m.Boot(ctx); err != nil {
		t.Errorf("a second Boot = %v, want nil", err)
	}
	if err := m.Close(ctx); err != nil {
		t.Fatalf("Close = %v", err)
	}
	if err := m.Close(ctx); err != nil {
		t.Errorf("a second Close = %v, want nil", err)
	}
	if err := m.Health(ctx); !errors.Is(err, face.ErrClosed) {
		t.Errorf("Health after Close = %v, want ErrClosed", err)
	}
	img := cartoonFace(320, 320, 160, 160, 60)
	if _, err := m.Detect(ctx, img); !errors.Is(err, face.ErrClosed) {
		t.Errorf("Detect after Close = %v, want ErrClosed", err)
	}
	if _, err := m.Embed(ctx, patternCrop(1)); !errors.Is(err, face.ErrClosed) {
		t.Errorf("Embed after Close = %v, want ErrClosed", err)
	}
	if err := m.Boot(ctx); !errors.Is(err, face.ErrClosed) {
		t.Errorf("Boot after Close = %v, want ErrClosed", err)
	}
	if len(m.Models()) != 0 {
		t.Error("a closed module still lists models")
	}

	fresh := unbootedModule(t, face.Config{ModelsDir: t.TempDir()})
	if err := fresh.Health(ctx); !errors.Is(err, face.ErrNotBooted) {
		t.Errorf("Health before Boot = %v, want ErrNotBooted", err)
	}
	if _, err := fresh.Detect(ctx, img); !errors.Is(err, face.ErrNotBooted) {
		t.Errorf("Detect before Boot = %v, want ErrNotBooted", err)
	}
}

func TestHealthFailsWhenTheRuntimeIsClosedUnderneath(t *testing.T) {
	cfg := face.Config{ModelsDir: modelsDir(t)}
	rt := newRuntime(t)
	m, err := face.New(cfg, rt)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Boot(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := rt.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Health(context.Background()); !errors.Is(err, onnx.ErrClosed) {
		t.Errorf("Health with the runtime closed = %v, want onnx.ErrClosed", err)
	}
}

func TestBootRefusesAMissingOrAlteredModel(t *testing.T) {
	rt := newRuntime(t)
	ctx := context.Background()

	empty := t.TempDir()
	m, err := face.New(face.Config{ModelsDir: empty}, rt)
	if err != nil {
		t.Fatal(err)
	}
	err = m.Boot(ctx)
	if err == nil || !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), face.ModelYuNet) {
		t.Errorf("Boot on an empty directory = %v, want a missing file naming %s", err, face.ModelYuNet)
	}
	if err := m.Health(ctx); !errors.Is(err, face.ErrNotBooted) {
		t.Errorf("Health after a failed Boot = %v, want ErrNotBooted", err)
	}

	altered := t.TempDir()
	manifest, err := face.LookupManifest(face.ModelYuNet)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(altered, manifest.File)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("not the detector"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err = face.New(face.Config{ModelsDir: altered}, rt)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Boot(ctx); !errors.Is(err, onnx.ErrDigestMismatch) {
		t.Errorf("Boot on an altered file = %v, want onnx.ErrDigestMismatch", err)
	}
}
