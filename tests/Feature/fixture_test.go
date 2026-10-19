package feature_test

import (
	"context"
	"errors"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	face "github.com/hyz-is/arandu-face"
	onnx "github.com/hyz-is/arandu-onnx"
)

// These helpers load the real models and draw the images the tests read.
//
// The models are not in the repository. The tests look for them in the
// directory FACE_MODELS_DIR names, or in models/ two levels above this
// checkout (where a workspace that holds the module under modules/ keeps its
// model cache), and skip, saying where they looked, when neither
// has them. Every image is drawn here, procedurally: no photograph of a person
// and no face dataset is read.

// modelsDir is the directory holding the model files, or a skip.
func modelsDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("FACE_MODELS_DIR")
	if dir == "" {
		dir = filepath.Join(packageRoot(t), "..", "..", "models")
	}
	manifests, err := face.Manifests()
	if err != nil {
		t.Fatalf("listing the manifests: %v", err)
	}
	for _, manifest := range manifests {
		if _, err := os.Stat(filepath.Join(dir, manifest.File)); err != nil {
			t.Skipf("the face models are not in %s (set FACE_MODELS_DIR, or fill it with go run ./cmd/face-models -dir %s): %v", dir, dir, err)
		}
	}
	return dir
}

// newRuntime returns an initialized ONNX runtime, or a skip when the library
// is not on this machine.
func newRuntime(t testing.TB) *onnx.Runtime {
	t.Helper()
	rt, err := onnx.NewRuntime(onnx.Config{})
	if err != nil {
		t.Fatalf("building the runtime: %v", err)
	}
	if err := rt.Init(); err != nil {
		if errors.Is(err, onnx.ErrLibraryUnavailable) {
			t.Skipf("ONNX Runtime is not available here, so nothing that runs a model can be tested: %v", err)
		}
		t.Fatalf("initialising the runtime: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("closing the runtime: %v", err)
		}
	})
	return rt
}

// bootModule builds and boots the module on the cached models.
func bootModule(t testing.TB, cfg face.Config) *face.Module {
	t.Helper()
	cfg.ModelsDir = modelsDir(t)
	rt := newRuntime(t)
	m, err := face.New(cfg, rt)
	if err != nil {
		t.Fatalf("building the module: %v", err)
	}
	if err := m.Boot(context.Background()); err != nil {
		t.Fatalf("booting the module: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(context.Background()); err != nil {
			t.Errorf("closing the module: %v", err)
		}
	})
	return m
}

// shade is one layer of a procedural drawing: a colour where inside is true.
type shade struct {
	inside func(x, y float64) bool
	colour func(x, y float64) color.RGBA
}

// render draws the layers, later over earlier, on a w by h image with 4×4
// supersampling so that the edges are smooth.
func render(w, h int, background color.RGBA, layers []shade) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	const n = 4
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var r, g, b float64
			for sy := 0; sy < n; sy++ {
				for sx := 0; sx < n; sx++ {
					px := float64(x) + (float64(sx)+0.5)/n
					py := float64(y) + (float64(sy)+0.5)/n
					c := background
					for _, l := range layers {
						if l.inside(px, py) {
							c = l.colour(px, py)
						}
					}
					r += float64(c.R)
					g += float64(c.G)
					b += float64(c.B)
				}
			}
			o := y*img.Stride + x*4
			img.Pix[o] = uint8(r / (n * n))
			img.Pix[o+1] = uint8(g / (n * n))
			img.Pix[o+2] = uint8(b / (n * n))
			img.Pix[o+3] = 255
		}
	}
	return img
}

// ellipse is the inside of an ellipse centred at (cx, cy), turned by angle
// degrees.
func ellipse(cx, cy, rx, ry, angle float64) func(x, y float64) bool {
	c, s := math.Cos(angle*math.Pi/180), math.Sin(angle*math.Pi/180)
	return func(x, y float64) bool {
		dx, dy := x-cx, y-cy
		u := (dx*c + dy*s) / rx
		v := (-dx*s + dy*c) / ry
		return u*u+v*v <= 1
	}
}

// flat is a uniform colour.
func flat(c color.RGBA) func(x, y float64) color.RGBA {
	return func(float64, float64) color.RGBA { return c }
}

// cartoonFace draws an invented cartoon face, centred at (cx, cy) with a head
// half-width of size, on a w by h canvas: a shaded head, hair, ears, eyes with
// irises, brows, a nose, a mouth, a neck and shoulders. It is a drawing, not
// a person.
func cartoonFace(w, h int, cx, cy, size float64) *image.RGBA {
	k := size / 95
	skin := color.RGBA{R: 222, G: 170, B: 138, A: 255}
	shaded := func(x, y float64) color.RGBA {
		// Darker toward the edge of the head, as light falling from the front.
		dx, dy := (x-cx)/(95*k), (y-cy)/(125*k)
		f := 1 - 0.35*math.Min(1, dx*dx+dy*dy)
		return color.RGBA{R: uint8(float64(skin.R) * f), G: uint8(float64(skin.G) * f), B: uint8(float64(skin.B) * f), A: 255}
	}
	dark := color.RGBA{R: 40, G: 28, B: 22, A: 255}
	layers := []shade{
		{ellipse(cx, cy+260*k, 190*k, 110*k, 0), flat(color.RGBA{R: 60, G: 90, B: 140, A: 255})},
		{func(x, y float64) bool { return math.Abs(x-cx) < 42*k && y > cy+80*k && y < cy+200*k }, flat(color.RGBA{R: 190, G: 140, B: 112, A: 255})},
		{ellipse(cx, cy-25*k, 112*k, 128*k, 0), flat(color.RGBA{R: 52, G: 36, B: 26, A: 255})},
		{ellipse(cx-95*k, cy+5*k, 16*k, 30*k, 0), shaded},
		{ellipse(cx+95*k, cy+5*k, 16*k, 30*k, 0), shaded},
		{ellipse(cx, cy+10*k, 95*k, 125*k, 0), shaded},
		{func(x, y float64) bool {
			return ellipse(cx, cy-70*k, 104*k, 70*k, 0)(x, y) && y < cy-45*k
		}, flat(color.RGBA{R: 52, G: 36, B: 26, A: 255})},
		{ellipse(cx-40*k, cy-37*k, 24*k, 6*k, -8), flat(dark)},
		{ellipse(cx+40*k, cy-37*k, 24*k, 6*k, 8), flat(dark)},
		{ellipse(cx-40*k, cy-15*k, 21*k, 11*k, 0), flat(color.RGBA{R: 245, G: 245, B: 240, A: 255})},
		{ellipse(cx+40*k, cy-15*k, 21*k, 11*k, 0), flat(color.RGBA{R: 245, G: 245, B: 240, A: 255})},
		{ellipse(cx-40*k, cy-15*k, 9*k, 9*k, 0), flat(color.RGBA{R: 90, G: 60, B: 40, A: 255})},
		{ellipse(cx+40*k, cy-15*k, 9*k, 9*k, 0), flat(color.RGBA{R: 90, G: 60, B: 40, A: 255})},
		{ellipse(cx-40*k, cy-15*k, 4*k, 4*k, 0), flat(color.RGBA{A: 255})},
		{ellipse(cx+40*k, cy-15*k, 4*k, 4*k, 0), flat(color.RGBA{A: 255})},
		{ellipse(cx, cy+18*k, 9*k, 26*k, 0), flat(color.RGBA{R: 200, G: 148, B: 118, A: 255})},
		{ellipse(cx-8*k, cy+40*k, 5*k, 3*k, 0), flat(color.RGBA{R: 120, G: 70, B: 60, A: 255})},
		{ellipse(cx+8*k, cy+40*k, 5*k, 3*k, 0), flat(color.RGBA{R: 120, G: 70, B: 60, A: 255})},
		{ellipse(cx, cy+75*k, 33*k, 11*k, 0), flat(color.RGBA{R: 170, G: 80, B: 80, A: 255})},
		{ellipse(cx, cy+75*k, 31*k, 2*k, 0), flat(color.RGBA{R: 90, G: 30, B: 35, A: 255})},
	}
	return render(w, h, color.RGBA{R: 200, G: 210, B: 220, A: 255}, layers)
}

// noFaceScenes are images with nothing a face detector should find: a flat
// grey, a gradient, a checkerboard and pseudo-random noise.
func noFaceScenes() map[string]*image.RGBA {
	const w, h = 640, 480
	scenes := map[string]*image.RGBA{}
	grey := image.NewRGBA(image.Rect(0, 0, w, h))
	gradient := image.NewRGBA(image.Rect(0, 0, w, h))
	checker := image.NewRGBA(image.Rect(0, 0, w, h))
	noise := image.NewRGBA(image.Rect(0, 0, w, h))
	state := uint32(2463534242)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := y*w*4 + x*4
			copy(grey.Pix[o:], []uint8{128, 128, 128, 255})
			copy(gradient.Pix[o:], []uint8{uint8(x * 255 / w), uint8(y * 255 / h), 128, 255})
			v := uint8(30)
			if (x/40+y/40)%2 == 0 {
				v = 220
			}
			copy(checker.Pix[o:], []uint8{v, v, v, 255})
			for c := 0; c < 3; c++ {
				state ^= state << 13
				state ^= state >> 17
				state ^= state << 5
				noise.Pix[o+c] = uint8(state)
			}
			noise.Pix[o+3] = 255
		}
	}
	scenes["grey"], scenes["gradient"], scenes["checkerboard"], scenes["noise"] = grey, gradient, checker, noise
	return scenes
}

// patternCrop is a 112×112 synthetic image, different for every seed: smooth
// sinusoidal bands, not a face.
func patternCrop(seed float64) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 112, 112))
	for y := 0; y < 112; y++ {
		for x := 0; x < 112; x++ {
			o := y*img.Stride + x*4
			for c := 0; c < 3; c++ {
				v := 128 + 100*math.Sin(float64(x)*0.11*seed+float64(y)*0.07*float64(c+1)+seed)
				img.Pix[o+c] = uint8(v)
			}
			img.Pix[o+3] = 255
		}
	}
	return img
}

func packageRoot(t testing.TB) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("finding package root")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}
