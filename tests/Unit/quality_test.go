package unit_test

import (
	"image"
	"math"
	"strings"
	"testing"

	face "github.com/kuaa-io/arandu-face"
)

// uniform is a w by h image of one grey.
func uniform(w, h int, v uint8) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		copy(img.Pix[i:], []uint8{v, v, v, 255})
	}
	return img
}

// checker is a w by h image alternating 0 and 255 pixel by pixel.
func checker(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(0)
			if (x+y)%2 == 1 {
				v = 255
			}
			copy(img.Pix[y*img.Stride+x*4:], []uint8{v, v, v, 255})
		}
	}
	return img
}

// frontal is a face box at (50, 50) of 100×100 with level eyes 50 px apart
// and the nose midway between them.
func frontal() face.Face {
	return face.Face{
		Box: face.Box{X: 50, Y: 50, Width: 100, Height: 100},
		Landmarks: [5]face.Point{
			{X: 75, Y: 90}, {X: 125, Y: 90}, {X: 100, Y: 110}, {X: 80, Y: 130}, {X: 120, Y: 130},
		},
	}
}

func check(t *testing.T, r face.QualityReport, signal string) face.QualityCheck {
	t.Helper()
	for _, c := range r.Checks {
		if c.Signal == signal {
			return c
		}
	}
	t.Fatalf("no check for %s", signal)
	return face.QualityCheck{}
}

func TestQualitySignalsOfAFlatImage(t *testing.T) {
	t.Parallel()

	thresholds := face.DefaultQualityThresholds()
	r, err := face.Quality(uniform(200, 200, 128), frontal(), thresholds)
	if err != nil {
		t.Fatal(err)
	}
	s := r.Signals
	// 100×100 of 200×200 is a quarter; the eyes are 50 apart and level; the
	// nose is on their midpoint; flat grey has no Laplacian and no spread.
	if s.FaceWidthPx != 100 || s.FaceHeightPx != 100 || !near(s.FaceAreaRatio, 0.25, eps) || !s.InsideFrame {
		t.Errorf("geometry = %+v", s)
	}
	if !near(s.InterOcularPx, 50, eps) || !near(s.RollDeg, 0, eps) || !near(s.YawProxy, 0, eps) || s.LandmarksInBox != 5 {
		t.Errorf("landmark signals = %+v", s)
	}
	if !near(s.Brightness, 128, 1e-9) || !near(s.Contrast, 0, 1e-9) || s.Sharpness != 0 {
		t.Errorf("pixel signals = %+v, want brightness 128, contrast 0, sharpness 0", s)
	}
	for signal, pass := range map[string]bool{
		"FaceWidthPx": true, "FaceAreaRatio": true, "InsideFrame": true, "InterOcularPx": true,
		"RollDeg": true, "YawProxy": true, "LandmarksInBox": true, "Brightness": true,
		"Sharpness": false, "Contrast": false,
	} {
		if c := check(t, r, signal); c.Pass != pass {
			t.Errorf("%s: pass %v, want %v (%+v)", signal, c.Pass, pass, c)
		}
	}
	if r.Thresholds.Version != thresholds.Version || !strings.Contains(r.Thresholds.Version, "uncalibrated") {
		t.Errorf("the report carries thresholds %q", r.Thresholds.Version)
	}
	if c := check(t, r, "Sharpness"); c.Rule != ">= 40" {
		t.Errorf("the sharpness rule reads %q", c.Rule)
	}
}

func TestQualitySharpnessOfACheckerboard(t *testing.T) {
	t.Parallel()

	r, err := face.Quality(checker(200, 200), frontal(), face.DefaultQualityThresholds())
	if err != nil {
		t.Fatal(err)
	}
	// Every interior pixel is 0 among four 255s (Laplacian +1020) or 255
	// among four 0s (−1020), in equal numbers: mean 0, variance 1020².
	if !near(r.Signals.Sharpness, 1020*1020, 1e-3) {
		t.Errorf("sharpness = %v, want 1040400", r.Signals.Sharpness)
	}
	if !near(r.Signals.Brightness, 127.5, 1e-6) || !near(r.Signals.Contrast, 127.5, 1e-6) {
		t.Errorf("brightness %v contrast %v, want 127.5 and 127.5", r.Signals.Brightness, r.Signals.Contrast)
	}
	if !check(t, r, "Sharpness").Pass || !check(t, r, "Contrast").Pass {
		t.Error("a checkerboard fails sharpness or contrast")
	}
}

func TestQualityRollYawAndFrame(t *testing.T) {
	t.Parallel()

	// Eyes from (40, 50) to (60, 70): a 45° eye line, 20√2 apart. The nose
	// sits 0.1 inter-ocular distances along that line from their midpoint.
	iod := 20 * math.Sqrt2
	mid := face.Point{X: 50, Y: 60}
	along := face.Point{X: math.Sqrt2 / 2, Y: math.Sqrt2 / 2}
	f := face.Face{
		Box: face.Box{X: 150, Y: 20, Width: 80, Height: 80},
		Landmarks: [5]face.Point{
			{X: 40, Y: 50}, {X: 60, Y: 70},
			{X: mid.X + 0.1*iod*along.X, Y: mid.Y + 0.1*iod*along.Y},
			{X: 30, Y: 80}, {X: 50, Y: 100},
		},
	}
	r, err := face.Quality(uniform(200, 200, 90), f, face.DefaultQualityThresholds())
	if err != nil {
		t.Fatal(err)
	}
	s := r.Signals
	if !near(s.RollDeg, 45, 1e-9) || !near(s.InterOcularPx, iod, 1e-9) || !near(s.YawProxy, 0.1, 1e-9) {
		t.Errorf("roll %v iod %v yaw %v, want 45, %v, 0.1", s.RollDeg, s.InterOcularPx, s.YawProxy, iod)
	}
	// The box runs past the right edge and holds none of the landmarks.
	if s.InsideFrame || s.LandmarksInBox != 0 || check(t, r, "RollDeg").Pass || !check(t, r, "YawProxy").Pass {
		t.Errorf("signals %+v, checks %+v", s, r.Checks)
	}
}

func TestQualityRefusesUnnamedThresholds(t *testing.T) {
	t.Parallel()

	bad := face.DefaultQualityThresholds()
	bad.Version = ""
	if _, err := face.Quality(uniform(10, 10, 0), frontal(), bad); err == nil {
		t.Error("thresholds without a version were accepted")
	}
	bad = face.DefaultQualityThresholds()
	bad.MinBrightness, bad.MaxBrightness = 200, 100
	if err := bad.Validate(); err == nil {
		t.Error("an inverted brightness range was accepted")
	}
	bad = face.DefaultQualityThresholds()
	bad.MinSharpness = math.NaN()
	if err := bad.Validate(); err == nil {
		t.Error("a NaN threshold was accepted")
	}
	if got := face.LaplacianVariance([]float64{1, 2, 3, 4}, 2, 2); got != 0 {
		t.Errorf("the Laplacian of a 2×2 plane = %v, want 0", got)
	}
}
