package face

import (
	"fmt"
	"image"
	"math"
)

// QualityThresholds are the bounds Quality checks each signal against. They
// are versioned parameters, not a policy: the report carries them, and what a
// failed check means for a case is the application's decision.
type QualityThresholds struct {
	// Version names this set of values. A change to any value is a new
	// version.
	Version string
	// MinFaceSidePx is the smallest face box side, in pixels, that passes.
	MinFaceSidePx float64
	// MinFaceAreaRatio is the smallest share of the frame the face box may
	// cover.
	MinFaceAreaRatio float64
	// MinInterOcularPx is the smallest distance between the eyes, in pixels.
	MinInterOcularPx float64
	// MaxAbsRollDeg is the largest tilt of the eye line, in degrees.
	MaxAbsRollDeg float64
	// MaxAbsYawProxy is the largest horizontal offset of the nose from the
	// eyes' midpoint, in inter-ocular distances.
	MaxAbsYawProxy float64
	// MinSharpness is the smallest variance of the Laplacian of the face's
	// luma.
	MinSharpness float64
	// MinBrightness and MaxBrightness bound the face's mean luma, 0..255.
	MinBrightness, MaxBrightness float64
	// MinContrast is the smallest standard deviation of the face's luma.
	MinContrast float64
}

// DefaultQualityThresholds returns the thresholds Quality uses when the
// configuration names none. They are starting values, not calibrated on any
// population; their version says so.
func DefaultQualityThresholds() QualityThresholds {
	return QualityThresholds{
		Version:          "kuaa-face-quality-0.1-uncalibrated",
		MinFaceSidePx:    80,
		MinFaceAreaRatio: 0.02,
		MinInterOcularPx: 40,
		MaxAbsRollDeg:    20,
		MaxAbsYawProxy:   0.25,
		MinSharpness:     40,
		MinBrightness:    50,
		MaxBrightness:    210,
		MinContrast:      20,
	}
}

// Validate reports a set of thresholds that cannot be checked against.
func (q QualityThresholds) Validate() error {
	if q.Version == "" {
		return fmt.Errorf("face: QualityThresholds.Version is required: a threshold nobody can name is one nobody can audit")
	}
	for _, f := range []struct {
		name  string
		value float64
	}{
		{"MinFaceSidePx", q.MinFaceSidePx}, {"MinFaceAreaRatio", q.MinFaceAreaRatio},
		{"MinInterOcularPx", q.MinInterOcularPx}, {"MaxAbsRollDeg", q.MaxAbsRollDeg},
		{"MaxAbsYawProxy", q.MaxAbsYawProxy}, {"MinSharpness", q.MinSharpness},
		{"MinBrightness", q.MinBrightness}, {"MaxBrightness", q.MaxBrightness},
		{"MinContrast", q.MinContrast},
	} {
		if math.IsNaN(f.value) || math.IsInf(f.value, 0) || f.value < 0 {
			return fmt.Errorf("face: QualityThresholds.%s is %v, and has to be a finite value of at least 0", f.name, f.value)
		}
	}
	if q.MinBrightness > q.MaxBrightness {
		return fmt.Errorf("face: QualityThresholds.MinBrightness %v is above MaxBrightness %v", q.MinBrightness, q.MaxBrightness)
	}
	return nil
}

// QualitySignals are the measurements Quality takes of one face.
type QualitySignals struct {
	// FaceWidthPx and FaceHeightPx are the face box's size.
	FaceWidthPx, FaceHeightPx float64
	// FaceAreaRatio is the face box's area over the frame's.
	FaceAreaRatio float64
	// InsideFrame is whether the whole face box lies within the image.
	InsideFrame bool
	// InterOcularPx is the distance between the two eye landmarks.
	InterOcularPx float64
	// RollDeg is the angle of the line from the right eye to the left eye
	// against the horizontal, in degrees, positive when the left eye (image
	// right) is lower on screen.
	RollDeg float64
	// YawProxy is the nose tip's horizontal offset from the eyes' midpoint,
	// once the roll is undone, in inter-ocular distances: 0 for a frontal
	// face, positive when the nose points toward the image's right. It is a
	// geometric proxy, not an angle.
	YawProxy float64
	// LandmarksInBox is how many of the five landmarks lie inside the face
	// box, a cheap consistency check of the detection.
	LandmarksInBox int
	// Sharpness is the variance of the 4-neighbour Laplacian of the luma of
	// the face box's pixels: higher is sharper. It depends on the face's
	// size in pixels.
	Sharpness float64
	// Brightness is the mean luma (BT.601) of the face box's pixels, 0..255.
	Brightness float64
	// Contrast is the standard deviation of that luma.
	Contrast float64
}

// QualityCheck is one signal held against its bound.
type QualityCheck struct {
	// Signal is the QualitySignals field the check reads.
	Signal string
	// Value is the signal's value.
	Value float64
	// Rule is the bound, written out, such as ">= 40" or "in [50, 210]".
	Rule string
	// Pass is whether the value satisfies the rule.
	Pass bool
}

// QualityReport is what Quality returns: the signals, each check, and the
// thresholds the checks used. It holds no overall verdict.
type QualityReport struct {
	// Signals are the measurements.
	Signals QualitySignals
	// Checks hold each signal against its bound, in a fixed order.
	Checks []QualityCheck
	// Thresholds are the bounds the checks used, with their version.
	Thresholds QualityThresholds
}

// Quality measures one detected face on the image it was detected in, in
// pure Go, and checks every signal against thresholds.
//
// Geometry comes from the face's box and landmarks; sharpness, brightness and
// contrast from the luma of the pixels inside the box, clipped to the image.
// Nothing runs a model.
func Quality(img image.Image, face Face, thresholds QualityThresholds) (QualityReport, error) {
	if err := thresholds.Validate(); err != nil {
		return QualityReport{}, err
	}
	b := img.Bounds()
	if b.Empty() {
		return QualityReport{}, fmt.Errorf("face: Quality of an empty image")
	}
	box := face.Box
	frame := Box{X: float64(b.Min.X), Y: float64(b.Min.Y), Width: float64(b.Dx()), Height: float64(b.Dy())}

	var s QualitySignals
	s.FaceWidthPx, s.FaceHeightPx = box.Width, box.Height
	s.FaceAreaRatio = box.Area() / frame.Area()
	s.InsideFrame = box.X >= frame.X && box.Y >= frame.Y &&
		box.X+box.Width <= frame.X+frame.Width && box.Y+box.Height <= frame.Y+frame.Height

	re, le, nose := face.Landmarks[0], face.Landmarks[1], face.Landmarks[2]
	dx, dy := le.X-re.X, le.Y-re.Y
	s.InterOcularPx = math.Hypot(dx, dy)
	s.RollDeg = math.Atan2(dy, dx) * 180 / math.Pi
	if s.InterOcularPx > 0 {
		// Undo the roll about the eyes' midpoint: the nose's offset along the
		// eye line is the yaw proxy.
		mid := Point{X: (re.X + le.X) / 2, Y: (re.Y + le.Y) / 2}
		ux, uy := dx/s.InterOcularPx, dy/s.InterOcularPx
		s.YawProxy = ((nose.X-mid.X)*ux + (nose.Y-mid.Y)*uy) / s.InterOcularPx
	}
	for _, p := range face.Landmarks {
		if box.Contains(p) {
			s.LandmarksInBox++
		}
	}

	luma, w, h := lumaOf(img, box.Rect().Intersect(b))
	s.Brightness, s.Contrast = meanStd(luma)
	s.Sharpness = LaplacianVariance(luma, w, h)

	checks := []QualityCheck{
		atLeast("FaceWidthPx", s.FaceWidthPx, thresholds.MinFaceSidePx),
		atLeast("FaceHeightPx", s.FaceHeightPx, thresholds.MinFaceSidePx),
		atLeast("FaceAreaRatio", s.FaceAreaRatio, thresholds.MinFaceAreaRatio),
		{Signal: "InsideFrame", Value: boolValue(s.InsideFrame), Rule: "== 1", Pass: s.InsideFrame},
		atLeast("InterOcularPx", s.InterOcularPx, thresholds.MinInterOcularPx),
		absAtMost("RollDeg", s.RollDeg, thresholds.MaxAbsRollDeg),
		absAtMost("YawProxy", s.YawProxy, thresholds.MaxAbsYawProxy),
		{Signal: "LandmarksInBox", Value: float64(s.LandmarksInBox), Rule: "== 5", Pass: s.LandmarksInBox == 5},
		atLeast("Sharpness", s.Sharpness, thresholds.MinSharpness),
		{
			Signal: "Brightness", Value: s.Brightness,
			Rule: fmt.Sprintf("in [%g, %g]", thresholds.MinBrightness, thresholds.MaxBrightness),
			Pass: s.Brightness >= thresholds.MinBrightness && s.Brightness <= thresholds.MaxBrightness,
		},
		atLeast("Contrast", s.Contrast, thresholds.MinContrast),
	}
	return QualityReport{Signals: s, Checks: checks, Thresholds: thresholds}, nil
}

// atLeast is the check value >= limit.
func atLeast(signal string, value, limit float64) QualityCheck {
	return QualityCheck{Signal: signal, Value: value, Rule: fmt.Sprintf(">= %g", limit), Pass: value >= limit}
}

// absAtMost is the check |value| <= limit.
func absAtMost(signal string, value, limit float64) QualityCheck {
	return QualityCheck{Signal: signal, Value: value, Rule: fmt.Sprintf("|x| <= %g", limit), Pass: math.Abs(value) <= limit}
}

// boolValue is 1 for true and 0 for false.
func boolValue(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// lumaOf returns the BT.601 luma of the pixels of img inside r, row by row,
// with r's width and height.
func lumaOf(img image.Image, r image.Rectangle) ([]float64, int, int) {
	if r.Empty() {
		return nil, 0, 0
	}
	w, h := r.Dx(), r.Dy()
	out := make([]float64, 0, w*h)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			cr, cg, cb, _ := img.At(x, y).RGBA()
			out = append(out, Luma(uint8(cr>>8), uint8(cg>>8), uint8(cb>>8)))
		}
	}
	return out, w, h
}

// Luma is the BT.601 luma of an 8-bit RGB colour, 0..255.
func Luma(r, g, b uint8) float64 {
	return 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
}

// meanStd is the mean and the population standard deviation of values, or
// zeros for none.
func meanStd(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	sq := 0.0
	for _, v := range values {
		sq += (v - mean) * (v - mean)
	}
	return mean, math.Sqrt(sq / float64(len(values)))
}

// LaplacianVariance is the population variance of the 4-neighbour Laplacian
// (up + down + left + right − 4·centre) of a w by h plane of values, taken
// over the interior pixels. A plane smaller than 3×3 has no interior and
// gives 0.
func LaplacianVariance(plane []float64, w, h int) float64 {
	if w < 3 || h < 3 || len(plane) < w*h {
		return 0
	}
	values := make([]float64, 0, (w-2)*(h-2))
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			values = append(values, plane[i-w]+plane[i+w]+plane[i-1]+plane[i+1]-4*plane[i])
		}
	}
	_, std := meanStd(values)
	return std * std
}
