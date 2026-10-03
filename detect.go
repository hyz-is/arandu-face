package face

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"

	onnx "github.com/hyz-is/arandu-onnx"
)

// The landmark schemes a Face may carry.
const (
	// LandmarksFivePoint is five measured landmarks: right eye, left eye,
	// nose tip, right mouth corner, left mouth corner. YuNet emits it.
	LandmarksFivePoint = "five-point"
	// LandmarksMouthCentre is BlazeFace's: eyes and nose tip measured, and
	// the mouth centre placed in both mouth-corner slots, because the model
	// emits no corners. Aligned through it, the mouth centre is fitted to the
	// midpoint of the template's corners with twice the weight.
	LandmarksMouthCentre = "mouth-centre"
)

// DetectorParameters are the values a detection was made with, carried with
// every Face so the result can be audited and reproduced.
type DetectorParameters struct {
	// ScoreThreshold is the score a candidate needed to be kept.
	ScoreThreshold float64
	// NMSThreshold is the IoU above which a candidate was suppressed.
	NMSThreshold float64
	// TopK is how many candidates suppression considered; 0 for BlazeFace,
	// whose weighted suppression has no cut.
	TopK int
	// InputWidth and InputHeight are the size of the tensor the detector
	// read, padding included.
	InputWidth, InputHeight int
	// Scale is the factor the image was resized by before padding or
	// letterboxing.
	Scale float64
}

// Face is one detected face, in the coordinates of the image as given.
type Face struct {
	// Box is the face's bounding box. It may extend past the image's edges.
	Box Box
	// Score is the detector's confidence, between 0 and 1.
	Score float64
	// Landmarks are, in order, the right eye, the left eye, the nose tip, the
	// right mouth corner and the left mouth corner, "right" being the
	// subject's right: the image's left on a photograph of someone facing the
	// camera. LandmarkScheme says which of them were measured.
	Landmarks [5]Point
	// LandmarkScheme is LandmarksFivePoint or LandmarksMouthCentre.
	LandmarkScheme string
	// Keypoints are every keypoint the detector emitted, in its own order.
	Keypoints []Point
	// Detector identifies the model that found the face.
	Detector onnx.ModelInfo
	// Parameters are the values the detection was made with.
	Parameters DetectorParameters
}

// Detect finds the faces in an image with the configured detector and returns
// them in descending score order. An image without faces returns none and no
// error.
//
// With YuNet the image is scaled down so its longer side is at most
// Config.DetectMaxSide (never up), padded with black at the bottom and right
// to multiples of 32, given as raw BGR 0..255, decoded by DecodeYuNet and
// suppressed by NMS. With BlazeFace it is letterboxed onto 128×128 black,
// given as RGB in −1..1, decoded by DecodeBlazeFace and suppressed by
// WeightedNMS.
func (m *Module) Detect(ctx context.Context, img image.Image) ([]Face, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	session, err := m.session(m.detector, "Detect", "Detector")
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	if b.Empty() {
		return nil, errors.New("face: Detect of an empty image")
	}
	if int64(b.Dx())*int64(b.Dy()) > onnx.MaxImagePixels {
		return nil, fmt.Errorf("%w: %dx%d is more than %d pixels", onnx.ErrImageTooLarge, b.Dx(), b.Dy(), onnx.MaxImagePixels)
	}
	if m.cfg.Detector == DetectorBlazeFace {
		return m.detectBlazeFace(ctx, session, img)
	}
	return m.detectYuNet(ctx, session, img)
}

// detectYuNet is Detect with YuNet.
func (m *Module) detectYuNet(ctx context.Context, session *onnx.Session, img image.Image) ([]Face, error) {
	b := img.Bounds()
	scale := math.Min(1, float64(m.cfg.DetectMaxSide)/float64(max(b.Dx(), b.Dy())))
	w := max(1, int(math.Round(float64(b.Dx())*scale)))
	h := max(1, int(math.Round(float64(b.Dy())*scale)))
	var resized *image.RGBA
	if w == b.Dx() && h == b.Dy() {
		resized = asRGBA(img)
	} else {
		var err error
		if resized, err = onnx.Resize(img, w, h); err != nil {
			return nil, err
		}
	}
	padW, padH := YuNetPaddedSize(w, h)
	padded := image.NewRGBA(image.Rect(0, 0, padW, padH))
	draw.Draw(padded, padded.Bounds(), image.Black, image.Point{}, draw.Src)
	draw.Draw(padded, image.Rect(0, 0, w, h), resized, image.Point{}, draw.Src)

	result, err := session.Run(ctx, map[string]onnx.Tensor{"input": RawPlanes(padded, true)})
	if err != nil {
		return nil, err
	}
	candidates, err := DecodeYuNet(result.Outputs, padW, padH, m.cfg.ScoreThreshold)
	if err != nil {
		return nil, err
	}
	kept := NMS(candidates, m.cfg.ScoreThreshold, m.cfg.NMSThreshold, m.cfg.TopK)

	params := DetectorParameters{
		ScoreThreshold: m.cfg.ScoreThreshold, NMSThreshold: m.cfg.NMSThreshold, TopK: m.cfg.TopK,
		InputWidth: padW, InputHeight: padH, Scale: float64(w) / float64(b.Dx()),
	}
	// Resize aligns pixel centres, so a point maps back through them.
	sx, sy := float64(b.Dx())/float64(w), float64(b.Dy())/float64(h)
	toSource := func(p Point) Point {
		return Point{X: (p.X+0.5)*sx - 0.5 + float64(b.Min.X), Y: (p.Y+0.5)*sy - 0.5 + float64(b.Min.Y)}
	}
	faces := make([]Face, 0, len(kept))
	for _, c := range kept {
		f := Face{
			Score:          c.Score,
			LandmarkScheme: LandmarksFivePoint,
			Detector:       result.Model,
			Parameters:     params,
		}
		origin := toSource(Point{X: c.Box.X, Y: c.Box.Y})
		f.Box = Box{X: origin.X, Y: origin.Y, Width: c.Box.Width * sx, Height: c.Box.Height * sy}
		for i, p := range c.Keypoints {
			f.Keypoints = append(f.Keypoints, toSource(p))
			if i < 5 {
				f.Landmarks[i] = toSource(p)
			}
		}
		faces = append(faces, f)
	}
	return faces, nil
}

// detectBlazeFace is Detect with BlazeFace.
func (m *Module) detectBlazeFace(ctx context.Context, session *onnx.Session, img image.Image) ([]Face, error) {
	canvas, info, err := onnx.Letterbox(img, BlazeFaceInputSize, BlazeFaceInputSize, color.RGBA{A: 255})
	if err != nil {
		return nil, err
	}
	input, err := onnx.NCHW(canvas, onnx.Normalization{Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}})
	if err != nil {
		return nil, err
	}
	result, err := session.Run(ctx, map[string]onnx.Tensor{"input": input})
	if err != nil {
		return nil, err
	}
	candidates, err := DecodeBlazeFace(result.Outputs["regressors"].Float32, result.Outputs["scores"].Float32, m.anchors, m.cfg.ScoreThreshold)
	if err != nil {
		return nil, err
	}
	kept := WeightedNMS(candidates, m.cfg.ScoreThreshold, m.cfg.NMSThreshold)

	params := DetectorParameters{
		ScoreThreshold: m.cfg.ScoreThreshold, NMSThreshold: m.cfg.NMSThreshold,
		InputWidth: BlazeFaceInputSize, InputHeight: BlazeFaceInputSize, Scale: info.Scale,
	}
	b := img.Bounds()
	toSource := func(p Point) Point {
		// Letterbox resizes with aligned pixel centres, so a point maps back
		// through them.
		x := (p.X-float64(info.PadX)+0.5)/info.Scale - 0.5
		y := (p.Y-float64(info.PadY)+0.5)/info.Scale - 0.5
		return Point{X: x + float64(b.Min.X), Y: y + float64(b.Min.Y)}
	}
	faces := make([]Face, 0, len(kept))
	for _, c := range kept {
		f := Face{
			Score:          c.Score,
			LandmarkScheme: LandmarksMouthCentre,
			Detector:       result.Model,
			Parameters:     params,
		}
		origin := toSource(Point{X: c.Box.X, Y: c.Box.Y})
		f.Box = Box{X: origin.X, Y: origin.Y, Width: c.Box.Width / info.Scale, Height: c.Box.Height / info.Scale}
		for _, p := range c.Keypoints {
			f.Keypoints = append(f.Keypoints, toSource(p))
		}
		f.Landmarks = [5]Point{f.Keypoints[0], f.Keypoints[1], f.Keypoints[2], f.Keypoints[3], f.Keypoints[3]}
		faces = append(faces, f)
	}
	return faces, nil
}

// RawPlanes lays an image out as a float32 tensor of shape [1, 3, height,
// width] holding the 8-bit channel values as they are, 0..255, in BGR order
// when bgr is set and RGB otherwise. Alpha is dropped. It is the input YuNet
// (BGR) and SFace (RGB) read.
func RawPlanes(img image.Image, bgr bool) onnx.Tensor {
	in := asRGBA(img)
	w, h := in.Rect.Dx(), in.Rect.Dy()
	order := [3]int{0, 1, 2}
	if bgr {
		order = [3]int{2, 1, 0}
	}
	plane := w * h
	data := make([]float32, 3*plane)
	for y := 0; y < h; y++ {
		row := in.Pix[y*in.Stride:]
		for x := 0; x < w; x++ {
			for c, source := range order {
				data[c*plane+y*w+x] = float32(row[x*4+source])
			}
		}
	}
	return onnx.Tensor{Shape: []int64{1, 3, int64(h), int64(w)}, Float32: data}
}
