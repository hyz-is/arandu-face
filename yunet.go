package face

import (
	"fmt"
	"math"

	onnx "github.com/hyz-is/arandu-onnx"
)

// YuNetStrides are the feature-map strides YuNet predicts at, in the order
// its outputs are named.
var YuNetStrides = [3]int{8, 16, 32}

// YuNetDivisor is the multiple both sides of YuNet's input are padded to.
const YuNetDivisor = 32

// YuNetPaddedSize is the size of the input YuNet is given for an image of
// width by height: each side rounded up to a multiple of YuNetDivisor, the
// padding added to the bottom and the right.
func YuNetPaddedSize(width, height int) (int, int) {
	pad := func(n int) int { return ((n-1)/YuNetDivisor + 1) * YuNetDivisor }
	return pad(width), pad(height)
}

// YuNetOutputNames are the twelve outputs DecodeYuNet reads, by stride:
// cls_s, obj_s, bbox_s and kps_s for s in YuNetStrides.
func YuNetOutputNames() []string {
	out := []string{}
	for _, kind := range []string{"cls", "obj", "bbox", "kps"} {
		for _, s := range YuNetStrides {
			out = append(out, fmt.Sprintf("%s_%d", kind, s))
		}
	}
	return out
}

// DecodeYuNet turns YuNet's raw outputs, for an input padded to padW by padH,
// into candidates scoring at least minScore, in input-pixel coordinates.
//
// It is OpenCV's FaceDetectorYN post-processing. For each stride s the
// feature map has padW/s columns; anchor idx sits at row idx/cols and column
// idx%cols. The score is the square root of the class and objectness
// outputs, each clamped to 0..1. The box centre is ((c+b0)·s, (r+b1)·s) and
// its size (exp(b2)·s, exp(b3)·s). Landmark n is ((k[2n]+c)·s, (k[2n+1]+r)·s),
// in YuNet's order: right eye, left eye, nose tip, right mouth corner, left
// mouth corner, "right" being the subject's right, which is the image's left
// on a photograph of someone facing the camera.
//
// No suppression is applied; NMS does that.
func DecodeYuNet(outputs map[string]onnx.Tensor, padW, padH int, minScore float64) ([]Candidate, error) {
	if padW <= 0 || padH <= 0 || padW%YuNetDivisor != 0 || padH%YuNetDivisor != 0 {
		return nil, fmt.Errorf("face: YuNet input %dx%d is not padded to a multiple of %d", padW, padH, YuNetDivisor)
	}
	out := []Candidate{}
	for _, s := range YuNetStrides {
		cols, rows := padW/s, padH/s
		n := cols * rows
		get := func(kind string, per int) ([]float32, error) {
			name := fmt.Sprintf("%s_%d", kind, s)
			t, ok := outputs[name]
			if !ok {
				return nil, fmt.Errorf("face: YuNet output %s is missing", name)
			}
			if len(t.Float32) != n*per {
				return nil, fmt.Errorf("face: YuNet output %s has %d values, want %d for a %dx%d input", name, len(t.Float32), n*per, padW, padH)
			}
			return t.Float32, nil
		}
		cls, err := get("cls", 1)
		if err != nil {
			return nil, err
		}
		obj, err := get("obj", 1)
		if err != nil {
			return nil, err
		}
		bbox, err := get("bbox", 4)
		if err != nil {
			return nil, err
		}
		kps, err := get("kps", 10)
		if err != nil {
			return nil, err
		}
		stride := float64(s)
		for idx := 0; idx < n; idx++ {
			score := math.Sqrt(clamp01(float64(cls[idx])) * clamp01(float64(obj[idx])))
			if score < minScore {
				continue
			}
			r, c := float64(idx/cols), float64(idx%cols)
			b := bbox[idx*4 : idx*4+4]
			cx := (c + float64(b[0])) * stride
			cy := (r + float64(b[1])) * stride
			w := math.Exp(float64(b[2])) * stride
			h := math.Exp(float64(b[3])) * stride
			points := make([]Point, 5)
			for k := range points {
				points[k] = Point{
					X: (float64(kps[idx*10+2*k]) + c) * stride,
					Y: (float64(kps[idx*10+2*k+1]) + r) * stride,
				}
			}
			out = append(out, Candidate{
				Box:       Box{X: cx - w/2, Y: cy - h/2, Width: w, Height: h},
				Score:     score,
				Keypoints: points,
			})
		}
	}
	return out, nil
}

// clamp01 limits v to 0..1; NaN becomes 0.
func clamp01(v float64) float64 {
	if !(v > 0) {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
