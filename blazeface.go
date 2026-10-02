package face

import (
	"fmt"
	"math"
)

// BlazeFaceInputSize is the side of the square image BlazeFace short range
// reads.
const BlazeFaceInputSize = 128

// BlazeFaceAnchorCount is the number of anchors of BlazeFace short range.
const BlazeFaceAnchorCount = 896

// BlazeFaceKeypoints is the number of keypoints BlazeFace emits per face, in
// this order: right eye, left eye, nose tip, mouth centre, right ear
// tragion, left ear tragion.
const BlazeFaceKeypoints = 6

// blazeFaceScoreClip is the raw score MediaPipe clips to before the sigmoid.
const blazeFaceScoreClip = 100

// Anchor is the centre of one SSD anchor on the unit square, with the anchor
// size BlazeFace short range uses: 1 by 1.
type Anchor struct {
	X, Y, Width, Height float64
}

// BlazeFaceAnchors returns the 896 anchors of BlazeFace short range, in the
// order of the model's outputs.
//
// They are MediaPipe's SSD anchors for the model: input 128, strides 8, 16,
// 16 and 16, offset 0.5, fixed size. Layers that share a stride are merged,
// so stride 8 gives a 16×16 grid of 2 anchors per cell (512) and stride 16 an
// 8×8 grid of 6 per cell (384).
func BlazeFaceAnchors() []Anchor {
	strides := []int{8, 16, 16, 16}
	anchors := make([]Anchor, 0, BlazeFaceAnchorCount)
	for layer := 0; layer < len(strides); {
		perCell := 0
		last := layer
		for last < len(strides) && strides[last] == strides[layer] {
			// Each layer contributes one aspect ratio and one interpolated
			// scale: two anchors per cell.
			perCell += 2
			last++
		}
		grid := int(math.Ceil(float64(BlazeFaceInputSize) / float64(strides[layer])))
		for y := 0; y < grid; y++ {
			for x := 0; x < grid; x++ {
				for k := 0; k < perCell; k++ {
					anchors = append(anchors, Anchor{
						X:      (float64(x) + 0.5) / float64(grid),
						Y:      (float64(y) + 0.5) / float64(grid),
						Width:  1,
						Height: 1,
					})
				}
			}
		}
		layer = last
	}
	return anchors
}

// DecodeBlazeFace turns BlazeFace's raw outputs for one image into
// candidates scoring at least minScore, on the 128×128 input, in pixels.
//
// regressors holds 16 values per anchor and scores one; anchors are
// BlazeFaceAnchors. It is MediaPipe's TensorsToDetections for the model: the
// score is the sigmoid of the raw score clipped to ±100; the box centre is
// raw/128·anchor size + anchor centre, its size raw/128·anchor size; keypoint
// k is raw[4+2k]/128 and raw[5+2k]/128 offset by the anchor centre. The unit
// square is then scaled to the input's 128 pixels.
//
// No suppression is applied; WeightedNMS does that.
func DecodeBlazeFace(regressors, scores []float32, anchors []Anchor, minScore float64) ([]Candidate, error) {
	n := len(anchors)
	if len(regressors) != n*16 || len(scores) != n {
		return nil, fmt.Errorf("face: BlazeFace outputs have %d regressors and %d scores, want %d and %d for %d anchors",
			len(regressors), len(scores), n*16, n, n)
	}
	const size = float64(BlazeFaceInputSize)
	out := []Candidate{}
	for i, a := range anchors {
		raw := math.Max(-blazeFaceScoreClip, math.Min(blazeFaceScoreClip, float64(scores[i])))
		score := 1 / (1 + math.Exp(-raw))
		if score < minScore {
			continue
		}
		r := regressors[i*16 : i*16+16]
		cx := float64(r[0])/size*a.Width + a.X
		cy := float64(r[1])/size*a.Height + a.Y
		w := float64(r[2]) / size * a.Width
		h := float64(r[3]) / size * a.Height
		points := make([]Point, BlazeFaceKeypoints)
		for k := range points {
			points[k] = Point{
				X: (float64(r[4+2*k])/size*a.Width + a.X) * size,
				Y: (float64(r[5+2*k])/size*a.Height + a.Y) * size,
			}
		}
		out = append(out, Candidate{
			Box:       Box{X: (cx - w/2) * size, Y: (cy - h/2) * size, Width: w * size, Height: h * size},
			Score:     score,
			Keypoints: points,
		})
	}
	return out, nil
}
