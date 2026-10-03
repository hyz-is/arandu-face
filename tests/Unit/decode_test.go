package unit_test

import (
	"fmt"
	"math"
	"testing"

	face "github.com/hyz-is/arandu-face"
	onnx "github.com/hyz-is/arandu-onnx"
)

// yunetOutputs builds zeroed YuNet outputs for a padW by padH input.
func yunetOutputs(padW, padH int) map[string]onnx.Tensor {
	out := map[string]onnx.Tensor{}
	for _, s := range face.YuNetStrides {
		n := (padW / s) * (padH / s)
		for kind, per := range map[string]int{"cls": 1, "obj": 1, "bbox": 4, "kps": 10} {
			out[fmt.Sprintf("%s_%d", kind, s)] = onnx.Tensor{Shape: []int64{1, int64(n), int64(per)}, Float32: make([]float32, n*per)}
		}
	}
	return out
}

func TestYuNetPadsToMultiplesOf32(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct{ w, h, pw, ph int }{
		{1, 1, 32, 32}, {32, 32, 32, 32}, {33, 64, 64, 64}, {640, 480, 640, 480}, {641, 481, 672, 512},
	} {
		if pw, ph := face.YuNetPaddedSize(tc.w, tc.h); pw != tc.pw || ph != tc.ph {
			t.Errorf("YuNetPaddedSize(%d, %d) = %d, %d; want %d, %d", tc.w, tc.h, pw, ph, tc.pw, tc.ph)
		}
	}
	if got := face.YuNetOutputNames(); len(got) != 12 || got[0] != "cls_8" || got[11] != "kps_32" {
		t.Errorf("YuNetOutputNames() = %v", got)
	}
}

func TestDecodeYuNetOnSyntheticTensors(t *testing.T) {
	t.Parallel()

	// A 64×32 input: stride 8 has 8 columns and 4 rows, stride 32 has 2×1.
	out := yunetOutputs(64, 32)
	// Stride 8, anchor 11: row 11/8 = 1, column 11%8 = 3.
	out["cls_8"].Float32[11] = 0.81
	out["obj_8"].Float32[11] = 1
	copy(out["bbox_8"].Float32[11*4:], []float32{0.5, 0.25, float32(math.Log(2)), float32(math.Log(4))})
	copy(out["kps_8"].Float32[11*10:], []float32{0.1, 0.2, 0.9, 0.2, 0.5, 0.6, 0.2, 1.0, 0.8, 1.0})
	// Stride 32, anchor 1: row 0, column 1. cls 1.5 clamps to 1.
	out["cls_32"].Float32[1] = 1.5
	out["obj_32"].Float32[1] = 0.64
	// Stride 16, anchor 5: cls −0.2 clamps to 0, so the score is 0.
	out["cls_16"].Float32[5] = -0.2
	out["obj_16"].Float32[5] = 1

	got, err := face.DecodeYuNet(out, 64, 32, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("decoded %d candidates, want 2: %+v", len(got), got)
	}
	const tol = 1e-5
	// score √(0.81·1) = 0.9; centre ((3+0.5)·8, (1+0.25)·8) = (28, 10);
	// size (2·8, 4·8) = (16, 32); top-left (20, −6).
	c := got[0]
	if !near(c.Score, 0.9, tol) || !near(c.Box.X, 20, tol) || !near(c.Box.Y, -6, tol) || !near(c.Box.Width, 16, tol) || !near(c.Box.Height, 32, tol) {
		t.Errorf("stride-8 candidate = %+v, want score 0.9 box {20 -6 16 32}", c)
	}
	// landmark n = ((k[2n]+3)·8, (k[2n+1]+1)·8).
	want := []face.Point{{X: 24.8, Y: 9.6}, {X: 31.2, Y: 9.6}, {X: 28, Y: 12.8}, {X: 25.6, Y: 16}, {X: 30.4, Y: 16}}
	for i, p := range want {
		if !near(c.Keypoints[i].X, p.X, tol) || !near(c.Keypoints[i].Y, p.Y, tol) {
			t.Errorf("landmark %d = %+v, want %+v", i, c.Keypoints[i], p)
		}
	}
	// score √(1·0.64) = 0.8; centre ((1+0)·32, 0) = (32, 0), size 32.
	c = got[1]
	if !near(c.Score, 0.8, tol) || !near(c.Box.X, 16, tol) || !near(c.Box.Y, -16, tol) || !near(c.Box.Width, 32, tol) {
		t.Errorf("stride-32 candidate = %+v, want score 0.8 box {16 -16 32 32}", c)
	}

	if got, _ := face.DecodeYuNet(out, 64, 32, 0.85); len(got) != 1 {
		t.Errorf("at 0.85, %d candidates, want 1", len(got))
	}
}

func TestDecodeYuNetRefusesMalformedOutputs(t *testing.T) {
	t.Parallel()

	if _, err := face.DecodeYuNet(yunetOutputs(64, 32), 60, 32, 0.5); err == nil {
		t.Error("an input not padded to 32 was accepted")
	}
	out := yunetOutputs(64, 32)
	delete(out, "kps_16")
	if _, err := face.DecodeYuNet(out, 64, 32, 0.5); err == nil {
		t.Error("a missing output was accepted")
	}
	if _, err := face.DecodeYuNet(yunetOutputs(64, 32), 64, 64, 0.5); err == nil {
		t.Error("outputs for another size were accepted")
	}
}

func TestBlazeFaceHas896Anchors(t *testing.T) {
	t.Parallel()

	a := face.BlazeFaceAnchors()
	if face.BlazeFaceAnchorCount != 896 || len(a) != face.BlazeFaceAnchorCount {
		t.Fatalf("%d anchors, want 896", len(a))
	}
	at := func(i int, x, y float64) {
		t.Helper()
		if !near(a[i].X, x, eps) || !near(a[i].Y, y, eps) || a[i].Width != 1 || a[i].Height != 1 {
			t.Errorf("anchor %d = %+v, want (%v, %v) size 1", i, a[i], x, y)
		}
	}
	// Stride 8: a 16×16 grid, two anchors per cell, centres (x+0.5)/16.
	at(0, 0.5/16, 0.5/16)
	at(1, 0.5/16, 0.5/16)
	at(2, 1.5/16, 0.5/16)
	at(511, 15.5/16, 15.5/16)
	// Stride 16: an 8×8 grid, six anchors per cell, centres (x+0.5)/8.
	for i := 512; i < 518; i++ {
		at(i, 0.5/8, 0.5/8)
	}
	at(518, 1.5/8, 0.5/8)
	at(895, 7.5/8, 7.5/8)
}

func TestDecodeBlazeFaceOnSyntheticTensors(t *testing.T) {
	t.Parallel()

	anchors := face.BlazeFaceAnchors()
	regressors := make([]float32, 896*16)
	scores := make([]float32, 896)
	for i := range scores {
		scores[i] = -1000 // clipped to −100: sigmoid ≈ 0
	}
	scores[0] = float32(math.Log(4)) // sigmoid = 4/5
	copy(regressors, []float32{12.8, -6.4, 25.6, 38.4, 6.4, 12.8})
	scores[1] = 1000 // clipped to 100: sigmoid ≈ 1

	got, err := face.DecodeBlazeFace(regressors, scores, anchors, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("decoded %d candidates, want 2", len(got))
	}
	const tol = 1e-5
	// Anchor 0 is at (1/32, 1/32). Centre (12.8/128 + 1/32)·128 = 16.8 and
	// (−6.4/128 + 1/32)·128 = −2.4; size 25.6 × 38.4; top-left (4, −21.6).
	c := got[0]
	if !near(c.Score, 0.8, tol) || !near(c.Box.X, 4, tol) || !near(c.Box.Y, -21.6, tol) || !near(c.Box.Width, 25.6, tol) || !near(c.Box.Height, 38.4, tol) {
		t.Errorf("candidate = %+v, want score 0.8 box {4 -21.6 25.6 38.4}", c)
	}
	// Keypoint 0: (6.4/128 + 1/32)·128 = 10.4, (12.8/128 + 1/32)·128 = 16.8.
	if len(c.Keypoints) != 6 || !near(c.Keypoints[0].X, 10.4, tol) || !near(c.Keypoints[0].Y, 16.8, tol) {
		t.Errorf("keypoints = %+v, want 6 with the first at (10.4, 16.8)", c.Keypoints)
	}
	// Zero regressors put keypoints on the anchor centre: (4, 4) for anchor 1.
	if !near(got[1].Score, 1, 1e-9) || !near(got[1].Keypoints[5].X, 4, tol) {
		t.Errorf("clipped candidate = %+v", got[1])
	}

	if _, err := face.DecodeBlazeFace(regressors[:10], scores, anchors, 0.5); err == nil {
		t.Error("short regressors were accepted")
	}
}
