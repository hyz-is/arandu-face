package unit_test

import (
	"math"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

const eps = 1e-9

func near(a, b, tolerance float64) bool { return math.Abs(a-b) <= tolerance }

func TestIoU(t *testing.T) {
	t.Parallel()

	a := face.Box{X: 0, Y: 0, Width: 10, Height: 10}
	if got := face.IoU(a, a); !near(got, 1, eps) {
		t.Errorf("IoU of a box with itself = %v, want 1", got)
	}
	if got := face.IoU(a, face.Box{X: 20, Y: 20, Width: 5, Height: 5}); got != 0 {
		t.Errorf("IoU of disjoint boxes = %v, want 0", got)
	}
	// (1..11)×(1..11) against (0..10)×(0..10): 9×9 = 81 shared, 100+100−81 = 119 covered.
	if got := face.IoU(a, face.Box{X: 1, Y: 1, Width: 10, Height: 10}); !near(got, 81.0/119, eps) {
		t.Errorf("IoU = %v, want 81/119", got)
	}
	if got := face.IoU(a, face.Box{X: 0, Y: 0, Width: 0, Height: 10}); got != 0 {
		t.Errorf("IoU with an empty box = %v, want 0", got)
	}
}

func TestNMSKeepsTheBestOfEachOverlap(t *testing.T) {
	t.Parallel()

	a := face.Candidate{Box: face.Box{X: 0, Y: 0, Width: 10, Height: 10}, Score: 0.9}
	b := face.Candidate{Box: face.Box{X: 1, Y: 1, Width: 10, Height: 10}, Score: 0.8}
	c := face.Candidate{Box: face.Box{X: 20, Y: 20, Width: 10, Height: 10}, Score: 0.7}
	d := face.Candidate{Box: face.Box{X: 5, Y: 0, Width: 10, Height: 10}, Score: 0.95}
	e := face.Candidate{Box: face.Box{X: 40, Y: 40, Width: 10, Height: 10}, Score: 0.2}
	in := []face.Candidate{a, b, c, d, e}

	scores := func(cs []face.Candidate) []float64 {
		out := []float64{}
		for _, c := range cs {
			out = append(out, c.Score)
		}
		return out
	}
	equal := func(got []face.Candidate, want ...float64) bool {
		s := scores(got)
		if len(s) != len(want) {
			return false
		}
		for i := range s {
			if s[i] != want[i] {
				return false
			}
		}
		return true
	}

	// D (0.95) first. A overlaps D by 50/150 = 0.333 and B by 54/146 = 0.370,
	// both over 0.3; C is apart; E scores under 0.5.
	if got := face.NMS(in, 0.5, 0.3, 0); !equal(got, 0.95, 0.7) {
		t.Errorf("NMS at IoU 0.3 kept %v, want [0.95 0.7]", scores(got))
	}
	// At 0.35, A's 0.333 with D survives; B still overlaps A by 0.68.
	if got := face.NMS(in, 0.5, 0.35, 0); !equal(got, 0.95, 0.9, 0.7) {
		t.Errorf("NMS at IoU 0.35 kept %v, want [0.95 0.9 0.7]", scores(got))
	}
	if got := face.NMS(in, 0.5, 0.3, 1); !equal(got, 0.95) {
		t.Errorf("NMS with topK 1 kept %v, want [0.95]", scores(got))
	}
	if got := face.NMS(nil, 0.5, 0.3, 0); len(got) != 0 {
		t.Errorf("NMS of nothing kept %v", got)
	}
}

func TestWeightedNMSAveragesEachGroupByScore(t *testing.T) {
	t.Parallel()

	p := face.Candidate{Box: face.Box{X: 0, Y: 0, Width: 10, Height: 10}, Score: 0.8, Keypoints: []face.Point{{X: 1, Y: 1}}}
	q := face.Candidate{Box: face.Box{X: 2, Y: 0, Width: 10, Height: 10}, Score: 0.2, Keypoints: []face.Point{{X: 3, Y: 1}}}
	r := face.Candidate{Box: face.Box{X: 50, Y: 50, Width: 10, Height: 10}, Score: 0.6, Keypoints: []face.Point{{X: 55, Y: 55}}}
	low := face.Candidate{Box: face.Box{X: 0, Y: 0, Width: 10, Height: 10}, Score: 0.1, Keypoints: []face.Point{{X: 9, Y: 9}}}

	got := face.WeightedNMS([]face.Candidate{q, r, p, low}, 0.15, 0.3)
	if len(got) != 2 {
		t.Fatalf("WeightedNMS returned %d groups, want 2: %+v", len(got), got)
	}
	// P and Q overlap by 80/120 = 0.667. Weights 0.8 and 0.2 (total 1):
	// x0 = 0·0.8 + 2·0.2 = 0.4, x1 = 10·0.8 + 12·0.2 = 10.4, keypoint x =
	// 1·0.8 + 3·0.2 = 1.4. The score stays P's. "low" scores under 0.15.
	m := got[0]
	if !near(m.Box.X, 0.4, eps) || !near(m.Box.Y, 0, eps) || !near(m.Box.Width, 10, eps) || !near(m.Box.Height, 10, eps) {
		t.Errorf("merged box = %+v, want {0.4 0 10 10}", m.Box)
	}
	if !near(m.Keypoints[0].X, 1.4, eps) || !near(m.Keypoints[0].Y, 1, eps) || m.Score != 0.8 {
		t.Errorf("merged keypoint %+v score %v, want {1.4 1} 0.8", m.Keypoints[0], m.Score)
	}
	if got[1].Score != 0.6 || got[1].Box != r.Box {
		t.Errorf("the apart candidate became %+v", got[1])
	}
}
