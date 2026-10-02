package face

import (
	"image"
	"math"
	"sort"
)

// Point is a position on an image, in pixels, with the origin at the
// top-left corner and Y pointing down. Integer coordinates are pixel centres.
type Point struct {
	X, Y float64
}

// Box is an axis-aligned rectangle on an image, in pixels: its top-left
// corner and its size.
type Box struct {
	X, Y, Width, Height float64
}

// Area is the box's width times its height, or 0 for a box with no extent.
func (b Box) Area() float64 {
	if b.Width <= 0 || b.Height <= 0 {
		return 0
	}
	return b.Width * b.Height
}

// Center is the middle of the box.
func (b Box) Center() Point {
	return Point{X: b.X + b.Width/2, Y: b.Y + b.Height/2}
}

// Contains reports whether p lies inside the box or on its edge.
func (b Box) Contains(p Point) bool {
	return p.X >= b.X && p.X <= b.X+b.Width && p.Y >= b.Y && p.Y <= b.Y+b.Height
}

// Rect is the smallest integer rectangle that holds the box, which is what a
// crop of it reads.
func (b Box) Rect() image.Rectangle {
	return image.Rect(
		int(math.Floor(b.X)), int(math.Floor(b.Y)),
		int(math.Ceil(b.X+b.Width)), int(math.Ceil(b.Y+b.Height)),
	)
}

// IoU is the intersection over union of two boxes: the area they share
// divided by the area either covers. Two boxes with no area give 0.
func IoU(a, b Box) float64 {
	x0 := math.Max(a.X, b.X)
	y0 := math.Max(a.Y, b.Y)
	x1 := math.Min(a.X+a.Width, b.X+b.Width)
	y1 := math.Min(a.Y+a.Height, b.Y+b.Height)
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	inter := (x1 - x0) * (y1 - y0)
	union := a.Area() + b.Area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// Candidate is one face a detector's raw outputs propose before suppression:
// a box, a score between 0 and 1, and the detector's keypoints in its own
// order.
type Candidate struct {
	// Box is the face's bounding box.
	Box Box
	// Score is the detector's confidence, between 0 and 1.
	Score float64
	// Keypoints are the landmarks the detector emits, in the order it emits
	// them: five for YuNet, six for BlazeFace.
	Keypoints []Point
}

// NMS is greedy non-maximum suppression as OpenCV's NMSBoxes performs it.
//
// Candidates scoring below minScore are dropped, the rest are sorted by score
// from the highest (ties keep their input order), cut to the topK best when
// topK is positive, and then each one is kept unless its IoU with a box
// already kept exceeds iouThreshold. The result is in descending score order.
func NMS(candidates []Candidate, minScore, iouThreshold float64, topK int) []Candidate {
	order := make([]int, 0, len(candidates))
	for i, c := range candidates {
		if c.Score >= minScore {
			order = append(order, i)
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		return candidates[order[i]].Score > candidates[order[j]].Score
	})
	if topK > 0 && len(order) > topK {
		order = order[:topK]
	}
	kept := []Candidate{}
	for _, i := range order {
		c := candidates[i]
		suppressed := false
		for _, k := range kept {
			if IoU(c.Box, k.Box) > iouThreshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			kept = append(kept, c)
		}
	}
	return kept
}

// WeightedNMS is the weighted non-maximum suppression MediaPipe applies to
// BlazeFace detections.
//
// Candidates scoring below minScore are dropped and the rest sorted by score.
// The best remaining candidate gathers every remaining one whose IoU with it
// exceeds iouThreshold, itself included; the box corners and keypoints of the
// group are averaged with the scores as weights, and the group's result keeps
// the best candidate's score. Gathered candidates leave the pool, and the
// next best starts the next group.
func WeightedNMS(candidates []Candidate, minScore, iouThreshold float64) []Candidate {
	pool := []Candidate{}
	for _, c := range candidates {
		if c.Score >= minScore {
			pool = append(pool, c)
		}
	}
	sort.SliceStable(pool, func(i, j int) bool { return pool[i].Score > pool[j].Score })

	out := []Candidate{}
	for len(pool) > 0 {
		best := pool[0]
		group := []Candidate{}
		rest := []Candidate{}
		for _, c := range pool {
			if IoU(best.Box, c.Box) > iouThreshold {
				group = append(group, c)
			} else {
				rest = append(rest, c)
			}
		}
		// The best candidate overlaps itself unless its box is empty; an empty
		// box forms a group of its own so the loop always advances.
		if len(group) == 0 {
			group = []Candidate{best}
			rest = pool[1:]
		}

		var total, x0, y0, x1, y1 float64
		keypoints := make([]Point, len(best.Keypoints))
		for _, c := range group {
			w := c.Score
			total += w
			x0 += c.Box.X * w
			y0 += c.Box.Y * w
			x1 += (c.Box.X + c.Box.Width) * w
			y1 += (c.Box.Y + c.Box.Height) * w
			for k := range keypoints {
				if k < len(c.Keypoints) {
					keypoints[k].X += c.Keypoints[k].X * w
					keypoints[k].Y += c.Keypoints[k].Y * w
				}
			}
		}
		merged := best
		if total > 0 {
			merged.Box = Box{X: x0 / total, Y: y0 / total, Width: (x1 - x0) / total, Height: (y1 - y0) / total}
			for k := range keypoints {
				keypoints[k].X /= total
				keypoints[k].Y /= total
			}
			merged.Keypoints = keypoints
		}
		out = append(out, merged)
		pool = rest
	}
	return out
}
