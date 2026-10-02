package feature_test

import (
	"context"
	"image"
	"math"
	"testing"

	face "github.com/kuaa-io/arandu-face"
	onnx "github.com/kuaa-io/arandu-onnx"
)

// These tests run the real detectors on images drawn here. None is a
// photograph of a person: the scenes hold no face at all, and the face is a
// cartoon assembled from ellipses. What they show is that the pre- and
// post-processing is wired the way each model expects -- a detector given the
// wrong channel order, scale or decoding finds nothing on the cartoon, or
// faces in noise. They say nothing about accuracy on real faces, which has to
// be measured on consented data.

// drawn is where cartoonFace puts the landmarks of a face centred at (cx, cy)
// with half-width size: eye centres, nose tip, mouth corners.
func drawn(cx, cy, size float64) [5]face.Point {
	k := size / 95
	return [5]face.Point{
		{X: cx - 40*k, Y: cy - 15*k}, {X: cx + 40*k, Y: cy - 15*k}, {X: cx, Y: cy + 40*k},
		{X: cx - 33*k, Y: cy + 75*k}, {X: cx + 33*k, Y: cy + 75*k},
	}
}

func dist(a, b face.Point) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

func TestNoFaceIsFoundWhereThereIsNone(t *testing.T) {
	ctx := context.Background()
	for _, detector := range []string{face.DetectorYuNet, face.DetectorBlazeFace} {
		m := bootModule(t, face.Config{Detector: detector})
		for name, img := range noFaceScenes() {
			faces, err := m.Detect(ctx, img)
			if err != nil {
				t.Fatalf("%s on %s: %v", detector, name, err)
			}
			if len(faces) != 0 {
				t.Errorf("%s found %d faces in %s at its default threshold, the first scoring %v", detector, len(faces), name, faces[0].Score)
			}
		}
	}
}

func TestYuNetFindsTheCartoonFaceAndItsLandmarks(t *testing.T) {
	m := bootModule(t, face.Config{})
	ctx := context.Background()
	for _, tc := range []struct {
		w, h         int
		cx, cy, size float64
	}{
		{480, 560, 240, 250, 95},
		{640, 480, 300, 230, 70},
		// Larger than DetectMaxSide: detected at half size and mapped back.
		{1280, 1120, 640, 560, 190},
	} {
		img := cartoonFace(tc.w, tc.h, tc.cx, tc.cy, tc.size)
		faces, err := m.Detect(ctx, img)
		if err != nil {
			t.Fatal(err)
		}
		if len(faces) != 1 {
			t.Fatalf("%dx%d: %d faces, want 1", tc.w, tc.h, len(faces))
		}
		f := faces[0]
		if f.Score < face.DefaultYuNetScoreThreshold || f.LandmarkScheme != face.LandmarksFivePoint || len(f.Keypoints) != 5 {
			t.Errorf("%dx%d: score %v scheme %s keypoints %d", tc.w, tc.h, f.Score, f.LandmarkScheme, len(f.Keypoints))
		}
		// Each landmark lands within a fifth of the eye distance of where it
		// was drawn, in the order the template expects: the subject's right
		// eye on the image's left.
		want := drawn(tc.cx, tc.cy, tc.size)
		tolerance := 0.2 * dist(want[0], want[1])
		for i := range want {
			if d := dist(f.Landmarks[i], want[i]); d > tolerance {
				t.Errorf("%dx%d: landmark %d at %+v, %.1f px from the drawn %+v", tc.w, tc.h, i, f.Landmarks[i], d, want[i])
			}
		}
		for _, p := range want {
			if !f.Box.Contains(p) {
				t.Errorf("%dx%d: box %+v does not hold the drawn feature at %+v", tc.w, tc.h, f.Box, p)
			}
		}
		if f.Detector.ID != face.ModelYuNet || f.Detector.Status != onnx.StatusApproved || f.Detector.SHA256 == "" {
			t.Errorf("the face carries the model %+v", f.Detector)
		}
		if f.Parameters.ScoreThreshold != 0.9 || f.Parameters.NMSThreshold != 0.3 || f.Parameters.TopK != 5000 ||
			f.Parameters.InputWidth%32 != 0 || f.Parameters.InputHeight%32 != 0 {
			t.Errorf("the face carries the parameters %+v", f.Parameters)
		}
	}
}

func TestDetectReportsCoordinatesOfTheImageAsGiven(t *testing.T) {
	m := bootModule(t, face.Config{})
	ctx := context.Background()
	whole := cartoonFace(640, 560, 320, 280, 95)
	// A sub-image keeps the parent's coordinates: its bounds start at (64, 32).
	// The offset is a multiple of YuNet's largest stride, so the detector
	// sees the same pixels on the same grid and only the mapping back differs.
	sub := whole.SubImage(image.Rect(64, 32, 640, 560))
	a, err := m.Detect(ctx, whole)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Detect(ctx, sub)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1 || len(b) != 1 {
		t.Fatalf("%d and %d faces, want 1 and 1", len(a), len(b))
	}
	for i := range a[0].Landmarks {
		if d := dist(a[0].Landmarks[i], b[0].Landmarks[i]); d > 0.5 {
			t.Errorf("landmark %d moved %.1f px between the image and its sub-image", i, d)
		}
	}
}

func TestBlazeFaceFindsTheCartoonSelfie(t *testing.T) {
	m := bootModule(t, face.Config{Detector: face.DetectorBlazeFace})
	ctx := context.Background()
	for _, size := range []float64{60, 150} {
		img := cartoonFace(480, 560, 240, 250, size)
		faces, err := m.Detect(ctx, img)
		if err != nil {
			t.Fatal(err)
		}
		if len(faces) != 1 {
			t.Fatalf("size %v: %d faces, want 1", size, len(faces))
		}
		f := faces[0]
		if f.LandmarkScheme != face.LandmarksMouthCentre || len(f.Keypoints) != face.BlazeFaceKeypoints {
			t.Errorf("scheme %s with %d keypoints", f.LandmarkScheme, len(f.Keypoints))
		}
		// The mouth centre stands in both mouth-corner slots.
		if f.Landmarks[3] != f.Landmarks[4] || f.Landmarks[3] != f.Keypoints[3] {
			t.Errorf("mouth slots %+v and %+v, keypoint %+v", f.Landmarks[3], f.Landmarks[4], f.Keypoints[3])
		}
		want := drawn(240, 250, size)
		tolerance := 0.25 * dist(want[0], want[1])
		mouth := face.Point{X: 240, Y: 250 + 75*size/95}
		for i, p := range []face.Point{want[0], want[1], want[2], mouth} {
			if d := dist(f.Keypoints[i], p); d > tolerance {
				t.Errorf("size %v: keypoint %d at %+v, %.1f px from the drawn %+v", size, i, f.Keypoints[i], d, p)
			}
		}
		if f.Detector.ID != face.ModelBlazeFace || f.Parameters.InputWidth != 128 {
			t.Errorf("model %+v parameters %+v", f.Detector, f.Parameters)
		}
	}
}

func TestEachStageOnTheCartoonFace(t *testing.T) {
	m := bootModule(t, face.Config{EnableEmbedding: true, EnableAge: true})
	ctx := context.Background()
	img := cartoonFace(480, 560, 240, 250, 95)
	faces, err := m.Detect(ctx, img)
	if err != nil || len(faces) != 1 {
		t.Fatalf("Detect = %d faces, %v", len(faces), err)
	}
	f := faces[0]

	aligned, err := face.Align(img, f.Landmarks)
	if err != nil {
		t.Fatal(err)
	}
	// The drawing's proportions are not a real face's, so the landmarks do
	// not sit exactly on the template; they sit within a few pixels.
	if aligned.Residual > 5 {
		t.Errorf("alignment residual %.2f px", aligned.Residual)
	}

	report, err := m.Quality(img, f)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range report.Checks {
		if !c.Pass {
			t.Errorf("the cartoon fails %s: %v against %s", c.Signal, c.Value, c.Rule)
		}
	}

	e, err := m.Embed(ctx, aligned.Image)
	if err != nil {
		t.Fatal(err)
	}
	if e.Model.ID != face.ModelSFace || e.Model.Status != onnx.StatusDevOnly {
		t.Errorf("the embedding carries %+v, want SFace marked dev-only", e.Model)
	}

	age, err := m.EstimateAge(ctx, img, f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if age.Model.ID != face.ModelMiVOLOAge || age.Model.Status != onnx.StatusDevOnly {
		t.Errorf("the age estimate carries %+v, want MiVOLO marked dev-only", age.Model)
	}
}
