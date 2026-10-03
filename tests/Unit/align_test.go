package unit_test

import (
	"errors"
	"image"
	"math"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

// similarity is s·R(θ) plus (tx, ty), θ counter-clockwise in degrees.
func similarity(s, degrees, tx, ty float64) face.Affine {
	c, n := math.Cos(degrees*math.Pi/180), math.Sin(degrees*math.Pi/180)
	return face.Affine{{s * c, -s * n, tx}, {s * n, s * c, ty}}
}

func affineNear(a, b face.Affine, tolerance float64) bool {
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			if !near(a[r][c], b[r][c], tolerance) {
				return false
			}
		}
	}
	return true
}

func TestSVD2ReconstructsTheMatrix(t *testing.T) {
	t.Parallel()

	for _, a := range [][2][2]float64{
		{{3, 1}, {1, 3}},
		{{2, 0}, {0, -5}},
		{{1, 2}, {3, 4}},  // negative determinant
		{{1, 2}, {2, 4}},  // rank one
		{{0, -1}, {1, 0}}, // a rotation
		{{1e-3, 7}, {-2, 0.5}},
	} {
		svd := face.DecomposeSVD2(a)
		if svd.S[0] < svd.S[1] || svd.S[1] < 0 {
			t.Errorf("%v: singular values %v are not ordered and non-negative", a, svd.S)
		}
		for r := 0; r < 2; r++ {
			for c := 0; c < 2; c++ {
				got := svd.U[r][0]*svd.S[0]*svd.Vt[0][c] + svd.U[r][1]*svd.S[1]*svd.Vt[1][c]
				if !near(got, a[r][c], 1e-12) {
					t.Errorf("%v: U·S·Vt[%d][%d] = %v", a, r, c, got)
				}
				// Columns of U and rows of Vt are orthonormal.
				uu := svd.U[0][r]*svd.U[0][c] + svd.U[1][r]*svd.U[1][c]
				vv := svd.Vt[r][0]*svd.Vt[c][0] + svd.Vt[r][1]*svd.Vt[c][1]
				want := 0.0
				if r == c {
					want = 1
				}
				if !near(uu, want, 1e-12) || !near(vv, want, 1e-12) {
					t.Errorf("%v: U or Vt is not orthogonal", a)
				}
			}
		}
	}
}

func TestEstimateSimilarityRecoversAKnownTransform(t *testing.T) {
	t.Parallel()

	for _, want := range []face.Affine{
		similarity(1.7, 20, 12, -5),
		similarity(0.35, -170, -40, 300),
		similarity(1, 0, 0, 0),
		similarity(2.5, 90, 7, 7),
	} {
		dst := make([]face.Point, 5)
		for i, p := range face.ArcFaceTemplate {
			dst[i] = want.Apply(p)
		}
		got, err := face.EstimateSimilarity(face.ArcFaceTemplate[:], dst)
		if err != nil {
			t.Fatal(err)
		}
		if !affineNear(got, want, 1e-9) {
			t.Errorf("recovered %v, want %v", got, want)
		}
	}
}

func TestEstimateSimilarityNeverReflects(t *testing.T) {
	t.Parallel()

	src := face.ArcFaceTemplate[:]
	mirrored := make([]face.Point, len(src))
	for i, p := range src {
		mirrored[i] = face.Point{X: 200 - p.X, Y: p.Y}
	}
	m, err := face.EstimateSimilarity(src, mirrored)
	if err != nil {
		t.Fatal(err)
	}
	if det := m[0][0]*m[1][1] - m[0][1]*m[1][0]; det <= 0 {
		t.Errorf("the fit to a mirror image has determinant %v: it reflects", det)
	}

	// Collinear points have a rank-one covariance; the fit is still exact.
	line := []face.Point{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 3, Y: 0}}
	want := similarity(2, 30, 5, 5)
	moved := []face.Point{want.Apply(line[0]), want.Apply(line[1]), want.Apply(line[2])}
	got, err := face.EstimateSimilarity(line, moved)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range line {
		if q := got.Apply(p); !near(q.X, moved[i].X, 1e-9) || !near(q.Y, moved[i].Y, 1e-9) {
			t.Errorf("collinear point %d maps to %+v, want %+v", i, q, moved[i])
		}
	}
}

func TestEstimateSimilarityRefusesDegeneratePoints(t *testing.T) {
	t.Parallel()

	same := []face.Point{{X: 3, Y: 3}, {X: 3, Y: 3}, {X: 3, Y: 3}}
	for name, tc := range map[string][2][]face.Point{
		"one point":   {{{X: 1, Y: 1}}, {{X: 2, Y: 2}}},
		"mismatched":  {same, same[:2]},
		"coincident":  {same, same},
		"not finite":  {{{X: math.NaN(), Y: 0}, {X: 1, Y: 1}}, {{X: 0, Y: 0}, {X: 1, Y: 1}}},
		"no elements": {nil, nil},
	} {
		if _, err := face.EstimateSimilarity(tc[0], tc[1]); !errors.Is(err, face.ErrDegenerateLandmarks) {
			t.Errorf("%s: err = %v, want ErrDegenerateLandmarks", name, err)
		}
	}
}

// grey2x2 is a 2×2 image whose pixels are 0, 100 / 200, 40 on every channel.
func grey2x2() *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i, v := range []uint8{0, 100, 200, 40} {
		copy(img.Pix[i*4:], []uint8{v, v, v, 255})
	}
	return img
}

func greys(img *image.RGBA) []uint8 {
	out := []uint8{}
	for i := 0; i < len(img.Pix); i += 4 {
		out = append(out, img.Pix[i])
	}
	return out
}

func TestWarpAffineGoldenValues(t *testing.T) {
	t.Parallel()

	// Shift by half a pixel right and down: output (x, y) reads the source
	// at (x−0.5, y−0.5), a quarter from each of four pixels, outside ones 0.
	//   (0,0): ¼·0 = 0      (1,0): ¼·100 = 25
	//   (0,1): ¼·200 = 50   (1,1): ¼·(0+100+200+40) = 85
	got, err := face.WarpAffine(grey2x2(), face.Affine{{1, 0, 0.5}, {0, 1, 0.5}}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if g := greys(got); g[0] != 0 || g[1] != 25 || g[2] != 50 || g[3] != 85 {
		t.Errorf("half-pixel shift = %v, want [0 25 50 85]", g)
	}
	for i := 3; i < len(got.Pix); i += 4 {
		if got.Pix[i] != 255 {
			t.Fatal("the warp is not opaque")
		}
	}

	// Scale by 2: output x reads source x/2. Row 0: 0, ½(0+100) = 50, 100,
	// then 1.5 sits between 100 and the black border: 50.
	got, err = face.WarpAffine(grey2x2(), face.Affine{{2, 0, 0}, {0, 2, 0}}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	if g := greys(got); g[0] != 0 || g[1] != 50 || g[2] != 100 || g[3] != 50 {
		t.Errorf("scale by 2 = %v, want [0 50 100 50]", g)
	}

	// The identity reproduces the image.
	got, err = face.WarpAffine(grey2x2(), face.Affine{{1, 0, 0}, {0, 1, 0}}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if g := greys(got); g[0] != 0 || g[1] != 100 || g[2] != 200 || g[3] != 40 {
		t.Errorf("identity = %v, want [0 100 200 40]", g)
	}

	if _, err := face.WarpAffine(grey2x2(), face.Affine{{0, 0, 0}, {0, 0, 0}}, 2, 2); err == nil {
		t.Error("a singular transform was accepted")
	}
}

func TestAlignMapsTheLandmarksOntoTheTemplate(t *testing.T) {
	t.Parallel()

	// Landmarks of a face drawn by the inverse of a known similarity: Align
	// must find that similarity and land every landmark on the template.
	want := similarity(0.5, -12, 20, -8)
	inv, err := want.Invert()
	if err != nil {
		t.Fatal(err)
	}
	var landmarks [5]face.Point
	for i, p := range face.ArcFaceTemplate {
		landmarks[i] = inv.Apply(p)
	}
	img := image.NewRGBA(image.Rect(0, 0, 300, 300))
	got, err := face.Align(img, landmarks)
	if err != nil {
		t.Fatal(err)
	}
	if got.Image.Rect.Dx() != face.AlignedSize || got.Image.Rect.Dy() != face.AlignedSize {
		t.Errorf("the crop is %v, want 112×112", got.Image.Rect)
	}
	if !affineNear(got.Transform, want, 1e-9) || got.Residual > 1e-9 {
		t.Errorf("transform %v residual %v, want %v and 0", got.Transform, got.Residual, want)
	}
	if !near(got.Transform.Scale(), 0.5, 1e-12) || !near(got.Transform.Rotation(), -12, 1e-9) {
		t.Errorf("scale %v rotation %v, want 0.5 and -12", got.Transform.Scale(), got.Transform.Rotation())
	}
	if got.Template != face.ArcFaceTemplate {
		t.Error("the result does not carry the template")
	}
}
