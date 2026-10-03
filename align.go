package face

import (
	"errors"
	"fmt"
	"image"
	"math"

	onnx "github.com/hyz-is/arandu-onnx"
)

// AlignedSize is the side of the square crop Align produces, which is the
// input SFace reads.
const AlignedSize = 112

// ArcFaceTemplate is where the five landmarks of a frontal face sit on a
// 112×112 crop, in the order right eye, left eye, nose tip, right mouth
// corner, left mouth corner. It is the template OpenCV's FaceRecognizerSF and
// the ArcFace family align to.
var ArcFaceTemplate = [5]Point{
	{X: 38.2946, Y: 51.6963},
	{X: 73.5318, Y: 51.5014},
	{X: 56.0252, Y: 71.7366},
	{X: 41.5493, Y: 92.3655},
	{X: 70.7299, Y: 92.2041},
}

// ErrDegenerateLandmarks is a set of points from which no similarity can be
// estimated: fewer than two, mismatched in number, not finite, or all at one
// place.
var ErrDegenerateLandmarks = errors.New("face: degenerate landmarks")

// Affine is a 2×3 affine transform: a point (x, y) maps to
// (M[0][0]x + M[0][1]y + M[0][2], M[1][0]x + M[1][1]y + M[1][2]).
type Affine [2][3]float64

// Apply maps a point through the transform.
func (m Affine) Apply(p Point) Point {
	return Point{
		X: m[0][0]*p.X + m[0][1]*p.Y + m[0][2],
		Y: m[1][0]*p.X + m[1][1]*p.Y + m[1][2],
	}
}

// Invert returns the inverse transform, or an error when the linear part is
// singular.
func (m Affine) Invert() (Affine, error) {
	det := m[0][0]*m[1][1] - m[0][1]*m[1][0]
	if det == 0 || math.IsNaN(det) || math.IsInf(det, 0) {
		return Affine{}, fmt.Errorf("face: the transform %v is not invertible", m)
	}
	a, b := m[1][1]/det, -m[0][1]/det
	c, d := -m[1][0]/det, m[0][0]/det
	return Affine{
		{a, b, -(a*m[0][2] + b*m[1][2])},
		{c, d, -(c*m[0][2] + d*m[1][2])},
	}, nil
}

// Scale is the factor a similarity transform scales by: the square root of
// the determinant of its linear part.
func (m Affine) Scale() float64 {
	return math.Sqrt(math.Abs(m[0][0]*m[1][1] - m[0][1]*m[1][0]))
}

// Rotation is the angle, in degrees, a similarity transform rotates by,
// counter-clockwise in a frame whose Y points up (clockwise on screen).
func (m Affine) Rotation() float64 {
	return math.Atan2(m[1][0], m[0][0]) * 180 / math.Pi
}

// SVD2 is the singular value decomposition of a 2×2 matrix: A = U·diag(S)·Vt,
// with U and Vt orthogonal and S[0] ≥ S[1] ≥ 0.
type SVD2 struct {
	U  [2][2]float64
	S  [2]float64
	Vt [2][2]float64
}

// DecomposeSVD2 computes the singular value decomposition of a 2×2 matrix in
// closed form.
//
// A is written as R(φ)·diag(σ1, σ2)·R(θ), with E = (a+d)/2, F = (a−d)/2,
// G = (c+b)/2, H = (c−b)/2, Q = √(E²+H²), R = √(F²+G²), σ1 = Q+R,
// σ2 = Q−R, φ = (atan2(H,E)+atan2(G,F))/2 and θ = (atan2(H,E)−atan2(G,F))/2.
// A negative σ2 is made positive by negating U's second column.
func DecomposeSVD2(a [2][2]float64) SVD2 {
	e := (a[0][0] + a[1][1]) / 2
	f := (a[0][0] - a[1][1]) / 2
	g := (a[1][0] + a[0][1]) / 2
	h := (a[1][0] - a[0][1]) / 2
	q := math.Hypot(e, h)
	r := math.Hypot(f, g)
	s1, s2 := q+r, q-r
	a1 := math.Atan2(g, f)
	a2 := math.Atan2(h, e)
	theta := (a2 - a1) / 2
	phi := (a2 + a1) / 2

	cp, sp := math.Cos(phi), math.Sin(phi)
	ct, st := math.Cos(theta), math.Sin(theta)
	out := SVD2{
		U:  [2][2]float64{{cp, -sp}, {sp, cp}},
		S:  [2]float64{s1, s2},
		Vt: [2][2]float64{{ct, -st}, {st, ct}},
	}
	if s2 < 0 {
		out.S[1] = -s2
		out.U[0][1], out.U[1][1] = -out.U[0][1], -out.U[1][1]
	}
	return out
}

// EstimateSimilarity returns the similarity transform (rotation, uniform
// scale and translation, no reflection) that maps src onto dst with the least
// squared error, by Umeyama's closed form.
//
// It follows OpenCV's FaceRecognizerSF: the covariance of the demeaned point
// sets, its SVD, a reflection guard on the sign of its determinant, and the
// scale from the singular values over the variance of src. At least two
// distinct points are needed.
func EstimateSimilarity(src, dst []Point) (Affine, error) {
	n := len(src)
	if n < 2 || len(dst) != n {
		return Affine{}, fmt.Errorf("%w: %d source and %d destination points", ErrDegenerateLandmarks, len(src), len(dst))
	}
	var sm, dm Point
	for i := range src {
		for _, v := range []float64{src[i].X, src[i].Y, dst[i].X, dst[i].Y} {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return Affine{}, fmt.Errorf("%w: a coordinate is not finite", ErrDegenerateLandmarks)
			}
		}
		sm.X += src[i].X
		sm.Y += src[i].Y
		dm.X += dst[i].X
		dm.Y += dst[i].Y
	}
	fn := float64(n)
	sm = Point{sm.X / fn, sm.Y / fn}
	dm = Point{dm.X / fn, dm.Y / fn}

	var cov [2][2]float64
	variance := 0.0
	for i := range src {
		sx, sy := src[i].X-sm.X, src[i].Y-sm.Y
		dx, dy := dst[i].X-dm.X, dst[i].Y-dm.Y
		cov[0][0] += dx * sx
		cov[0][1] += dx * sy
		cov[1][0] += dy * sx
		cov[1][1] += dy * sy
		variance += sx*sx + sy*sy
	}
	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			cov[r][c] /= fn
		}
	}
	variance /= fn
	if variance <= 0 {
		return Affine{}, fmt.Errorf("%w: every source point is at the same place", ErrDegenerateLandmarks)
	}

	svd := DecomposeSVD2(cov)
	d := [2]float64{1, 1}
	if cov[0][0]*cov[1][1]-cov[0][1]*cov[1][0] < 0 {
		d[1] = -1
	}
	tol := math.Max(svd.S[0], svd.S[1]) * 2 * 1.1754943508222875e-38 // 2·FLT_MIN, as OpenCV
	rank := 0
	for _, s := range svd.S {
		if s > tol {
			rank++
		}
	}
	if rank == 0 {
		return Affine{}, fmt.Errorf("%w: the points carry no direction", ErrDegenerateLandmarks)
	}
	det2 := func(m [2][2]float64) float64 { return m[0][0]*m[1][1] - m[0][1]*m[1][0] }
	rotationD := d
	if rank == 1 {
		// With one direction the determinant is zero, and the reflection guard
		// is the sign of U and Vt instead.
		if det2(svd.U)*det2(svd.Vt) > 0 {
			rotationD = [2]float64{1, 1}
		} else {
			rotationD = [2]float64{1, -1}
		}
	}
	var t [2][2]float64
	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			t[r][c] = svd.U[r][0]*rotationD[0]*svd.Vt[0][c] + svd.U[r][1]*rotationD[1]*svd.Vt[1][c]
		}
	}
	scale := (svd.S[0]*d[0] + svd.S[1]*d[1]) / variance

	ts := Point{X: t[0][0]*sm.X + t[0][1]*sm.Y, Y: t[1][0]*sm.X + t[1][1]*sm.Y}
	return Affine{
		{scale * t[0][0], scale * t[0][1], dm.X - scale*ts.X},
		{scale * t[1][0], scale * t[1][1], dm.Y - scale*ts.Y},
	}, nil
}

// WarpAffine produces a width by height image whose pixel (x, y) is the
// source sampled at the inverse of m applied to (x, y): m maps source
// coordinates onto the output. Sampling is bilinear, and a position outside
// the source reads opaque black, as OpenCV's warpAffine with INTER_LINEAR and
// a constant zero border does (but in floating point rather than its 1/32
// fixed-point steps). Pixel centres are at integer coordinates.
func WarpAffine(src image.Image, m Affine, width, height int) (*image.RGBA, error) {
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("face: warp to %dx%d: the size has to be positive", width, height)
	}
	inv, err := m.Invert()
	if err != nil {
		return nil, err
	}
	in := asRGBA(src)
	sw, sh := in.Rect.Dx(), in.Rect.Dy()
	out := image.NewRGBA(image.Rect(0, 0, width, height))
	at := func(x, y, c int) float64 {
		if x < 0 || y < 0 || x >= sw || y >= sh {
			return 0
		}
		return float64(in.Pix[y*in.Stride+x*4+c])
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			p := inv.Apply(Point{X: float64(x), Y: float64(y)})
			x0, y0 := math.Floor(p.X), math.Floor(p.Y)
			fx, fy := p.X-x0, p.Y-y0
			ix, iy := int(x0), int(y0)
			o := y*out.Stride + x*4
			for c := 0; c < 3; c++ {
				top := at(ix, iy, c)*(1-fx) + at(ix+1, iy, c)*fx
				bottom := at(ix, iy+1, c)*(1-fx) + at(ix+1, iy+1, c)*fx
				out.Pix[o+c] = toByte(top*(1-fy) + bottom*fy)
			}
			out.Pix[o+3] = 255
		}
	}
	return out, nil
}

// Aligned is a face warped onto ArcFaceTemplate.
type Aligned struct {
	// Image is the 112×112 crop.
	Image *image.RGBA
	// Transform maps the source image's coordinates onto the crop's.
	Transform Affine
	// Template is the landmark positions the transform was fitted to.
	Template [5]Point
	// Residual is the root-mean-square distance, in crop pixels, between
	// each transformed landmark and its template position: how far the face's
	// landmark geometry is from the template's.
	Residual float64
}

// Align warps the face whose five landmarks are given onto ArcFaceTemplate
// and returns the 112×112 crop with the transform it used.
//
// The landmarks are in the order Face.Landmarks holds them. The transform is
// EstimateSimilarity from the landmarks to the template; the crop is
// WarpAffine of the image through it.
func Align(img image.Image, landmarks [5]Point) (Aligned, error) {
	m, err := EstimateSimilarity(landmarks[:], ArcFaceTemplate[:])
	if err != nil {
		return Aligned{}, err
	}
	crop, err := WarpAffine(img, m, AlignedSize, AlignedSize)
	if err != nil {
		return Aligned{}, err
	}
	sum := 0.0
	for i, p := range landmarks {
		q := m.Apply(p)
		dx, dy := q.X-ArcFaceTemplate[i].X, q.Y-ArcFaceTemplate[i].Y
		sum += dx*dx + dy*dy
	}
	return Aligned{
		Image:     crop,
		Transform: m,
		Template:  ArcFaceTemplate,
		Residual:  math.Sqrt(sum / float64(len(landmarks))),
	}, nil
}

// asRGBA returns the image as an *image.RGBA whose origin is (0, 0), copying
// only when it is not one already.
func asRGBA(src image.Image) *image.RGBA {
	if rgba, ok := src.(*image.RGBA); ok && rgba.Rect.Min == (image.Point{}) {
		return rgba
	}
	b := src.Bounds()
	out, err := onnx.Crop(src, b)
	if err != nil {
		return image.NewRGBA(image.Rectangle{})
	}
	return out
}

// toByte rounds to the nearest integer and clamps to 0..255.
func toByte(v float64) uint8 {
	v = math.Round(v)
	if !(v > 0) {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
