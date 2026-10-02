package face

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"

	onnx "github.com/kuaa-io/arandu-onnx"
)

// MiVOLOInputSize is the side of each square crop MiVOLO reads.
const MiVOLOInputSize = 384

// imageNet is the normalization MiVOLO's crops are given: RGB/255, ImageNet
// mean and standard deviation.
var imageNet = onnx.Normalization{
	Mean: [3]float32{0.485, 0.456, 0.406},
	Std:  [3]float32{0.229, 0.224, 0.225},
}

// AgeEstimate is an apparent age estimated from a face, with what is known of
// its error. It is a signal, not an age verification.
type AgeEstimate struct {
	// Years is the model's estimate.
	Years float64
	// BodyUsed is whether a body crop was given beside the face.
	BodyUsed bool
	// Model identifies the model, file and runtime that produced it.
	Model onnx.ModelInfo
	// ReportedMAE is the mean absolute error, in years, the model's authors
	// report, quoted as metadata; nothing here measured it.
	ReportedMAE float64
	// Uncertainty says what ReportedMAE was measured on and why it does not
	// bound this estimate.
	Uncertainty string
}

// mivoloReportedMAE is the MAE MiVOLO v2's authors report with face and body
// on LAGENDA.
const mivoloReportedMAE = 3.65

// EstimateAge estimates the apparent age of a detected face with MiVOLO v2.
//
// The face box, clipped to the image, is cropped and letterboxed onto
// 384×384 black; so is the body box when one is given, and an all-black image
// stands in for it otherwise. Both are normalized as RGB/255 with ImageNet
// mean and standard deviation and stacked as six channels. It needs
// Config.EnableAge.
func (m *Module) EstimateAge(ctx context.Context, img image.Image, face Face, body *Box) (AgeEstimate, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	session, err := m.session(m.age, "EstimateAge", "EnableAge")
	if err != nil {
		return AgeEstimate{}, err
	}
	faceTensor, err := letterboxedCrop(img, face.Box)
	if err != nil {
		return AgeEstimate{}, fmt.Errorf("face: the face crop: %w", err)
	}
	var bodyTensor onnx.Tensor
	if body != nil {
		if bodyTensor, err = letterboxedCrop(img, *body); err != nil {
			return AgeEstimate{}, fmt.Errorf("face: the body crop: %w", err)
		}
	} else {
		black := image.NewRGBA(image.Rect(0, 0, MiVOLOInputSize, MiVOLOInputSize))
		for i := 3; i < len(black.Pix); i += 4 {
			black.Pix[i] = 255
		}
		if bodyTensor, err = onnx.NCHW(black, imageNet); err != nil {
			return AgeEstimate{}, err
		}
	}
	data := make([]float32, 0, 2*len(faceTensor.Float32))
	data = append(data, faceTensor.Float32...)
	data = append(data, bodyTensor.Float32...)
	input := onnx.Tensor{Shape: []int64{1, 6, MiVOLOInputSize, MiVOLOInputSize}, Float32: data}

	result, err := session.Run(ctx, map[string]onnx.Tensor{"faces_bodies": input})
	if err != nil {
		return AgeEstimate{}, err
	}
	out := result.Outputs["age"].Float32
	if len(out) != 1 || math.IsNaN(float64(out[0])) || math.IsInf(float64(out[0]), 0) {
		return AgeEstimate{}, fmt.Errorf("face: %s returned %v, want one finite value", result.Model.ID, out)
	}
	return AgeEstimate{
		Years:       float64(out[0]),
		BodyUsed:    body != nil,
		Model:       result.Model,
		ReportedMAE: mivoloReportedMAE,
		Uncertainty: "ReportedMAE is the authors' figure with face and body on LAGENDA; it is not measured on this " +
			"population, does not bound a single estimate, and is larger without a body crop.",
	}, nil
}

// letterboxedCrop crops a box out of the image, clipped to it, letterboxes it
// onto MiVOLO's black square and normalizes it.
func letterboxedCrop(img image.Image, box Box) (onnx.Tensor, error) {
	if box.Area() <= 0 || math.IsNaN(box.Area()) || math.IsInf(box.Area(), 0) {
		return onnx.Tensor{}, fmt.Errorf("the box %+v has no area", box)
	}
	crop, err := onnx.Crop(img, box.Rect())
	if err != nil {
		return onnx.Tensor{}, err
	}
	canvas, _, err := onnx.Letterbox(crop, MiVOLOInputSize, MiVOLOInputSize, color.RGBA{A: 255})
	if err != nil {
		return onnx.Tensor{}, err
	}
	return onnx.NCHW(canvas, imageNet)
}
