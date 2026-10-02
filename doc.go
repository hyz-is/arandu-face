// Package face is an Arandu module that measures faces with ONNX models run
// through arandu-onnx, and returns every measurement as a separate technical
// signal with the provenance of the model behind it.
//
// Each stage is its own call: Detect finds faces and their landmarks, Align
// warps one onto the 112×112 template, Quality measures it in pure Go, Embed
// turns an aligned crop into a vector, Compare measures two vectors, and
// EstimateAge estimates an apparent age. No call runs two stages for the
// caller, and none decides anything: thresholds travel with the results as
// versioned parameters, and what they mean for a case is the application's
// policy.
//
// An application builds it once in bootstrap/app.go, after arandu-onnx, and
// registers it there. There is no service provider, no container and no
// discovery.
package face
