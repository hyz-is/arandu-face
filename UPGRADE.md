# Upgrade Guide

Every release that changes what an installer has to write is described here,
newest first. A release with nothing to change still has a heading, so this file
and [CHANGELOG.md](CHANGELOG.md) describe the same set of versions.

## v0.2.0

Nothing in the code has to change: v0.1.0 configurations and call sites build
and behave as before, except which MiVOLO file the age stage opens.

1. Require both modules: `go get github.com/hyz-is/arandu-face@v0.2.0`
   (it brings `github.com/hyz-is/arandu-onnx` v0.3.0).
2. With `EnableAge`, put the new file on every node that runs the
   application: `age/mivolo_v2_age_convt.onnx`, SHA-256
   `cd7826cff39be95bbcbf8e7855a9971137cd0fb5bea8b536e2efee3e201e8333`, in
   the models directory beside `age/mivolo_v2_age.onnx`. Either run
   `go run github.com/hyz-is/arandu-face/cmd/face-models@v0.2.0 -dir <models>
   -only mivolo_v2_age_convt`, which makes it from the original (fetching the
   original if it is missing) and verifies the digest, or copy the verified
   file there. Without it, `Boot` stops naming `mivolo_v2_age_convt`.
3. To keep running the file as published, set
   `AgeModel: face.ModelMiVOLOAge`. The estimates are the same; only an
   accelerator's placement differs.
4. SFace and MiVOLO are now `approved`. A configuration with
   `RefuseDevOnly` that enables embedding or age, which v0.1.0 refused with
   `ErrDevOnlyRefused`, now boots. Evidence records show
   `Status: "approved"` and, for age, the new model id and digest; code that
   compared `AgeEstimate.Model.ID` with `ModelMiVOLOAge` compares it with
   `ModelMiVOLOAgeConvTranspose`, or with the configured `AgeModel`.
5. A calibration table made for `mivolo_v2_age` does not carry over to the
   rewritten file: calibrate it again before running it in INT8.

## v0.1.0

The first public release. Nothing to upgrade from.

A checkout that imported the module before it was published under its current
path replaces the old import paths with `github.com/hyz-is/arandu-face` and
`github.com/hyz-is/arandu-onnx`, and requires both at `v0.1.0`. The default
quality thresholds are versioned `arandu-face-quality-0.1-uncalibrated`; a
record that compared the version string with an earlier development name
compares it with this one. Tests that set the models directory through an
environment variable now set `FACE_MODELS_DIR`.
