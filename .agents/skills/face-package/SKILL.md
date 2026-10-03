---
name: face-package
description: Install, wire and use the Arandu Face package (Go, Arandu) in an application. Use when the request is to "install arandu-face", "detect a face", "align a face", "compare two faces", "embed a selfie", "estimate age", "check selfie quality", "go get github.com/hyz-is/arandu-face", "wire it into bootstrap/app.go", "download the face models", "the boot fails opening yunet_2026may", "dev-only model refused", "model digest mismatch", "no face found", or when a project's go.mod already requires github.com/hyz-is/arandu-face. Covers the build requirement, the models directory, the wiring and where it goes, the Config fields, each stage's result, and what each error means.
license: MIT
---

# Using Arandu Face

A module that measures faces with ONNX models on `arandu-onnx`'s runtime. It
returns separate signals — detection, alignment, quality, embedding,
comparison, age — each with the model behind it, and decides nothing. It is
registered by hand — there is no service provider, no container and no
discovery.

## Build requirement

`CGO_ENABLED=1`, a C compiler, and the ONNX Runtime shared library (1.26 or
later) on the machine that runs the application, as `arandu-onnx` requires.

## Install and fetch the models

```bash
go get github.com/hyz-is/arandu-face@v0.1.0
go run github.com/hyz-is/arandu-face/cmd/face-models@v0.1.0 -dir /srv/models
```

The command downloads every model over HTTPS from its pinned source and keeps
a file only when its SHA-256 matches; `-check` verifies without connecting,
`-only` takes ids. Run it where the application is deployed or when its image is
built. The library itself never downloads.

## The wiring, and where it goes

All of it is in `bootstrap/app.go`, after the `arandu-onnx` module.

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
	face "github.com/hyz-is/arandu-face"
)
```

In `Build`, before `k.Register`, with every value from the application's typed
configuration (the package reads no environment variable):

```go
	faceModule, err := face.New(face.Config{
		ModelsDir:       cfg.Face.ModelsDir,
		EnableEmbedding: cfg.Face.Embedding,
		EnableAge:       cfg.Face.Age,
		RefuseDevOnly:   cfg.Face.Production,
	}, onnxModule.Runtime())
	if err != nil {
		return App{}, err
	}
```

Inside the `k.Register(...)` call, after `onnxModule`:

```go
		faceModule,
```

## The stages

```go
faces, err := faceModule.Detect(ctx, img)             // []Face: Box, Score, Landmarks, Detector, Parameters
aligned, err := face.Align(img, faces[0].Landmarks)   // 112×112 crop, Transform, Residual
report, err := faceModule.Quality(img, faces[0])      // Signals, Checks (per signal), Thresholds
e, err := faceModule.Embed(ctx, aligned.Image)        // Vector (unit), Dim 128, Model
cmp, err := faceModule.Compare(e, reference)          // Cosine, L2, Model, Thresholds
age, err := faceModule.EstimateAge(ctx, img, faces[0], nil) // Years, Model, ReportedMAE
```

Record each result's `ModelInfo` beside the signal as evidence. The policy
that turns signals into a decision lives in the application, with the
thresholds' versions recorded.

## Symptoms to recognise

- **`dev-only model refused`.** `RefuseDevOnly` is set and the configuration
  enables SFace (`EnableEmbedding`) or MiVOLO (`EnableAge`), which are
  dev-only. That is the production switch working.
- **Boot fails opening `<id>` with "no such file".** The models directory is
  not filled, or `ModelsDir` points elsewhere. Run the command with `-check`.
- **`model digest mismatch`.** The file is not the reviewed one. Fetch it again
  with the command; never edit a digest.
- **`stage not enabled`.** Embed or EstimateAge without `EnableEmbedding` or
  `EnableAge`.
- **`embeddings from different models`.** Two embeddings from different model
  files cannot be compared; re-embed the reference with the current model.
- **`ONNX Runtime library unavailable`.** See `arandu-onnx`.
- **No face on a large photograph.** YuNet sees at most `DetectMaxSide` pixels
  on the longer side; crop or raise it. BlazeFace is for selfies only.
