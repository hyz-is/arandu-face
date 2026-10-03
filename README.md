<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Arandu Face

An Arandu package that measures faces with ONNX models, run through ONNX
Runtime by [`arandu-onnx`](https://github.com/hyz-is/arandu-onnx). It
returns **separate, auditable technical signals**, each with the provenance of
the model behind it: where a face is and where its landmarks are, how it was
aligned, how sharp, lit and posed it is, its embedding, the distance between
two embeddings, and an apparent age. It decides nothing. Thresholds come back
as versioned parameters beside the measurements, and whether two faces are the
same person, a selfie is good enough or someone is an adult is the
application's policy, not this package's.

Each stage is its own call, with its own result:

| Stage | Call | Model | Result |
| --- | --- | --- | --- |
| 1. Detection | `Module.Detect(ctx, img)` | YuNet 2026may or BlazeFace short range | `[]Face{Box, Score, Landmarks [5]Point, Keypoints, Detector ModelInfo, Parameters}` |
| 2. Alignment | `Align(img, landmarks)` | none (Go) | `Aligned{Image 112×112, Transform, Template, Residual}` |
| 3. Quality | `Quality(img, face, thresholds)` / `Module.Quality` | none (Go) | `QualityReport{Signals, Checks, Thresholds}` |
| 4. Embedding | `Module.Embed(ctx, aligned)` | SFace 2021dec (**dev-only**) | `Embedding{Vector, Dim 128, Model, Normalization "l2", RawNorm}` |
| 5. Comparison | `Compare(a, b, thresholds)` / `Module.Compare` | none (Go) | `Comparison{Cosine, L2, Model, Thresholds}` |
| 6. Age | `Module.EstimateAge(ctx, img, face, body)` | MiVOLO v2 age (**dev-only**) | `AgeEstimate{Years, BodyUsed, Model, ReportedMAE, Uncertainty}` |

No function runs two stages for the caller. Nothing returns a boolean
verdict.

## How each stage works

1. **Detection with YuNet** (default). The image is scaled down so its longer
   side is at most `DetectMaxSide` (640; never up), padded with black at the
   bottom and right to multiples of 32, and given as raw BGR 0..255. For each
   stride s ∈ {8, 16, 32} the feature map has padW/s columns; anchor idx sits
   at row idx/cols, column idx%cols. The score is √(clamp(cls)·clamp(obj));
   the box centre ((c+b0)·s, (r+b1)·s), its size (e^b2·s, e^b3·s); landmark n
   ((k[2n]+c)·s, (k[2n+1]+r)·s). Candidates under the score threshold (0.9)
   are dropped, the rest go through greedy NMS (IoU 0.3, top 5000), as
   OpenCV's `FaceDetectorYN` does. Landmarks are right eye, left eye, nose
   tip, right mouth corner, left mouth corner — the subject's right, which is
   the image's left.
2. **Detection with BlazeFace** (selfies). The image is letterboxed onto
   128×128 black, RGB in −1..1. MediaPipe's 896 SSD anchors (strides 8 and 16,
   2 and 6 anchors per cell) decode the regressors; scores are sigmoids of the
   raw value clipped to ±100; candidates under 0.5 are dropped and the rest
   merged by MediaPipe's weighted NMS (IoU 0.3). BlazeFace has six keypoints
   and no mouth corners, so `Face.Landmarks` holds the mouth centre in both
   corner slots and `LandmarkScheme` says `mouth-centre`; aligned through it,
   the mouth centre is fitted to the midpoint of the template's corners with
   twice the weight.
3. **Alignment.** Umeyama's least-squares similarity from the five landmarks
   onto the ArcFace template — (38.2946, 51.6963), (73.5318, 51.5014),
   (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041) — with the 2×2
   SVD in closed form and OpenCV's reflection guard, then a bilinear
   `WarpAffine` onto 112×112 with a black border. `Residual` is the RMS
   distance, in crop pixels, between the transformed landmarks and the
   template.
4. **Quality**, all in Go: face box size and share of the frame, whether it is
   inside the frame, inter-ocular distance, roll of the eye line, a yaw proxy
   (the nose's offset from the eyes' midpoint along the eye line, in
   inter-ocular distances — a proxy, not an angle), how many landmarks fall
   in the box, sharpness (variance of the 4-neighbour Laplacian of the box's
   BT.601 luma), brightness (mean luma) and contrast (its standard deviation).
   Every signal is checked against `QualityThresholds` and reported as a
   `QualityCheck{Signal, Value, Rule, Pass}`. The default thresholds are
   versioned `arandu-face-quality-0.1-uncalibrated`: starting values, not
   calibrated on any population. There is no overall pass.
5. **Embedding.** The aligned crop goes to SFace as RGB 0..255 (SFace
   normalizes inside its graph); `fc1` [1, 128] is divided by its norm.
6. **Comparison.** Cosine similarity and the Euclidean distance between the
   two unit vectors. Embeddings of different models, versions or files are
   refused (`ErrModelMismatch`), and so is a `ThresholdSet` stated for another
   model. The result carries `SFaceReferenceThresholds()` — version
   `sface-opencv-2021dec`, cosine 0.363, L2 1.128, opencv_zoo's reference
   values — beside the measurement, never as a verdict.
7. **Age.** The face box (and the body box, when given) is cropped,
   letterboxed onto 384×384 black, normalized as RGB/255 with ImageNet mean
   and std, and stacked as six channels; a missing body is an all-black image
   normalized the same way. The output is years. The authors' MAE of 3.65
   (face and body, LAGENDA) is returned as metadata, not as a bound.

The constants are those of OpenCV's `FaceDetectorYN` and `FaceRecognizerSF`,
MediaPipe's short-range detector graph, and MiVOLO's preprocessing; the unit
tests pin them with values worked out by hand.

## Build requirement

Those of `arandu-onnx`: `CGO_ENABLED=1`, a C compiler, and the ONNX Runtime
shared library, 1.26 or later, on the machine that runs the application.
Development and the tests use 1.30.0.

## Install

```bash
go get github.com/hyz-is/arandu-face
```

Until `arandu-onnx` is tagged, `go.mod` resolves it from the sibling checkout:

```
replace github.com/hyz-is/arandu-onnx => ../arandu-onnx
```

That line is removed, and `arandu-onnx` required at its tag, once that module
is published. The release workflow refuses to publish this one while it is
there.

## The models

The model files are not in this repository. Each has a manifest, embedded in
the package under `manifests/`, recording its source revision, SHA-256,
licences, status and contract; `Runtime.Open` verifies every file against it
before ONNX Runtime reads a byte. What the manifest has no field for —
training data, reported accuracy, caveats — is in its `ModelCard`
(`LookupModelCard`).

| id | model | source revision | sha256 | size | status |
| --- | --- | --- | --- | --- | --- |
| `yunet_2026may` | YuNet 2026may, dynamic input | opencv_zoo `26cc381e4d25` | `ebafce4e3c11…` | 230 KB | approved |
| `blazeface_short_range` | BlazeFace short range | yakhyo release `weights` | `2f2689b040be…` | 471 KB | approved |
| `sface_2021dec` | SFace 2021dec | opencv_zoo `ba91a3b91d00` | `0ba9fbfa01b5…` | 39 MB | **dev-only** |
| `mivolo_v2_age` | MiVOLO v2, age output | HF `8eb4cd8f5dd4` | `2db5e05be33b…` | 118 MB | **dev-only** |

- **YuNet**: MIT. Trained on WIDER FACE, which is CC BY-NC-ND 4.0; whether that
  term reaches the weights is unsettled and recorded as a risk.
- **BlazeFace short range**: Google's model, Apache-2.0, exported to ONNX by
  yakhyo/mediapipe-face-mesh-onnx. The release tag is mutable; the digest,
  which equals the release asset's published digest, is what pins it.
- **SFace**: Apache-2.0 as opencv_zoo distributes it, but the upstream weights
  carry no licence and the training data is undocumented: commercial use is
  unknown, so it is `dev-only`.
- **MiVOLO v2 age**: Apache-2.0, a third-party export of iitolstykh/mivolo_v2
  trained partly on proprietary data: `dev-only`.

No model here infers race, ethnicity or gender, and no non-commercial
recognizer (InsightFace, ArcFace) is shipped; a test refuses them.

A models directory holds each file at the path its manifest's `file` says
(`face/face_detection_yunet_2026may.onnx`, `age/mivolo_v2_age.onnx`, …).
`cmd/face-models` fills one:

```bash
go run github.com/hyz-is/arandu-face/cmd/face-models -dir /srv/models
go run github.com/hyz-is/arandu-face/cmd/face-models -dir /srv/models -check
```

It downloads each file over HTTPS from the source its manifest records,
follows only HTTPS redirects, hashes while it writes, and keeps the file only
when the digest matches. `-only` takes a comma-separated list of ids; `-check`
verifies what is there and connects to nothing. **The command is the only code
here that opens a network connection.** The library an application links reads
files from the directory it is given, which is why `arandu.mod.toml` declares
`network = false`.

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go` and there are no others.

The imports, with the other module imports:

```go
import (
	onnx "github.com/hyz-is/arandu-onnx"
	face "github.com/hyz-is/arandu-face"
)
```

The construction, in `Build`, before `k.Register`, after the ONNX module. Every
value comes from the application's own typed configuration; this package reads
no environment variable:

```go
	faceModule, err := face.New(face.Config{
		ModelsDir:       cfg.Face.ModelsDir,
		Detector:        face.DetectorYuNet,
		EnableEmbedding: cfg.Face.Embedding,
		EnableAge:       cfg.Face.Age,
		RefuseDevOnly:   cfg.Face.Production,
	}, onnxModule.Runtime())
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there, after
`onnxModule`:

```go
		onnxModule,
		faceModule,
```

`Boot` initializes the runtime if `arandu-onnx` has not yet, and opens the
configured models, verifying each; a missing or altered file stops the boot,
naming the model. With `RefuseDevOnly` set, a configuration that needs a
dev-only model (SFace, MiVOLO) stops the boot before any file is read, with
`ErrDevOnlyRefused`. `Health` reports whether the models are open and the
runtime under them is usable. `Models` lists the `ModelInfo` of each open
model. `Close` closes the sessions. Registering it adds no route, no table and
no background loop.

## Measure a face

```go
img, err := onnx.DecodeImage(upload) // JPEG or PNG, turned upright by EXIF

faces, err := faceModule.Detect(ctx, img)       // stage 1: boxes, landmarks, detector ModelInfo
aligned, err := face.Align(img, faces[0].Landmarks) // stage 2: 112×112 crop and transform
report, err := faceModule.Quality(img, faces[0])     // stage 3: signals and per-signal checks
probe, err := faceModule.Embed(ctx, aligned.Image)   // stage 4: unit vector, SFace ModelInfo
cmp, err := faceModule.Compare(probe, reference)     // stage 5: cosine, L2, ThresholdSet
age, err := faceModule.EstimateAge(ctx, img, faces[0], nil) // stage 6: years, MiVOLO ModelInfo
```

Every model-backed result carries an `onnx.ModelInfo` — ID, Version, SHA256,
Runtime, Precision, Provider and Status — which is what the application
records as evidence beside the signal. A `dev-only` status in that record says
the signal came from a model not cleared for production.

Every `Face` is in the coordinates of the image as given, including a
sub-image's offset. Its `Parameters` record the score threshold, NMS
threshold, top-k, the input size the detector read and the scale applied.

The pure-Go pieces are exported for reuse and testing: `DecodeYuNet`,
`YuNetPaddedSize`, `BlazeFaceAnchors`, `DecodeBlazeFace`, `NMS`, `WeightedNMS`,
`IoU`, `DecomposeSVD2`, `EstimateSimilarity`, `WarpAffine`, `LaplacianVariance`,
`L2Normalize` and `RawPlanes`.

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `ModelsDir` | required | The directory the model files are in |
| `Detector` | `yunet` | `yunet` or `blazeface` (selfies) |
| `EnableEmbedding` | `false` | Open SFace at Boot, for Embed |
| `EnableAge` | `false` | Open MiVOLO at Boot, for EstimateAge |
| `RefuseDevOnly` | `false` | Refuse every configured dev-only model at Boot — the production switch |
| `DetectMaxSide` | `640` | Longest side YuNet sees, 32 to 4096 |
| `ScoreThreshold` | `0.9` YuNet, `0.5` BlazeFace | Score a detection needs to be kept |
| `NMSThreshold` | `0.3` | IoU above which a detection is suppressed |
| `TopK` | `5000` | Candidates YuNet's suppression considers |
| `Thresholds` | `SFaceReferenceThresholds()` | The reference values Compare returns |
| `Quality` | `DefaultQualityThresholds()` | The bounds Quality checks against |

A numeric field left at zero, and a threshold set left without a version,
takes its default.

## What was measured, and what was not

The feature tests run every model on images drawn in Go: flat grey, a
gradient, a checkerboard, noise, and a cartoon face assembled from ellipses
(`tests/Feature/fixture_test.go`). No photograph of a person and no face
dataset is used.

- Neither detector finds a face in the four scenes without one at its default
  threshold (BlazeFace scores up to about 0.38 on noise, under its 0.5).
- YuNet finds the cartoon at three sizes and positions, including a
  1280×1120 image detected at half size, with score ≥ 0.9 and every landmark
  within a fifth of the eye distance of where it was drawn, in the template's
  order. BlazeFace finds it at two sizes.
- SFace is deterministic (one crop, bit-identical vectors), its vectors have
  norm 1 within 1e-5, and the cartoon at two placements embeds at cosine ≈ 0.98
  against ≈ 0.11 to a non-face pattern — a check that alignment is wired, not a
  recognition result.
- MiVOLO returns one finite value in 0..120 years, identical across runs.
- Concurrent calls from eight goroutines give the same answers as sequential
  ones, under the race detector.

On an Apple M4 Max, CPU provider, ONNX Runtime 1.30.0, on the 480×560 cartoon
(`go test -run '^$' -bench . ./tests/Feature/`): Detect with YuNet ≈ 3.6 ms,
with BlazeFace ≈ 1.0 ms; Align ≈ 0.4 ms; Quality ≈ 1.9 ms; Embed ≈ 7.4 ms;
EstimateAge ≈ 440 ms.

**Detection accuracy, the false-match and false-non-match rates of the
embedding, the calibration of any threshold, the error of the age estimate
and any difference between demographic groups have not been measured.** They
must be measured by the owner on consented data before any of these signals
weighs on a decision.

## Concurrency

`Detect`, `Embed` and `EstimateAge` are safe for concurrent use. Calls share
the ONNX Runtime sessions, whose runs are thread-safe; the module's
reader-writer lock lets them proceed together and makes `Close` wait for
them. `Align`, `Quality` and `Compare` are pure functions.

## Why there is no table, policy or route

This repository has the shape of the Arandu package skeleton without its demo
entity, like `arandu-onnx`. It wraps a technology and owns no record, so there
is nothing for a policy to decide about; a route would measure any face for
anybody who sent one. Faces are measured through the application's
controllers and Grants, and biometric data is the application's to minimize,
encrypt and audit.

## Limitations

- YuNet sees at most `DetectMaxSide` pixels on the longer side; a small face on
  a large photograph may need a crop or a larger limit.
- BlazeFace is meant for selfies, where the face covers a large part of the
  image, and has no mouth corners.
- The yaw signal is a geometric proxy from 2D landmarks, not a head-pose
  angle; there is no pitch signal and no occlusion signal.
- The warp is bilinear in floating point; OpenCV quantizes to 1/32 pixel, so
  crops differ from OpenCV's by rounding.
- SFace and MiVOLO are dev-only: their licences or training data are not
  settled for commercial use.
- Every figure above is on synthetic images.

## Gates

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

The model tests read the models from `FACE_MODELS_DIR`, or from the KUAA
workspace's `models/` two levels above this checkout, and skip, saying where
they looked, when the files are not there; they skip too when ONNX Runtime is
absent. A skip is a reason to fetch the models, not a green result. No test
downloads anything.

## Licence

MIT. See [LICENSE.md](LICENSE.md). The model files, which this repository does
not contain, are under their own licences, listed above and in their
manifests.
