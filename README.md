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
| 4. Embedding | `Module.Embed(ctx, aligned)` | SFace 2021dec | `Embedding{Vector, Dim 128, Model, Normalization "l2", RawNorm}` |
| 5. Comparison | `Compare(a, b, thresholds)` / `Module.Compare` | none (Go) | `Comparison{Cosine, L2, Model, Thresholds}` |
| 6. Age | `Module.EstimateAge(ctx, img, face, body)` | MiVOLO v2 age, Col2Im rewritten (default) or as published | `AgeEstimate{Years, BodyUsed, Model, ReportedMAE, Uncertainty}` |

No function runs two stages for the caller. Nothing returns a boolean
verdict.

## What it is not

- **Not a decision.** It never says two faces are the same person, a selfie is
  acceptable or someone is an adult. Signals, never decisions: thresholds come
  back as versioned reference parameters beside each measurement, and the
  application's decision layer applies its own policy and records the evidence.
- **Not a demographic classifier.** No model here infers race, ethnicity or
  gender, and a unit test refuses any manifest for one, or for a
  non-commercial recognizer (InsightFace, ArcFace, EdgeFace weights).
- **Not a model distribution.** The model weights are not in this repository.
  A separate command downloads them from their pinned sources and verifies
  their digests; see [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).
- **Not a store.** It keeps no image, no embedding and no result, and opens no
  network connection.

## Requirements

| | |
| --- | --- |
| Go | 1.26 or later |
| cgo | `CGO_ENABLED=1` and a C compiler (the ONNX Runtime binding is cgo) |
| ONNX Runtime | the shared library, **1.26 or later**; development and CI use **1.30.0**. Installation per operating system is in the [`arandu-onnx` README](https://github.com/hyz-is/arandu-onnx#installing-onnx-runtime) (`brew install onnxruntime` on macOS; the release archive on Linux) |
| Arandu | framework 0.50 or later, and `github.com/hyz-is/arandu-onnx` v0.3.0 or later |
| Models | 230 KB (YuNet) or 471 KB (BlazeFace) for detection alone; 39 MB more for embedding, 118 MB more for age. Fetched once with `cmd/face-models` |

## Install

```bash
go get github.com/hyz-is/arandu-face@v0.2.0
go run github.com/hyz-is/arandu-face/cmd/face-models@v0.2.0 -dir /srv/models
```

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
dev-only model stops the boot before any file is read, with
`ErrDevOnlyRefused`; every model this version ships is approved, so none is
refused today. `Health` reports whether the models are open and the
runtime under them is usable. `Models` lists the `ModelInfo` of each open
model. `Close` closes the sessions. Registering it adds no route, no table and
no background loop.

## Configuration

| Field | Default | What it does |
| --- | --- | --- |
| `ModelsDir` | required | The directory the model files are in |
| `Detector` | `yunet` | `yunet` or `blazeface` (selfies) |
| `EnableEmbedding` | `false` | Open SFace at Boot, for Embed |
| `EnableAge` | `false` | Open MiVOLO at Boot, for EstimateAge |
| `AgeModel` | `mivolo_v2_age_convt` | The MiVOLO file `EnableAge` opens: `mivolo_v2_age_convt` (`ModelMiVOLOAgeConvTranspose`, Col2Im rewritten, runs whole on TensorRT and CUDA) or `mivolo_v2_age` (`ModelMiVOLOAge`, as published). Same weights, same estimates |
| `RefuseDevOnly` | `false` | Refuse every configured dev-only model at Boot — the production switch |
| `DetectMaxSide` | `640` | Longest side YuNet sees, 32 to 4096 |
| `ScoreThreshold` | `0.9` YuNet, `0.5` BlazeFace | Score a detection needs to be kept |
| `NMSThreshold` | `0.3` | IoU above which a detection is suppressed |
| `TopK` | `5000` | Candidates YuNet's suppression considers |
| `Thresholds` | `SFaceReferenceThresholds()` | The reference values Compare returns |
| `Quality` | `DefaultQualityThresholds()` | The bounds Quality checks against |

A numeric field left at zero, and a threshold set left without a version,
takes its default.

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
   (face and body, LAGENDA) is returned as metadata, not as a bound. By
   default the file is `mivolo_v2_age_convt`: MiVOLO's four `Col2Im` nodes,
   which TensorRT and ONNX Runtime's CUDA provider have no kernel for, are
   rewritten into a `Reshape` and a grouped `ConvTranspose` with one-hot
   weights, so an accelerator runs the whole graph instead of handing those
   nodes to the CPU. The weights and the numbers are the original's
   (`AgeModel: face.ModelMiVOLOAge` opens the file as published).

The constants are those of OpenCV's `FaceDetectorYN` and `FaceRecognizerSF`,
MediaPipe's short-range detector graph, and MiVOLO's preprocessing; the unit
tests pin them with values worked out by hand.

## The models

The model files are not in this repository and are not distributed with it.
Each has a manifest, embedded in the package under `manifests/`, recording its
source, SHA-256, licences, status and contract; `Runtime.Open` verifies every
file against it before ONNX Runtime reads a byte. What the manifest has no
field for — training data, reported accuracy, caveats — is in its `ModelCard`
(`LookupModelCard`).

| id | file | sha256 | size | used by | status |
| --- | --- | --- | --- | --- | --- |
| `yunet_2026may` | `face/face_detection_yunet_2026may.onnx` | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` | 230 KB | `Detect` (default) | approved |
| `blazeface_short_range` | `face/face_detection_short_range.onnx` | `2f2689b040becf555706d2cb978d2f0e3296ea82413734fba9a856c66c5f2b17` | 471 KB | `Detect` with `Detector: blazeface` | approved |
| `sface_2021dec` | `face/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | 39 MB | `Embed`, with `EnableEmbedding` | approved |
| `mivolo_v2_age_convt` | `age/mivolo_v2_age_convt.onnx` | `cd7826cff39be95bbcbf8e7855a9971137cd0fb5bea8b536e2efee3e201e8333` | 118 MB | `EstimateAge`, with `EnableAge` (default `AgeModel`) | approved |
| `mivolo_v2_age` | `age/mivolo_v2_age.onnx` | `2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951` | 118 MB | `EstimateAge`, with `AgeModel: mivolo_v2_age`; the original `mivolo_v2_age_convt` is made from | approved |

| id | source (pinned) | code licence | model licence | training data | commercial use |
| --- | --- | --- | --- | --- | --- |
| `yunet_2026may` | opencv_zoo, commit [`26cc381e4d25`](https://github.com/opencv/opencv_zoo/tree/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet) | MIT | MIT | WIDER FACE, distributed under CC BY-NC-ND 4.0. Whether that non-commercial term reaches the weights is unsettled: a recorded risk | yes, by the weights' licence; see the risk |
| `blazeface_short_range` | Google MediaPipe, exported to ONNX by [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/tag/weights), release `weights` | Apache-2.0 | Apache-2.0 | described by Google's BlazeFace model card | yes |
| `sface_2021dec` | opencv_zoo, commit [`ba91a3b91d00`](https://github.com/opencv/opencv_zoo/tree/ba91a3b91d00d76e86540d4013f944bd6b514e39/models/face_recognition_sface) | Apache-2.0 | Apache-2.0 as opencv_zoo distributes it; the upstream weights carry no licence of their own | undocumented: an accepted, recorded risk | yes, under the licence opencv_zoo distributes it with |
| `mivolo_v2_age` | iitolstykh/mivolo_v2, age output exported by [Sam-Apostel/mivolo-v2-age-onnx](https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/tree/8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36), revision `8eb4cd8f5dd4` | Apache-2.0 | Apache-2.0 | partly proprietary, as the authors describe it; a third-party export: an accepted, recorded risk | yes |
| `mivolo_v2_age_convt` | `mivolo_v2_age` (sha256 `2db5e05b…6951`), its Col2Im rewritten by `github.com/hyz-is/arandu-onnx` v0.3.0 (`rewrite.Col2ImToConvTranspose`) | Apache-2.0 | Apache-2.0, a modification it allows | as `mivolo_v2_age` | yes |

The BlazeFace release tag is mutable; the digest, which equals the release
asset's published digest, is what pins the file. Full source URLs are in the
manifests and in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

SFace and MiVOLO were `dev-only` until v0.2.0. They are approved under the
maintainer's decision that open-source weights under a permissive licence
(Apache-2.0, MIT, BSD) are eligible for production; what is not documented
about their training data stays recorded as an accepted risk in each
`ModelCard.Review`, not resolved. Non-commercial weights (InsightFace,
EdgeFace, UTKFace-trained models) stay out, and a unit test refuses an
approved manifest whose `commercial_use` is not `yes`.

### `dev-only` and `RefuseDevOnly`

A `dev-only` model loads, and every `ModelInfo` it produces says
`Status: "dev-only"`, so the evidence the application records shows the signal
came from a model not cleared for production. With `Config.RefuseDevOnly` set —
the production switch — `Boot` refuses a configuration that needs a dev-only
model before any file is read, with `ErrDevOnlyRefused` naming the model.
Every model this version ships is approved, so every configuration boots with
the switch set; it stays for the next model that starts as dev-only.

### The derived age model

`mivolo_v2_age_convt` has no download address of its own: its manifest's
`source_url` is the original's, and its `ModelCard.Derivation` records the
original's id and SHA-256, the rewrite, the tool and the equivalence measured
(the rewritten and original files differ by 0 on the CPU). `cmd/face-models`
makes it from the verified original — fetching the original first when it is
missing — with the same pure-Go rewrite, no ONNX Runtime needed, and keeps it
only when its SHA-256 is the manifest's. A node can also be given the file
directly; the manifest's digest is what is checked either way.

### Download and verify

```bash
go run github.com/hyz-is/arandu-face/cmd/face-models@v0.2.0 -dir /srv/models
go run github.com/hyz-is/arandu-face/cmd/face-models@v0.2.0 -dir /srv/models -check
go run github.com/hyz-is/arandu-face/cmd/face-models@v0.2.0 -dir /srv/models -only yunet_2026may
```

A models directory holds each file at the path its manifest's `file` says
(`face/face_detection_yunet_2026may.onnx`, `age/mivolo_v2_age.onnx`, …). The
command downloads each file over HTTPS from the source its manifest records,
follows only HTTPS redirects, hashes while it writes, and renames the file into
place only when the digest matches. A file already present with the right
digest is left alone. `-check` verifies what is there, connects to nothing, and
exits non-zero when a file is missing or altered. **The command is the only
code in this repository that opens a network connection**, and an application
never links it, which is why `arandu.mod.toml` declares `network = false`.

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
  The rewritten file has no `Col2Im`, four `ConvTranspose`, and every other
  operator count of the original, and gives the original's answer exactly
  (difference 0) on four seeded tensors and eight drawn scenes, on the CPU.
- Concurrent calls from eight goroutines give the same answers as sequential
  ones, under the race detector.

Speed on the 480×560 cartoon, CPU provider, ONNX Runtime 1.30.0, on an Apple
M4 Max (`go test -run '^$' -bench . ./tests/Feature/`), over several runs:

| call | per image |
| --- | --- |
| `Detect`, YuNet | about 3.5–5 ms |
| `Detect`, BlazeFace | about 0.6–1 ms |
| `Align` | about 0.1–0.4 ms |
| `Quality` | about 0.6–1.9 ms |
| `Embed` | about 3.5–7.5 ms |
| `EstimateAge` | about 200–440 ms |

These figures are indicative: one machine, synthetic images, no accelerator.

**Detection accuracy, the false-match and false-non-match rates of the
embedding, the calibration of any threshold, the error of the age estimate
and any difference between demographic groups have not been measured.** They
must be measured by the owner on consented, representative data before any of
these signals weighs on a decision. `SFaceReferenceThresholds()` and
`DefaultQualityThresholds()` are starting values, versioned as such
(`sface-opencv-2021dec`, `arandu-face-quality-0.1-uncalibrated`), and are not
calibrated on any population.

## Concurrency

`Detect`, `Embed` and `EstimateAge` are safe for concurrent use. Calls share
the ONNX Runtime sessions, whose runs are thread-safe; the module's
reader-writer lock lets them proceed together and makes `Close` wait for
them. `Align`, `Quality` and `Compare` are pure functions.

## Limitations, and what needs evaluation

- YuNet sees at most `DetectMaxSide` pixels on the longer side; a small face on
  a large photograph may need a crop or a larger limit.
- BlazeFace is meant for selfies, where the face covers a large part of the
  image, and has no mouth corners.
- The yaw signal is a geometric proxy from 2D landmarks, not a head-pose
  angle; there is no pitch signal and no occlusion signal.
- The warp is bilinear in floating point; OpenCV quantizes to 1/32 pixel, so
  crops differ from OpenCV's by rounding.
- SFace's upstream weights carry no licence of their own and its training
  data is undocumented; MiVOLO was trained partly on proprietary data. Both
  are approved with those facts recorded as accepted risks, not resolved.
- Every figure above is on synthetic images.
- Nothing here has been evaluated on photographs of real people. Detection
  rate, embedding error rates, age error and their variation across age, skin
  tone, sex, capture device and lighting need a consented, representative
  evaluation before use.

## Security and privacy

Face images and the embeddings derived from them are biometric data, which
LGPD, GDPR and similar laws treat as sensitive personal data. This package
holds none of it beyond the call that measures it: it does not store, log or
transmit an image, an embedding or a result. Collecting only what is needed,
encrypting the raw image apart from the normalized facts, controlling and
auditing every access, obtaining a lawful basis or consent, and deleting on
schedule are the application's responsibility.

An embedding is a biometric template: treat it like the image it came from.
Comparisons across models, versions or files are refused (`ErrModelMismatch`),
so a template is never silently compared with one it cannot be compared with.

`DecodeImage` bounds the pixel count before decoding, and every model file is
verified against its digest before the native runtime parses it. Report
vulnerabilities privately; see [SECURITY.md](SECURITY.md).

## Why there is no table, policy or route

This repository has the shape of the Arandu package skeleton without its demo
entity, like `arandu-onnx`. It wraps a technology and owns no record, so there
is nothing for a policy to decide about; a route would measure any face for
anybody who sent one. Faces are measured through the application's
controllers and Grants, and biometric data is the application's to minimize,
encrypt and audit.

## Tests

```bash
export GOWORK=off CGO_ENABLED=1
test -z "$(gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go'))"
go build ./...
go vet ./...
go test -race -count=1 ./...
```

The model tests read the models from the directory `FACE_MODELS_DIR` names, or
from `../../models` when it is unset, and skip, saying where they looked, when
the files are not there; they skip too when ONNX Runtime is absent. No test
downloads anything. To run them:

```bash
go run ./cmd/face-models -dir /tmp/face-models
FACE_MODELS_DIR=/tmp/face-models go test -race -count=1 ./...
```

CI installs ONNX Runtime 1.30.0, fetches and verifies the four models with the
same command, and runs the suite with nothing skipped. Every image the tests
read is drawn in Go; no photograph of a person is read or committed.

## Versioning and releases

The module follows [Semantic Versioning](https://semver.org/). While the major
version is `v0`, a minor release may break the API, and every break is named in
[UPGRADE.md](UPGRADE.md); CI compares the exported API with the last tag. A new
model, a new model revision, a changed default threshold or a changed threshold
version changes what every installer records as evidence, and is always a
changelog line.

A release is a tag `vX.Y.Z` on `main` whose notes are the `## [X.Y.Z] - date`
entry already committed in [CHANGELOG.md](CHANGELOG.md). The release workflow
refuses a `go.mod` that still replaces a module with a local path, re-runs the
gates on the exact tag archive with the models fetched, audits known
vulnerabilities, and publishes those notes. A published version is immutable.

## Licence

The code is MIT; see [LICENSE.md](LICENSE.md). Copyright HYZIS - Serviços
Digitais Ltda.

The model weights are not part of this repository; each is under its own
licence, listed in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md). The
algorithmic constants this code follows are attributed in [NOTICE](NOTICE).
