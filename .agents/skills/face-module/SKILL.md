---
name: face-module
description: Change this Arandu package's face stages. Use when the request is to "add a model", "change the detector", "tune a threshold", "the landmarks are off", "add a quality signal", "add liveness", "return whether the faces match", "add an is-adult flag", "verify a face in one call", "add gender or ethnicity", or when a change touches detect.go, yunet.go, blazeface.go, geometry.go, align.go, quality.go, embed.go, age.go, models.go, config.go, module.go, manifests/ or cmd/face-models. Covers what may and may not change, where each constant comes from, and the tests that hold each property.
license: MIT
---

# Changing a stage

## Refused outright

- **A decision.** No `Match`, `Verified`, `Approved`, `IsAdult`, `Live`
  field or function. Each stage returns its measurement, its `ModelInfo` and
  the versioned parameters it used. `tests/Unit/decision_test.go` refuses the
  names.
- **A function that hides stages.** No `VerifyFace(img, ref)`. Detect, Align,
  Quality, Embed, Compare and EstimateAge stay separate calls, so each signal
  can be recorded and audited on its own.
- **A sensitive or non-commercial model.** Nothing that infers race, ethnicity
  or gender (LGPD art. 11), no FairFace, no InsightFace/ArcFace/buffalo/
  EdgeFace weights. `tests/Unit/models_test.go` holds the list.
- **A model without a manifest and a card,** or a manifest without a pinned
  revision, a digest and its licences.
- **A download in the library or in a test.** `cmd/face-models` is the only
  network code, and the tests run it in `-check` mode only.
- **A real face in a fixture.** Tests draw their images in Go; the face is the
  cartoon in `tests/Feature/fixture_test.go`.
- **Python,** anywhere, including "just to compare against OpenCV".
- **A route.** Faces are measured through the application's controllers.

## Where the constants come from

- YuNet: OpenCV's `FaceDetectorYN` (`face_detect.cpp`): divisor 32, strides
  8/16/32, score √(clamp(cls)·clamp(obj)), exp box sizes, `NMSBoxes` with
  score 0.9, IoU 0.3, top-k 5000. Input raw BGR 0..255.
- BlazeFace short range: MediaPipe's `face_detection_short_range` graph: SSD
  anchors (input 128, strides 8,16,16,16, offset 0.5, fixed size), scale 128,
  sigmoid with clipping at 100, min score 0.5, weighted NMS at 0.3. Input RGB
  −1..1, letterboxed.
- Alignment: OpenCV's `FaceRecognizerSF::alignCrop`: the five-point ArcFace
  template, Umeyama with the reflection guard and the rank-one branch, bilinear
  `warpAffine`.
- SFace: input RGB 0..255 (`blobFromImage(..., swapRB=true)`), output `fc1`,
  reference thresholds cosine 0.363 and L2 1.128.
- MiVOLO v2 age: the export's model card: letterbox 384 on black, RGB/255,
  ImageNet mean and std, face then body, black body when absent.

A change to any of them names its source in the commit message.

## Thresholds

Every threshold is a versioned value: `ThresholdSet.Version`,
`QualityThresholds.Version`, `Face.Parameters`. Changing a value is a new
version string, a changelog line, and never a boolean added to a result.

## Adding a model

1. Pin the revision; take the digest from the Git LFS object id, the release
   asset digest or the Hugging Face linked etag, and check it against the file
   you have.
2. Write `manifests/<id>.json`: id, version (naming the revision), file (the
   path under the models directory), sha256, `source_url`, licences, `status`,
   inputs and outputs as the graph declares them, precision.
3. Add its `ModelCard` in models.go: source, training data, reported accuracy,
   caveats.
4. Extend `tests/Unit/models_test.go`, and a feature test that runs it on a
   drawn image.

## Golden values

Unit tests compute their expected values by hand, in a comment beside the
assertion, never by running the code under test.

## Concurrency

`Detect`, `Embed` and `EstimateAge` hold the module's read lock for the whole
call; `Close` the write lock. Sessions are shared. Any change here runs
`go test -race ./...` with the library and the models present, and reads the
skip lines.
