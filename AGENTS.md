# Working on Arandu Face

This is an Arandu package that measures faces with ONNX models run through
ONNX Runtime by `arandu-onnx`. It returns separate technical signals — a
detection, an alignment, quality measurements, an embedding, a distance, an
age estimate — each with the `ModelInfo` of the model behind it and the
versioned parameters it was made with.
It is a Go module somebody `go get`s and registers by hand in their own
`bootstrap/app.go`, which is the whole difference from working in an
application. There is no service provider, no container and no discovery — if a
line of wiring is not written in the installer's repository, it does not happen.

Read `.agents/skills/` before writing code. Each skill is a procedure, and the
one you need is named by the situation you are in.

## The KUAA rules

These come from the project, not from taste. A change that breaks one is not
merged.

1. **Signals, never decisions.** Every stage is its own exported call with its
   own result. Nothing here returns "match", "verified", "approved", "live" or
   "adult", and no function runs several stages for the caller (there is no
   `VerifyFace`). Thresholds travel with the results as versioned parameters
   (`ThresholdSet`, `QualityThresholds`); what they mean for a case is the
   policy engine's business, in the application.
   `tests/Unit/decision_test.go` refuses the names that would break this.
2. **No Python.** Not as a runtime, not in a gate, not in a test, not to
   generate a fixture. Models arrive as already-exported ONNX files; test
   images are drawn in Go.
3. **Model provenance is mandatory.** Every model is opened through its
   embedded manifest, which `Runtime.Open` verifies. A new model is a manifest
   with a pinned revision, a digest and its licences, plus a `ModelCard` for
   what the manifest has no field for (training data, reported accuracy),
   reviewed before its status is `approved`. `dev-only` models are refused by
   `Boot` when `Config.RefuseDevOnly` is set.
4. **No sensitive or non-commercial model.** No model that infers race or
   ethnicity (sensitive data under LGPD art. 11), no gender classifier, and no
   non-commercial recognizer (InsightFace, ArcFace, buffalo, EdgeFace).
   `tests/Unit/models_test.go` holds the list.
5. **The library does not touch the network.** `cmd/face-models` is the one
   program that downloads, and an application never links it. The library
   reads model files from `Config.ModelsDir`.
6. **Configuration is typed.** `Config` comes from the application. This
   package reads no environment variable; only the tests read
   `KUAA_MODELS_DIR`.
7. **Fixtures are synthetic.** No photograph of a person, no face dataset, no
   name or number in a test, an example or a log line. The face the tests
   detect is a cartoon drawn from ellipses in `tests/Feature/fixture_test.go`.

## The gates

Nothing is finished until all four exit zero.

```sh
export GOWORK=off CGO_ENABLED=1
gofmt -l $(find . -name '*.go' -not -path '*/testdata/*' -not -name '*.kyse.go')
go build ./...
go vet ./...
go test -race ./...
```

`CGO_ENABLED=1` is required: arandu-onnx's binding is cgo. The model tests
need the ONNX Runtime shared library and the model files, from
`KUAA_MODELS_DIR` or the KUAA workspace's `models/`; where either is absent
they skip saying why, and the skip is a reason to fix the machine, not a green
result for the models.

`GOWORK=off` matters because this checkout may sit beside a Go workspace that
does not list it.

`go.mod` replaces `github.com/kuaa-io/arandu-onnx` with `../arandu-onnx` until
that module is tagged. Build from a checkout that has it beside this one.

`aru doctor` reads applications, not libraries. That is why
`tests/Unit/audit_test.go` compares `arandu.mod.toml` with what the code calls.

## What this repository holds

```
module.go     registration: Name, Routes (none), Boot, Health, Models, Close
config.go     what the application passes in, and the defaults
models.go     the embedded manifests and the model cards
detect.go     Detect: YuNet or BlazeFace, pre-processing, Face
yunet.go      YuNet decoding
blazeface.go  BlazeFace anchors and decoding
geometry.go   Point, Box, IoU, NMS, WeightedNMS
align.go      2×2 SVD, Umeyama similarity, WarpAffine, Align
quality.go    Quality: pure-Go signals and versioned checks
embed.go      Embed, Compare, ThresholdSet
age.go        EstimateAge
manifests/    one manifest per model
cmd/face-models/  downloads and verifies the model files
```

Tests live under `tests/Unit` (decoding, suppression, alignment, quality,
comparison, manifests, config, the audit, the no-decision rule) and
`tests/Feature` (the module, the command's check mode, and every test that
runs a model).

## What does not exist here, on purpose

| A change reaches for | What is here instead |
| --- | --- |
| a `match`, `verified`, `approved` or `live` field | the measurement and the `ThresholdSet` beside it. The policy engine decides |
| one call that detects, aligns, embeds and compares | five calls, each with its own result and model |
| an age gate ("is adult") | `AgeEstimate.Years` with the reported MAE as metadata |
| FairFace, a gender head, InsightFace weights | nothing. They are refused by test |
| a model file in the repository | its manifest; the file is fetched by `cmd/face-models` |
| a download in the library or a test | `cmd/face-models`, run once where the application is deployed |
| loading a model by path alone | `Runtime.Open(manifest, ModelsDir)` |
| a route that reads a selfie | nothing. Faces are measured through the application's controllers |
| OpenCV, a Python reference, a photograph as fixture | the Go implementations and images drawn in Go |

## Writing code

Everything in the source is in English: identifiers, doc comments, internal
comments, error messages, log messages, and the names and messages of tests.

Every exported symbol carries a doc comment, and the comment documents the
symbol and nothing else. A date, an issue number, a version in progress or the
name of another repository does not belong there.

Tests go under `tests/`, in a capitalized category directory declaring a
lowercase external package: `tests/Unit` holds `package unit_test`,
`tests/Feature` holds `package feature_test`. Both import the package by its
module path, which is what makes them see exactly what a caller sees.
