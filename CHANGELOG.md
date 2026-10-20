# Changelog

Everything worth knowing about a release of Arandu Face is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.2.0] - 2026-10-19

SFace and MiVOLO are approved, and the age stage runs a MiVOLO graph without
`Col2Im` by default, so TensorRT and the CUDA provider run it whole. It
requires `github.com/hyz-is/arandu-onnx` v0.3.0. Every v0.1.0 call site
builds unchanged; what changes at run time is the default age file, which a
models directory has to hold (see `UPGRADE.md`).

### Added

- Manifest `mivolo_v2_age_convt` (`age/mivolo_v2_age_convt.onnx`, sha256
  `cd7826cff39be95bbcbf8e7855a9971137cd0fb5bea8b536e2efee3e201e8333`,
  approved): `mivolo_v2_age` with its four `Col2Im` nodes rewritten into
  `Reshape` and a grouped `ConvTranspose` by arandu-onnx v0.3.0. Same
  weights, same contract, identical outputs on the CPU.
  `ModelMiVOLOAgeConvTranspose`.
- `Config.AgeModel`: the MiVOLO file `EnableAge` opens,
  `ModelMiVOLOAgeConvTranspose` (default, `DefaultAgeModel`) or
  `ModelMiVOLOAge`; `Validate` refuses any other.
- `ModelCard.Review` (who set the status, when, on what grounds, and the
  risks accepted) and `ModelCard.Derivation` (`Derivation{From, FromSHA256,
  Rewrite, Tool, Equivalence}`, for a file made from another model).
  `RewriteCol2ImToConvTranspose`.
- `cmd/face-models` makes a derived model from its verified original
  (fetched first when missing) with the pure-Go rewrite, and keeps it only
  when its SHA-256 is the manifest's; it prints `derived`.
- Tests: the default age graph has no `Col2Im`; the rewritten and original
  files agree on the CPU; no manifest is approved unless its
  `commercial_use` is `yes`; the derived manifest points at its original.

### Changed

- `sface_2021dec` and `mivolo_v2_age` move from `dev-only` to `approved`, by
  the maintainer's decision of 2026-10-19 (ADR-0010 of KUAA) that open-source
  weights under a permissive licence are eligible for production. The
  training-data gaps stay recorded in each `ModelCard` as accepted risks.
  `sface_2021dec`'s `commercial_use` is now `yes`, following the Apache-2.0
  licence opencv_zoo distributes it with; its `model_license` still says the
  upstream weights carry no licence of their own.
- `EstimateAge` runs `mivolo_v2_age_convt` unless `Config.AgeModel` says
  otherwise; its `AgeEstimate.Model.ID` is that id.
- With every shipped model approved, `RefuseDevOnly` refuses no
  configuration; the switch stays for a model added as dev-only.
- Requires `github.com/hyz-is/arandu-onnx` v0.3.0 (was v0.1.0).
- `.gitignore` ignores `tmp/`.

## [0.1.0] - 2026-10-03

The first public release, as `github.com/hyz-is/arandu-face`. It requires
`github.com/hyz-is/arandu-onnx` v0.1.0 and ONNX Runtime 1.26 or later (CI and
release run 1.30.0). SFace and MiVOLO are `dev-only`; see
`THIRD_PARTY_MODELS.md`.

### Added

- `Module`: `foundation.Module` with no route, plus `Bootable` (opens the
  configured models, verified against their manifests; with `RefuseDevOnly`,
  refuses dev-only models before reading any file), `Health`, `Models` and
  `Closable`. `New(Config, *onnx.Runtime)`.
- `Module.Detect`: YuNet 2026may (default) or BlazeFace short range, returning
  `[]Face{Box, Score, Landmarks, LandmarkScheme, Keypoints, Detector,
  Parameters}`.
- `Align`: Umeyama similarity onto the ArcFace five-point template and a
  bilinear warp to 112×112, returning `Aligned{Image, Transform, Template,
  Residual}`.
- `Quality` and `Module.Quality`: pure-Go signals (size, frame share,
  inter-ocular distance, roll, yaw proxy, landmarks in box, sharpness,
  brightness, contrast) with per-signal checks against versioned
  `QualityThresholds` (`arandu-face-quality-0.1-uncalibrated`).
- `Module.Embed`: L2-normalized SFace 2021dec embeddings (dev-only).
- `Compare` and `Module.Compare`: cosine and L2 between embeddings of one
  model, with the versioned `ThresholdSet` (`sface-opencv-2021dec`) beside
  them and no verdict.
- `Module.EstimateAge`: MiVOLO v2 apparent age (dev-only), with the reported
  MAE as metadata.
- `Config`: `ModelsDir`, `Detector` (`yunet`, `blazeface`), `EnableEmbedding`,
  `EnableAge`, `RefuseDevOnly`, `DetectMaxSide`, `ScoreThreshold`,
  `NMSThreshold`, `TopK`, `Thresholds`, `Quality`, with `Validate` and
  `WithDefaults`.
- Manifests for yunet_2026may, blazeface_short_range, sface_2021dec and
  mivolo_v2_age, and their `ModelCard`s. `Manifests`, `LookupManifest`,
  `LookupModelCard`.
- Pure-Go pieces: `DecodeYuNet`, `YuNetPaddedSize`, `YuNetOutputNames`,
  `BlazeFaceAnchors`, `DecodeBlazeFace`, `IoU`, `NMS`, `WeightedNMS`,
  `DecomposeSVD2`, `EstimateSimilarity`, `WarpAffine`, `LaplacianVariance`,
  `Luma`, `L2Normalize`, `RawPlanes`.
- `cmd/face-models`: downloads the model files over HTTPS and verifies their
  digests; `-check` verifies without connecting.
- `NOTICE`, attributing the published constants the stages follow, and
  `THIRD_PARTY_MODELS.md`, listing each model's source, digest, licences,
  training data and status. The weights are not distributed with the module.
