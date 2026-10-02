# Changelog

Everything worth knowing about a release of Arandu Face is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

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
  `QualityThresholds` (`kuaa-face-quality-0.1-uncalibrated`).
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
