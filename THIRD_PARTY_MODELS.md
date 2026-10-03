# Third-party models

The model weights this package runs are **not** in this repository and are not
distributed with it. `cmd/face-models` downloads each file from the source its
manifest records and keeps it only when its SHA-256 matches. Whoever downloads
and deploys them accepts their licence. The table below is what was read before
each status was set; it is not legal advice, and the open questions are stated
as such.

| id | source URL | sha256 | code licence | model licence | commercial use | redistribution | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `yunet_2026may` | <https://media.githubusercontent.com/media/opencv/opencv_zoo/26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/face_detection_yunet_2026may.onnx> | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` | MIT | MIT | yes, with a recorded training-data risk | allowed under MIT, with its notice | approved |
| `blazeface_short_range` | <https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_detection_short_range.onnx> | `2f2689b040becf555706d2cb978d2f0e3296ea82413734fba9a856c66c5f2b17` | Apache-2.0 | Apache-2.0 | yes | allowed under Apache-2.0, with its licence and notice | approved |
| `sface_2021dec` | <https://media.githubusercontent.com/media/opencv/opencv_zoo/ba91a3b91d00d76e86540d4013f944bd6b514e39/models/face_recognition_sface/face_recognition_sface_2021dec.onnx> | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 | Apache-2.0 as opencv_zoo distributes it; the upstream weights carry no licence | **unknown** | unsettled | **dev-only** |
| `mivolo_v2_age` | <https://huggingface.co/Sam-Apostel/mivolo-v2-age-onnx/resolve/8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36/mivolo_v2_age.onnx> | `2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951` | Apache-2.0 | Apache-2.0 | permitted by the licence | allowed under Apache-2.0, with its licence and notice | **dev-only** |

## Training data and open questions

- **YuNet** was trained on WIDER FACE, which is distributed under CC BY-NC-ND
  4.0. The weights are MIT; whether the dataset's non-commercial term reaches a
  model trained on it is unsettled, and is recorded as a risk rather than
  resolved here.
- **BlazeFace short range** is Google's model, described by Google's own model
  card, including its fairness evaluation. The file comes from a third-party
  ONNX export whose release tag is mutable; the SHA-256, which equals the
  release asset's published digest, is what pins it.
- **SFace**: opencv_zoo distributes the file under Apache-2.0, but the upstream
  weights carry no licence and the training data is undocumented. Commercial
  use is unknown, so it is `dev-only`.
- **MiVOLO v2 (age)**: the authors' weights are Apache-2.0, and were trained
  partly on proprietary data. The file is a third-party export of the age
  output. It stays `dev-only` until that export and its data are reviewed.

`dev-only` models load and say so in every `ModelInfo.Status`; with
`Config.RefuseDevOnly` set, `Boot` refuses them before reading a file.

None of these models has been evaluated by this project on photographs of
people. Accuracy, error rates and their variation across demographic groups
must be measured on consented, representative data before any signal is used
in a decision.

## Changing a model

A new model or a new revision is a new manifest under `manifests/` and a model
card in `models.go`, with the source pinned, the digest of the exact file, and
the licence fields filled from what the upstream says. It starts as `dev-only`
until its licence and training data have been reviewed, and moving it to
`approved` is a changelog line. Models that infer race, ethnicity or gender,
and non-commercial recognizers, are refused by a unit test.
