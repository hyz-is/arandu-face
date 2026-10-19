package face

import (
	"embed"
	"fmt"
	"io/fs"
	"path"
	"sort"
	"strings"

	onnx "github.com/hyz-is/arandu-onnx"
)

// The identifiers of the models this package ships manifests for.
const (
	// ModelYuNet is YuNet 2026may, the default face detector: dynamic input,
	// five landmarks.
	ModelYuNet = "yunet_2026may"
	// ModelBlazeFace is MediaPipe's BlazeFace short range, a detector for
	// selfies, where the face fills much of a 128×128 view.
	ModelBlazeFace = "blazeface_short_range"
	// ModelSFace is SFace 2021dec, a 128-dimensional face embedding.
	ModelSFace = "sface_2021dec"
	// ModelMiVOLOAge is the age output of MiVOLO v2, as its export was
	// published. Its graph has four Col2Im nodes, which TensorRT and ONNX
	// Runtime's CUDA provider have no kernel for.
	ModelMiVOLOAge = "mivolo_v2_age"
	// ModelMiVOLOAgeConvTranspose is ModelMiVOLOAge with every Col2Im node
	// rewritten into a Reshape and a grouped ConvTranspose: the same weights
	// and the same numbers, in a graph an accelerator runs whole. It is the
	// age model EstimateAge runs by default.
	ModelMiVOLOAgeConvTranspose = "mivolo_v2_age_convt"
)

// RewriteCol2ImToConvTranspose names, in a Derivation, the rewrite that
// replaces every Col2Im node with a Reshape and a grouped ConvTranspose:
// rewrite.Col2ImToConvTranspose, run with its default options.
const RewriteCol2ImToConvTranspose = "col2im-to-convtranspose"

//go:embed manifests/*.json
var manifestFiles embed.FS

// Manifests returns the manifest of every model this package can run, sorted
// by identifier. Each one records where its file was obtained, its SHA-256,
// its licences and its status, and Runtime.Open verifies the file against it.
func Manifests() ([]onnx.Manifest, error) {
	entries, err := fs.ReadDir(manifestFiles, "manifests")
	if err != nil {
		return nil, fmt.Errorf("face: listing the embedded manifests: %w", err)
	}
	out := make([]onnx.Manifest, 0, len(entries))
	for _, entry := range entries {
		data, err := manifestFiles.ReadFile(path.Join("manifests", entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("face: reading the embedded manifest %s: %w", entry.Name(), err)
		}
		m, err := onnx.LoadManifest(data)
		if err != nil {
			return nil, fmt.Errorf("face: embedded manifest %s: %w", entry.Name(), err)
		}
		if want := strings.TrimSuffix(entry.Name(), ".json"); m.ID != want {
			return nil, fmt.Errorf("face: embedded manifest %s carries the id %q", entry.Name(), m.ID)
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// LookupManifest returns the manifest of one model by its identifier.
func LookupManifest(id string) (onnx.Manifest, error) {
	all, err := Manifests()
	if err != nil {
		return onnx.Manifest{}, err
	}
	for _, m := range all {
		if m.ID == id {
			return m, nil
		}
	}
	return onnx.Manifest{}, fmt.Errorf("face: no model is called %q", id)
}

// ModelCard is what the manifest has no field for: the model's training data,
// what its authors report about it, and the caveats a reviewer recorded.
type ModelCard struct {
	// ID is the manifest's model identifier.
	ID string
	// Source is where the model comes from, upstream of the file.
	Source string
	// TrainingData is what is known of the data the weights were trained on,
	// and under which terms.
	TrainingData string
	// Reported is what the model's authors report about its accuracy, quoted
	// as metadata; nothing here measured it.
	Reported string
	// Notes are the reviewer's caveats.
	Notes string
	// Review is who set the manifest's status, when, and on what grounds,
	// including the risks that were accepted rather than resolved.
	Review string
	// Derivation is how the file was made from another model this package
	// pins, or nil for a file used as it was published.
	Derivation *Derivation
}

// Derivation records a model file made from another one rather than
// downloaded: what it was made from, by what, and how the two were compared.
//
// A derived file has no address of its own. Its manifest's source_url is the
// original's, and cmd/face-models makes it from the verified original by
// running the named rewrite, keeping the result only when its SHA-256 is the
// derived manifest's.
type Derivation struct {
	// From is the manifest id of the original model.
	From string
	// FromSHA256 is the SHA-256 of the original file the derived one was made
	// from.
	FromSHA256 string
	// Rewrite names the graph rewrite applied, such as
	// RewriteCol2ImToConvTranspose.
	Rewrite string
	// Tool is the module and version that implements the rewrite.
	Tool string
	// Equivalence is how the derived model was compared with the original,
	// and the largest difference seen.
	Equivalence string
}

// openSourceReview is the ground the embedding and age models were approved
// on.
const openSourceReview = "Approved on 2026-10-19 by the maintainer (decision ADR-0010 of KUAA, the application these " +
	"engines were built for): open-source weights under a permissive licence (Apache-2.0, MIT or BSD) are eligible " +
	"for production, and no paid licence is used. "

// mivoloCard is what both MiVOLO files share: one set of weights.
var mivoloCard = ModelCard{
	Source: "iitolstykh/mivolo_v2 (Apache-2.0 since 2026-03-11), age output exported to ONNX by Sam-Apostel/mivolo-v2-age-onnx, revision 8eb4cd8f5dd4bd28df2c9a43b24bc54d6d139d36",
	TrainingData: "Partly proprietary, as MiVOLO's authors describe it, and not disclosed further. A third-party " +
		"export of the weights.",
	Reported: "MAE 3.65 years with face and body on LAGENDA, as reported by the authors; not measured here.",
	Notes: "Input is the face crop and the body crop, each letterboxed to 384x384 on black, RGB/255, ImageNet " +
		"mean and std; a missing body is an all-black image normalized the same way. Output is years.",
	Review: openSourceReview + "The weights are Apache-2.0. The partly proprietary training data is a risk the " +
		"maintainer accepted and recorded, not one resolved here. Apparent age is a signal, never an age verification.",
}

// cards are the model cards, by identifier.
var cards = map[string]ModelCard{
	ModelYuNet: {
		ID:     ModelYuNet,
		Source: "opencv/opencv_zoo models/face_detection_yunet, commit 26cc381e4d2594bb9f47a26eb8fd96c94a13660d (dynamic-input export)",
		TrainingData: "WIDER FACE, distributed under CC BY-NC-ND 4.0. The weights are MIT; whether the dataset's " +
			"non-commercial term reaches a model trained on it is unsettled, and is recorded as a risk.",
		Reported: "opencv_zoo reports WIDER FACE validation AP for its YuNet releases; not measured here.",
		Notes: "Input is raw BGR 0..255 padded bottom and right to a multiple of 32. Landmarks: right eye, left eye, " +
			"nose tip, right mouth corner, left mouth corner (subject's right = image left).",
		Review: "Approved at the model review of 2026-10-02: MIT weights; the training data's non-commercial term is " +
			"a recorded risk.",
	},
	ModelBlazeFace: {
		ID:     ModelBlazeFace,
		Source: "Google MediaPipe face_detection_short_range, exported to ONNX by yakhyo/mediapipe-face-mesh-onnx (release \"weights\")",
		TrainingData: "Described by Google's model card for BlazeFace short range (Apache-2.0). The release tag the file " +
			"comes from is not immutable; the digest is what pins it.",
		Reported: "Google's model card reports its own fairness and accuracy evaluation; not measured here.",
		Notes: "Meant for selfies, where the face covers a large part of the image. Emits six keypoints: right eye, " +
			"left eye, nose tip, mouth centre, right ear tragion, left ear tragion; no mouth corners.",
		Review: "Approved at the model review of 2026-10-02: Apache-2.0 weights, training data described by Google.",
	},
	ModelSFace: {
		ID:     ModelSFace,
		Source: "opencv/opencv_zoo models/face_recognition_sface, commit ba91a3b91d00d76e86540d4013f944bd6b514e39",
		TrainingData: "Undocumented. opencv_zoo distributes the file under Apache-2.0; the upstream weights carry no " +
			"licence of their own.",
		Reported: "opencv_zoo gives reference thresholds of cosine >= 0.363 and L2 <= 1.128; not calibrated here.",
		Notes:    "Input is the 112x112 aligned crop, RGB, raw 0..255: the normalization is inside the graph.",
		Review: openSourceReview + "The file is used under the Apache-2.0 licence opencv_zoo distributes it with. " +
			"The upstream weights carrying no licence of their own, and the undocumented training data, are risks the " +
			"maintainer accepted and recorded, not ones resolved here.",
	},
	ModelMiVOLOAge: withID(mivoloCard, ModelMiVOLOAge, nil),
	ModelMiVOLOAgeConvTranspose: withID(mivoloCard, ModelMiVOLOAgeConvTranspose, &Derivation{
		From:       ModelMiVOLOAge,
		FromSHA256: "2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951",
		Rewrite:    RewriteCol2ImToConvTranspose,
		Tool:       "github.com/hyz-is/arandu-onnx v0.3.0, package rewrite (also cmd/arandu-onnx-rewrite)",
		Equivalence: "Run beside the original on the ONNX Runtime 1.30.0 CPU provider over 15 synthetic inputs " +
			"in two runs of the rewrite command (5 inputs, then 10 with seed 7 in -3..3): max |original - rewritten| " +
			"= 0 on the age output. tests/Feature/age_test.go repeats the comparison on every run with the models.",
	}),
}

// withID returns a copy of a shared card for one of its files.
func withID(card ModelCard, id string, derivation *Derivation) ModelCard {
	card.ID = id
	card.Derivation = derivation
	return card
}

// LookupModelCard returns the model card of one model by its identifier.
func LookupModelCard(id string) (ModelCard, error) {
	card, ok := cards[id]
	if !ok {
		return ModelCard{}, fmt.Errorf("face: no model card for %q", id)
	}
	return card, nil
}
