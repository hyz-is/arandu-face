package unit_test

import (
	"net/url"
	"regexp"
	"strings"
	"testing"

	face "github.com/kuaa-io/arandu-face"
	onnx "github.com/kuaa-io/arandu-onnx"
)

func TestEveryShippedManifestRecordsItsProvenance(t *testing.T) {
	t.Parallel()

	manifests, err := face.Manifests()
	if err != nil {
		t.Fatal(err)
	}
	type want struct {
		status     onnx.Status
		commercial onnx.CommercialUse
		file       string
		inputs     []string
		outputs    []string
	}
	expected := map[string]want{
		face.ModelBlazeFace: {onnx.StatusApproved, onnx.CommercialUseYes, "face/face_detection_short_range.onnx",
			[]string{"input"}, []string{"regressors", "scores"}},
		face.ModelMiVOLOAge: {onnx.StatusDevOnly, onnx.CommercialUseYes, "age/mivolo_v2_age.onnx",
			[]string{"faces_bodies"}, []string{"age"}},
		face.ModelSFace: {onnx.StatusDevOnly, onnx.CommercialUseUnknown, "face/face_recognition_sface_2021dec.onnx",
			[]string{"data"}, []string{"fc1"}},
		face.ModelYuNet: {onnx.StatusApproved, onnx.CommercialUseYes, "face/face_detection_yunet_2026may.onnx",
			[]string{"input"}, face.YuNetOutputNames()},
	}
	if len(manifests) != len(expected) {
		t.Fatalf("%d manifests, want %d", len(manifests), len(expected))
	}
	names := func(specs []onnx.TensorSpec) string {
		out := []string{}
		for _, s := range specs {
			out = append(out, s.Name)
		}
		return strings.Join(out, ",")
	}
	for _, m := range manifests {
		w, ok := expected[m.ID]
		if !ok {
			t.Errorf("unexpected manifest %s", m.ID)
			continue
		}
		if m.Status != w.status || m.CommercialUse != w.commercial || m.File != w.file || m.Precision != onnx.PrecisionFP32 {
			t.Errorf("%s: status %s, commercial use %s, file %s, precision %s", m.ID, m.Status, m.CommercialUse, m.File, m.Precision)
		}
		if names(m.Inputs) != strings.Join(w.inputs, ",") || names(m.Outputs) != strings.Join(w.outputs, ",") {
			t.Errorf("%s: contract %s -> %s", m.ID, names(m.Inputs), names(m.Outputs))
		}
		if u, err := url.Parse(m.SourceURL); err != nil || u.Scheme != "https" {
			t.Errorf("%s: source_url %s is not https", m.ID, m.SourceURL)
		}
		if _, err := face.LookupModelCard(m.ID); err != nil {
			t.Errorf("%s has no model card: %v", m.ID, err)
		}
	}
}

func TestTheManifestsRecordTheReviewedDigests(t *testing.T) {
	t.Parallel()

	// The digests the model review recorded, which are also the Git LFS object
	// ids at the pinned commits, the GitHub release asset digest and the
	// Hugging Face linked etag.
	for id, digest := range map[string]string{
		face.ModelYuNet:     "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
		face.ModelBlazeFace: "2f2689b040becf555706d2cb978d2f0e3296ea82413734fba9a856c66c5f2b17",
		face.ModelSFace:     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
		face.ModelMiVOLOAge: "2db5e05be33b3f120518a86f29a4a65eb0a01c2967219b8787a7dead9b796951",
	} {
		m, err := face.LookupManifest(id)
		if err != nil {
			t.Fatal(err)
		}
		if m.SHA256 != digest {
			t.Errorf("%s: sha256 %s, want %s", id, m.SHA256, digest)
		}
	}
	if _, err := face.LookupManifest("no-such-model"); err == nil {
		t.Error("an unknown id was found")
	}
}

func TestTheSourcesArePinnedToARevision(t *testing.T) {
	t.Parallel()

	pinned := regexp.MustCompile(`/[0-9a-f]{40}/`)
	for _, id := range []string{face.ModelYuNet, face.ModelSFace, face.ModelMiVOLOAge} {
		m, err := face.LookupManifest(id)
		if err != nil {
			t.Fatal(err)
		}
		match := pinned.FindString(m.SourceURL)
		if match == "" {
			t.Errorf("%s: source_url %s pins no 40-character revision", id, m.SourceURL)
			continue
		}
		// The version names the revision's first twelve characters.
		if !strings.Contains(m.Version, match[1:13]) {
			t.Errorf("%s: version %s does not name revision %s", id, m.Version, match[1:41])
		}
	}
}

// TestNoSensitiveOrNonCommercialModelIsShipped holds two exclusions: no model
// that infers race or ethnicity (sensitive data under LGPD art. 11), and none
// of the non-commercial recognizers.
func TestNoSensitiveOrNonCommercialModelIsShipped(t *testing.T) {
	t.Parallel()

	manifests, err := face.Manifests()
	if err != nil {
		t.Fatal(err)
	}
	forbidden := []string{"fairface", "race", "ethnic", "arcface", "insightface", "buffalo", "genderage", "edgeface", "gender"}
	for _, m := range manifests {
		text := strings.ToLower(m.ID + " " + m.File + " " + m.SourceURL)
		for _, word := range forbidden {
			if strings.Contains(text, word) {
				t.Errorf("%s mentions %q", m.ID, word)
			}
		}
		for _, out := range m.Outputs {
			for _, word := range forbidden {
				if strings.Contains(strings.ToLower(out.Name), word) {
					t.Errorf("%s has the output %s", m.ID, out.Name)
				}
			}
		}
	}
}

func TestTheModelCardsRecordWhatTheManifestCannot(t *testing.T) {
	t.Parallel()

	yunet, err := face.LookupModelCard(face.ModelYuNet)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(yunet.TrainingData, "WIDER FACE") || !strings.Contains(yunet.TrainingData, "CC BY-NC-ND") {
		t.Errorf("the YuNet card does not record its training data's licence: %q", yunet.TrainingData)
	}
	age, err := face.LookupModelCard(face.ModelMiVOLOAge)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(age.Reported, "3.65") || !strings.Contains(age.Reported, "LAGENDA") {
		t.Errorf("the MiVOLO card does not quote its reported MAE: %q", age.Reported)
	}
	if _, err := face.LookupModelCard("fairface"); err == nil {
		t.Error("a card for an unshipped model was found")
	}
}
