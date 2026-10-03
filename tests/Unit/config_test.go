package unit_test

import (
	"math"
	"strings"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

func TestConfigValidateRefusesWhatCannotWork(t *testing.T) {
	t.Parallel()

	valid := face.Config{ModelsDir: "models"}
	if err := valid.Validate(); err != nil {
		t.Fatalf("a minimal config is refused: %v", err)
	}
	for name, tc := range map[string]struct {
		mutate func(*face.Config)
		field  string
	}{
		"no models dir":      {func(c *face.Config) { c.ModelsDir = "" }, "ModelsDir"},
		"unknown detector":   {func(c *face.Config) { c.Detector = "retina" }, "Detector"},
		"max side too small": {func(c *face.Config) { c.DetectMaxSide = 31 }, "DetectMaxSide"},
		"max side too large": {func(c *face.Config) { c.DetectMaxSide = 4097 }, "DetectMaxSide"},
		"score above one":    {func(c *face.Config) { c.ScoreThreshold = 1.1 }, "ScoreThreshold"},
		"nms negative":       {func(c *face.Config) { c.NMSThreshold = -0.1 }, "NMSThreshold"},
		"nms NaN":            {func(c *face.Config) { c.NMSThreshold = math.NaN() }, "NMSThreshold"},
		"negative top k":     {func(c *face.Config) { c.TopK = -1 }, "TopK"},
		"thresholds without a model": {func(c *face.Config) {
			c.Thresholds = face.ThresholdSet{Version: "x", CosineReference: 0.5, L2Reference: 1}
		}, "ThresholdSet"},
		"quality with a negative bound": {func(c *face.Config) {
			c.Quality = face.DefaultQualityThresholds()
			c.Quality.MinContrast = -1
		}, "MinContrast"},
	} {
		c := valid
		tc.mutate(&c)
		err := c.Validate()
		if err == nil {
			t.Errorf("%s: accepted", name)
			continue
		}
		if !strings.Contains(err.Error(), tc.field) {
			t.Errorf("%s: %v does not name %s", name, err, tc.field)
		}
	}
}

func TestConfigDefaultsDependOnTheDetector(t *testing.T) {
	t.Parallel()

	yunet := face.Config{ModelsDir: "m"}.WithDefaults()
	if yunet.Detector != face.DetectorYuNet || yunet.ScoreThreshold != 0.9 || yunet.NMSThreshold != 0.3 ||
		yunet.TopK != 5000 || yunet.DetectMaxSide != 640 {
		t.Errorf("YuNet defaults = %+v", yunet)
	}
	if yunet.Thresholds != face.SFaceReferenceThresholds() || yunet.Quality != face.DefaultQualityThresholds() {
		t.Errorf("the default thresholds are %+v and %+v", yunet.Thresholds, yunet.Quality)
	}
	blaze := face.Config{ModelsDir: "m", Detector: face.DetectorBlazeFace}.WithDefaults()
	if blaze.ScoreThreshold != 0.5 {
		t.Errorf("BlazeFace's default score threshold = %v, want 0.5", blaze.ScoreThreshold)
	}
	set := face.Config{ModelsDir: "m", ScoreThreshold: 0.7, DetectMaxSide: 1280}.WithDefaults()
	if set.ScoreThreshold != 0.7 || set.DetectMaxSide != 1280 {
		t.Errorf("WithDefaults overwrote a set field: %+v", set)
	}
}
