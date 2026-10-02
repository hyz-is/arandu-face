package face

import (
	"fmt"
	"math"
)

// The detectors a Config may name.
const (
	// DetectorYuNet is YuNet 2026may, for any photograph.
	DetectorYuNet = "yunet"
	// DetectorBlazeFace is BlazeFace short range, for selfies.
	DetectorBlazeFace = "blazeface"
)

// The defaults Config falls back to for a zero value.
const (
	// DefaultDetectMaxSide is the longest side, in pixels, of the image YuNet
	// is given; a larger image is scaled down to it.
	DefaultDetectMaxSide = 640
	// DefaultYuNetScoreThreshold is the score a YuNet candidate needs to be
	// kept, OpenCV's default.
	DefaultYuNetScoreThreshold = 0.9
	// DefaultBlazeFaceScoreThreshold is the score a BlazeFace candidate needs
	// to be kept, MediaPipe's default.
	DefaultBlazeFaceScoreThreshold = 0.5
	// DefaultNMSThreshold is the IoU above which a candidate is suppressed by
	// a better one.
	DefaultNMSThreshold = 0.3
	// DefaultTopK is how many candidates suppression considers.
	DefaultTopK = 5000
)

// Config is what the application passes when it wires this package.
//
// Every value comes from the application's own typed configuration. This
// package reads no environment variable. A numeric field left at zero, and a
// threshold set left without a version, takes the default named beside it.
type Config struct {
	// ModelsDir is the directory the model files are in, laid out the way
	// their manifests' file fields say (face/… and age/…). cmd/face-models
	// fills one.
	ModelsDir string

	// Detector is DetectorYuNet (the default) or DetectorBlazeFace.
	Detector string
	// EnableEmbedding opens SFace at Boot, which Embed needs.
	EnableEmbedding bool
	// EnableAge opens MiVOLO at Boot, which EstimateAge needs.
	EnableAge bool
	// RefuseDevOnly makes Boot refuse every configured model whose manifest
	// is dev-only, before reading any file: the production switch.
	RefuseDevOnly bool

	// DetectMaxSide caps the longer side of the image YuNet sees. Default
	// DefaultDetectMaxSide, between 32 and 4096. BlazeFace always sees 128.
	DetectMaxSide int
	// ScoreThreshold is the score a detection needs to be kept. Default
	// DefaultYuNetScoreThreshold or DefaultBlazeFaceScoreThreshold, by
	// detector.
	ScoreThreshold float64
	// NMSThreshold is the IoU above which a detection is suppressed by a
	// better one. Default DefaultNMSThreshold.
	NMSThreshold float64
	// TopK caps the candidates YuNet's suppression considers. Default
	// DefaultTopK, at most 100000.
	TopK int

	// Thresholds are the reference values Compare returns beside its
	// measurements. Default SFaceReferenceThresholds().
	Thresholds ThresholdSet
	// Quality are the bounds Quality checks signals against. Default
	// DefaultQualityThresholds().
	Quality QualityThresholds
}

// Validate reports what the configuration cannot be used with. New calls it,
// so a setting that cannot work fails where it is wired.
func (c Config) Validate() error {
	if c.ModelsDir == "" {
		return fmt.Errorf("face: Config.ModelsDir is required: it is the directory the model files are in")
	}
	switch c.Detector {
	case "", DetectorYuNet, DetectorBlazeFace:
	default:
		return fmt.Errorf("face: Config.Detector is %q, and has to be %q or %q", c.Detector, DetectorYuNet, DetectorBlazeFace)
	}
	if c.DetectMaxSide != 0 && (c.DetectMaxSide < 32 || c.DetectMaxSide > 4096) {
		return fmt.Errorf("face: Config.DetectMaxSide is %d, and has to be between 32 and 4096, or 0 for %d", c.DetectMaxSide, DefaultDetectMaxSide)
	}
	for _, unit := range []struct {
		name  string
		value float64
	}{
		{"ScoreThreshold", c.ScoreThreshold},
		{"NMSThreshold", c.NMSThreshold},
	} {
		if math.IsNaN(unit.value) || unit.value < 0 || unit.value > 1 {
			return fmt.Errorf("face: Config.%s is %v, and has to be between 0 and 1", unit.name, unit.value)
		}
	}
	if c.TopK < 0 || c.TopK > 100000 {
		return fmt.Errorf("face: Config.TopK is %d, and has to be between 0 and 100000", c.TopK)
	}
	if c.Thresholds.Version != "" {
		if err := c.Thresholds.Validate(); err != nil {
			return err
		}
	}
	if c.Quality.Version != "" {
		if err := c.Quality.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// WithDefaults returns the configuration with every zero field set to its
// default. It does not validate.
func (c Config) WithDefaults() Config {
	if c.Detector == "" {
		c.Detector = DetectorYuNet
	}
	if c.DetectMaxSide == 0 {
		c.DetectMaxSide = DefaultDetectMaxSide
	}
	if c.ScoreThreshold == 0 {
		c.ScoreThreshold = DefaultYuNetScoreThreshold
		if c.Detector == DetectorBlazeFace {
			c.ScoreThreshold = DefaultBlazeFaceScoreThreshold
		}
	}
	if c.NMSThreshold == 0 {
		c.NMSThreshold = DefaultNMSThreshold
	}
	if c.TopK == 0 {
		c.TopK = DefaultTopK
	}
	if c.Thresholds.Version == "" {
		c.Thresholds = SFaceReferenceThresholds()
	}
	if c.Quality.Version == "" {
		c.Quality = DefaultQualityThresholds()
	}
	return c
}

// detectorModel is the manifest id of the configured detector.
func (c Config) detectorModel() string {
	if c.Detector == DetectorBlazeFace {
		return ModelBlazeFace
	}
	return ModelYuNet
}
