package feature_test

import (
	"context"
	"sync"
	"testing"

	face "github.com/hyz-is/arandu-face"
)

// TestStagesRunConcurrently runs every model stage from many goroutines on
// one module at once. Under -race it holds that the calls share nothing
// mutable; the comparison with a sequential run holds that sharing the
// sessions does not change an answer.
func TestStagesRunConcurrently(t *testing.T) {
	m := bootModule(t, face.Config{EnableEmbedding: true, EnableAge: true})
	ctx := context.Background()
	img := cartoonFace(480, 560, 240, 250, 95)

	faces, err := m.Detect(ctx, img)
	if err != nil || len(faces) != 1 {
		t.Fatalf("Detect = %d faces, %v", len(faces), err)
	}
	aligned, err := face.Align(img, faces[0].Landmarks)
	if err != nil {
		t.Fatal(err)
	}
	wantEmbedding, err := m.Embed(ctx, aligned.Image)
	if err != nil {
		t.Fatal(err)
	}
	wantAge, err := m.EstimateAge(ctx, img, faces[0], nil)
	if err != nil {
		t.Fatal(err)
	}

	const workers, rounds = 8, 3
	var wg sync.WaitGroup
	errs := make(chan error, workers*rounds*4)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				got, err := m.Detect(ctx, img)
				if err != nil {
					errs <- err
					continue
				}
				if len(got) != 1 || got[0].Box != faces[0].Box {
					t.Errorf("a concurrent Detect found %+v", got)
					continue
				}
				a, err := face.Align(img, got[0].Landmarks)
				if err != nil {
					errs <- err
					continue
				}
				e, err := m.Embed(ctx, a.Image)
				if err != nil {
					errs <- err
					continue
				}
				for i := range e.Vector {
					if e.Vector[i] != wantEmbedding.Vector[i] {
						t.Errorf("a concurrent Embed differs at %d", i)
						break
					}
				}
				age, err := m.EstimateAge(ctx, img, got[0], nil)
				if err != nil {
					errs <- err
					continue
				}
				if age.Years != wantAge.Years {
					t.Errorf("a concurrent EstimateAge gave %v, want %v", age.Years, wantAge.Years)
				}
				if _, err := m.Quality(img, got[0]); err != nil {
					errs <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
