package unit_test

import (
	"go/ast"
	"regexp"
	"testing"
)

// TestNothingHereDecides holds the rule this package exists under: it returns
// separate, auditable signals, and the decision belongs to the application's
// policy engine.
//
// Two shapes would break it, and both are visible in syntax. A result type
// carrying a verdict -- a field called Match, Verified, Approved, Decision --
// is a decision with a model's name on it. A function that runs several
// stages behind one name -- VerifyFace, MatchFaces, CheckFace -- hides which
// signal said what. The names are what an assistant reaches for first, which
// is why they are refused by name.
func TestNothingHereDecides(t *testing.T) {
	t.Parallel()

	verdict := regexp.MustCompile(`(?i)(match|verif|approv|reject|decision|decide|verdict|issame|ismatch|accept|liveness|genuine|spoof)`)
	for _, source := range productionGoFiles(t, packageRoot(t)) {
		ast.Inspect(source.file, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.FuncDecl:
				if node.Name.IsExported() && verdict.MatchString(node.Name.Name) {
					t.Errorf("%s declares %s, a name that decides or hides stages", source.path, node.Name.Name)
				}
			case *ast.TypeSpec:
				if node.Name.IsExported() && verdict.MatchString(node.Name.Name) {
					t.Errorf("%s declares the type %s, a name that decides", source.path, node.Name.Name)
				}
				if st, ok := node.Type.(*ast.StructType); ok {
					for _, field := range st.Fields.List {
						for _, name := range field.Names {
							if name.IsExported() && verdict.MatchString(name.Name) {
								t.Errorf("%s: %s.%s is a verdict field", source.path, node.Name.Name, name.Name)
							}
						}
					}
				}
			}
			return true
		})
	}
}
