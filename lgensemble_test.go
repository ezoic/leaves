package leaves

import (
	"bufio"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ezoic/leaves/mat"
	"github.com/ezoic/leaves/util"
)

func TestReadLGTree(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	// Read ensemble header (to skip)
	_, err = util.ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	// Read first tree only
	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if len(tree.nodes) != 2 {
		t.Fatalf("tree.nodes != 2 (got %d)", len(tree.nodes))
	}
	if tree.nCategorical != 1 {
		t.Fatalf("tree.nCategorical != 1 (got %d)", tree.nCategorical)
	}
	trueLeavesValues := []float64{0.56697267424823339, 0.3584987837673016, 0.41213915936587919}
	if err := util.AlmostEqualFloat64Slices(tree.leafValues, trueLeavesValues, 1e-10); err != nil {
		t.Fatalf("tree.leavesValues incorrect: %s", err.Error())
	}
	if tree.nodes[0].Flags&categorical == 0 {
		t.Fatal("first node should have categorical threshold")
	}
	if tree.nodes[0].Flags&catOneHot == 0 {
		t.Fatal("first node should have one hot decision rule")
	}
	if tree.nodes[0].Flags&leftLeaf == 0 {
		t.Fatal("first node should have right leaf")
	}
	if tree.nodes[0].Left != 0 {
		t.Fatal("first node should have leaf index 0")
	}
	if tree.nodes[0].Flags&missingNan == 0 {
		t.Fatal("first node should have missing nan")
	}
	if uint32(tree.nodes[0].Threshold) != 100 {
		t.Fatal("first node should have threshold = 100")
	}
	if tree.nodes[1].Flags&categorical != 0 {
		t.Fatal("second node should have numerical threshold")
	}
	if tree.nodes[1].Flags&defaultLeft == 0 {
		t.Fatal("second node should have default left")
	}
	if tree.nodes[1].Flags&rightLeaf == 0 {
		t.Fatal("second node should have left leaf")
	}
	if tree.nodes[1].Right != 2 {
		t.Fatal("second node should have leaf index 2")
	}
	if tree.nodes[1].Flags&leftLeaf == 0 {
		t.Fatal("second node should have right leaf")
	}
	if tree.nodes[1].Left != 1 {
		t.Fatal("second node should have leaf index 1")
	}
}

func TestLGTreeLeaf1(t *testing.T) {
	path := filepath.Join("testdata", "tree_1leaf.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 1 {
		t.Fatalf("expected tree with 1 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 0 {
		t.Fatalf("expected tree with 0 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0}
	check := func(truePred float64) {
		p, _ := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.123)
	fvals[0] = 10.0
	check(0.123)
	fvals[0] = -10.0
	check(0.123)
	fvals[0] = math.NaN()
	check(0.123)
}

func TestLGTreeLeaves2(t *testing.T) {
	path := filepath.Join("testdata", "tree_2leaves.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 2 {
		t.Fatalf("expected tree with 2 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 1 {
		t.Fatalf("expected tree with 1 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0}
	check := func(truePred float64) {
		p, _ := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.43)
	fvals[0] = 5.1
	check(0.59)
	fvals[0] = math.NaN()
	check(0.43)
}

func TestLGTreeLeaves3(t *testing.T) {
	path := filepath.Join("testdata", "tree_3leaves.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	if tree.nLeaves() != 3 {
		t.Fatalf("expected tree with 3 leaves (got %d)", tree.nLeaves())
	}
	if tree.nNodes() != 2 {
		t.Fatalf("expected tree with 2 node (got %d)", tree.nNodes())
	}

	fvals := []float64{0.0, 0.0}
	check := func(truePred float64) {
		p, _ := tree.predict(fvals)
		if !util.AlmostEqualFloat64(p, truePred, 1e-3) {
			t.Errorf("expected prediction %f, got %f", truePred, p)
		}
	}

	check(0.35)
	fvals[0] = 1000.0
	check(0.38)
	fvals[0] = math.NaN()
	check(0.35)
	fvals[1] = 10.0
	check(0.35)
	fvals[1] = 100.0
	check(0.54)
}

func checkPredLeaves(t *testing.T, predicted []float64, trueIds *mat.DenseMat) {
	rows := trueIds.Rows
	cols := trueIds.Cols
	if len(predicted) != rows*cols {
		t.Fatalf("predeicted size mismatch")
	}

	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if uint32(trueIds.Values[row*cols+col]) != uint32(predicted[row*cols+col]) {
				t.Fatalf("Predicted leaves don't match %v, %v at row = %d, col = %d", predicted, trueIds.Values, row, col)
			}
		}
	}
}

func TestLGPredLeaf(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	testPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")
	predLeavesTruthPath := filepath.Join("testdata", "lg_breast_cancer_data_pred_leaves.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}
	model = model.EnsembleWithLeafPredictions()

	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, " ", 0.0)
	predLeavesTruth, err := mat.DenseMatFromCsvFile(predLeavesTruthPath, 0, false, " ", 0.0)

	// Test Single
	fvals := test.Values[:test.Cols]
	res := model.PredictSingle(fvals, 0)
	if res != 0.0 {
		t.Errorf("Failed PredictSingle should return 0.0")
	}

	// Test Single
	predictions := make([]float64, 1*model.NOutputGroups())
	err = model.Predict(fvals, 0, predictions)
	if err != nil {
		t.Fatal(err)
	}
	checkPredLeaves(t, predictions, &mat.DenseMat{Values: predLeavesTruth.Values[0:predLeavesTruth.Cols], Cols: predLeavesTruth.Cols, Rows: 1})

	// Test Dense
	predictionsDense := make([]float64, test.Rows*model.NOutputGroups())
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictionsDense, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	checkPredLeaves(t, predictionsDense, predLeavesTruth)

	// Test batch and multi thread
	err = model.PredictDense(test.Values, test.Rows, test.Cols, predictionsDense, 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	checkPredLeaves(t, predictionsDense, predLeavesTruth)

	testCSR, err := mat.CSRMatFromArray(test.Values, test.Rows, test.Cols)
	if err != nil {
		t.Fatal(err)
	}

	// Test CSR
	predictionsCSR := make([]float64, testCSR.Rows()*model.NOutputGroups())
	err = model.PredictCSR(testCSR.RowHeaders, testCSR.ColIndexes, testCSR.Values, predictionsCSR, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	checkPredLeaves(t, predictionsCSR, predLeavesTruth)

	// Test batch and multi thread
	err = model.PredictCSR(testCSR.RowHeaders, testCSR.ColIndexes, testCSR.Values, predictionsCSR, 0, 5)
	if err != nil {
		t.Fatal(err)
	}
	checkPredLeaves(t, predictionsCSR, predLeavesTruth)
}

func TestLGEnsemble(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	model, err := LGEnsembleFromFile(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 2 {
		t.Fatalf("expected 2 trees (got %d)", model.NEstimators())
	}

	denseValues := []float64{0.0, 0.0,
		1000.0, 0.0,
		800.0, 0.0,
		800.0, 100,
		0.0, 100,
		1000, math.NaN(),
		math.NaN(), math.NaN(),
	}

	denseRows := 7
	denseCols := 2

	// check predictions
	predictions := make([]float64, denseRows)
	model.PredictDense(denseValues, denseRows, denseCols, predictions, 0, 0)

	truePredictions := []float64{0.29462594, 0.39565483, 0.39565483, 0.69580371, 0.69580371, 0.39565483, 0.29462594}
	if err := util.AlmostEqualFloat64Slices(predictions, truePredictions, 1e-7); err != nil {
		t.Fatalf("predictions on dense not correct (all trees): %s", err.Error())
	}

	// check prediction only on first tree
	model.PredictDense(denseValues, denseRows, denseCols, predictions, 1, 0)
	truePredictions = []float64{0.35849878, 0.41213916, 0.41213916, 0.56697267, 0.56697267, 0.41213916, 0.35849878}
	if err := util.AlmostEqualFloat64Slices(predictions, truePredictions, 1e-7); err != nil {
		t.Fatalf("predictions on dense not correct (all trees): %s", err.Error())
	}
}

func TestLGEnsembleJSON1tree1leaf(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_1tree_1leaf.json")
	// loading model
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 1 {
		t.Fatalf("expected 1 trees (got %d)", model.NEstimators())
	}

	if model.NOutputGroups() != 1 {
		t.Fatalf("expected 1 class (got %d)", model.NOutputGroups())
	}

	if model.NFeatures() != 41 {
		t.Fatalf("expected 41 class (got %d)", model.NFeatures())
	}

	features := make([]float64, model.NFeatures())
	pred := model.PredictSingle(features, 0)
	if pred != 0.42 {
		t.Fatalf("expected prediction 0.42 (got %f)", pred)
	}
}

func TestLGEnsembleJSON1tree(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_1tree.json")
	// loading model
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()
	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}
	if model.NEstimators() != 1 {
		t.Fatalf("expected 1 trees (got %d)", model.NEstimators())
	}

	if model.NOutputGroups() != 1 {
		t.Fatalf("expected 1 class (got %d)", model.NOutputGroups())
	}

	if model.NFeatures() != 2 {
		t.Fatalf("expected 2 class (got %d)", model.NFeatures())
	}

	check := func(features []float64, trueAnswer float64) {
		pred := model.PredictSingle(features, 0)
		if pred != trueAnswer {
			t.Fatalf("expected prediction %f (got %f)", trueAnswer, pred)
		}
	}

	check([]float64{0.0, 0.0}, 0.4242)
	check([]float64{0.0, 11.0}, 0.4242)
	check([]float64{0.13, 11.0}, 0.4242)
	check([]float64{0.0, 1.0}, 0.4703)
	check([]float64{0.0, 10.0}, 0.4703)
	check([]float64{0.0, 100.0}, 0.4703)
	check([]float64{0.15, 0.0}, 1.1111)
	check([]float64{0.15, 11.0}, 1.1111)
}

func TestLeafCounts(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	model, err := LGEnsembleFromFile(path, false)
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	leafCounts := model.LeafCounts()
	if leafCounts == nil {
		t.Fatal("LeafCounts returned nil")
	}
	// model_simple.txt has 2 trees
	if len(leafCounts) != 2 {
		t.Fatalf("Expected 2 trees, got %d", len(leafCounts))
	}
	// Both trees have leaf_count=200 341 459
	expected := []int64{200, 341, 459}
	for treeIdx, treeCounts := range leafCounts {
		if len(treeCounts) != len(expected) {
			t.Fatalf("Tree %d: expected %d leaf counts, got %d", treeIdx, len(expected), len(treeCounts))
		}
		for i, exp := range expected {
			if treeCounts[i] != exp {
				t.Errorf("Tree %d, Leaf %d: expected %d, got %d", treeIdx, i, exp, treeCounts[i])
			}
		}
	}
}

func TestLeafCountsTree2Leaves(t *testing.T) {
	path := filepath.Join("testdata", "tree_2leaves.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	tree, err := lgTreeFromReader(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	// tree_2leaves.txt has leaf_count=300 700
	expected := []int64{300, 700}
	if len(tree.leafCounts) != len(expected) {
		t.Fatalf("Expected %d leaf counts, got %d", len(expected), len(tree.leafCounts))
	}
	for i, exp := range expected {
		if tree.leafCounts[i] != exp {
			t.Errorf("Leaf %d: expected %d, got %d", i, exp, tree.leafCounts[i])
		}
	}
}

func TestPredictWithLeafIndices(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_breast_cancer.txt")
	testPath := filepath.Join("testdata", "lg_breast_cancer_data.txt")
	predLeavesTruthPath := filepath.Join("testdata", "lg_breast_cancer_data_pred_leaves.txt")

	model, err := LGEnsembleFromFile(modelPath, false)
	if err != nil {
		t.Fatal(err)
	}

	test, err := mat.DenseMatFromCsvFile(testPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}
	predLeavesTruth, err := mat.DenseMatFromCsvFile(predLeavesTruthPath, 0, false, " ", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	nEstimators := model.NEstimators()
	nGroups := model.NRawOutputGroups()
	nLeafIndices := nGroups * nEstimators

	for row := 0; row < test.Rows; row++ {
		fvals := test.Values[row*test.Cols : (row+1)*test.Cols]

		// Two-pass approach (existing)
		predTwoPass := make([]float64, model.NOutputGroups())
		err := model.Predict(fvals, 0, predTwoPass)
		if err != nil {
			t.Fatal(err)
		}

		leafModel := model.EnsembleWithLeafPredictions()
		leafTwoPass := make([]float64, leafModel.NOutputGroups())
		err = leafModel.Predict(fvals, 0, leafTwoPass)
		if err != nil {
			t.Fatal(err)
		}

		// Single-pass approach (new)
		predSinglePass := make([]float64, model.NOutputGroups())
		leafSinglePass := make([]float64, nLeafIndices)
		err = model.PredictWithLeafIndices(fvals, 0, predSinglePass, leafSinglePass)
		if err != nil {
			t.Fatal(err)
		}

		// Predictions must match exactly
		if err := util.AlmostEqualFloat64Slices(predTwoPass, predSinglePass, 1e-15); err != nil {
			t.Fatalf("row %d: predictions differ: %s", row, err)
		}

		// Leaf indices must match exactly
		if err := util.AlmostEqualFloat64Slices(leafTwoPass, leafSinglePass, 0); err != nil {
			t.Fatalf("row %d: leaf indices differ: %s", row, err)
		}

		// Also verify against known truth
		trueLeaves := predLeavesTruth.Values[row*predLeavesTruth.Cols : (row+1)*predLeavesTruth.Cols]
		for col := 0; col < predLeavesTruth.Cols; col++ {
			if uint32(trueLeaves[col]) != uint32(leafSinglePass[col]) {
				t.Fatalf("row %d, col %d: leaf index %d != truth %d", row, col, uint32(leafSinglePass[col]), uint32(trueLeaves[col]))
			}
		}
	}
}

func TestPredictWithLeafIndicesSimple(t *testing.T) {
	path := filepath.Join("testdata", "model_simple.txt")
	model, err := LGEnsembleFromFile(path, false)
	if err != nil {
		t.Fatal(err)
	}

	denseValues := []float64{0.0, 0.0,
		1000.0, 0.0,
		800.0, 0.0,
		800.0, 100,
		0.0, 100,
		1000, math.NaN(),
		math.NaN(), math.NaN(),
	}
	truePredictions := []float64{0.29462594, 0.39565483, 0.39565483, 0.69580371, 0.69580371, 0.39565483, 0.29462594}

	nEstimators := model.NEstimators()
	nGroups := model.NRawOutputGroups()
	nLeafIndices := nGroups * nEstimators

	for row := 0; row < 7; row++ {
		fvals := denseValues[row*2 : (row+1)*2]

		predictions := make([]float64, model.NOutputGroups())
		leafIndices := make([]float64, nLeafIndices)
		err := model.PredictWithLeafIndices(fvals, 0, predictions, leafIndices)
		if err != nil {
			t.Fatal(err)
		}

		if !util.AlmostEqualFloat64(predictions[0], truePredictions[row], 1e-7) {
			t.Errorf("row %d: expected %f, got %f", row, truePredictions[row], predictions[0])
		}
	}
}

func TestCatMediumDecision(t *testing.T) {
	tree := lgTree{
		nodes: []lgNode{
			categoricalNode(0, 0, math.Float64frombits(
				uint64(1<<5|1<<10|1<<31) | uint64(1<<(33-32)|1<<(50-32))<<32,
			), catMedium),
		},
		leafValues: []float64{1.0, 2.0},
	}
	tree.nodes[0].Flags |= leftLeaf | rightLeaf
	tree.nodes[0].Left = 0
	tree.nodes[0].Right = 1

	tests := []struct {
		fval     float64
		expected float64
	}{
		{5.0, 1.0},   // bit 5 set → left
		{10.0, 1.0},  // bit 10 set → left
		{31.0, 1.0},  // bit 31 set → left
		{33.0, 1.0},  // bit 33 set → left
		{50.0, 1.0},  // bit 50 set → left
		{0.0, 2.0},   // bit 0 not set → right
		{6.0, 2.0},   // bit 6 not set → right
		{32.0, 2.0},  // bit 32 not set → right
		{63.0, 2.0},  // bit 63 not set → right
		{64.0, 2.0},  // out of range → right
		{100.0, 2.0}, // out of range → right
		{-1.0, 2.0},  // negative → right
	}

	for _, tc := range tests {
		fvals := []float64{tc.fval}
		pred, _ := tree.predict(fvals)
		if pred != tc.expected {
			t.Errorf("fval=%f: expected %f, got %f", tc.fval, tc.expected, pred)
		}
	}
}

func TestLeafCountsJSON(t *testing.T) {
	modelPath := filepath.Join("testdata", "lg_1tree.json")
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	defer modelFile.Close()

	model, err := LGEnsembleFromJSON(modelFile, false)
	if err != nil {
		t.Fatal(err)
	}

	leafCounts := model.LeafCounts()
	if leafCounts == nil {
		t.Fatal("LeafCounts returned nil")
	}
	if len(leafCounts) != 1 {
		t.Fatalf("Expected 1 tree, got %d", len(leafCounts))
	}
	// lg_1tree.json has 3 leaves, each with leaf_count=38
	if len(leafCounts[0]) != 3 {
		t.Fatalf("Expected 3 leaf counts, got %d", len(leafCounts[0]))
	}
	for i, count := range leafCounts[0] {
		if count != 38 {
			t.Errorf("Leaf %d: expected count 38, got %d", i, count)
		}
	}
}
