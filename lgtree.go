package leaves

import (
	"math"

	"github.com/ezoic/leaves/util"
)

const (
	categorical = 1 << 0
	defaultLeft = 1 << 1
	leftLeaf    = 1 << 2
	rightLeaf   = 1 << 3
	missingZero = 1 << 4
	missingNan  = 1 << 5
	catOneHot   = 1 << 6
	catSmall    = 1 << 7
	catMedium   = 1 << 8 // 64-bit inline bitset stored in Threshold via Float64frombits
)

const zeroThreshold = 1e-35

type lgNode struct {
	Threshold float64
	Left      uint32
	Right     uint32
	Feature   uint32
	Flags     uint16
}

type lgTree struct {
	nodes         []lgNode
	leafValues    []float64
	leafCounts    []int64
	catBoundaries []uint32
	catThresholds []uint32
	nCategorical  uint32
}

func (t *lgTree) numericalDecision(node *lgNode, fval float64) bool {
	if node.Flags&(missingZero|missingNan) == 0 {
		if math.IsNaN(fval) {
			fval = 0.0
		}
		return fval <= node.Threshold
	}
	if math.IsNaN(fval) {
		if node.Flags&missingNan > 0 {
			return node.Flags&defaultLeft > 0
		}
		fval = 0.0
	}
	if (node.Flags&missingZero > 0) && isZero(fval) {
		return node.Flags&defaultLeft > 0
	}
	// Note: LightGBM uses `<=`, but XGBoost uses `<`
	return fval <= node.Threshold
}

func (t *lgTree) categoricalDecision(node *lgNode, fval float64) bool {
	ifval := int32(fval)
	if ifval < 0 {
		return false
	}
	// int32(NaN) == 0 in Go, so NaN is only possible when ifval == 0.
	// Only check IsNaN when the node has missingNan set AND ifval is 0.
	if ifval == 0 && node.Flags&missingNan > 0 && math.IsNaN(fval) {
		return false
	}
	if node.Flags&catOneHot > 0 {
		return int32(node.Threshold) == ifval
	} else if node.Flags&catMedium > 0 {
		if ifval >= 64 {
			return false
		}
		// Safe: Float64bits/Float64frombits are Go-spec-guaranteed bitwise
		// identity operations. The packed uint64 is never passed through FP
		// arithmetic, so NaN bit patterns are preserved unchanged.
		bits := math.Float64bits(node.Threshold)
		return (bits>>uint(ifval))&1 != 0
	} else if node.Flags&catSmall > 0 {
		return util.FindInBitsetUint32(uint32(node.Threshold), uint32(ifval))
	}
	return t.findInBitset(uint32(node.Threshold), uint32(ifval))
}

func (t *lgTree) predict(fvals []float64) (float64, uint32) {
	nodes := t.nodes
	if len(nodes) == 0 {
		return t.leafValues[0], 0
	}
	leafValues := t.leafValues
	idx := uint32(0)
	for {
		node := &nodes[idx]
		var left bool
		if node.Flags&categorical > 0 {
			left = t.categoricalDecision(node, fvals[node.Feature])
		} else {
			left = t.numericalDecision(node, fvals[node.Feature])
		}
		if left {
			if node.Flags&leftLeaf > 0 {
				return leafValues[node.Left], node.Left
			}
			idx = node.Left
		} else {
			if node.Flags&rightLeaf > 0 {
				return leafValues[node.Right], node.Right
			}
			idx++
		}
	}
}

func (t *lgTree) findInBitset(idx uint32, pos uint32) bool {
	boundaries := t.catBoundaries
	thresholds := t.catThresholds
	idxE := boundaries[idx+1]
	idxS := boundaries[idx]
	span := idxE - idxS
	bit := pos & 31
	// Most large leaf-growth categorical splits use a single uint32 word; hot-path
	// that case without a general (i1 >= span) check when span is 1.
	if span == 1 {
		if pos>>5 != 0 {
			return false
		}
		return (thresholds[idxS]>>bit)&1 != 0
	}
	i1 := pos >> 5
	if i1 >= span {
		return false
	}
	// Slice once so i1 is proved against len(words); reduces repeated base+idx
	// bound checks versus thresholds[idxS+i1] after profile showed this path hot.
	words := thresholds[idxS:idxE]
	return (words[i1]>>bit)&1 != 0
}

func (t *lgTree) nLeaves() int {
	return len(t.nodes) + 1
}

func (t *lgTree) nNodes() int {
	return len(t.nodes)
}

func isZero(fval float64) bool {
	return (fval > -zeroThreshold && fval <= zeroThreshold)
}

func categoricalNode(feature uint32, missingType uint16, threshold float64, catType uint16) lgNode {
	node := lgNode{}
	node.Feature = feature
	node.Flags = categorical | missingType | catType
	node.Threshold = threshold
	return node
}

func numericalNode(feature uint32, missingType uint16, threshold float64, defaultType uint16) lgNode {
	node := lgNode{}
	node.Feature = feature
	node.Flags = missingType | defaultType
	node.Threshold = threshold
	return node
}
