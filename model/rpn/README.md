### `rpn` module overview

#### `anchor_generation.py`
Generates a regular grid of multi-scale, multi-aspect anchor boxes.

#### `proposal.py`
Applies RPN net outputs (per-anchor scores and bbox regression estimates) to the anchor boxes, thus convert them into object proposals (RoIs).

#### `anchor_refine.py`
Generates training targets/labels for each anchor. Classification labels are 1 (object), 0 (not object) or -1 (ignore). Bbox regression targets are specified when the classification label is > 0. Refines the anchor generation and improves the quality of anchors.

#### `proposal_refine.py`
Generates training targets/labels for each object proposal: classification labels 0 - K (bg or object class 1, ... , K)
and bbox regression targets in that case that the label is > 0. Refines the class-specific proposal and improves the quality of proposals.

#### `rpn.py`
The main RPN model.

#### `utils.py`
Common bounding box operations.

#### Notes
* other versions use centers & topleft and bottomright corners to locate a bounding box, which introduces more conversions for bbox operations. I used topleft corner & height-width to represent box, which is easier for calculation and more straightforward for computing the regression coefficients.
* Other versions generate anchors in both proposal layer and anchor target layer, which is redundant. I cached the generated anchors in RPN and pass to both layers.
* what's the point of setting foreground/background fraction?
* no need to unmap, just return the indices of kept anchor indices after bbox_drop
* jwang's implementation is wrong during the calculation of classification loss in RPN loss. It uses the don't care class as well, where we should rather mask out foreground + background anchors only.
