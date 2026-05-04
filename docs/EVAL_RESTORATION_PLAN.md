# Evaluation Bias Analysis & Remediation Plan

## 🔍 The "Fake Results" Problem
During the investigation of the `lexigaze/scripts/geco` codebase, we identified a critical flaw in the evaluation pipeline that explains the suspiciously high (near 100%) accuracy results.

### Root Cause: State Space Collapsing
In `run_full_neurips_pipeline.py` and `evaluate_neurips_final.py`, the candidate "word boxes" passed to the decoder are derived directly from the **fixation sequence** rather than the **text layout**.

```python
# The current "cheating" implementation:
word_boxes = [[row.true_x - 20, ..., row.true_y + 15] for row in df.itertuples()]
```

1. **State-Target Alignment**: `word_boxes[i]` corresponds exactly to `gaze_sequence[i]`.
2. **Reduced Search Space**: If a trial has 50 fixations, the decoder only chooses between those 50 specific boxes.
3. **Diagonal Bias**: Since the Viterbi decoder uses a narrow transition window (e.g., $j \pm 3$), and the "correct" box for step $t$ is always at index $t$, the model is mathematically forced to succeed.
4. **Invisible Skips**: Words that were not fixated are completely missing from the state space. The model never has to decide to "skip" a word because the unvisited words don't exist in its world.

## 🛠️ Remediation Strategy: "Full Layout Evaluation"

To produce valid scientific results for the NeurIPS paper, the pipeline must be refactored to use a realistic state space.

### 1. Data Extraction Update (`data/batch_extract_features.py`)
- **Current**: Filters GECO data to only rows with fixations.
- **Required**: Must extract the **complete sequence of words** for the trial (the "Layout"), including words that received zero fixations.
- **Challenge**: The raw GECO `.xlsx` files likely contain the full sentence. We need to preserve the `WORD_ID` and coordinates for *every* word in the trial.

### 2. Decoder Input Update
- `word_boxes`: Should be the list of boxes for **all** words in the paragraph layout (e.g., 200 words).
- `base_cm`: Should be a vector of cognitive mass for **all** words in the layout.
- `targets`: Should be the **indices** of the words in the full layout that were actually fixated.
  - Example: If the user fixates words 1, 2, 4, 5 (skipping 3), `targets` = `[0, 1, 3, 4]`.

### 3. Transition Matrix Alignment
- The `PsycholinguisticTransitionMatrix` and `ReadingTransitionMatrix` must be sized to the number of words in the **layout**, not the number of fixations.

## 📋 Next Session TODOs
1.  **Locate Raw Data**: Confirm the absolute path to `L1ReadingData.xlsx` and `L2ReadingData.xlsx` (likely in `D:/projects/lexigaze/data/geco/`).
2.  **Verify Layout Info**: Check if these files contain a "Master Word List" or if we need to reconstruct the sentence from the `WORD` column while including skipped IDs.
3.  **Refactor Extraction**: Modify `batch_extract_features.py` to output a `layout.csv` (all words) and a `fixations.csv` (mapping gaze to layout indices).
4.  **Re-run Pipeline**: Execute the ablation study with the new realistic setup. Expect accuracy to drop to a scientifically meaningful range (e.g., 70-85%).

---
*Date: 2026-05-04*
*Status: Root cause identified. Remediation plan drafted.*
