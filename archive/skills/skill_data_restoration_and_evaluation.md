# Skill: Data Restoration and Pipeline Re-Execution

## Description
This skill guides the user through the process of restoring corrupted GECO datasets and executing the newly refactored, scientifically valid evaluation pipeline for the NeurIPS submission. The new pipeline utilizes the full text layout to provide realistic accuracy metrics.

## Prerequisites
- The raw `L1ReadingData.xlsx` and `L2ReadingData.xlsx` files must be fully downloaded and placed in the `data/geco/` directory.
- Confirm that the file sizes are > 100MB to ensure they are not truncated.

## Workflow

### Step 1: Verify Data Integrity
Check the file size of the raw datasets to ensure they are completely downloaded and not corrupted.
```powershell
Get-Item data/geco/L1ReadingData.xlsx | Select Length
Get-Item data/geco/L2ReadingData.xlsx | Select Length
```
*(If sizes are around 37MB/26MB, they are truncated and must be redownloaded).*

### Step 2: Batch Extract Features (Full Layout)
Run the updated extraction script. This will generate `layout.csv` (containing the complete sequence of words with interpolated coordinates) and `fixations.csv` (mapping gaze to layout indices) for all trials, creating the proper state space for evaluation.
```bash
python scripts/geco/data/batch_extract_features.py
```

### Step 3: Run Single Trial Final Evaluation
Test the pipeline on the standard pp01 trial to ensure the dual-metric evaluation works with the new full-layout format.
```bash
python scripts/geco/evaluate_neurips_final.py
```

### Step 4: Execute Full Population Pipeline
Run the full NeurIPS ablation pipeline to process all subjects and generate the final manuscript-ready LaTeX table.
```bash
python scripts/geco/run_full_neurips_pipeline.py
```

## Expected Outcome
The pipeline should output a scientifically valid accuracy range (e.g., 70-85% for L2, 90%+ for L1) reflecting true model performance across the complete text layout. The final results will be saved to `data/geco/benchmark/neurips_final_ablation_N37.csv`, ready for insertion into the NeurIPS paper.