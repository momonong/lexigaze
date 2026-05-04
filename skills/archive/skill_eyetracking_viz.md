# Skill 25: Eye-Tracking Visualizations (NeurIPS Standard)

# Role & Task
You are an expert Data Visualization Engineer specializing in eye-tracking and psycholinguistic research for top-tier conferences (e.g., NeurIPS). 
Your task is to write a highly polished Python script using `matplotlib` to generate Figure 5 ("Scanpath Trajectory & Line-Locking Correction") for our paper on the STOCK-T algorithm.

# Visual Architecture & Requirements
We need to visualize how our algorithm corrects severe vertical hardware drift (+45px) during reading, snapping the gaze back to the correct words using "Cognitive Mass" (Semantic Gravity).

## 1. Canvas Setup
- `figsize=(8, 3.5)` or similar wide aspect ratio.
- Strictly NO axes, NO ticks, NO gridlines, NO borders (spines disabled). Clean white background.

## 2. Text and Bounding Boxes (The Stimulus)
- Simulate two lines of text. 
  - Line 1: "The unexpected consequence of the global phenomenon..."
  - Line 2: "was felt across multiple technological sectors."
- Draw precise, faint dashed gray bounding boxes (`edgecolor='lightgray'`, `linestyle='--'`) around each word.
- **Cognitive Mass Highlight:** Fill the background of complex words (e.g., "unexpected", "consequence", "phenomenon") with a very faint orange/yellow (`alpha=0.15`) to represent higher cognitive mass/surprisal.

## 3. Raw Gaze Trajectory (The Problem)
- Represents uncalibrated webcam drift.
- Style: Red, dashed line (`color='#E63946'`, `linestyle='--'`, `alpha=0.6`).
- Markers: Small red crosses (`marker='x'`).
- Path: The trajectory should start near the text of Line 1, but gradually drift downwards (y-axis) as it moves right, eventually overlapping entirely with the physical space of Line 2 (demonstrating the "Line-Locking" failure).

## 4. STOCK-T Corrected Trajectory (The Solution)
- Represents our algorithm's output.
- Style: Emerald Green or Royal Blue solid thick line (`color='#2A9D8F'` or `#1D3557'`, `linewidth=2`, `alpha=0.9`).
- Markers: Circles (`marker='o'`). The scatter size (`s`) should vary slightly to represent fixation duration (e.g., larger on the word "phenomenon").
- Path: Snapped perfectly to the horizontal center (y-axis) of the bounding boxes in Line 1.

## 5. Correction Vectors (The Mechanism)
- Draw thin, dotted gray arrows (`annotate` with `arrowprops`) pointing from each Raw Gaze marker to its corresponding Corrected Gaze marker. This explicitly shows the algorithm "pulling" the gaze back to the correct semantic target.

## 6. Legend & Polish
- Add a clean, minimalist legend at the top right or bottom right: "Raw Gaze (Hardware Drift)", "STOCK-T Corrected", "Correction Vector".
- Use a high-quality serif or sans-serif font standard for LaTeX papers (e.g., Times New Roman or DejaVu Sans).

# Instructions for the AI
1. Create mock data (x, y coordinates for word centers, raw gaze points, and corrected gaze points) that logically fits the narrative.
2. Output ONLY the complete, executable Python code.
3. Save the output as a high-resolution PDF (`fig_scanpath_correction.pdf`) with `bbox_inches='tight'`.