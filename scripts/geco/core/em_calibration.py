import numpy as np
from .viterbi_decoder import viterbi_gaze_decode
from .dynamic_field import DynamicCognitiveField

class AutoCalibratingDecoder:
    """
    Skill 6: EM-based Dynamic Drift Auto-Calibration.
    Iteratively estimates systematic drift and re-decodes the sequence.
    """
    def __init__(self, calibration_window_size=40):
        self.window_size = calibration_window_size

    def calibrate_and_decode(self, raw_gaze_sequence, word_boxes, base_cm, transition_matrix, sigma_gaze=[40, 30], use_ovp=True):
        # Step 1: E-Step (Expectation)
        # Run Viterbi on an initial window to guess intended targets
        window = raw_gaze_sequence[:self.window_size]
        initial_indices = viterbi_gaze_decode(window, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp)
        
        # Step 2: M-Step (Maximization / Drift Estimation)
        # Skill 11 FIX: Access biologically aligned OVP centers for drift estimation
        if use_ovp:
            dfield = DynamicCognitiveField(word_boxes, base_cm, use_ovp=True)
            word_centers = dfield.word_centers # OVP Centers (35% width)
        else:
            word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
            
        predicted_centers = word_centers[initial_indices]
        
        # Calculate error vector
        errors = window - predicted_centers
        
        # Robust Mean Drift (Median)
        drift_x = np.nanmedian(errors[:, 0])
        drift_y = np.nanmedian(errors[:, 1])
        
        # Handle case where user isn't reading (high variance)
        if np.nanstd(errors) > 150:
            print("⚠️ High variance detected in initial window. Skipping auto-calibration.")
            drift_x, drift_y = 0, 0
            
        print(f"🔧 Auto-Calibration: Detected Drift ({drift_x:.1f}px, {drift_y:.1f}px)")
        
        # Step 3: Update & Final Decode
        corrected_gaze = raw_gaze_sequence - np.array([drift_x, drift_y])
        final_indices = viterbi_gaze_decode(corrected_gaze, word_boxes, base_cm, transition_matrix, sigma_gaze, use_ovp=use_ovp)
        
        return final_indices, (drift_x, drift_y)
