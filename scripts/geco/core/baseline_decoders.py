import numpy as np
from scipy.stats import multivariate_normal

class NearestBoundingBoxDecoder:
    """Baseline 1: Pure Spatial Heuristic (Geometric Closeness)."""
    def decode(self, gaze_sequence, word_boxes, base_cm=None):
        word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
        preds = []
        for gaze in gaze_sequence:
            # Handle NaN (blinks/missing data)
            if np.isnan(gaze).any():
                preds.append(preds[-1] if preds else 0)
                continue
            dists = np.sum((word_centers - gaze)**2, axis=1)
            preds.append(np.argmin(dists))
        return preds

class StandardKalmanDecoder:
    """Baseline 2: Temporal smoothing using Kalman Filter, then nearest box."""
    def decode(self, gaze_sequence, word_boxes, base_cm=None):
        word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
        
        # Simple Constant Velocity Kalman Filter
        dt = 1.0
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = np.eye(4) * 0.1
        R = np.eye(2) * 10.0
        P = np.eye(4) * 100.0
        x = np.array([gaze_sequence[0][0], gaze_sequence[0][1], 0, 0])
        
        smoothed_gaze = []
        for gaze in gaze_sequence:
            if np.isnan(gaze).any():
                # Predict only
                x = F @ x
                P = F @ P @ F.T + Q
            else:
                # Predict
                x_pre = F @ x
                P_pre = F @ P @ F.T + Q
                # Update
                y = gaze - H @ x_pre
                S = H @ P_pre @ H.T + R
                K = P_pre @ H.T @ np.linalg.inv(S)
                x = x_pre + K @ y
                P = (np.eye(4) - K @ H) @ P_pre
            
            smoothed_gaze.append(H @ x)
            
        preds = []
        for gaze in smoothed_gaze:
            dists = np.sum((word_centers - gaze)**2, axis=1)
            preds.append(np.argmin(dists))
        return preds

class StaticBayesianDecoder:
    """Baseline 3: Cognitive Mass Prior (Point-by-point, no temporal sequence)."""
    def __init__(self, sigma_x=40, sigma_y=30):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def decode(self, gaze_sequence, word_boxes, base_cm):
        word_centers = np.array([[ (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 ] for box in word_boxes])
        cov = [[self.sigma_x**2, 0], [0, self.sigma_y**2]]
        
        preds = []
        for gaze in gaze_sequence:
            if np.isnan(gaze).any():
                preds.append(preds[-1] if preds else 0)
                continue
                
            posteriors = []
            for i in range(len(word_boxes)):
                like = multivariate_normal.pdf(gaze, mean=word_centers[i], cov=cov)
                posteriors.append(like * base_cm[i])
            
            preds.append(np.argmax(posteriors))
        return preds
