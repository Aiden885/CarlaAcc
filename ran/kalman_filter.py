import numpy as np
from scipy.spatial import distance

class KalmanFilter:
    def __init__(self, dt=0.1):  # dt = 0.1 seconds for 10 Hz
        # State vector [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        # State transition matrix (constant velocity model with dt)
        self.F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        # Measurement matrix (observing x, y, vx, vy)
        self.H = np.array([[1, 0, 0, 0],  # x
                          [0, 1, 0, 0],  # y
                          [0, 0, 1, 0],  # vx
                          [0, 0, 0, 1]]) # vy
        # Process noise covariance
        # self.Q = np.eye(4) * 0.1
        self.Q = np.array([[(dt ** 4) / 4, 0, (dt ** 3) / 2, 0],
                          [0, (dt ** 4) / 4, 0, (dt ** 3) / 2],
                          [(dt ** 3) / 2, 0, dt ** 2, 0],
                          [0, (dt ** 3) / 2, 0, dt ** 2]])
        # Measurement noise covariance (x, y, vx, vy)
        self.R = np.eye(4) * 5
        # Error covariance
        self.P = np.eye(4)

    def predict(self):
        # Predict next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x.flatten()  # Return x, y, vx, vy

    def update(self, z):
        # z is [x, y, vx, vy], reshape to column vector
        z = z.reshape(4, 1)
        # Measurement update
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x.flatten()  # Return x, y, vx, vy

class RadarTracker:
    def __init__(self):
        self.trackers = {}  # Dictionary to store active trackers
        self.next_id = 0    # Next available tracking ID
        self.frame_counts = {}  # Count of frames per track
        self.miss_counts = {}   # Count of consecutive missed frames
        self.max_distance =  4 # Maximum distance for matching
        self.max_misses = 3     # Keep tracking for 3 frames after disappearance
        self.attributes = {}    # Store z, w, l, h, vz for each track

    def update(self, radar_points):
        """
        radar_points: numpy array of shape (n, 9) containing [x, y, z, w, l, h, vx, vy, vz]
        Returns: list of tuples (x, y, z, w, l, h, vx, vy, vz, track_id)
        """
        radar_points = np.array(radar_points)
        if radar_points is None:
            return 
        # Predict positions for existing trackers
        predictions = {}
        for track_id, kf in self.trackers.items():
            pred = kf.predict()
            predictions[track_id] = pred

        # Calculate distances between predictions and new measurements (using x, y only)
        matched = set()
        assignments = []
        
        if len(predictions) > 0 and len(radar_points) > 0:
            pred_points = np.array([p[:2] for p in predictions.values()])  # Use x, y
            meas_points = radar_points[:, :2]  # Use x, y
            dist_matrix = distance.cdist(pred_points, meas_points)
            
            for pred_idx, pred_id in enumerate(predictions.keys()):
                if pred_idx >= dist_matrix.shape[0]:
                    continue
                distances = dist_matrix[pred_idx]
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]
                
                if min_dist < self.max_distance and min_dist_idx not in matched:
                    assignments.append((pred_id, min_dist_idx))
                    matched.add(min_dist_idx)

        # Update matched trackers and reset miss counts
        tracked_points = []
        matched_ids = set()
        for track_id, point_idx in assignments:
            # Use only x, y, vx, vy for Kalman update
            z = radar_points[point_idx][[0, 1, 6, 7]]  # [x, y, vx, vy]
            pred = self.trackers[track_id].update(z)
            self.frame_counts[track_id] += 1
            self.miss_counts[track_id] = 0  # Reset miss count
            # Store z, w, l, h, vz
            self.attributes[track_id] = radar_points[point_idx][2:6].tolist() + [radar_points[point_idx][8]]
            matched_ids.add(track_id)
            
            if self.frame_counts[track_id] >= 3:
                x, y, vx, vy = pred
                z, w, l, h, vz = self.attributes[track_id]
                tracked_points.append((x, -y, z, w, l, h, vx, vy, vz, track_id))

        tmp_tracker = self.trackers.copy()
        # Handle unmatched trackers (predict only, up to 3 frames)
        for track_id in tmp_tracker:
            if track_id not in matched_ids:
                self.miss_counts[track_id] += 1
                if self.miss_counts[track_id] <= self.max_misses:
                    pred = predictions[track_id]  # Use predicted position
                    if self.frame_counts[track_id] >= 3:
                        x, y, vx, vy = pred
                        z, w, l, h, vz = self.attributes[track_id]
                        tracked_points.append((x, -y, z, w, l, h, vx, vy, vz, track_id))
                else:
                    # Remove tracker after max_misses exceeded
                    del self.trackers[track_id]
                    del self.frame_counts[track_id]
                    del self.miss_counts[track_id]
                    del self.attributes[track_id]

        # Create new trackers for unmatched points
        for i, point in enumerate(radar_points):
            if i not in matched:
                kf = KalmanFilter(dt=0.1)
                kf.x = point[[0, 1, 6, 7]].reshape(4, 1)  # Initialize x, y, vx, vy
                new_id = self.next_id
                self.trackers[new_id] = kf
                self.frame_counts[new_id] = 1
                self.miss_counts[new_id] = 0
                self.attributes[new_id] = point[2:6].tolist() + [point[8]]  # Store z, w, l, h, vz
                self.next_id += 1

        # Do not remove trackers immediately to ensure persistence
        return tracked_points

# Test data (updated for [x, y, z, w, l, h, vx, vy, vz])
test_frames = [
    # Frame 0: Both objects visible
    [[100.0, 100, 50, 2, 3, 1, 5, 0, 0], [200.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 1: Both visible
    [[100.5, 100, 50, 2, 3, 1, 5, 0, 0], [201.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 2: Both visible
    [[101.0, 100, 50, 2, 3, 1, 5, 0, 0], [202.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 3: Both visible
    [[101.5, 100, 50, 2, 3, 1, 5, 0, 0], [203.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 4: Object 1 disappears
    [[204.0, 200, 100, 4, 5, 2, 10, 0, 0]],  # Only Object 2
    # Frame 5: Object 1 still gone
    [[205.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 6: Object 1 still gone
    [[206.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 7: Object 1 still gone (should stop tracking Object 1)
    [[207.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 8: Only Object 2
    [[208.0, 200, 100, 4, 5, 2, 10, 0, 0]],
    # Frame 9: Only Object 2
    [[209.0, 200, 100, 4, 5, 2, 10, 0, 0]]
]

# Test the tracker
if __name__ == "__main__":
    tracker = RadarTracker()
    for i, frame in enumerate(test_frames):
        result = tracker.update(frame)
        print(f"Frame {i}: {result}")