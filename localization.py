import cv2
import numpy as np
from database import Database

class Localizer:
    def __init__(self, database):
        self.db = database
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.confidence_threshold = 30  # Minimum good matches
    
    def extract_features(self, frame):
        """Extract ORB keypoints and descriptors from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def localize(self, current_frame, location_name):
        """
        Determine current position within saved location
        Returns: (matched_frame_id, confidence, matched_sequence)
        """
        current_kp, current_desc = self.extract_features(current_frame)
        
        if current_desc is None or len(current_kp) < 10:
            return None, 0, None
        
        # Get all saved frames for this location
        saved_frames = self.db.get_location_frames(location_name)
        
        if not saved_frames:
            return None, 0, None
        
        best_match = None
        best_match_count = 0
        best_sequence = 0
        
        # Compare against each saved frame
        for frame_data in saved_frames:
            saved_desc = frame_data['descriptors']
            
            if saved_desc is None:
                continue
            
            # Match features using Brute Force Matcher
            matches = self.matcher.match(current_desc, saved_desc)
            
            # Filter good matches (distance < threshold)
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_match = frame_data['id']
                best_sequence = frame_data['sequence']
        
        # Check confidence
        confidence = best_match_count
        if confidence >= self.confidence_threshold:
            return best_match, confidence, best_sequence
        else:
            return None, confidence, None
    
    def draw_features(self, frame, keypoints):
        """Draw detected keypoints on frame for visualization"""
        return cv2.drawKeypoints(frame, keypoints, None, 
                                color=(0, 255, 0), 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
