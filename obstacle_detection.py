import cv2
import numpy as np
import tensorflow as tf

class ObstacleDetector:
    def __init__(self, model_path='models/yolov4-tiny.tflite'):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.load_model()
        
        # COCO class names (subset relevant for indoor navigation)
        self.class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                           'bus', 'train', 'truck', 'boat', 'traffic light',
                           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                           'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                           'kite', 'baseball bat', 'baseball glove', 'skateboard',
                           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                           'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
                           'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                           'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                           'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        # Priority obstacles (higher priority = more urgent warning)
        self.obstacle_priority = {
            'person': 3,
            'chair': 2,
            'sofa': 2,
            'diningtable': 2,
            'bed': 2,
            'pottedplant': 1,
            'bottle': 1
        }
    
    def load_model(self):
        """Load TFLite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            print("Obstacle detection will use mock data for demo")
            self.interpreter = None
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles in frame
        Returns: list of (class_name, confidence, bbox, distance_estimate)
        """
        if self.interpreter is None:
            # Mock detection for demo if model not available
            return self._mock_detection(frame)
        
        # Preprocess frame
        input_shape = self.input_details[0]['shape']
        input_size = (input_shape[1], input_shape[2])
        
        resized = cv2.resize(frame, input_size)
        input_data = np.expand_dims(resized, axis=0)
        input_data = (input_data.astype(np.float32) - 127.5) / 127.5
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        obstacles = []
        height, width = frame.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                class_id = int(classes[i])
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    
                    # Convert normalized coordinates to pixel coordinates
                    ymin, xmin, ymax, xmax = boxes[i]
                    bbox = (int(xmin * width), int(ymin * height),
                           int(xmax * width), int(ymax * height))
                    
                    # Estimate distance based on bounding box size
                    box_area = (xmax - xmin) * (ymax - ymin)
                    distance = self._estimate_distance(box_area)
                    
                    obstacles.append({
                        'class': class_name,
                        'confidence': float(scores[i]),
                        'bbox': bbox,
                        'distance': distance,
                        'priority': self.obstacle_priority.get(class_name, 1)
                    })
        
        # Sort by priority then distance
        obstacles.sort(key=lambda x: (-x['priority'], x['distance']))
        return obstacles
    
    def _estimate_distance(self, box_area):
        """
        Rough distance estimation based on bounding box size
        Returns distance in meters (0.5 - 3.0 m)
        """
        # Larger box = closer object
        if box_area > 0.3:
            return 0.5
        elif box_area > 0.15:
            return 1.0
        elif box_area > 0.05:
            return 1.5
        else:
            return 2.5
    
    def _mock_detection(self, frame):
        """Mock obstacle detection for demo (when model unavailable)"""
        # Simulate random obstacle detection for demo purposes
        import random
        if random.random() < 0.3:  # 30% chance of "detecting" obstacle
            height, width = frame.shape[:2]
            return [{
                'class': random.choice(['chair', 'person', 'table']),
                'confidence': 0.85,
                'bbox': (width//3, height//3, 2*width//3, 2*height//3),
                'distance': random.uniform(1.0, 2.5),
                'priority': 2
            }]
        return []
    
    def draw_detections(self, frame, obstacles):
        """Draw bounding boxes and labels on frame"""
        for obs in obstacles:
            x1, y1, x2, y2 = obs['bbox']
            
            # Color based on distance (red = close, yellow = medium, green = far)
            if obs['distance'] < 1.0:
                color = (0, 0, 255)  # Red
            elif obs['distance'] < 2.0:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{obs['class']}: {obs['distance']:.1f}m"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
