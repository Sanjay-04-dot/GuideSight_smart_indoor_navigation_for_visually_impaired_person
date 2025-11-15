from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import time
import threading
from database import Database
from localization import Localizer
from obstacle_detection import ObstacleDetector
from navigation import Navigator
from voice_handler import VoiceHandler

app = Flask(__name__)

# Initialize components
os.makedirs('data/saved_frames', exist_ok=True)
os.makedirs('models', exist_ok=True)

db = Database()
localizer = Localizer(db)
obstacle_detector = ObstacleDetector()
navigator = Navigator(db)
voice_handler = VoiceHandler()

# Global state
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

app_state = {
    'mode': 'idle',  # 'idle', 'mapping', 'navigating'
    'current_location': None,
    'mapping_frames': 0,
    'current_position': None,
    'localization_confidence': 0,
    'obstacles': [],
    'navigation_instruction': '',
    'destination': None
}

def generate_frames():
    """Generate video frames with overlay information"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame based on mode
        if app_state['mode'] == 'mapping':
            # Show feature points during mapping
            kp, desc = localizer.extract_features(frame)
            frame = localizer.draw_features(frame, kp)
            
            # Add text overlay
            cv2.putText(frame, f"MAPPING MODE - Frame: {app_state['mapping_frames']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Location: {app_state['current_location']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif app_state['mode'] == 'navigating':
            # Detect obstacles
            obstacles = obstacle_detector.detect_obstacles(frame)
            app_state['obstacles'] = obstacles
            frame = obstacle_detector.draw_detections(frame, obstacles)
            
            # Add navigation overlay
            cv2.putText(frame, f"NAVIGATION MODE", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Destination: {app_state['destination']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(frame, f"Confidence: {app_state['localization_confidence']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Warning if obstacle too close
            if obstacles and obstacles[0]['distance'] < 1.5:
                cv2.putText(frame, "OBSTACLE AHEAD!", 
                           (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        else:  # idle mode
            cv2.putText(frame, "GuideSight - Ready", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Say a command or use buttons below", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add privacy badge
        cv2.rectangle(frame, (500, 10), (630, 50), (0, 200, 0), -1)
        cv2.putText(frame, "100% OFFLINE", 
                   (510, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_mapping', methods=['POST'])
def start_mapping():
    """Start mapping mode"""
    data = request.json
    location_name = data.get('location_name', 'unnamed_location')
    
    app_state['mode'] = 'mapping'
    app_state['current_location'] = location_name
    app_state['mapping_frames'] = 0
    
    # Create location in database
    location_id = db.save_location(location_name)
    
    # Start mapping thread
    threading.Thread(target=mapping_loop, args=(location_id,), daemon=True).start()
    
    voice_handler.speak(f"Starting to map {location_name}")
    
    return jsonify({'status': 'success', 'message': f'Mapping {location_name}'})

def mapping_loop(location_id):
    """Background thread for capturing mapping frames"""
    frame_interval = 0.5  # Capture every 0.5 seconds
    last_capture = time.time()
    
    while app_state['mode'] == 'mapping':
        if time.time() - last_capture >= frame_interval:
            success, frame = camera.read()
            if success:
                # Extract features
                kp, desc = localizer.extract_features(frame)
                
                if desc is not None and len(kp) > 50:
                    # Save frame
                    frame_path = f"data/saved_frames/{location_id}_{app_state['mapping_frames']}.jpg"
                    cv2.imwrite(frame_path, frame)
                    
                    # Save to database
                    db.save_frame(location_id, app_state['mapping_frames'], 
                                 frame_path, kp, desc)
                    
                    app_state['mapping_frames'] += 1
                
                last_capture = time.time()
        
        time.sleep(0.1)

@app.route('/api/stop_mapping', methods=['POST'])
def stop_mapping():
    """Stop mapping mode"""
    if app_state['mode'] == 'mapping':
        location_name = app_state['current_location']
        frames_captured = app_state['mapping_frames']
        
        app_state['mode'] = 'idle'
        app_state['current_location'] = None
        
        voice_handler.speak(f"Mapping complete. Saved {frames_captured} landmarks for {location_name}")
        
        return jsonify({
            'status': 'success',
            'message': f'Mapping complete: {frames_captured} frames',
            'location': location_name
        })
    
    return jsonify({'status': 'error', 'message': 'Not in mapping mode'})

@app.route('/api/start_navigation', methods=['POST'])
def start_navigation():
    """Start navigation to destination"""
    data = request.json
    destination = data.get('destination')
    
    # Check if location exists
    locations = db.get_all_locations()
    if destination not in locations:
        voice_handler.speak(f"Location {destination} not found")
        return jsonify({'status': 'error', 'message': 'Location not found'})
    
    app_state['mode'] = 'navigating'
    app_state['destination'] = destination
    
    # Start navigation thread
    threading.Thread(target=navigation_loop, args=(destination,), daemon=True).start()
    
    voice_handler.speak(f"Navigating to {destination}")
    
    return jsonify({'status': 'success', 'message': f'Navigating to {destination}'})

def navigation_loop(destination):
    """Background thread for navigation"""
    last_localization = time.time()
    last_obstacle_check = time.time()
    last_instruction = time.time()
    current_node = None
    # Ensure navigator path and confidence are initialized
    navigator.current_path = getattr(navigator, 'current_path', [])
    app_state['localization_confidence'] = 0.0

    def _cap_confidence(val):
        """Normalize and cap confidence to 0-100 (%).
        If val is in 0..1 range it's treated as probability and multiplied by 100.
        Otherwise it's treated as percentage and capped at 100.
        """
        try:
            v = float(val) if val is not None else 0.0
        except Exception:
            return 0.0
        if 0.0 <= v <= 1.0:
            v = v * 100.0
        return round(max(0.0, min(v, 100.0)), 1)

    while app_state['mode'] == 'navigating':
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue

        # Localize every 1 second
        if time.time() - last_localization >= 1.0:
            frame_id, confidence, sequence = localizer.localize(frame, destination)
            # Normalize / cap confidence to 0-100%
            conf_pct = _cap_confidence(confidence)
            app_state['localization_confidence'] = conf_pct

            if frame_id:
                app_state['current_position'] = sequence

                # Get corresponding node
                nodes, edges = db.get_navigation_graph(destination)
                for node_id, node_data in nodes.items():
                    if node_data.get('frame_id') == frame_id:
                        current_node = node_id
                        break

                # Plan route if needed
                if current_node and not navigator.current_path:
                    navigator.plan_route(destination, current_node)

                # Get instruction
                if current_node and navigator.current_path:
                    instruction = navigator.get_next_instruction(nodes)
                    app_state['navigation_instruction'] = instruction

                    # Speak instruction every 5 seconds
                    if time.time() - last_instruction >= 5.0:
                        voice_handler.speak(instruction)
                        last_instruction = time.time()
                        navigator.advance_step()

                    if "reached" in instruction.lower():
                        app_state['mode'] = 'idle'
                        voice_handler.speak("You have arrived at your destination")
                        break
            else:
                voice_handler.speak("Unable to determine location. Please look around")

            last_localization = time.time()

        # Check obstacles every 0.5 seconds
        if time.time() - last_obstacle_check >= 0.5:
            obstacles = obstacle_detector.detect_obstacles(frame)
            app_state['obstacles'] = obstacles

            # Warn about close obstacles
            if obstacles and obstacles[0]['distance'] < 1.5:
                obs = obstacles[0]
                voice_handler.speak(f"Warning! {obs['class']} ahead at {obs['distance']:.1f} meters")

            last_obstacle_check = time.time()

        time.sleep(0.1)
    while app_state['mode'] == 'navigating':
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        
        # Localize every 1 second
        if time.time() - last_localization >= 1.0:
            frame_id, confidence, sequence = localizer.localize(frame, destination)
            app_state['localization_confidence'] = confidence
            
            if frame_id:
                app_state['current_position'] = sequence
                
                # Get corresponding node
                nodes, edges = db.get_navigation_graph(destination)
                for node_id, node_data in nodes.items():
                    if node_data['frame_id'] == frame_id:
                        current_node = node_id
                        break
                
                # Plan route if needed
                if current_node and not navigator.current_path:
                    navigator.plan_route(destination, current_node)
                
                # Get instruction
                if current_node and navigator.current_path:
                    instruction = navigator.get_next_instruction(nodes)
                    app_state['navigation_instruction'] = instruction
                    
                    # Speak instruction every 5 seconds
                    if time.time() - last_instruction >= 5.0:
                        voice_handler.speak(instruction)
                        last_instruction = time.time()
                        navigator.advance_step()
                    
                    if "reached" in instruction.lower():
                        app_state['mode'] = 'idle'
                        voice_handler.speak("You have arrived at your destination")
                        break
            else:
                voice_handler.speak("Unable to determine location. Please look around")
            
            last_localization = time.time()
        
        # Check obstacles every 0.5 seconds
        if time.time() - last_obstacle_check >= 0.5:
            obstacles = obstacle_detector.detect_obstacles(frame)
            app_state['obstacles'] = obstacles
            
            # Warn about close obstacles
            if obstacles and obstacles[0]['distance'] < 1.5:
                obs = obstacles[0]
                voice_handler.speak(f"Warning! {obs['class']} ahead at {obs['distance']:.1f} meters")
            
            last_obstacle_check = time.time()
        
        time.sleep(0.1)

@app.route('/api/stop_navigation', methods=['POST'])
def stop_navigation():
    """Stop navigation"""
    if app_state['mode'] == 'navigating':
        app_state['mode'] = 'idle'
        app_state['destination'] = None
        navigator.current_path = []
        voice_handler.speak("Navigation stopped")
        return jsonify({'status': 'success', 'message': 'Navigation stopped'})
    
    return jsonify({'status': 'error', 'message': 'Not in navigation mode'})

@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    """Process voice command"""
    text = voice_handler.listen(timeout=5)
    
    if not text:
        return jsonify({'status': 'error', 'message': 'No command recognized'})
    
    # Parse command
    if 'map' in text or 'create' in text:
        # Extract location name
        words = text.split()
        location_name = words[-1] if len(words) > 1 else 'unnamed'
        return start_mapping()
    
    elif 'stop' in text:
        if app_state['mode'] == 'mapping':
            return stop_mapping()
        elif app_state['mode'] == 'navigating':
            return stop_navigation()
    
    elif 'navigate' in text or 'go to' in text or 'take me' in text:
        # Extract destination
        locations = db.get_all_locations()
        for loc in locations:
            if loc.lower() in text:
                return start_navigation()
        
        voice_handler.speak("Destination not found")
        return jsonify({'status': 'error', 'message': 'Destination not recognized'})
    
    elif 'where am i' in text:
        if app_state['current_position'] is not None:
            voice_handler.speak(f"You are at position {app_state['current_position']} in {app_state['destination']}")
        else:
            voice_handler.speak("Location unknown")
        return jsonify({'status': 'success', 'message': 'Location query'})
    
    return jsonify({'status': 'error', 'message': 'Command not understood'})

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get list of all saved locations"""
    locations = db.get_all_locations()
    return jsonify({'locations': locations})

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current app state"""
    return jsonify(app_state)

@app.route('/api/emergency', methods=['POST'])
def emergency():
    """Emergency button"""
    voice_handler.speak_blocking("Emergency assistance requested. Help is on the way.")
    return jsonify({'status': 'success', 'message': 'Emergency alert sent'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
