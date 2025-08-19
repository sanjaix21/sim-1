import asyncio
import json
import websockets
from flask import Flask, request, jsonify
import threading
import time
import base64
from collections import deque
import requests
import numpy as np
import math

try:
    import cv2
except ImportError:
    cv2 = None
    print("WARNING: OpenCV not installed. Install with: pip install opencv-python")

app = Flask(__name__)

# --- CORS ---
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

# ---------------------------
# Globals
# ---------------------------
connected = set()
async_loop = None
collision_count = 0

FLOOR_HALF = 50

def corner_to_coords(corner: str, margin=5):
    c = corner.upper()
    if c in ("NE", "EN", "TR"): 
        return {"x": FLOOR_HALF - margin, "y": 0, "z": -(FLOOR_HALF - margin)}
    elif c in ("NW", "WN", "TL"): 
        return {"x": -(FLOOR_HALF - margin), "y": 0, "z": -(FLOOR_HALF - margin)}
    elif c in ("SE", "ES", "BR"): 
        return {"x": FLOOR_HALF - margin, "y": 0, "z": FLOOR_HALF - margin}
    elif c in ("SW", "WS", "BL"): 
        return {"x": -(FLOOR_HALF - margin), "y": 0, "z": FLOOR_HALF - margin}
    return {"x": 0, "y": 0, "z": 0}

# Shared state
agent_state = {
    "last_image_b64": None,
    "goal_reached": False,
    "robot_position": {"x": 0, "y": 0, "z": 0}
}
image_queue = deque(maxlen=1)

# ---------------------------
# Enhanced Computer Vision
# ---------------------------
class RobotVision:
    def __init__(self):
        self.obstacle_color_range = {
            'lower': np.array([35, 50, 50]),
            'upper': np.array([85, 255, 255])
        }
        self.goal_color_range = {
            'lower': np.array([85, 50, 50]),
            'upper': np.array([130, 255, 255])
        }
    
    def analyze_frame(self, frame_b64: str) -> dict:
        if cv2 is None:
            return {"obstacles": {"center": {"blocked": False}}, "goal": {"visible": False}, "clear_path": {"best_direction": 0}}
        
        try:
            # Fix: Remove data URL prefix if present
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',', 1)[1]
            
            img = base64.b64decode(frame_b64)
            arr = np.frombuffer(img, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Image decode error: {e}")
            return {"obstacles": {"center": {"blocked": False}}, "goal": {"visible": False}, "clear_path": {"best_direction": 0}}
        
        if frame is None:
            return {"obstacles": {"center": {"blocked": False}}, "goal": {"visible": False}, "clear_path": {"best_direction": 0}}
        
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        analysis = {
            'obstacles': self._detect_obstacles(hsv, h, w),
            'goal': self._detect_goal(hsv, h, w),
            'clear_path': self._find_clear_path(hsv, h, w)
        }
        
        return analysis
    
    def _detect_obstacles(self, hsv_frame, h, w) -> dict:
        mask = cv2.inRange(hsv_frame, self.obstacle_color_range['lower'], 
                          self.obstacle_color_range['upper'])
        
        regions = {
            'left': mask[int(h*0.4):int(h*0.8), 0:int(w*0.4)],
            'center': mask[int(h*0.4):int(h*0.8), int(w*0.3):int(w*0.7)],
            'right': mask[int(h*0.4):int(h*0.8), int(w*0.6):w],
            'near': mask[int(h*0.7):h, int(w*0.2):int(w*0.8)]
        }
        
        obstacles = {}
        for region, mask_region in regions.items():
            if mask_region.size > 0:
                obstacle_ratio = (mask_region > 0).mean()
                obstacles[region] = {
                    'blocked': obstacle_ratio > 0.03,
                    'density': obstacle_ratio
                }
            else:
                obstacles[region] = {'blocked': False, 'density': 0}
        
        return obstacles
    
    def _detect_goal(self, hsv_frame, h, w) -> dict:
        mask = cv2.inRange(hsv_frame, self.goal_color_range['lower'], 
                          self.goal_color_range['upper'])
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        goal_info = {
            'visible': False,
            'direction': 0,
            'distance': 'unknown',
            'size': 0
        }
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 50:
                goal_info['visible'] = True
                goal_info['size'] = area
                
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    center_x = w // 2
                    
                    if cx < center_x - w * 0.15:
                        goal_info['direction'] = -1
                    elif cx > center_x + w * 0.15:
                        goal_info['direction'] = 1
                    else:
                        goal_info['direction'] = 0
                
                if area > 3000:
                    goal_info['distance'] = 'close'
                elif area > 800:
                    goal_info['distance'] = 'medium'
                else:
                    goal_info['distance'] = 'far'
        
        return goal_info
    
    def _find_clear_path(self, hsv_frame, h, w) -> dict:
        obstacle_mask = cv2.inRange(hsv_frame, self.obstacle_color_range['lower'], 
                                   self.obstacle_color_range['upper'])
        
        sectors = {}
        sector_angles = [-45, -22, 0, 22, 45]
        
        for i, angle in enumerate(sector_angles):
            if angle == -45:
                sector_mask = obstacle_mask[int(h*0.4):int(h*0.8), 0:int(w*0.3)]
            elif angle == -22:
                sector_mask = obstacle_mask[int(h*0.4):int(h*0.8), int(w*0.15):int(w*0.45)]
            elif angle == 0:
                sector_mask = obstacle_mask[int(h*0.4):int(h*0.8), int(w*0.35):int(w*0.65)]
            elif angle == 22:
                sector_mask = obstacle_mask[int(h*0.4):int(h*0.8), int(w*0.55):int(w*0.85)]
            else:
                sector_mask = obstacle_mask[int(h*0.4):int(h*0.8), int(w*0.7):w]
            
            if sector_mask.size > 0:
                clearness = 1.0 - (sector_mask > 0).mean()
            else:
                clearness = 1.0
            sectors[angle] = clearness
        
        best_direction = max(sectors, key=sectors.get)
        
        return {
            'sectors': sectors,
            'best_direction': best_direction,
            'clearness': sectors[best_direction]
        }

# ---------------------------
# Smart Navigator (FIXED)
# ---------------------------
class SmartNavigator:
    def __init__(self, host="http://localhost:5000"):
        self.host = host
        self.vision = RobotVision()
        self.running = False
        self.thread = None
        self.goal_position = None
        self.last_positions = deque(maxlen=5)  # Track recent positions
        
    def start(self, corner="NE"):
        if self.running:
            return
        self.running = True
        agent_state["goal_reached"] = False
        
        try:
            response = requests.post(f"{self.host}/goal", json={"corner": corner}, timeout=5)
            self.goal_position = corner_to_coords(corner)
            print(f"Goal set at {corner}: {self.goal_position}")
        except Exception as e:
            print(f"Failed to set goal: {e}")
            return
        
        self.thread = threading.Thread(target=self._navigate, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        try:
            requests.post(f"{self.host}/stop", timeout=2)
        except:
            pass
    
    def _navigate(self):
        stuck_counter = 0
        step_count = 0
        exploration_mode = False
        exploration_steps = 0
        last_action_time = 0
        
        while self.running and not agent_state["goal_reached"] and step_count < 300:
            step_count += 1
            
            # Get current robot position from simulator
            current_pos = agent_state["robot_position"]
            self.last_positions.append((current_pos['x'], current_pos['z']))
            
            # Check if stuck (same position for multiple steps)
            if len(self.last_positions) >= 3:  # Reduced from 5 to detect stuck faster
                recent_positions = list(self.last_positions)[-3:]
                if self._positions_clustered(recent_positions, threshold=0.5):  # Reduced threshold
                    stuck_counter += 1
                    if stuck_counter > 1:  # Reduced from 2 to react faster
                        print(f"Robot stuck at {current_pos}, entering exploration mode")
                        exploration_mode = True
                        exploration_steps = 5 + int(np.random.randint(3))  # Shorter exploration
                        stuck_counter = 0
                        # Force a different action when stuck
                        action = {'type': 'turn', 'angle': int(np.random.choice([60, -60, 90, -90]))}
                        self._execute_turn(action['angle'])
                        time.sleep(0.8)
                        continue
                else:
                    stuck_counter = 0
                    if not exploration_mode:
                        exploration_steps = 0

            # Capture image with better error handling
            try:
                response = requests.post(f"{self.host}/capture", timeout=5)
                if response.status_code != 200:
                    print(f"Capture request failed: {response.status_code}")
                    continue
            except Exception as e:
                print(f"Capture failed: {e}")
                continue
            
            # Wait for image with longer timeout
            frame_b64 = self._wait_for_image(timeout=2.0)
            if frame_b64 is None:
                print("No image received, using fallback navigation")
                # Fallback: simple obstacle avoidance without vision
                action = self._fallback_action()
            else:
                # Analyze frame
                try:
                    analysis = self.vision.analyze_frame(frame_b64)
                except Exception as e:
                    print(f"Vision analysis failed: {e}")
                    action = self._fallback_action()
                    continue
                
                # Make decision based on mode
                if exploration_mode and exploration_steps > 0:
                    action = self._exploration_action(analysis)
                    exploration_steps -= 1
                    if exploration_steps <= 0:
                        exploration_mode = False
                        print("Exploration complete, resuming goal-directed navigation")
                else:
                    action = self._goal_directed_action(analysis, current_pos)
            
            # Ensure action has minimum distance to actually move the robot
            if action['type'] == 'move' and action.get('distance', 0) < 1.0:
                action['distance'] = 1.5
            elif action['type'] == 'turn_and_move' and action.get('distance', 0) < 1.0:
                action['distance'] = 1.5
                
            print(f"Step {step_count}: Executing {action} at position {current_pos}")
            
            # Execute action with better error handling and timing
            current_time = time.time()
            if current_time - last_action_time < 0.8:  # Ensure minimum time between actions
                time.sleep(0.8 - (current_time - last_action_time))
                
            try:
                if action['type'] == 'turn':
                    self._execute_turn(action['angle'])
                    time.sleep(0.8)
                elif action['type'] == 'move':
                    self._execute_move(action['distance'])
                    time.sleep(1.0)  # Longer wait for move commands
                elif action['type'] == 'turn_and_move':
                    self._execute_turn(action['angle'])
                    time.sleep(0.6)
                    self._execute_move(action['distance'])
                    time.sleep(1.0)
                last_action_time = time.time()
            except Exception as e:
                print(f"Action execution failed: {e}")
            
            time.sleep(0.2)  # Small delay between steps
        
        print(f"Navigation ended. Steps: {step_count}, Goal reached: {agent_state['goal_reached']}")
    
    def _fallback_action(self) -> dict:
        """Simple fallback when vision is not available"""
        return {'type': 'turn_and_move', 'angle': 45, 'distance': 2.0}  # Increased distance
    
    def _goal_directed_action(self, analysis, current_pos) -> dict:
        """Goal-directed navigation with improved logic"""
        obstacles = analysis['obstacles']
        goal = analysis['goal']
        clear_path = analysis['clear_path']
        
        # Priority 1: Goal visible and close
        if goal['visible'] and goal['distance'] == 'close':
            if not obstacles['center']['blocked']:
                return {'type': 'move', 'distance': 1.5}
            elif goal['direction'] != 0:
                angle = 15 if goal['direction'] > 0 else -15
                return {'type': 'turn', 'angle': angle}
        
        # Priority 2: Goal visible - align and approach
        if goal['visible']:
            if goal['direction'] == 0:
                if not obstacles['center']['blocked']:
                    distance = 3.0 if goal['distance'] == 'far' else 2.0
                    return {'type': 'move', 'distance': distance}
                else:
                    # Obstacle in center, try to go around
                    if not obstacles['right']['blocked']:
                        return {'type': 'turn', 'angle': 30}
                    elif not obstacles['left']['blocked']:
                        return {'type': 'turn', 'angle': -30}
                    else:
                        return {'type': 'turn', 'angle': 60}
            else:
                # Turn toward goal
                angle = 25 if goal['direction'] > 0 else -25
                return {'type': 'turn', 'angle': angle}
        
        # Priority 3: Navigate toward goal using position
        if self.goal_position:
            dx = self.goal_position['x'] - current_pos['x']
            dz = self.goal_position['z'] - current_pos['z']
            
            # Calculate desired turn angle
            desired_angle = math.degrees(math.atan2(dx, dz))
            
            # Check if we need to turn significantly
            if abs(desired_angle) > 15:
                turn_angle = max(-45, min(45, desired_angle))
                if not obstacles['center']['blocked']:
                    return {'type': 'turn_and_move', 'angle': turn_angle, 'distance': 2.0}
                else:
                    return {'type': 'turn', 'angle': turn_angle}
        
        # Priority 4: Obstacle avoidance
        if obstacles['center']['blocked'] or obstacles['near']['blocked']:
            if clear_path['clearness'] > 0.7:
                if abs(clear_path['best_direction']) <= 30:
                    return {'type': 'turn', 'angle': clear_path['best_direction']}
            
            # Simple avoidance
            if not obstacles['right']['blocked']:
                return {'type': 'turn', 'angle': 35}
            elif not obstacles['left']['blocked']:
                return {'type': 'turn', 'angle': -35}
            else:
                return {'type': 'turn', 'angle': 90}
        
        # Priority 5: Move forward if path is clear
        return {'type': 'move', 'distance': 2.5}
    
    def _exploration_action(self, analysis) -> dict:
        """Exploration mode to get unstuck"""
        obstacles = analysis['obstacles']
        clear_path = analysis['clear_path']
        
        # Try to find a clear direction
        if clear_path['clearness'] > 0.8:
            angle = clear_path['best_direction']
            if abs(angle) > 20:
                return {'type': 'turn', 'angle': angle}
            else:
                return {'type': 'move', 'distance': 2.0}
        
        # Random exploration
        if not obstacles['right']['blocked']:
            return {'type': 'turn_and_move', 'angle': int(np.random.randint(20, 60)), 'distance': 2.0}
        elif not obstacles['left']['blocked']:
            return {'type': 'turn_and_move', 'angle': int(np.random.randint(-60, -20)), 'distance': 2.0}
        else:
            return {'type': 'turn', 'angle': int(np.random.choice([90, -90, 120, -120]))}
    
    def _wait_for_image(self, timeout=3.0):  # Increased default timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            if image_queue:
                return image_queue[-1]
            time.sleep(0.1)  # Slightly longer sleep
        return None
    
    def _execute_turn(self, angle):
        try:
            response = requests.post(f"{self.host}/move_rel", 
                         json={"turn": int(angle), "distance": 0}, timeout=5)
            if response.status_code != 200:
                print(f"Turn request failed: {response.status_code}")
        except Exception as e:
            print(f"Turn failed: {e}")
    
    def _execute_move(self, distance):
        try:
            response = requests.post(f"{self.host}/move_rel", 
                         json={"turn": 0, "distance": float(distance)}, timeout=5)
            if response.status_code != 200:
                print(f"Move request failed: {response.status_code}")
        except Exception as e:
            print(f"Move failed: {e}")
    
    def _positions_clustered(self, positions, threshold=0.5):  # Reduced threshold
        """Check if recent positions are clustered (indicating stuck)"""
        if len(positions) < 2:  # Reduced minimum positions needed
            return False
        
        x_coords = [p[0] for p in positions]
        z_coords = [p[1] for p in positions]
        
        x_range = max(x_coords) - min(x_coords)
        z_range = max(z_coords) - min(z_coords)
        
        return (x_range + z_range) < threshold

navigator = SmartNavigator()

# ---------------------------
# WebSocket Handler (FIXED)
# ---------------------------
async def ws_handler(websocket, path=None):
    global collision_count
    print("Client connected via WebSocket")
    connected.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if isinstance(data, dict):
                    t = data.get("type")
                    if t == "collision" and data.get("collision"):
                        collision_count += 1
                        print(f"Collision detected! Total: {collision_count}")
                    elif t == "capture_image_response" and data.get("image"):
                        img_b64 = data["image"]
                        # Fix: Handle data URL format properly
                        if img_b64.startswith("data:"):
                            img_b64 = img_b64.split(",", 1)[-1]
                        agent_state["last_image_b64"] = img_b64
                        image_queue.clear()
                        image_queue.append(img_b64)
                        print("Image received and queued")  # Debug log
                        if data.get("position"):
                            agent_state["robot_position"] = data["position"]
                    elif t == "goal_reached":
                        agent_state["goal_reached"] = True
                        print("🎉 GOAL REACHED!")
                        navigator.stop()  # Stop navigation when goal is reached
            except Exception as e:
                print(f"WebSocket message error: {e}")
                pass
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket handler error: {e}")
    finally:
        connected.discard(websocket)  # Use discard instead of remove to avoid KeyError

def broadcast(msg: dict):
    if not connected:
        return False
    disconnected = set()
    for ws in list(connected):
        try:
            if async_loop and not async_loop.is_closed():
                asyncio.run_coroutine_threadsafe(ws.send(json.dumps(msg)), async_loop)
        except Exception as e:
            print(f"Failed to send to websocket: {e}")
            disconnected.add(ws)
    
    # Clean up disconnected websockets
    for ws in disconnected:
        connected.discard(ws)
    
    return len(connected) > 0

# ---------------------------
# Flask Endpoints
# ---------------------------
@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    if not data or 'x' not in data or 'z' not in data:
        return jsonify({'error': 'Missing parameters. Provide "x" and "z".'}), 400
    msg = {"command": "move", "target": {"x": data['x'], "y": 0, "z": data['z']}}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move command sent', 'command': msg})

@app.route('/move_rel', methods=['POST'])
def move_rel():
    data = request.get_json()
    if not data or 'turn' not in data or 'distance' not in data:
        return jsonify({'error': 'Missing parameters. Provide "turn" and "distance".'}), 400
    msg = {"command": "move_relative", "turn": data['turn'], "distance": data['distance']}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move relative command sent', 'command': msg})

@app.route('/stop', methods=['POST'])
def stop():
    msg = {"command": "stop"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'stop command sent', 'command': msg})

@app.route('/capture', methods=['POST'])
def capture():
    msg = {"command": "capture_image"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'capture command sent', 'command': msg})

@app.route('/goal', methods=['POST'])
def set_goal():
    data = request.get_json() or {}
    if 'corner' in data:
        pos = corner_to_coords(str(data['corner']))
    elif 'x' in data and 'z' in data:
        pos = {"x": float(data['x']), "y": float(data.get('y', 0)), "z": float(data['z'])}
    else:
        return jsonify({'error': 'Provide {"corner":"NE|NW|SE|SW"} OR {"x":..,"z":..}'}), 400
    msg = {"command": "set_goal", "position": pos}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'goal set', 'goal': pos})

@app.route('/obstacles/motion', methods=['POST'])
def set_obstacle_motion():
    data = request.get_json() or {}
    if 'enabled' not in data:
        return jsonify({'error': 'Missing "enabled" boolean.'}), 400
    msg = {
        "command": "set_obstacle_motion",
        "enabled": bool(data['enabled']),
        "speed": float(data.get('speed', 0.05)),
        "velocities": data.get('velocities'),
        "bounds": data.get('bounds', {"minX": -45, "maxX": 45, "minZ": -45, "maxZ": 45}),
        "bounce": bool(data.get('bounce', True)),
    }
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'obstacle motion updated', 'config': msg})

@app.route('/collisions', methods=['GET'])
def get_collisions():
    return jsonify({'count': collision_count})

@app.route('/reset', methods=['POST'])
def reset():
    global collision_count
    collision_count = 0
    agent_state["goal_reached"] = False
    agent_state["robot_position"] = {"x": 0, "y": 0, "z": 0}
    navigator.last_positions.clear()
    if not broadcast({"command": "reset"}):
        return jsonify({'status': 'reset done (no simulators connected)', 'collisions': collision_count})
    return jsonify({'status': 'reset broadcast', 'collisions': collision_count})

@app.route("/autonomous_start", methods=["POST"])
def autonomous_start():
    data = request.get_json() or {}
    corner = data.get("corner", "NE")
    print(f"🚀 Starting autonomous navigation to {corner}")
    navigator.start(corner=corner)
    return jsonify({"status": "autonomous navigation started", "target": corner})

@app.route("/autonomous_stop", methods=["POST"])
def autonomous_stop():
    print("🛑 Stopping autonomous navigation")
    navigator.stop()
    return jsonify({"status": "autonomous navigation stopped"})

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "running": navigator.running,
        "collisions": collision_count,
        "goal_reached": agent_state["goal_reached"],
        "robot_position": agent_state.get("robot_position", {"x": 0, "y": 0, "z": 0})
    })

# ---------------------------
# Flask Thread
# ---------------------------
def start_flask():
    app.run(port=5000, debug=False, use_reloader=False)  # Added use_reloader=False

# ---------------------------
# Main Async for WebSocket
# ---------------------------
async def main():
    global async_loop
    async_loop = asyncio.get_running_loop()
    try:
        ws_server = await websockets.serve(ws_handler, "localhost", 8080)
        print("🌐 WebSocket server started on ws://localhost:8080")
        await ws_server.wait_closed()
    except Exception as e:
        print(f"WebSocket server error: {e}")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    print("🤖 Robot Navigation Server Starting...")
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("🌐 Flask API started on http://localhost:5000")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutting down...")
