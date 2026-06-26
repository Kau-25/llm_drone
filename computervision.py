#!/usr/bin/env python3
"""
Autonomous Drone Landing with Computer Vision
This script:
1. Receives video stream from Raspberry Pi camera over ZeroTier
2. Performs real-time object detection using YOLO
3. Automatically lands the drone when target object is detected
Usage:
python3 autonomous_landing.py \
--pi-ip 10.147.17.123 \
--target person \
--confidence 0.7
"""
import argparse
import sys
import time
import cv2
from ultralytics import YOLO
from pymavlink import mavutil


class DroneController:
    """Handles MAVLink communication with the drone"""

    def __init__(self, drone_ip, drone_port=5678):
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        self.master = None
        self.connected = False

    def connect(self):
        """Connect to the drone via MAVLink"""
        connection_string = f"tcp:{self.drone_ip}:{self.drone_port}"
        print(f"Connecting to drone at {connection_string}...")
        try:
            self.master = mavutil.mavlink_connection(connection_string)
            self.master.wait_heartbeat(timeout=10)
            self.connected = True
            print(f"✓ Connected to drone (System ID: {self.master.target_system})")
            return True
        except Exception as e:
            print(f"ERROR: Could not connect to drone: {e}")
            return False

    def land(self):
        """Command the drone to land"""
        if not self.connected:
            print("ERROR: Not connected to drone")
            return False
        try:
            modes = self.master.mode_mapping() or {}
            if "LAND" not in modes:
                print(f"ERROR: LAND mode not available. Available: {list(modes.keys())}")
                return False

            mode_id = modes["LAND"]
            print("\n" + "=" * 60)
            print("🚨 TARGET DETECTED - INITIATING LANDING SEQUENCE 🚨")
            print("=" * 60)
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id
            )
            print("✓ LAND command sent to flight controller")
            print("=" * 60 + "\n")
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"ERROR: Could not send land command: {e}")
            return False

    def close(self):
        """Close connection to drone"""
        if self.master:
            try:
                self.master.close()
                print("Drone connection closed")
            except:
                pass


class VideoStreamDetector:
    """Handles video stream reception and object detection"""

    def __init__(self, pi_ip, stream_port=8554, model_path='yolov8n.pt'):
        self.pi_ip = pi_ip
        self.stream_port = stream_port
        self.stream_url = f"tcp://{pi_ip}:{stream_port}"
        self.model = YOLO(model_path)
        self.cap = None

    def connect_stream(self):
        """Connect to Pi video stream"""
        print(f"Connecting to video stream: {self.stream_url}")
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            print("ERROR: Cannot open video stream")
            print("Check:")
            print("  1. Is rpicam-vid running on Pi?")
            print("  2. Can you ping the Pi?")
            print(f"  Try: ping {self.pi_ip}")
            return False
        print("✓ Video stream connected")
        return True

    def detect_objects(self, frame):
        """Run object detection on a frame"""
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls_id]
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'box': box.xyxy[0].cpu().numpy()
                })
        return detections

    def stop_stream(self):
        """Stop video stream"""
        if self.cap:
            self.cap.release()
            print("Video stream closed")


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Autonomous drone landing with computer vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Land when person detected with 70%+ confidence
  python3 autonomous_landing.py \\
      --pi-ip 10.147.17.123 \\
      --target person \\
      --confidence 0.7

  # Land when sports ball detected
  python3 autonomous_landing.py \\
      --pi-ip 10.147.17.123 \\
      --target "sports ball" \\
      --confidence 0.6

Common target objects:
  person, car, truck, dog, cat, bird, bottle, cup, sports ball,
  backpack, umbrella, handbag, chair, couch, bed, laptop, etc.
"""
    )
    parser.add_argument('--pi-ip', required=True,
                        help='ZeroTier IP address of Raspberry Pi')
    parser.add_argument('--target', required=True,
                        help='Target object to detect (e.g., "person", "car", "sports ball")')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Minimum confidence threshold (0.0-1.0, default: 0.7)')
    parser.add_argument('--stream-port', type=int, default=8554,
                        help='Video stream port (default: 8554)')
    parser.add_argument('--mavlink-port', type=int, default=5678,
                        help='MAVLink port (default: 5678)')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Cooldown period in seconds after landing command (default: 30)')
    parser.add_argument('--frame-skip', type=int, default=3,
                        help='Process every Nth frame (default: 3 = process 1 out of 3 frames)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("AUTONOMOUS LANDING SYSTEM")
    print("=" * 60)
    print(f"Pi IP address:   {args.pi_ip}")
    print(f"Video stream:    tcp://{args.pi_ip}:{args.stream_port}")
    print(f"Drone MAVLink:   tcp://{args.pi_ip}:{args.mavlink_port}")
    print(f"Target object:   {args.target}")
    print(f"Min confidence:  {args.confidence:.0%}")
    print(f"Frame skip:      Process 1 of every {args.frame_skip} frames")
    print(f"Cooldown period: {args.cooldown}s")
    print("=" * 60 + "\n")

    # Initialize components
    print("Initializing system...")
    detector = VideoStreamDetector(args.pi_ip, args.stream_port)
    controller = DroneController(args.pi_ip, args.mavlink_port)

    # Connect to video stream
    if not detector.connect_stream():
        print("FATAL: Cannot connect to video stream")
        return 1

    # Connect to drone
    if not controller.connect():
        print("FATAL: Cannot connect to drone")
        detector.stop_stream()
        return 1

    print("\n" + "=" * 60)
    print("✓ SYSTEM READY - Monitoring for target object")
    print("=" * 60)
    print(f"Looking for: {args.target}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Main detection loop
    frame_count = 0
    detection_count = 0
    landed = False
    land_time = 0

    try:
        while True:
            # Read frame
            ret, frame = detector.cap.read()
            if not ret:
                print("ERROR: Lost video stream")
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % args.frame_skip != 0:
                continue

            # Print status every 30 processed frames
            if (frame_count // args.frame_skip) % 30 == 0:
                print(f"[Frame {frame_count:05d}] Processing... (detection count: {detection_count})")

            # Check if cooldown period has passed
            if landed and (time.time() - land_time) < args.cooldown:
                continue
            elif landed and (time.time() - land_time) >= args.cooldown:
                # Cooldown expired, reset
                print(f"[Frame {frame_count:05d}] Cooldown expired. Resuming detection.")
                landed = False

            # Run detection
            detections = detector.detect_objects(frame)

            # Check for target object
            for det in detections:
                if det['class'].lower() == args.target.lower() and \
                        det['confidence'] >= args.confidence:
                    detection_count += 1
                    print(f"\n🎯 TARGET DETECTED! {det['class']} "
                          f"(confidence: {det['confidence']:.1%}) "
                          f"[Detection #{detection_count}]")

                    # Send land command
                    if controller.land():
                        landed = True
                        land_time = time.time()
                        print(f"\nCooldown active for {args.cooldown} seconds...")
                        print(f"(Will not send another land command during cooldown)")
                    break  # Only land once per detection cycle

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 60)
        print("SHUTTING DOWN")
        print("=" * 60)
        print(f"Total frames received: {frame_count}")
        print(f"Frames processed:      {frame_count // args.frame_skip}")
        print(f"Target detections:     {detection_count}")
        print(f"Land commands sent:    {1 if landed else 0}")
        print("=" * 60 + "\n")
        detector.stop_stream()
        controller.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
