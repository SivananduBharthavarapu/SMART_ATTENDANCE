import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import sqlite3
import face_recognition
import pickle
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore")

# Paths and configurations
DATA_PATH = "C:\\Users\\bsiva\\PycharmProjects\\ProtoType\\.venv\\training_images"
PICKLE_FILE = "FINAL_ENCODING.pkl"
DB_FILE = "attendance.db"

# Performance configurations
PROCESS_EVERY_N_FRAMES = 3
FACE_DETECTION_SCALE = 0.5
FACE_RECOGNITION_SCALE = 0.25


class GestureController:
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.7, maxHands=1)
        self.last_detection_time = 0
        self.cooldown_period = 5  # Reduced cooldown period
        self.gesture_active = False
        self.last_process_time = time.time()
        self.process_interval = 0.05

    def is_five_fingers_detected(self, hand_landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        status = []

        # Check thumb
        if hand_landmarks[4][0] > hand_landmarks[3][0]:
            status.append(1)
        else:
            status.append(0)

        # Check other fingers
        for id in range(1, 5):
            if hand_landmarks[finger_tips[id]][1] < hand_landmarks[finger_tips[id] - 2][1]:
                status.append(1)
            else:
                status.append(0)

        return sum(status) >= 4

    def detect_gesture(self, frame):
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return False, frame

        self.last_process_time = current_time
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        hands, _ = self.detector.findHands(small_frame, draw=False)

        if hands:
            hand_landmarks = hands[0]['lmList']
            if self.is_five_fingers_detected(hand_landmarks):
                if not self.gesture_active or (current_time - self.last_detection_time > self.cooldown_period):
                    self.last_detection_time = current_time
                    self.gesture_active = True
                    frame_with_hands = frame.copy()
                    self.detector.findHands(frame_with_hands)
                    cv2.putText(frame_with_hands, "✋ Five fingers detected!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    return True, frame_with_hands
            else:
                self.gesture_active = False

        return False, frame


class AttendanceSystem:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.present_students = set()
        self.student_list = [name.replace("_training_images", "") for name in os.listdir(DATA_PATH)]
        self._init_database()
        self.conn = sqlite3.connect(self.db_path)

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    status TEXT NOT NULL,
                    UNIQUE(student_name, date)
                )
            """)
            conn.commit()

    def mark_attendance(self, student_name, attendance_mode=False):
        if not attendance_mode:
            return False

        if student_name not in self.present_students:
            self.present_students.add(student_name)
            print(f"✅ Attendance marked for {student_name}")
            return True
        return False

    def get_attendance_summary(self):
        absent_students = set(self.student_list) - self.present_students
        return list(self.present_students), list(absent_students)


class FaceRecognitionSystem:
    def __init__(self, attendance_system: AttendanceSystem):
        self.attendance_system = attendance_system
        self.known_face_encodings = []
        self.known_face_names = []
        self.authorized_user = "Sivanandu"  # Host name
        self.host_detected = False
        self.frame_count = 0
        self.load_known_faces()
        self.last_face_locations = []
        self.last_face_names = []
        self.last_recognition_time = time.time()
        self.host_detection_color = (0, 255, 0)
        self.student_detection_color = (255, 165, 0)
        self.unknown_detection_color = (0, 0, 255)

    def load_known_faces(self):
        try:
            with open(PICKLE_FILE, "rb") as f:
                self.known_face_encodings, self.known_face_names = pickle.load(f)
                print(f"✅ Loaded {len(self.known_face_names)} trained faces")
        except FileNotFoundError:
            print("❌ No training data found. Run the training function first.")

    def process_frame(self, frame, attendance_mode):
        self.frame_count += 1

        if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            for (top, right, bottom, left), name in zip(self.last_face_locations, self.last_face_names):
                top = int(top / FACE_DETECTION_SCALE)
                right = int(right / FACE_DETECTION_SCALE)
                bottom = int(bottom / FACE_DETECTION_SCALE)
                left = int(left / FACE_DETECTION_SCALE)

                if name == self.authorized_user:
                    color = self.host_detection_color
                    label = f"{name} (HOST)"
                elif name == "Unknown":
                    color = self.unknown_detection_color
                    label = name
                else:
                    color = self.student_detection_color
                    label = name

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (left, bottom), (left + text_size[0], bottom + 25), color, -1)
                cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            return self.host_detected

        small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        self.last_face_locations = []
        self.last_face_names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if len(self.known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < 0.6:
                    name = self.known_face_names[best_match_index]

                    if name == self.authorized_user:
                        self.host_detected = True
                    elif attendance_mode:
                        self.attendance_system.mark_attendance(name, attendance_mode)
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            self.last_face_locations.append((top, right, bottom, left))
            self.last_face_names.append(name)

            scaled_top = int(top / FACE_DETECTION_SCALE)
            scaled_right = int(right / FACE_DETECTION_SCALE)
            scaled_bottom = int(bottom / FACE_DETECTION_SCALE)
            scaled_left = int(left / FACE_DETECTION_SCALE)

            if name == self.authorized_user:
                color = self.host_detection_color
                label = f"{name} (HOST)"
            elif name == "Unknown":
                color = self.unknown_detection_color
                label = name
            else:
                color = self.student_detection_color
                label = name

            cv2.rectangle(frame, (scaled_left, scaled_top), (scaled_right, scaled_bottom), color, 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (scaled_left, scaled_bottom),
                          (scaled_left + text_size[0], scaled_bottom + 25), color, -1)
            cv2.putText(frame, label, (scaled_left, scaled_bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return self.host_detected


def main():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    attendance_system = AttendanceSystem()
    face_recognition_system = FaceRecognitionSystem(attendance_system)
    gesture_controller = GestureController()

    if not os.path.exists(PICKLE_FILE):
        print("❌ No face encodings found. Please run training first.")
        return

    phone_camera_url = "http://192.0.0.2:8080/video"
    cap = cv2.VideoCapture(phone_camera_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Failed to access camera")
        return

    attendance_mode = False
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    print("👋 Host: Show 5 fingers to start attendance system. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # FPS calculation
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        # Process face recognition
        host_detected = face_recognition_system.process_frame(frame, attendance_mode)

        # Only check for gesture if host is detected and attendance mode is not active
        if host_detected and not attendance_mode:
            five_fingers_detected, frame = gesture_controller.detect_gesture(frame)
            if five_fingers_detected:
                attendance_mode = True
                print("✅ Attendance mode activated! Recording attendance...")

        # Status display
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status_text = "ACTIVE" if attendance_mode else "WAITING FOR GESTURE"
        status_color = (0, 255, 0) if attendance_mode else (0, 165, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        host_status = "DETECTED ✓" if host_detected else "NOT DETECTED ✗"
        host_color = (0, 255, 0) if host_detected else (0, 0, 255)
        cv2.putText(frame, f"Host: {host_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, host_color, 2)

        present_count = len(attendance_system.present_students)
        cv2.putText(frame, f"Students Present: {present_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Show final attendance
    present_students, absent_students = attendance_system.get_attendance_summary()

    print("\n📊 Final Attendance Report:")
    print("\n✅ Present Students:")
    for student in present_students:
        print(f"  - {student}")

    print("\n❌ Absent Students:")
    for student in absent_students:
        print(f"  - {student}")

    print(f"\n📈 Summary:")
    print(f"  Total Students: {len(attendance_system.student_list)}")
    print(f"  Present: {len(present_students)}")
    print(f"  Absent: {len(absent_students)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()