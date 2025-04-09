SMART_ATTENDANCE
Smart Attendance System using Gesture and Face Detection
Overview: The SMART_ATTENDANCE system is an intelligent and contactless attendance solution that combines gesture recognition and face detection to automate the attendance process. This system is designed to minimize manual intervention by allowing students or teachers to mark attendance simply by showing a specific hand gesture, followed by automatic face verification.

Once the gesture is recognized and the face is verified, the system records the attendance in a database and optionally uploads it to a portal. It enhances classroom efficiency, prevents proxy attendance, and supports real-time monitoring.

How It Works:
Gesture Detection:

The system uses a camera feed to continuously monitor for a predefined hand gesture (e.g., raised hand with specific fingers).

When the correct gesture is detected using a trained model or gesture-matching algorithm, it acts as a trigger to initiate attendance marking.

Face Detection & Recognition:

After detecting the gesture, the system scans the face of the person showing the gesture.

The face is compared against a pre-trained database of student images using face recognition algorithms (e.g., face_recognition library).

To reduce false positives, it requires multiple (e.g., 5) valid detections before confirming identity.

Attendance Marking:

Once a face is successfully verified, attendance is marked in a SQLite database or any other backend system.

The system logs the student name, date, time, and optionally the image of the person.

Portal Update & Reporting:

The attendance record can be automatically uploaded to a web portal or exported to an Excel sheet.

An email report can also be generated and sent to authorized personnel (e.g., teacher or admin).

Dynamic Updates:

The system dynamically loads student data based on folder names for easy scaling.

Students can be added or removed without retraining the entire model.

Technologies Used:
Python, OpenCV, face_recognition library

Gesture Recognition using CNN or MediaPipe

SQLite for attendance records

Tkinter/Streamlit for optional GUI

Email automation (e.g., smtplib)

Multi-threading for real-time performance
