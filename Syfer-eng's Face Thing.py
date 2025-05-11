import cv2
import mediapipe as mp
import numpy as np
import math

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def main():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Define custom drawing specs
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    # Define specific facial landmark indices
    # See MediaPipe Face Mesh documentation for index references
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    RIGHT_EYE = [33, 133]  # Right eye landmarks
    LEFT_EYE = [362, 263]  # Left eye landmarks
    LIPS = [61, 291]  # Lips landmarks
    RIGHT_EYEBROW = [70, 105]  # Right eyebrow landmarks
    LEFT_EYEBROW = [336, 300]  # Left eyebrow landmarks
    FACE_OVAL = [10, 338]  # Face oval landmarks
    NOSE = [1, 168]  # Nose landmarks
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure your webcam is connected and not in use by another application.")
        return
    
    # Set resolution to improve performance (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("Face Detection Started...")
        print("Press 'q' to quit.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Convert the image to RGB for MediaPipe processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, mark the image as not writeable
            image.flags.writeable = False
            results = face_mesh.process(image)
            
            # Convert back to BGR for OpenCV display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face mesh
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw the face contours
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Get image dimensions
                    h, w, c = image.shape
                    
                    # Calculate facial metrics
                    landmarks = face_landmarks.landmark
                    
                    # 1. Distance between eyes (using eye centers)
                    left_eye_center = landmarks[LEFT_EYE[0]]
                    right_eye_center = landmarks[RIGHT_EYE[0]]
                    eye_distance = calculate_distance(left_eye_center, right_eye_center) * w
                    
                    # 2. Face width (using face oval)
                    left_face_edge = landmarks[FACE_OVAL[0]]
                    right_face_edge = landmarks[FACE_OVAL[1]]
                    face_width = calculate_distance(left_face_edge, right_face_edge) * w
                    
                    # 3. Nose length
                    nose_top = landmarks[NOSE[0]]
                    nose_bottom = landmarks[NOSE[1]]
                    nose_length = calculate_distance(nose_top, nose_bottom) * h
                    
                    # 4. Mouth width
                    left_lip = landmarks[LIPS[0]]
                    right_lip = landmarks[LIPS[1]]
                    mouth_width = calculate_distance(left_lip, right_lip) * w
                    
                    # 5. Distance between eyebrows
                    left_eyebrow = landmarks[LEFT_EYEBROW[0]]
                    right_eyebrow = landmarks[RIGHT_EYEBROW[0]]
                    eyebrow_distance = calculate_distance(left_eyebrow, right_eyebrow) * w
                    
                    # Draw facial measurements on the image
                    metrics_color = (255, 255, 255)  # White color for text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    
                    # Display metrics on the image
                    cv2.putText(image, f"Eye Distance: {eye_distance:.1f}px", (10, 30), 
                                font, font_scale, metrics_color, thickness)
                    cv2.putText(image, f"Face Width: {face_width:.1f}px", (10, 60), 
                                font, font_scale, metrics_color, thickness)
                    cv2.putText(image, f"Nose Length: {nose_length:.1f}px", (10, 90), 
                                font, font_scale, metrics_color, thickness)
                    cv2.putText(image, f"Mouth Width: {mouth_width:.1f}px", (10, 120), 
                                font, font_scale, metrics_color, thickness)
                    cv2.putText(image, f"Eyebrow Distance: {eyebrow_distance:.1f}px", (10, 150), 
                                font, font_scale, metrics_color, thickness)
                    
                    # Calculate face symmetry (ratio of left-to-right side distances)
                    # This is a simplified measure - real symmetry is more complex
                    left_eye_to_center = calculate_distance(left_eye_center, nose_top)
                    right_eye_to_center = calculate_distance(right_eye_center, nose_top)
                    if right_eye_to_center > 0:
                        symmetry_ratio = left_eye_to_center / right_eye_to_center
                        symmetry_percentage = min(symmetry_ratio, 1/symmetry_ratio) * 100
                        cv2.putText(image, f"Face Symmetry: {symmetry_percentage:.1f}%", (10, 180), 
                                    font, font_scale, metrics_color, thickness)
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Display instructions
            cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the image
            cv2.imshow('Face Landmark Detection', image)
            
            # Exit on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Face Detection Terminated.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("An error occurred. Make sure you have installed all required packages:")
        print("pip install opencv-python mediapipe numpy")
        input("Press Enter to exit...")
