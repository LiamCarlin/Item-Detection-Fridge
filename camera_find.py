import cv2

def open_camera():
    # Initialize camera (0 is usually the default/built-in camera)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                break
                
            # Display the frame
            cv2.imshow('Camera Feed', frame)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera()