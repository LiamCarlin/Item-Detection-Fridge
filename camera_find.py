import cv2

def open_camera():
    # Initialize camera (0 is usually the default/built-in camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
