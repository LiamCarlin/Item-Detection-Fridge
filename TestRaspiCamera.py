import cv2
from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Set resolution
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow("Raspberry Pi Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
