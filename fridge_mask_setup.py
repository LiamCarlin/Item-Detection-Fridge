import cv2
import json
import numpy as np

MASK_FILE = "fridge_mask.json"
polygon_points = []  # Stores the polygon points
drawing = True  # Determines if the user is still drawing
first_point = None  # Stores the first point for easier closure
closing_threshold = 15  # Larger hitbox size for closing the shape

capture_now = False
latest_frame = None

def draw_mask_overlay(image, points, mouse_pos=None):
    """ Draws the polygon and previews the next line while drawing. """
    overlay = image.copy()

    # Draw gray exclusion overlay
    if len(points) > 1:
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], (255, 255, 255))  # White for included area
        overlay = cv2.addWeighted(overlay, 0.4, mask, 0.6, 0)

    # Draw existing lines
    for i in range(1, len(points)):
        cv2.line(overlay, points[i - 1], points[i], (0, 255, 0), 2)

    # Show preview line if mouse is moving and drawing is active
    if mouse_pos and len(points) > 0 and drawing:
        cv2.line(overlay, points[-1], mouse_pos, (0, 255, 255), 1)  # Yellow preview line

    return overlay


def draw_polygon(event, x, y, flags, param):
    """ Handles mouse clicks to create a polygon mask. """
    global polygon_points, drawing, first_point, original_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if first_point and abs(x - first_point[0]) < closing_threshold and abs(y - first_point[1]) < closing_threshold:
            # If user clicks near the first point, close the polygon
            polygon_points.append(first_point)  # Close the shape
            drawing = False  # Stop further drawing
            print("✅ Polygon closed. Press 'q' to save.")

        else:
            polygon_points.append((x, y))  # Add new point
            if first_point is None:
                first_point = (x, y)  # Store first point

            print(f"Point added: {x}, {y}")

        # Refresh display with new line
        updated_frame = draw_mask_overlay(original_frame.copy(), polygon_points)
        cv2.imshow("Setup Fridge Mask", updated_frame)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Show pending line
        updated_frame = draw_mask_overlay(original_frame.copy(), polygon_points, (x, y))
        cv2.imshow("Setup Fridge Mask", updated_frame)


def setup_mask():
    global original_frame, first_point, latest_frame, capture_now, polygon_points
    polygon_points = []
    first_point = None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Setup Fridge Mask")
    
    print("\n📝 INSTRUCTIONS:")
    print(" - Press the 'c' key to capture an image.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        latest_frame = frame.copy()
        cv2.imshow("Setup Fridge Mask", latest_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            original_frame = latest_frame.copy()
            print("Photo captured from live feed.")
            break
        elif key == ord('q'):  # Optional exit during live view
            print("Exiting without capturing photo.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n📝 INSTRUCTIONS:")
    print(" - Click to add points to the mask area.")
    print(f" - Click **near the first point** (within {closing_threshold}px) to close the shape.")
    print(" - Press 'r' to reset.")
    print(" - Press 'q' to save and exit.\n")
    
    cv2.imshow("Setup Fridge Mask", original_frame)
    cv2.setMouseCallback("Setup Fridge Mask", draw_polygon)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and not drawing:
            break
        elif key == ord('r'):
            polygon_points.clear()
            first_point = None
            cv2.imshow("Setup Fridge Mask", original_frame)
    
    with open(MASK_FILE, "w") as file:
        json.dump(polygon_points, file)
        
    print(f"\n✅ Mask saved to {MASK_FILE}")
    print("🎯 Run the main fridge tracking script to apply the mask.\n")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_mask()
