import os
import time
import base64
import re
import cv2
import numpy as np
import json
from openai import OpenAI
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# Firebase Setup
cred = credentials.Certificate("fridge-detection-firebase-adminsdk-fbsvc-a48c01fce7.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# OpenAI API Setup
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
if not API_KEY:
    raise ValueError("API key is missing! Set the OPENAI_API_KEY environment variable.")

# Load YOLO Model for Object Detection
yolo_model = YOLO("yolov8n.pt")

# Load the predefined fridge mask
MASK_FILE = "fridge_mask.json"
if os.path.exists(MASK_FILE):
    with open(MASK_FILE, "r") as file:
        fridge_mask = np.array(json.load(file), dtype=np.int32)
else:
    fridge_mask = None

# Firebase Inventory
master_inventory = {}
tracked_objects = {}
scanned_objects = set()  # Stores already scanned object IDs

# **1️⃣ Apply & Display Fridge Mask**
def apply_mask(image):
    """Applies the user-defined mask to exclude non-fridge areas & displays the mask."""
    if fridge_mask is not None:
        # Create a binary mask from the polygon points
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [fridge_mask], 255)
        
        # Apply mask to get the vivid fridge area
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Dim areas outside the mask on the overlay by setting them to dark gray
        overlay = image.copy()
        inverse_mask = cv2.bitwise_not(mask)
        overlay[inverse_mask == 255] = (50, 50, 50)
        
        # Draw the mask outline in green
        cv2.polylines(overlay, [fridge_mask], isClosed=True, color=(0, 255, 0), thickness=2)
        
        return masked_image, overlay

    return image, image  # If no mask is found, return original image

# **2️⃣ Convert Image to Base64**
def encode_image(image):
    """Converts an image into Base64 format for OpenAI processing."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

# **3️⃣ Detect & Track Moving Objects Using YOLO**
def detect_objects_with_yolo(image):
    """Detects objects and returns tracking data (bounding box, position, ID)."""
    masked_image, overlay = apply_mask(image)  # Apply mask & show it in live view
    # Lower confidence threshold to detect new/moving objects with lower confidence
    results = yolo_model.track(masked_image, persist=True, conf=0.2)
    
    detected_objects = {}
    for r in results:
        if r.boxes is None:
            continue  # Skip if no boxes detected

        track_ids = r.boxes.id if r.boxes.id is not None else [None] * len(r.boxes.xyxy)
        classes = r.boxes.cls if hasattr(r.boxes, 'cls') else [None] * len(r.boxes.xyxy)

        for box, track_id, cls in zip(r.boxes.xyxy, track_ids, classes):
            if track_id is None:
                continue  # Skip objects without tracking IDs

            # Skip detections for the refrigerator itself
            if cls is not None:
                label = yolo_model.names[int(cls)]
                if label.lower() in ("refrigerator", "fridge"):
                    continue

            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # Object center

            # Ensure detection is inside the fridge mask
            if fridge_mask is not None and cv2.pointPolygonTest(fridge_mask, (center_x, center_y), False) < 0:
                continue  # Skip objects outside the fridge mask

            detected_objects[int(track_id)] = {
                "bbox": (x1, y1, x2, y2),
                "position": (center_x, center_y)
            }

    return detected_objects, overlay

# **4️⃣ Detect When Objects Stop Moving**
def detect_stopped_objects(previous_objects, new_objects, stop_threshold=3):
    """Detects objects that have stopped moving inside the fridge."""
    stopped_objects = []

    for obj_id, obj_data in new_objects.items():
        if obj_id in previous_objects:
            prev_pos = previous_objects[obj_id]["position"]
            current_pos = obj_data["position"]

            # If the object moves less than the threshold, consider it stopped
            if abs(prev_pos[0] - current_pos[0]) < stop_threshold and abs(prev_pos[1] - current_pos[1]) < stop_threshold:
                stopped_objects.append((obj_id, obj_data["bbox"]))

    return stopped_objects

# **5️⃣ Send Stopped Objects to OpenAI for Classification**
def analyze_fridge(image, bounding_boxes):
    """Crops stopped objects and sends them to OpenAI Vision."""
    detected_items = {}

    for obj_id, (x1, y1, x2, y2) in bounding_boxes:
        if obj_id in scanned_objects:
            continue  # Skip already scanned objects

        cropped_item = image[y1:y2, x1:x2]  # Extract object
        base64_image = encode_image(cropped_item)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze the cropped food item image and return its name along with quantity. "
                                "Format output as 'ITEM: QUANTITY'. "
                                "Ignore objects that are not food items."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )

        # Parse response
        response_text = response.choices[0].message.content
        for line in response_text.split('\n'):
            if ": " in line:
                item, quantity = line.split(": ", 1)
                detected_items[item.strip()] = quantity.strip()
                scanned_objects.add(obj_id)  # Mark object as scanned

    return detected_items

# **6️⃣ Update Firebase Inventory**
def update_inventory(new_items):
    """Updates Firebase with new fridge items, avoiding duplicates."""
    stored_inventory = db.collection("fridge_inventory").document("current_inventory").get().to_dict() or {}

    for item, quantity in new_items.items():
        quantity = int(re.search(r'\d+', quantity).group()) if re.search(r'\d+', quantity) else 1

        # Add only new items
        if item not in stored_inventory:
            master_inventory[item] = quantity
        else:
            print(f"🟡 Skipping {item} - already in inventory.")

    db.collection("fridge_inventory").document("current_inventory").set(master_inventory)
    print("✅ Firebase inventory updated!")

# **7️⃣ Live Fridge Tracking**
def live_fridge_tracking():
    """Starts the real-time fridge monitoring system."""
    global tracked_objects

    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting fridge tracking... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame, skipping.")
            continue

        new_tracked_objects, overlay = detect_objects_with_yolo(frame)

        # Draw bounding boxes on moving objects
        for obj_id, obj_data in new_tracked_objects.items():
            x1, y1, x2, y2 = obj_data["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
            cv2.putText(overlay, f"Tracking {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        stopped_objects = detect_stopped_objects(tracked_objects, new_tracked_objects)

        if stopped_objects:
            new_items = analyze_fridge(frame, stopped_objects)
            if new_items:
                update_inventory(new_items)

        tracked_objects = new_tracked_objects  # Update tracking data

        cv2.imshow("Live Fridge Tracking", overlay)  # Show masked and tracked objects
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping fridge tracking.")
            break

    cap.release()
    cv2.destroyAllWindows()

# **Run Live Tracking**
if __name__ == "__main__":
    live_fridge_tracking()