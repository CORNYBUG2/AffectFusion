import cv2
import numpy as np
import tensorflow as tf
import time

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_MODEL_PATH = "models/video_emotion_model.keras"
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ==========================================
# LOAD MODEL
# ==========================================
print("Loading video model...")
model = tf.keras.models.load_model(VIDEO_MODEL_PATH)
print("Model loaded successfully.")

# ==========================================
# VIDEO CAPTURE
# ==========================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit.")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def preprocess_frame(frame):
    """Convert webcam frame to 48×48 grayscale, equalized, cropped face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) > 0:
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])  # largest face
        face = gray[y:y+h, x:x+w]
    else:
        face = gray  # fallback

    # Equalize histogram to normalize lighting
    face = cv2.equalizeHist(face)

    resized = cv2.resize(face, (48, 48))
    img = resized.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img, faces

def sharpen(p, T=0.5):
    """Sharpen probabilities for stronger class contrast."""
    p = np.power(p, 1.0 / T)
    return p / np.sum(p)

# ==========================================
# MAIN LOOP
# ==========================================
last_label = "Listening..."
last_conf = 0.0
last_update_time = 0
display_timeout = 2.0  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img, faces = preprocess_frame(frame)
    preds = model.predict(img, verbose=0)[0]
    preds = sharpen(preds, T=0.5)

    # Debug probabilities in console
    print("Video probs:", np.round(preds, 3))

    idx = int(np.argmax(preds))
    label = LABELS[idx] if idx < len(LABELS) else str(idx)
    conf = float(preds[idx])

    last_label, last_conf = label, conf
    last_update_time = time.time()

    # Draw bounding box if face found
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display predicted emotion
    if time.time() - last_update_time < display_timeout:
        text = f"{last_label.upper()} ({last_conf * 100:.1f}%)"
        color = (0, 255, 0)
    else:
        text = "Listening..."
        color = (255, 255, 255)

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2.imshow("Video Emotion Detection (Face Cropped)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
