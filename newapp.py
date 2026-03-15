import os
os.environ["LIBROSA_RESAMPLER"] = "scipy"
os.environ["NUMBA_DISABLE_JIT"] = "1"

import cv2
import numpy as np
import sounddevice as sd
import librosa, pickle, io, soundfile as sf
from keras.models import load_model
import tensorflow as tf

# ======================================================
# CONFIG
# ======================================================
VIDEO_MODEL_PATH = "models/video_emotion_model.keras"
AUDIO_MODEL_PATH = "models/audio_emotion_model.keras"
SCALER_PATH = "models/scaler.pkl"
SAMPLERATE = 22050
DURATION = 1.0          # seconds per capture
FUSION_RATIO = (0.65, 0.35)
EMOTIONS = ["Angry", "Calm", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


# ======================================================
# MODEL LOADING
# ======================================================
print("Loading models...")
video_model = load_model(VIDEO_MODEL_PATH)
audio_model = load_model(AUDIO_MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("Models and scaler loaded.")
print("Press 'q' to quit.")

# ======================================================
# AUDIO PREPROCESSING
# ======================================================
def preprocess_audio_librosa(wave, sr_target=22050, n_mfcc=40, n_frames=216):
    if wave.ndim > 1:
        wave = np.mean(wave, axis=1)
    mfcc = librosa.feature.mfcc(y=wave, sr=sr_target, n_mfcc=n_mfcc)
    if mfcc.shape[1] < n_frames:
        mfcc = np.pad(mfcc, ((0,0),(0,n_frames-mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :n_frames]
    flat = mfcc.flatten().reshape(1, -1)
    scaled = scaler.transform(flat)
    mfcc = scaled.reshape(1, n_mfcc, n_frames, 1)
    return mfcc

# ======================================================
# VIDEO PREPROCESSING
# ======================================================
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=(0, -1))

# ======================================================
# MAIN LOOP
# ======================================================
cap = cv2.VideoCapture(0)
print("Running fusion model...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Video Prediction ----
    vid_input = preprocess_frame(frame)
    vid_pred = video_model.predict(vid_input, verbose=0)[0]
    vid_pred /= np.sum(vid_pred)

    # ---- Audio Capture ----
    audio_data = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    audio_data = audio_data.flatten()

    # ---- Audio Prediction ----
    aud_feat = preprocess_audio_librosa(audio_data)
    aud_pred = audio_model.predict(aud_feat, verbose=0)[0]
    aud_pred /= np.sum(aud_pred)

    # ---- Debug: print raw probabilities ----
    print("Video:", np.round(vid_pred, 3))
    print("Audio:", np.round(aud_pred, 3))

    # ---- Weighted Fusion ----
    min_len = min(len(vid_pred), len(aud_pred))
    vid_pred, aud_pred = vid_pred[:min_len], aud_pred[:min_len]
    fused = FUSION_RATIO[0]*vid_pred + FUSION_RATIO[1]*aud_pred

    top_idx = int(np.argmax(fused))
    label = EMOTIONS[top_idx] if top_idx < len(EMOTIONS) else str(top_idx)
    conf = float(fused[top_idx])

    # ---- Display ----
    cv2.putText(frame, f"Emotion: {label} ({conf*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Top-left: video probs
    v_text = "Video: " + ", ".join([f"{e[:3]}:{p:.2f}" for e,p in zip(EMOTIONS, vid_pred)])
    cv2.putText(frame, v_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Bottom-right: audio probs
    a_text = "Audio: " + ", ".join([f"{e[:3]}:{p:.2f}" for e,p in zip(EMOTIONS, aud_pred)])
    (tw, th), _ = cv2.getTextSize(a_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, a_text, (frame.shape[1]-tw-10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Audio-Visual Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
