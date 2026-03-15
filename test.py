import numpy as np
import soundfile as sf
from keras.models import load_model
import pickle, librosa

model = load_model("models/audio_emotion_model.keras")
with open("models/scaler.pkl","rb") as f:
    sc = pickle.load(f)

wave, sr = librosa.load("03-01-06-01-01-01-03.wav", sr=22050)
mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=40)
mfcc = np.pad(mfcc, ((0,0),(0,216-mfcc.shape[1])), mode='constant')[:, :216]
flat = mfcc.flatten().reshape(1,-1)
scaled = sc.transform(flat).reshape(1,40,216,1)
print(model.predict(scaled))
