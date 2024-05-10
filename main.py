from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import joblib
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from pydub import AudioSegment
import speech_recognition as sr

# Create an instance of the FastAPI class
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load speaker names
speaker_folders = ["Benjamin_Netanyau", "Jens_Stoltenberg", "Julia_Gillard", "Magaret_Tarcher", "Nelson_Mandela"]

# Load your trained SVM model
loaded_svm_model = joblib.load('./Trained_ML_Models/RF.joblib')

# Define a function to extract MFCC features and make predictions
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None, duration=1)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features = mfccs.flatten().reshape(1, -1)  # Flatten and reshape features
    return features

# Define a function to convert mp3 to wav
def convert_to_wav(mp3_file_path):
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    AudioSegment.from_mp3(mp3_file_path).export(wav_file_path, format="wav")
    return wav_file_path

# Define a request body model
class FileInput(BaseModel):
    file: bytes

# Define an endpoint to receive audio files and return predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save the received audio file
    file_path = "./temp.mp3"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    # Convert mp3 to wav
    wav_file_path = convert_to_wav(file_path)
    # Extract features
    features = extract_features(wav_file_path)
    # Make prediction
    predicted_label = loaded_svm_model.predict(features)[0]
    predicted_speaker = speaker_folders[predicted_label]
    
    # Use speech recognition to extract text from the audio
    recognizer = sr.Recognizer()
    transcription = None
    with sr.AudioFile(wav_file_path) as source:
        try:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            transcription = None
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    return {"predicted_speaker": predicted_speaker, "transcription": transcription}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
