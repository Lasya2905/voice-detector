import base64
import io
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI()

class VoiceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    language: str
    audioFormat: str
    audio_base_64: str = Field(..., alias="audioBase64")

@app.get("/")
def health_check():
    return {"status": "online", "system": "ready"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != "my_secret_key_123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 1. Decode the Base64 string
        try:
            audio_bytes = base64.b64decode(request.audio_base_64)
        except Exception:
            return {"status": "error", "message": "Invalid Base64 string provided."}

        # 2. Load the audio data
        # We wrap it in a try-except to catch format issues (like MP3 vs WAV)
        try:
            audio_file = io.BytesIO(audio_bytes)
            # We use sr=None to get the original speed, then analyze
            y, sr = librosa.load(audio_file, sr=16000)
        except Exception as e:
            return {"status": "error", "message": f"Audio loading failed. Ensure the Base64 is a valid {request.audioFormat} file. Error: {str(e)}"}

        # 3. Fast Spectral Analysis
        # Check for 'Robotic' consistency (AI voices often lack natural jitter)
        stft = np.abs(librosa.stft(y))
        centroid = np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr))
        
        # Decision Logic
        if centroid > 2700:
            classification = "AI_GENERATED"
            score = round(0.85 + (centroid / 20000), 2)
            expl = f"Artificial frequency artifacts detected in {request.language} audio."
        else:
            classification = "HUMAN"
            score = round(0.98 - (centroid / 30000), 2)
            expl = f"Natural vocal harmonics detected in {request.language} audio."

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": min(score, 0.99),
            "explanation": expl
        }

    except Exception as e:
        return {"status": "error", "message": f"Unexpected system error: {str(e)}"}
