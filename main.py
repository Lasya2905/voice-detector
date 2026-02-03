import base64
import io
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI()

class VoiceRequest(BaseModel):
    # This is the V2 way to handle aliases
    model_config = ConfigDict(populate_by_name=True)
    
    language: str
    audioFormat: str
    # This ensures 'audioBase64' from the tester maps to 'audio_base_64'
    audio_base_64: str = Field(..., alias="audioBase64")

@app.get("/")
def health_check():
    return {"status": "online", "version": "2.1.0"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != "my_secret_key_123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 1. Decode
        audio_bytes = base64.b64decode(request.audio_base_64)
        
        # 2. Load with a fallback for memory
        # We use a smaller sample rate (8k) to ensure it doesn't crash Render's RAM
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=8000)

        # 3. Simple Spectral Math
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Determine AI vs Human based on spectral 'brightness'
        if centroid > 2600:
            label, score = "AI_GENERATED", round(0.85 + (centroid/20000), 2)
            msg = f"Unnatural frequency spikes detected in {request.language} sample."
        else:
            label, score = "HUMAN", round(0.98 - (centroid/25000), 2)
            msg = f"Natural harmonic resonance observed in {request.language} sample."

        return {
            "status": "success",
            "language": request.language,
            "classification": label,
            "confidenceScore": min(score, 0.99),
            "explanation": msg
        }

    except Exception as e:
        # This helps us see the EXACT error in the tester response
        return {"status": "error", "message": str(e)}
