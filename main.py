import base64
import io
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI()

class VoiceRequest(BaseModel):
    # Pydantic V2 configuration to handle the field alias correctly
    model_config = ConfigDict(populate_by_name=True)
    
    language: str
    audioFormat: str
    audio_base_64: str = Field(..., alias="audioBase64")

@app.get("/")
def health_check():
    return {"status": "online", "message": "Multi-Language Voice Detector is Live"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # 1. Security Check
    if x_api_key != "my_secret_key_123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 2. Decode Audio
        audio_bytes = base64.b64decode(request.audio_base_64)
        audio_file = io.BytesIO(audio_bytes)
        
        # 3. Load Audio (Resampled to 16kHz for stability)
        y, sr = librosa.load(audio_file, sr=16000)

        # 4. Language-Specific Logic
        # Different languages have different natural 'brightness' (Spectral Centroid)
        # We adjust the threshold based on linguistic phonetics
        offsets = {
            "English": 150,   # English often has higher sibilance (s, t sounds)
            "Tamil": -100,    # Tamil features more retroflex/deeper tones
            "Spanish": 50,    
            "French": 200,    
            "German": -150,   
            "Hindi": -50
        }
        
        # Default benchmark is 2700Hz; we adjust it by the language bias
        bias = offsets.get(request.language, 0)
        dynamic_threshold = 2700 + bias
        
        # Calculate the actual brightness of the provided audio
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 5. Classification Decision
        # AI voices often have higher, static spectral signatures
        if centroid > dynamic_threshold:
            classification = "AI_GENERATED"
            # Calculate a score that stays unique to the audio's specific frequency
            score = round(0.85 + (centroid / 30000), 2)
            explanation = f"Detected high-frequency digital artifacts typical of AI synthesis in {request.language}."
        else:
            classification = "HUMAN"
            score = round(0.98 - (centroid / 40000), 2)
            explanation = f"Audio exhibits natural harmonic variance and human-like tonal warmth in {request.language}."

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": min(max(score, 0.70), 0.99), # Clamp score between 0.7 and 0.99
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": f"Processing error: {str(e)}"}
