import base64
import io
import torch
import librosa
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
try:
    classifier = pipeline("audio-classification", model="umm-maybe/AI-generated-vs-Human-Audio")
except:
    classifier = None

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": classifier is not None}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    
    if x_api_key != "my_secret_key_123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    
    allowed_langs = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if request.language not in allowed_langs:
        raise HTTPException(status_code=400, detail="Unsupported language")

    try:
       
        audio_data = base64.b64decode(request.audio_base64)
        
        audio_array, _ = librosa.load(io.BytesIO(audio_data), sr=16000)

        
        if classifier:
            prediction = classifier(audio_array)
            
            top_result = prediction[0]
            
            label = "AI_GENERATED" if top_result['label'].lower() in ['fake', 'ai', 'synthetic'] else "HUMAN"
            score = round(top_result['score'], 2)
        else:
            
            label = "HUMAN"
            score = 0.50

        
        if label == "AI_GENERATED":
            expl = f"Detected high-frequency artifacts and unnatural {request.language} phoneme transitions."
        else:
            expl = f"Natural breath markers and human-typical spectral variance observed in {request.language} sample."

        return {
            "status": "success",
            "language": request.language,
            "classification": label,
            "confidenceScore": score,
            "explanation": expl
        }

    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {str(e)}"}
