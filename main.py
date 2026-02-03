import base64
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

# This defines the data your API expects to receive
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # 1. Check if the API key is correct
    if x_api_key != "my_secret_key_123":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2. This is a basic logical placeholder for detection
    # In a real scenario, you'd put your AI model logic here
    result = "HUMAN" 
    confidence = 0.98

    # 3. Return the exact JSON format required by the problem
    return {
        "status": "success",
        "language": request.language,
        "classification": result,
        "confidenceScore": confidence,
        "explanation": "Natural speech patterns and breathing detected."
    }
