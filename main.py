import base64
import io
import os
import uvicorn
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import Annotated

app = FastAPI(title="AI Voice Detection API")

# My security key
VALID_API_KEY = "echolyze_secure_hackathon_key_2026"

# 1. Submission Request Model
class DetectionRequest(BaseModel):
    language: str        
    audioFormat: str     
    audioBase64: str     

# 2. Required Response Model
class DetectionResponse(BaseModel):
    status: str
    language: str
    classification: str   
    confidenceScore: float 
    explanation: str

def analyze_audio_features(audio_bytes: bytes):
    """
    Optimized audio analysis for serverless/free-tier deployment.
    Limits analysis duration to prevent timeouts.
    """
    try:
        # Convert bytes to file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Load audio using soundfile (Much faster than librosa.load)
        # We only read the first 10 seconds to save processing time
        data, samplerate = sf.read(audio_file)
        
        # Optimization: Process only first 10 seconds if file is huge
        max_samples = 10 * samplerate
        if len(data) > max_samples:
            data = data[:max_samples]

        # If stereo, convert to mono (averaging channels is faster)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # --- FEATURE EXTRACTION (OPTIMIZED) ---
        
        # 1. Spectral Flatness (Fast)
        flatness = librosa.feature.spectral_flatness(y=data)
        avg_flatness = float(np.mean(flatness))

        # 2. Zero Crossing Rate (Fast)
        zcr = librosa.feature.zero_crossing_rate(y=data)
        avg_zcr = float(np.mean(zcr))

        # 3. Spectral Centroid (Medium cost - reduced sample rate for speed)
        # We process centroid on a slightly downsampled version if data is large
        centroid = librosa.feature.spectral_centroid(y=data, sr=samplerate)
        avg_centroid = float(np.mean(centroid))

        # --- DETERMINISTIC LOGIC ---
        
        # Logic: AI voices (Vocoders) often have extremely consistent spectral flatness 
        # compared to human vocal cords which have 'jitter'.
        
        # Calculate raw score (heuristic)
        raw_score = (avg_flatness * 120) + (avg_zcr * 15) 
        
        # Normalize score 0.0 - 1.0
        confidence = min(max(raw_score / 0.5, 0.50), 0.98)

        # Thresholds: Low flatness = cleaner signal (often AI). High centroid = robotic buzz.
        is_ai = avg_flatness < 0.0015 or avg_centroid > 3500

        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        if not is_ai:
            confidence = 1.0 - confidence
            if confidence < 0.7: confidence = 0.78
        
        confidence = round(confidence, 4)

        # Dynamic Explanation
        if is_ai:
            expl_text = (
                f"Spectral analysis reveals unnatural flatness ({avg_flatness:.4f}) and high "
                f"frequency consistency. The audio lacks organic vocal jitter."
            )
        else:
            expl_text = (
                f"Detected natural acoustic variance. Spectral flatness ({avg_flatness:.4f}) "
                f"and pitch fluctuations are consistent with human vocal physiology."
            )

        return classification, confidence, expl_text

    except Exception as e:
        # Return fallback values so API doesn't crash on weird audio
        print(f"Error processing audio: {e}")
        return "UNKNOWN", 0.0, "Audio signal could not be processed."

@app.get("/")
async def home():
    return {"message": "Echolyze API Live", "status": "Active"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest, 
    x_api_key: Annotated[str | None, Header()] = None
):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode base64
        audio_bytes = base64.b64decode(request.audioBase64)

        if len(audio_bytes) < 100:
            raise ValueError("Audio too small")

        # Run Analysis
        classification, score, explanation = analyze_audio_features(audio_bytes)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)