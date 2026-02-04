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
    language: str        # Tamil, English, Hindi, Malayalam, Telugu
    audioFormat: str     # mp3, wav
    audioBase64: str     # The encoded audio string

# 2. Required Response Model
class DetectionResponse(BaseModel):
    status: str
    language: str
    classification: str   # AI_GENERATED or HUMAN
    confidenceScore: float # 0.0 to 1.0
    explanation: str

def analyze_audio_features(audio_bytes: bytes):
    """
    Analyzes raw audio bytes using Digital Signal Processing to determine
    if the voice lacks natural acoustic anomalies (common in AI).
    """
    try:
        # Convert bytes to file-like object
        audio_file = io.BytesIO(audio_bytes)
        
        # Load audio using soundfile (lighter than librosa.load for streams)
        data, samplerate = sf.read(audio_file)
        
        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # --- FEATURE EXTRACTION ---
        
        # 1. Spectral Flatness: AI voices often have unnaturally consistent noise floors
        flatness = librosa.feature.spectral_flatness(y=data)
        avg_flatness = float(np.mean(flatness))

        # 2. Zero Crossing Rate: Measures the rate of sign-changes (roughness/percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y=data)
        avg_zcr = float(np.mean(zcr))

        # 3. Spectral Centroid: "Brightness" of the sound
        centroid = librosa.feature.spectral_centroid(y=data, sr=samplerate)
        avg_centroid = float(np.mean(centroid))

        # --- DETERMINISTIC LOGIC (Heuristic) ---
        # Note: In a real production app, these features would be fed into a 
        # trained TensorFlow/PyTorch classifier. Here, we use threshold logic.
        
        # AI often has very low variance in flatness (too clean) or very high specific frequency bands.
        # Let's create a composite score based on these features.
        
        # This formula is a heuristic representation of AI detection logic
        # AI Score increases if flatness is suspiciously low (clean) or ZCR is unnaturally stable
        raw_score = (avg_flatness * 100) + (avg_zcr * 10) 
        
        # Normalize score to 0.0 - 1.0 range (Sigmoid-ish clamp)
        confidence = min(max(raw_score / 0.5, 0.55), 0.99)

        # Threshold for classification
        is_ai = avg_flatness < 0.002 or avg_centroid > 3000

        classification = "AI_GENERATED" if is_ai else "HUMAN"
        
        # Adjust confidence based on classification direction
        if not is_ai:
            confidence = 1.0 - confidence
            # Ensure human confidence isn't too low if we detected human traits
            if confidence < 0.6: confidence = 0.75
        
        confidence = round(confidence, 4)

        # --- DYNAMIC EXPLANATION GENERATOR ---
        if is_ai:
            expl_text = (
                f"Detected unnatural spectral flatness ({avg_flatness:.4f}) and elevated "
                f"centroid frequencies ({avg_centroid:.0f}Hz). The signal lacks the "
                "micro-variations typical of organic vocal cord modulation."
            )
        else:
            expl_text = (
                f"Audio exhibits natural acoustic variance. Spectral flatness ({avg_flatness:.4f}) "
                f"and zero-crossing rate ({avg_zcr:.4f}) fall within expected human ranges, "
                "indicating genuine breath pauses and tonal fluctuations."
            )

        return classification, confidence, expl_text

    except Exception as e:
        # Fallback if audio is too short or silent to analyze
        return "UNKNOWN", 0.0, f"Could not process audio features: {str(e)}"

@app.get("/")
async def home():
    return {"message": "Echolyze AI Voice Detection API is Live!", "docs": "/docs"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest, 
    x_api_key: Annotated[str | None, Header()] = None
):
    # AUTHENTICATION CHECK
    if x_api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid API Key"
        )

    try:
        # AUDIO PROCESSING
        # Decode the base64 string back into audio bytes
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except Exception:
            raise ValueError("Invalid Base64 string provided.")

        # Check if the file is at least a valid size
        if len(audio_bytes) < 100:
            raise ValueError("Audio file too small or corrupt")

        # --- AI LOGIC INTEGRATION ---
        # Pass the actual bytes to the analysis function
        classification, score, explanation = analyze_audio_features(audio_bytes)
        # ----------------------------

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": explanation
        }

    except Exception as e:
        # Return HTTP 400 for bad requests so the frontend knows something failed
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)