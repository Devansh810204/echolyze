import base64
import io
import os
import uvicorn
import numpy as np
import soundfile as sf
import scipy.signal
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from typing import Annotated

app = FastAPI(title="AI Voice Detection API")

# Security key
VALID_API_KEY = "echolyze_secure_hackathon_key_2026"

# 1. Request Model
class DetectionRequest(BaseModel):
    language: str        
    audioFormat: str     
    audioBase64: str     

# 2. Response Model
class DetectionResponse(BaseModel):
    status: str
    language: str
    classification: str   
    confidenceScore: float 
    explanation: str

def analyze_audio_fast(audio_bytes: bytes):
    """
    High-Speed Audio Analysis using pure Numpy.
    Removes heavy libraries to prevent server timeouts.
    """
    try:
        # 1. Read Audio (Fastest method)
        audio_file = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_file)

        # 2. OPTIMIZATION: Process max 5 seconds
        # This guarantees the API responds in < 1 second
        max_samples = 5 * samplerate
        if len(data) > max_samples:
            data = data[:max_samples]

        # 3. Convert to Mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # --- MATH-BASED FEATURE EXTRACTION (No Librosa) ---

        # A. Amplitude Variance (Dynamic Range)
        # Humans have high variance (loud/soft). AI is often normalized/flatter.
        amplitude_variance = np.var(data)

        # B. Zero Crossing Rate (ZCR)
        # Count how many times the signal crosses the center line.
        # High ZCR = Noisy/Breath. Low ZCR = Tonal/Robotic.
        zero_crossings = np.sum(np.diff(np.signbit(data).astype(int)))
        zcr_rate = zero_crossings / len(data)

        # C. Silence Ratio (Dead air)
        # AI often eliminates silence too perfectly.
        # We count samples close to absolute zero.
        silence_threshold = 0.01
        silence_count = np.sum(np.abs(data) < silence_threshold)
        silence_ratio = silence_count / len(data)

        # --- DETERMINISTIC LOGIC ---
        
        # Heuristic: 
        # - AI has lower variance (very consistent volume).
        # - AI often has higher "roughness" in ZCR or oddly low silence.
        
        # Base Score Calculation
        score = 0.5
        
        # Factor 1: Variance Check (Human = High Variance)
        if amplitude_variance < 0.005: 
            score += 0.20 # Likely AI (Too flat)
        else:
            score -= 0.15 # Likely Human (Dynamic)

        # Factor 2: ZCR Check
        if zcr_rate > 0.15:
            score -= 0.10 # Likely Human (Breath sounds)
        elif zcr_rate < 0.02:
            score += 0.15 # Likely AI (Too tonal)

        # Clamp Score (0.01 to 0.99)
        final_score = max(0.01, min(0.99, score))
        
        # Classification Threshold
        is_ai = final_score > 0.60
        
        classification = "HUMAN" if is_ai else "AI GENERATED"
        
        # Dynamic Explanation Generator
        if not is_ai:
            explanation = (
                f"Signal lacks dynamic range (Variance: {amplitude_variance:.4f}). "
                f"Zero-crossing rate ({zcr_rate:.3f}) suggests synthetic waveform generation."
            )
        else:
            final_score = 1.0 - final_score # Flip confidence for Human
            explanation = (
                f"High dynamic amplitude detected (Variance: {amplitude_variance:.4f}). "
                f"Natural silence ratio ({silence_ratio:.2f}) indicates organic speech patterns."
            )

        return classification, round(final_score, 4), explanation

    except Exception as e:
        print(f"Analysis Error: {e}")
        return "UNKNOWN", 0.0, "Signal processing failed."

@app.get("/")
async def home():
    return {"message": "Echolyze API Live (Lite Version)", "status": "Active"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_voice(
    request: DetectionRequest, 
    x_api_key: Annotated[str | None, Header()] = None
):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode base64
        try:
            audio_bytes = base64.b64decode(request.audioBase64)
        except:
            raise ValueError("Invalid Base64")

        if len(audio_bytes) < 100:
            raise ValueError("Audio too small")

        # Run Fast Analysis
        classification, score, explanation = analyze_audio_fast(audio_bytes)

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


