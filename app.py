# main.py
import os
import tempfile
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Hugging Face Token Configuration ---
# Pyannote.audio models require a Hugging Face authentication token.
# 1. Go to hf.co/settings/tokens
# 2. Generate a new token with 'read' access.
# 3. Create a .env file in the same directory as this script and add:
#    HF_AUTH_TOKEN="YOUR_HUGGING_FACE_TOKEN_HERE"
#    Or set it as an environment variable directly in your deployment environment.
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
if not HF_AUTH_TOKEN:
    raise ValueError(
        "Hugging Face authentication token (HF_AUTH_TOKEN) not found. "
        "Please set it in a .env file or as an environment variable. "
        "See comments in main.py for instructions."
    )

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Speaker Diarization API",
    description="An API to perform speaker diarization on audio files using pyannote.audio.",
    version="1.0.0"
)

# --- Pyannote.audio Pipeline Loading ---
# Load the pre-trained pyannote.audio pipeline once when the application starts.
# This avoids reloading the model for every request, which is crucial for performance.
# Using 'pyannote/speaker-diarization-3.1' for state-of-the-art diarization.
try:
    diarization_pipeline = Pipeline(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_AUTH_TOKEN
    )
    # Move the pipeline to GPU if available for faster processing
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
        print("Pyannote pipeline moved to GPU.")
    else:
        print("Pyannote pipeline running on CPU.")
except Exception as e:
    print(f"Error loading pyannote.audio pipeline: {e}")
    print("Please ensure your Hugging Face token is valid and you have accepted the model's terms of use.")
    diarization_pipeline = None # Set to None to indicate failure to load

# --- Response Model for Diarization Results ---
class SpeakerSegment(BaseModel):
    speaker: str
    start_time: float
    end_time: float

class DiarizationResult(BaseModel):
    segments: List[SpeakerSegment]

# --- Health Check Endpoint ---
@app.get("/health", summary="Health Check")
async def health_check():
    """
    Checks the health of the API and if the pyannote pipeline is loaded.
    """
    if diarization_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pyannote pipeline not loaded. Check server logs for errors."
        )
    return {"status": "ok", "message": "API is running and pyannote pipeline is loaded."}

# --- Main Diarization Endpoint ---
@app.post(
    "/diarize_audio",
    response_model=DiarizationResult,
    summary="Perform Speaker Diarization",
    description="Receives an audio file, performs speaker diarization using pyannote.audio, "
                "and returns the detected speakers with their timestamps. "
                "Note: This API does not perform background noise reduction. "
                "For noise reduction, consider pre-processing the audio with other tools."
)
async def diarize_audio(audio_file: UploadFile = File(...)):
    """
    Processes an uploaded audio file to identify different speakers and their speech segments.

    Args:
        audio_file (UploadFile): The audio file to be processed.
                                 Supported formats typically include WAV, FLAC, MP3.

    Returns:
        DiarizationResult: A JSON object containing a list of speaker segments,
                           each with a speaker label, start time, and end time.

    Raises:
        HTTPException: If the pyannote pipeline is not loaded, or if there's an error
                       during file processing or diarization.
    """
    if diarization_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Speaker diarization service is not available. Pipeline failed to load."
        )

    # Validate file type (basic check, pyannote handles many formats)
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {audio_file.content_type}. Please upload an audio file."
        )

    # Create a temporary file to save the uploaded audio
    # Using NamedTemporaryFile ensures a unique file name and handles cleanup
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        try:
            # Read the uploaded file content and write it to the temporary file
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        except Exception as e:
            # Ensure the temporary file is cleaned up even if writing fails
            if os.path.exists(tmp_file.name):
                os.remove(tmp_file.name)
            raise HTTPException(status_code=500, detail=f"Error saving audio file: {e}")

    try:
        # Perform speaker diarization
        # The pipeline expects a path to the audio file or a dictionary with 'uri' and 'audio' keys.
        # We provide the path to the temporary file.
        diarization = diarization_pipeline(tmp_file_path)

        segments: List[SpeakerSegment] = []
        # Iterate through the diarization result to extract segments
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=label,
                    start_time=round(segment.start, 2), # Round to 2 decimal places for cleaner output
                    end_time=round(segment.end, 2)
                )
            )
        return DiarizationResult(segments=segments)

    except Exception as e:
        # Catch any errors during diarization and return a 500 error
        raise HTTPException(status_code=500, detail=f"Error during speaker diarization: {e}")
    finally:
        # Ensure the temporary file is deleted after processing
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
            print(f"Cleaned up temporary file: {tmp_file_path}")

