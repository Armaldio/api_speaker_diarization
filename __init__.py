import fastapi
import uvicorn
from fastapi import File, UploadFile, HTTPException
import soundfile as sf
import noisereduce as nr
import torch
import torchaudio
from pyannote.audio import Pipeline
import tempfile
import os
import numpy as np
from typing import List, Dict, Any

# --- Configuration ---
# IMPORTANT: Set your Hugging Face access token here or as an environment variable
# e.g., HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
# Ensure you have accepted the terms of use for:
# 1. pyannote/speaker-diarization-3.1 (or the specific model you use)
# 2. pyannote/segmentation-3.0 (dependency)
# on the Hugging Face Hub.
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE") # <-- REPLACE WITH YOUR ACTUAL TOKEN or use os.getenv

if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
    print("WARNING: Hugging Face token not set. Diarization will likely fail.")
    print("Please replace 'YOUR_HF_TOKEN_HERE' with your actual token or set the HF_TOKEN environment variable.")

# Initialize FastAPI app
app = fastapi.FastAPI(
    title="Speaker Diarization API",
    description="Upload an audio file to perform noise reduction and speaker diarization.",
)

# --- Global Variables & Model Loading ---
# Load the diarization pipeline (once at startup)
# This can take a few moments
DIARIZATION_PIPELINE = None
TARGET_SAMPLE_RATE = 16000  # Pyannote models typically expect 16kHz

try:
    if HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN_HERE":
        print("Initializing Pyannote Diarization Pipeline...")
        # Using a specific model version known to work well.
        # You can explore other models like 'pyannote/speaker-diarization-3.1'
        DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-pytorch",  # This is an older but stable alias
                                                # For newer versions, try "pyannote/speaker-diarization-3.1"
                                                # Ensure you have agreed to the terms on Hugging Face Hub
            use_auth_token=HF_TOKEN
        )
        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            DIARIZATION_PIPELINE = DIARIZATION_PIPELINE.to(torch.device("cuda"))
            print("Diarization pipeline moved to GPU.")
        else:
            print("Diarization pipeline running on CPU.")
        print("Pyannote Diarization Pipeline initialized successfully.")
    else:
        print("Skipping pipeline initialization due to missing HF_TOKEN.")
except Exception as e:
    print(f"Error initializing Pyannote Diarization Pipeline: {e}")
    DIARIZATION_PIPELINE = None # Ensure it's None if initialization fails


# --- Helper Functions ---

def preprocess_audio(audio_data: np.ndarray, original_sr: int) -> torch.Tensor:
    """
    Preprocesses audio data:
    1. Converts to mono by averaging channels if stereo.
    2. Resamples to the target sample rate (16kHz).
    3. Converts to a PyTorch tensor.
    """
    if audio_data.ndim > 1 and audio_data.shape[1] > 1: # Check if stereo
        # Convert to mono by averaging channels
        audio_data = np.mean(audio_data, axis=1)

    # Convert to PyTorch tensor
    audio_tensor = torch.from_numpy(audio_data).float()

    # Resample if necessary
    if original_sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=TARGET_SAMPLE_RATE)
        audio_tensor = resampler(audio_tensor)

    # Ensure waveform is 2D: (1, num_samples) for pyannote
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    return audio_tensor

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    Here, we mainly ensure the global pipeline variable is checked.
    """
    if DIARIZATION_PIPELINE is None and (HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN_HERE"):
        print("Re-attempting pipeline initialization on startup if it failed globally...")
        # This is a fallback, ideally it initializes correctly above.
        # Consider more robust initialization or error handling for production.
        try:
            global DIARIZATION_PIPELINE
            DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-pytorch",
                use_auth_token=HF_TOKEN
            )
            if torch.cuda.is_available():
                DIARIZATION_PIPELINE = DIARIZATION_PIPELINE.to(torch.device("cuda"))
            print("Diarization Pipeline initialized successfully on startup.")
        except Exception as e:
            print(f"Critical Error: Pyannote Diarization Pipeline could not be initialized on startup: {e}")
            # You might want to prevent the app from starting or run in a degraded mode
    elif not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN_HERE":
         print("Startup: HF_TOKEN not configured. Diarization endpoint will not function correctly.")


@app.post("/diarize/", summary="Perform Speaker Diarization", response_model=List[Dict[str, Any]])
async def diarize_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file, performs noise reduction, and speaker diarization.

    Returns a list of speaker segments with timestamps.
    Example: `[{"speaker": "SPEAKER_00", "start": 0.5, "end": 2.3}, ...]`
    """
    if DIARIZATION_PIPELINE is None:
        raise HTTPException(
            status_code=503,
            detail="Diarization pipeline is not available. Check server logs for initialization errors (e.g., Hugging Face token)."
        )

    # Check file type (optional, but good practice)
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_audio_file:
            content = await file.read()
            tmp_audio_file.write(content)
            tmp_audio_file_path = tmp_audio_file.name

        # 1. Load audio using soundfile
        try:
            audio_data, sample_rate = sf.read(tmp_audio_file_path, dtype='float32')
        except Exception as e:
            os.unlink(tmp_audio_file_path) # Clean up temp file
            raise HTTPException(status_code=400, detail=f"Could not read audio file: {e}. Ensure it's a valid audio format (e.g., WAV, FLAC, MP3 with ffmpeg).")

        # 2. Noise Reduction
        # Ensure audio_data is 1D for noisereduce if it's mono after sf.read
        if audio_data.ndim > 1 and audio_data.shape[1] == 1: # Mono but 2D
            audio_data_nr = audio_data[:,0]
        elif audio_data.ndim > 1 and audio_data.shape[1] > 1: # Stereo
             audio_data_nr = np.mean(audio_data, axis=1) # Convert to mono for noise reduction
        else: # Already mono 1D
            audio_data_nr = audio_data

        print(f"Original audio shape: {audio_data.shape}, Sample rate: {sample_rate}")
        print(f"Audio shape for noise reduction: {audio_data_nr.shape}")

        # Perform noise reduction
        # Note: noisereduce works best on stationary noise.
        # Parameters might need tuning based on your audio.
        reduced_noise_audio = nr.reduce_noise(y=audio_data_nr, sr=sample_rate, stationary=True)
        print(f"Audio shape after noise reduction: {reduced_noise_audio.shape}")

        # 3. Preprocess for Pyannote (resample, convert to tensor)
        # The reduced_noise_audio is already mono here.
        waveform_tensor = preprocess_audio(reduced_noise_audio, sample_rate)
        print(f"Waveform tensor shape for diarization: {waveform_tensor.shape}, Target SR: {TARGET_SAMPLE_RATE}")

        # Move tensor to GPU if pipeline is on GPU
        if torch.cuda.is_available() and DIARIZATION_PIPELINE.device.type == "cuda":
            waveform_tensor = waveform_tensor.to(DIARIZATION_PIPELINE.device)

        # 4. Perform Speaker Diarization
        # The input to the pipeline should be a dictionary
        audio_input_for_pipeline = {"waveform": waveform_tensor, "sample_rate": TARGET_SAMPLE_RATE}

        print("Starting diarization process...")
        diarization = DIARIZATION_PIPELINE(audio_input_for_pipeline)
        print("Diarization complete.")

        # 5. Format results
        results = []
        # The `diarization` object is an Annotation. We iterate over its segments.
        # `itertracks` yields (segment, track_name, speaker_label)
        # `yield_label=True` ensures speaker_label is the actual speaker ID (e.g., SPEAKER_00)
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": round(segment.start, 3), # seconds
                "end": round(segment.end, 3)      # seconds
            })

        return results

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        # Log the full error for debugging
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during processing: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'tmp_audio_file_path' in locals() and os.path.exists(tmp_audio_file_path):
            os.unlink(tmp_audio_file_path)

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Speaker Diarization API. Use the /docs endpoint for API documentation."}

# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    # It's better to run uvicorn from the command line for more control:
    # uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # However, this allows running with `python main.py` for simplicity in some cases.
    print("Starting Uvicorn server. Access API at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
