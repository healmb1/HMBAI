import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
from contextlib import asynccontextmanager

# Create thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Create temporary directory for processing
temp_dir = tempfile.mkdtemp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    shutil.rmtree(temp_dir)

app = FastAPI(
    title="OpenVoice API",
    description="Voice Cloning API powered by OpenVoice",
    lifespan=lifespan
)

# Initialize models with caching
@lru_cache(maxsize=1)
def init_models():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize base speaker TTS
    ckpt_base = 'checkpoints/base_speakers/EN'
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
    
    # Initialize tone color converter
    ckpt_converter = 'checkpoints/converter'
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    
    # Load source speaker embedding
    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
    
    return base_speaker_tts, tone_color_converter, source_se, device

# Initialize models at startup
base_speaker_tts, tone_color_converter, source_se, device = init_models()

async def process_audio(reference_path: str, text: str, style: str, language: str, speed: float):
    # Extract target speaker embedding
    target_se, _ = se_extractor.get_se(reference_path, tone_color_converter, target_dir='processed', vad=True)
    
    # Generate speech
    src_path = os.path.join(temp_dir, 'tmp.wav')
    base_speaker_tts.tts(text, src_path, speaker=style, language=language, speed=speed)
    
    # Convert tone color
    output_path = os.path.join(temp_dir, 'output.wav')
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message=encode_message
    )
    
    return output_path

@app.post("/clone-voice")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    text: str = Form(...),
    style: str = Form("default"),
    language: str = Form("English"),
    speed: float = Form(1.0)
):
    """
    Clone voice from reference audio and generate speech from text.
    
    Parameters:
    - reference_audio: Audio file containing the voice to clone
    - text: Text to convert to speech
    - style: Voice style (default, friendly, cheerful, excited, sad, angry, terrified, shouting, whispering)
    - language: Language of the output speech (English, Chinese, Spanish, French, Japanese, Korean)
    - speed: Speech speed (0.5 to 2.0)
    """
    try:
        # Save uploaded audio file to temporary directory
        reference_path = os.path.join(temp_dir, 'reference.wav')
        with open(reference_path, "wb") as f:
            f.write(await reference_audio.read())
        
        # Process audio in thread pool
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            thread_pool,
            lambda: asyncio.run(process_audio(reference_path, text, style, language, speed))
        )
        
        # Clean up reference file
        os.remove(reference_path)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="cloned_voice.wav"
        )
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "Welcome to OpenVoice API",
        "endpoints": {
            "/clone-voice": "POST - Clone voice from reference audio and generate speech from text"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "voice_clone_api:app",
        host="127.0.0.1",
        port=8000,
        workers=4,
        reload=True
    ) 