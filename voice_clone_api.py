import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

app = FastAPI(title="OpenVoice API", description="Voice Cloning API powered by OpenVoice")

# Initialize models
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

# Create output directory
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

class CloneRequest(BaseModel):
    text: str
    style: str = "default"
    language: str = "English"
    speed: float = 1.0

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
        # Save uploaded audio file
        reference_path = f"{output_dir}/reference.wav"
        with open(reference_path, "wb") as f:
            f.write(await reference_audio.read())
        
        # Extract target speaker embedding
        target_se, _ = se_extractor.get_se(reference_path, tone_color_converter, target_dir='processed', vad=True)
        
        # Generate speech
        src_path = f'{output_dir}/tmp.wav'
        base_speaker_tts.tts(text, src_path, speaker=style, language=language, speed=speed)
        
        # Convert tone color
        output_path = f'{output_dir}/output.wav'
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=output_path,
            message=encode_message
        )
        
        # Clean up temporary files
        os.remove(reference_path)
        os.remove(src_path)
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 