import torch
import torchaudio
import gradio as gr
from voice_encoder import VoiceEncoder, extract_voice_embedding
from tts_model import TTSModel, generate_speech
import os

# Create character mapping
char_to_id = {c: i for i, c in enumerate(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?')}
char_to_id['<unk>'] = len(char_to_id)

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
voice_encoder = VoiceEncoder().to(device)
tts_model = TTSModel(vocab_size=len(char_to_id)).to(device)

def convert_mel_to_audio(mel_spec):
    """Convert mel spectrogram to audio using Griffin-Lim algorithm."""
    # This is a simplified version - in practice, you'd want to use a proper vocoder
    n_fft = 1024
    hop_length = 256
    
    # Convert mel to linear spectrogram (simplified)
    linear_spec = torch.exp(mel_spec)
    
    # Griffin-Lim algorithm
    audio = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=1.0,
        n_iter=32
    )(linear_spec)
    
    return audio

def clone_voice(audio_file, text):
    """Clone voice from reference audio and generate speech from text."""
    # Extract voice embedding
    voice_embedding = extract_voice_embedding(audio_file, voice_encoder, device)
    
    # Generate mel spectrogram
    mel_spec = generate_speech(tts_model, text, voice_embedding, char_to_id, device)
    
    # Convert to audio
    audio = convert_mel_to_audio(mel_spec)
    
    # Save to temporary file
    output_path = 'output.wav'
    torchaudio.save(output_path, audio, 16000)
    
    return output_path

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Simple Voice Cloning") as demo:
        gr.Markdown("# Simple Voice Cloning Demo")
        gr.Markdown("Upload a reference audio file and enter text to generate speech in the same voice.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Reference Audio", type="filepath")
                text_input = gr.Textbox(label="Text to Speak", placeholder="Enter text here...")
                generate_btn = gr.Button("Generate Speech")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech")
        
        generate_btn.click(
            fn=clone_voice,
            inputs=[audio_input, text_input],
            outputs=audio_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 