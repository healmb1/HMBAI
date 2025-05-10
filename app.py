import os
import torch
import gradio as gr
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

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

def clone_voice(audio_file, text, language, style, speed):
    # Initialize models
    base_speaker_tts, tone_color_converter, source_se, device = init_models()
    
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract target speaker embedding
    target_se, _ = se_extractor.get_se(audio_file, tone_color_converter, target_dir='processed', vad=True)
    
    # Generate speech
    src_path = f'{output_dir}/tmp.wav'
    base_speaker_tts.tts(text, src_path, speaker=style, language=language, speed=speed)
    
    # Convert tone color
    save_path = f'{output_dir}/output.wav'
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message
    )
    
    return save_path

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="OpenVoice - Voice Cloning") as demo:
        gr.Markdown("# OpenVoice Voice Cloning Demo")
        gr.Markdown("Upload a reference audio file and enter text to generate speech in the same voice.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Reference Audio", type="filepath")
                text_input = gr.Textbox(label="Text to Speak", placeholder="Enter text here...")
                language = gr.Dropdown(
                    choices=["English", "Chinese", "Spanish", "French", "Japanese", "Korean"],
                    value="English",
                    label="Language"
                )
                style = gr.Dropdown(
                    choices=["default", "friendly", "cheerful", "excited", "sad", "angry", "terrified", "shouting", "whispering"],
                    value="default",
                    label="Voice Style"
                )
                speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speech Speed")
                generate_btn = gr.Button("Generate Speech")
            
            with gr.Column():
                audio_output = gr.Audio(label="Generated Speech")
        
        generate_btn.click(
            fn=clone_voice,
            inputs=[audio_input, text_input, language, style, speed],
            outputs=audio_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 