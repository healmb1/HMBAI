# Simple Voice Cloning System

This is a simplified voice cloning system that allows you to generate speech in someone else's voice. The system consists of three main components:

1. Voice Encoder: Extracts voice characteristics from a reference audio file
2. Text-to-Speech Model: Generates speech from text and voice characteristics
3. Audio Converter: Converts the generated mel spectrogram to audio

## Installation

1. Create a new conda environment:
```bash
conda create -n voice_clone python=3.9
conda activate voice_clone
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:7860)

3. In the web interface:
   - Upload a reference audio file (WAV format recommended)
   - Enter the text you want to convert to speech
   - Click "Generate Speech"

## How it Works

1. The voice encoder extracts a voice embedding from the reference audio
2. The text-to-speech model combines the text and voice embedding to generate a mel spectrogram
3. The audio converter transforms the mel spectrogram into audio

## Limitations

This is a simplified version of voice cloning and has several limitations:

1. The voice quality may not be as good as commercial solutions
2. The system requires a clear reference audio file
3. The generated speech may not perfectly match the reference voice
4. The system works best with English text

## Future Improvements

1. Add a proper vocoder for better audio quality
2. Implement more advanced voice encoding techniques
3. Add support for more languages
4. Improve the text-to-speech model architecture
5. Add voice style control (emotion, speaking rate, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 