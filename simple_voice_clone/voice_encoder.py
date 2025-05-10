import torch
import torch.nn as nn
import torchaudio
import numpy as np

class VoiceEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=256):
        super(VoiceEncoder, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Final projection layer
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        x = x.transpose(1, 2)  # (batch_size, input_dim, time_steps)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_dim)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Project to embedding space
        x = self.projection(x)
        
        # Average pooling over time
        x = torch.mean(x, dim=1)
        
        # L2 normalization
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x

def extract_voice_embedding(audio_path, model, device='cuda'):
    """
    Extract voice embedding from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        model (VoiceEncoder): The voice encoder model
        device (str): Device to run the model on
    
    Returns:
        torch.Tensor: Voice embedding
    """
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Extract mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    
    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0)
    
    # Move to device
    mel_spec = mel_spec.to(device)
    model = model.to(device)
    
    # Extract embedding
    with torch.no_grad():
        embedding = model(mel_spec)
    
    return embedding 