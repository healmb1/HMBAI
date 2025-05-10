import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=80):
        super(Decoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.projection(x)
        return x

class TTSModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super(TTSModel, self).__init__()
        
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim * 2 + embedding_dim, hidden_dim)
        
    def forward(self, text, voice_embedding):
        # Encode text
        text_features = self.text_encoder(text)
        
        # Expand voice embedding to match text length
        batch_size, seq_len, _ = text_features.shape
        voice_embedding = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate text features and voice embedding
        combined = torch.cat([text_features, voice_embedding], dim=-1)
        
        # Decode to mel spectrogram
        mel_spec = self.decoder(combined)
        
        return mel_spec

def text_to_sequence(text, char_to_id):
    """Convert text to sequence of character IDs."""
    sequence = [char_to_id.get(c, char_to_id['<unk>']) for c in text]
    return torch.tensor(sequence).unsqueeze(0)  # Add batch dimension

def generate_speech(model, text, voice_embedding, char_to_id, device='cuda'):
    """
    Generate speech from text and voice embedding.
    
    Args:
        model (TTSModel): The TTS model
        text (str): Input text
        voice_embedding (torch.Tensor): Voice embedding
        char_to_id (dict): Character to ID mapping
        device (str): Device to run the model on
    
    Returns:
        torch.Tensor: Generated mel spectrogram
    """
    model = model.to(device)
    voice_embedding = voice_embedding.to(device)
    
    # Convert text to sequence
    sequence = text_to_sequence(text, char_to_id).to(device)
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel_spec = model(sequence, voice_embedding)
    
    return mel_spec 