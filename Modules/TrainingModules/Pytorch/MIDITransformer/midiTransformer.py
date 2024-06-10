import miditoolkit
import numpy as np

def midi_to_tokens(midi_path):
    midi_obj = miditoolkit.MidiFile(midi_path)
    notes = midi_obj.instruments[0].notes
    tokens = []
    for note in notes:
        tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
    tokens.sort()  # Ensure the tokens are in order of time
    return tokens

def tokens_to_midi(tokens, output_path):
    midi_obj = miditoolkit.MidiFile()
    track = miditoolkit.Instrument(program=0, is_drum=False, name='piano')
    for token in tokens:
        start, pitch, velocity, duration = token
        note = miditoolkit.Note(
            start=start, end=start+duration, pitch=pitch, velocity=velocity)
        track.notes.append(note)
    midi_obj.instruments.append(track)
    midi_obj.dump(output_path)





import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)
        
    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_length, :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=-1)







def train(model, data_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            optimizer.zero_grad()
            input_seq, target_seq = batch[0], batch[1]
            output = model(input_seq)
            loss = criterion(output.view(-1, output.shape[-1]), target_seq.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Example usage:
# Define the hyperparameters
input_dim = 128  # assuming MIDI note numbers
model_dim = 512
num_heads = 8
num_layers = 6
output_dim = 128
max_seq_length = 1024

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim, max_seq_length)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()













import miditoolkit

def midi_to_tokens(midi_path):
    midi_obj = miditoolkit.MidiFile(midi_path)
    notes = midi_obj.instruments[0].notes
    tokens = []
    for note in notes:
        tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
    tokens.sort(key=lambda x: x[0])  # Ensure the tokens are in order of time
    return tokens

class MIDITokenizer:
    def __init__(self):
        self.pitch_offset = 21  # MIDI note number for A0
        self.num_pitches = 88   # Number of piano keys (A0 to C8)

    def encode(self, notes):
        tokens = []
        for note in notes:
            start, pitch, velocity, duration = note
            tokens.append(start)
            tokens.append(pitch - self.pitch_offset)
            tokens.append(velocity)
            tokens.append(duration)
        return tokens

    def decode(self, tokens):
        notes = []
        for i in range(0, len(tokens), 4):
            start = tokens[i]
            pitch = tokens[i+1] + self.pitch_offset
            velocity = tokens[i+2]
            duration = tokens[i+3]
            notes.append((start, pitch, velocity, duration))
        return notes

# Example usage
midi_path = 'FurElise.mid'
tokens = midi_to_tokens(midi_path)
tokenizer = MIDITokenizer()
encoded_tokens = tokenizer.encode(tokens)

from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, token_sequences, seq_length=512):
        self.token_sequences = token_sequences
        self.seq_length = seq_length

    def __len__(self):
        return len(self.token_sequences) // self.seq_length

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        return torch.tensor(self.token_sequences[start_idx:end_idx], dtype=torch.long)

# Example usage
dataset = MIDIDataset(encoded_tokens)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
train(model, data_loader, optimizer, criterion, epochs=20)






def generate(model, start_token, max_length=1024):
    model.eval()
    generated = [start_token]
    input_seq = torch.tensor(generated).unsqueeze(0)  # Add batch dimension
    for _ in range(max_length - 1):
        output = model(input_seq)
        next_token = torch.argmax(output[:, -1, :], dim=-1).item()
        generated.append(next_token)
        input_seq = torch.tensor(generated).unsqueeze(0)  # Update input sequence
    return generated

# Example usage:
# start_token = ...
# generated_tokens = generate(model, start_token)
# tokens_to_midi(generated_tokens, 'generated_music.mid')