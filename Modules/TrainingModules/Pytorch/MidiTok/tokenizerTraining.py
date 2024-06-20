from miditok import REMI, TokenizerConfig  # here we choose to use REMI
from pathlib import Path

# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
    # "save_to_json": "tokenizer.json"
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)



# Tokenize a MIDI file
tokens = tokenizer(Path('FurElise.mid'))  # automatically detects Score objects, paths, tokens

# print(tokens) 

# Convert to MIDI and save it
generated_midi = tokenizer(tokens)  # MidiTok can handle PyTorch/Numpy/Tensorflow tensors
generated_midi.dump_midi(Path("decoded_midi.mid"))

from miditok import REMI
from pathlib import Path

# Creates the tokenizer and list the file paths
tokenizer = REMI()  # using defaults parameters (constants.py)
# midi_paths = list(Path("Assets/Datasets/LakhMidiClean/").glob("**/*.mid"))
midi_paths = list(Path("Assets/Datasets/LakhMidiClean/3T/").glob("**/*.mid"))

# Builds the vocabulary with BPE
tokenizer.train(vocab_size=283, files_paths=midi_paths)

tokenizer.save("tokenizer.json")





