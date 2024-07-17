# https://github.com/Natooz/MidiTok

from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Train the tokenizer with Byte Pair Encoding (BPE)
files_paths = list(Path("C:/Users/Utilisateur/Desktop/Research/MusicGenerationFramework/Assets/Datasets/Maestro").glob("**/*.midi"))
tokenizer.train(vocab_size=30000, files_paths=files_paths)
tokenizer.save(Path("MIDITok", "tokenizer.json"))
# And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
tokenizer.push_to_hub("Progz/MidiTokTokenizer", private=True, token="")

# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path("MIDITOK", "midi_chunks")
split_files_for_training(
    files_paths=files_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=1024,
)

# Create a Dataset, a DataLoader and a collator to train a model
dataset = DatasetMIDI(
    files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

# Iterate over the dataloader to train a model
for batch in dataloader:
    print("Train your model on this batch...")