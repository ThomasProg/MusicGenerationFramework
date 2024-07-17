# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = ''

loadFromDisk = False


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load Existing Model

# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")


# from tokenizer import MIDItokenizer
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

tokenizer = REMI(params=Path("MIDITok", "tokenizer.json"))
# tokenizer = MIDItokenizer()


from transformers import AutoModelForCausalLM, GPT2Config

model = None

if (loadFromDisk):
    model = AutoModelForCausalLM.from_pretrained("ModelWithMidiTok/")
else:
    # Define the distilgpt2 model configuration
    # config = GPT2Config(
    #     vocab_size=50257,
    #     n_positions=1024,
    #     n_ctx=1024,
    #     n_embd=400,
    #     n_layer=6,  # distilgpt2 has fewer layers than GPT-2
    #     n_head=10,
    #     bos_token_id=50256,
    #     eos_token_id=50256
    # )

    config = GPT2Config(
        vocab_size=30000,#tokenizer.nbTokens,#721+128+5000,#50257, # min : 721
        n_positions=1024,
        n_ctx=1024,
        n_embd=400,
        n_layer=6,  # distilgpt2 has fewer layers than GPT-2
        n_head=10,
        n_inner=400*4,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.pad_token_id
    )

    # Instantiate the model from the configuration
    model = AutoModelForCausalLM.from_config(config)





# Load Dataset
from datasets import DatasetDict, Dataset

import os
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict

import miditoolkit

class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, tokenizer, transform=None):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    self.file_paths.append(os.path.join(subdir, file))
        self.transform = transform

    def __len__(self):
        return len(self.file_paths) // 10

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # midiFile = miditoolkit.MidiFile(file_path)

        # newTicksPerBeat = 4*4#96 #4*4*3
        # for note in midiFile.instruments[0].notes:
        #     note.start = round(note.start * newTicksPerBeat / midiFile.ticks_per_beat)
        #     note.end = round(note.end * newTicksPerBeat / midiFile.ticks_per_beat)
        # midiFile.ticks_per_beat = newTicksPerBeat
        
        # if self.transform:
        #     midiFile = self.transform(midiFile)
        # tokens = self.tokenizer(midiFile)

        tokens = tokenizer(file_path)

        return {"input_ids": tokens}




# midi_path = "K545 Piano Sonata.mid" # 'FurElise.mid'
# midi_path = "Andre_Andante.mid"
# # midi_path = "Assets/Datasets/LakhMidiClean/Ludwig_van_Beethoven/5th_Symphony.mid"

# midiFile = miditoolkit.MidiFile(midi_path)
# multSet = set()
# for change in midiFile.time_signature_changes:
#     multSet.add(change.denominator)
#     multSet.add(change.numerator)
# mult = 1
# for m in multSet:
#     mult *= m
# # assert(len(midiFile.time_signature_changes) == 1)
# # timeSign = midiFile.time_signature_changes[0]
    
# print("original:")
# newTicksPerBeat = mult # 4*4=16
# for note in midiFile.instruments[0].notes:
#     note.start = round(note.start * newTicksPerBeat / midiFile.ticks_per_beat)
#     note.end = round(note.end * newTicksPerBeat / midiFile.ticks_per_beat)
#     # print("%d / %d" % (note.start, note.end))
#     print(note.pitch, end=" ")
# midiFile.ticks_per_beat = newTicksPerBeat

# encodedMIDI = tokenizer.encode(midiFile)
# midiFile = tokenizer.decode(encodedMIDI)

# print("decoded:")
# for note in midiFile.instruments[0].notes:
#     print(note.pitch, end=" ")

# # encodedMIDI = []
# midiFile.ticks_per_beat = newTicksPerBeat
# midiFile.dump("test.mid")


dataset = MIDIDataset("Assets/Datasets/maestro-v3.0.0", tokenizer)

from datasets import Features, Value, Sequence
# Define your features according to your data structure
features = Features({
    "features": Value(dtype="string"),  # Replace with your actual feature
    # Add other features as needed
})

def gen():
    for idx in range(len(dataset)):
        yield dataset[idx]  # this has to be a dictionary

# def gen():
#     # Simulate the actual dataset with a placeholder
#     dataset = [{"features": f"example {i}"} for i in range(10)]  # Replace this with your actual dataset

#     for idx in range(len(dataset)):
#         yield dataset[idx]  # Yield each dictionary item

# dset2 = HFDataset.from_generator(gen, features=features)
dset2 = HFDataset.from_generator(gen)
# train_dataset = Dataset.from_dict({"input_ids": [encodedMIDI], "attention_mask": [[1] * len(encodedMIDI)]})
# test_dataset = train_dataset

# datasetDict =  DatasetDict({'train': train_dataset, 'test': test_dataset})
datasetDict =  DatasetDict({'train': dset2, 'test': dset2})

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = datasetDict.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# Train

training_args = TrainingArguments(
    output_dir="ModelWithMidiTok",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs = 30,
    fp16 = True # To go faster
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    # data_collator=data_collator,
)

trainer.train()


import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub("ModelWithMidiTok")
# tokenizer.push_to_hub("MIDICausalFinetuning4")

