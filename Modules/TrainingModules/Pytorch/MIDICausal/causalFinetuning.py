# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = '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'



from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load Existing Model

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")



# Load Dataset

import miditoolkit

def midi_get_notes(midi_path):
    midi_obj = miditoolkit.MidiFile(midi_path)
    notes = midi_obj.instruments[0].notes
    tokens = []
    for note in notes:
        tokens.append((note.start, note.pitch, note.velocity, note.end - note.start))
    tokens.sort(key=lambda x: x[0])  # Ensure the tokens are in order of time
    return tokens

import numpy as np
import transformers
from typing import List, Dict, Union, Optional

import torch
class MIDITokenizer:
    pad_token = 0
    pad_token_id = 0
    eos_token = 0

    def __init__(self):
        self.pitch_offset = 21  # MIDI note number for A0
        self.num_pitches = 88   # Number of piano keys (A0 to C8)
        self.model_input_names = ['input_ids']

    def encode(self, notes):
        tokens = []
        for note in notes:
            start, pitch, velocity, duration = note
            # tokens.append(start)
            # tokens.append(pitch - self.pitch_offset)
            # tokens.append(velocity)
            # tokens.append(duration)
            tokens.append(0)
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
    
    # def _pad(self, encoded_inputs: Dict[str, Union[List[int], List[List[int]]]], max_length: int,
    #          padding_strategy: str, pad_to_multiple_of: Optional[int] = None,
    #          return_attention_mask: Optional[bool] = None) -> Dict[str, Union[List[int], List[List[int]]]]:
    #     # required_input = encoded_inputs[self.model_input_names[0]]
    #     # if padding_strategy == 'max_length':
    #     #     padded_sequences = []
    #     #     attention_masks = []
    #     #     for seq in required_input:
    #     #         padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
    #     #         attention_mask = [1] * len(seq) + [0] * (max_length - len(seq))
    #     #         padded_sequences.append(padded_seq)
    #     #         attention_masks.append(attention_mask)
    #     #     encoded_inputs['input_ids'] = padded_sequences
    #     #     if return_attention_mask:
    #     #         encoded_inputs['attention_mask'] = attention_masks
    #     # return encoded_inputs

    #     encoded_inputs.append("special_tokens_mask")
    #     return encoded_inputs

    # def pad(self, encoded_inputs: Union[Dict[str, List[int]], List[Dict[str, List[int]]]],
    #         padding: Union[bool, str] = True,
    #         max_length: Optional[int] = None,
    #         pad_to_multiple_of: Optional[int] = None,
    #         return_attention_mask: Optional[bool] = None,
    #         return_tensors: Optional[str] = None,
    #         verbose: bool = True) -> Dict[str, Union[List[int], List[List[int]]]]:

    #     encoded_inputs.append("special_tokens_mask")

    #     return encoded_inputs
    #     # # If we have a list of dicts, let's convert it in a dict of lists
    #     # if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], dict):
    #     #     encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

    #     # # The model's main input name, usually `input_ids`, has to be passed for padding
    #     # if self.model_input_names[0] not in encoded_inputs:
    #     #     raise ValueError(
    #     #         f"You should supply an encoding or a list of encodings to this method that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
    #     #     )

    #     # required_input = encoded_inputs[self.model_input_names[0]]
    #     # if required_input is None or (isinstance(required_input, list) and len(required_input) == 0):
    #     #     if return_attention_mask:
    #     #         encoded_inputs["attention_mask"] = []
    #     #     return encoded_inputs

    #     # # Determine padding strategy
    #     # if isinstance(padding, bool) and padding:
    #     #     padding_strategy = 'longest'
    #     # elif padding == 'max_length':
    #     #     padding_strategy = 'max_length'
    #     # elif not padding:
    #     #     padding_strategy = 'do_not_pad'
    #     # else:
    #     #     raise ValueError("Invalid padding strategy")

    #     # if padding_strategy == 'longest':
    #     #     max_length = max(len(inputs) for inputs in required_input)
    #     #     padding_strategy = 'max_length'

    #     # encoded_inputs = self._pad(
    #     #     encoded_inputs,
    #     #     max_length=max_length,
    #     #     padding_strategy=padding_strategy,
    #     #     pad_to_multiple_of=pad_to_multiple_of,
    #     #     return_attention_mask=return_attention_mask
    #     # )

    #     # # Convert to tensors if required
    #     # if return_tensors is not None:
    #     #     for key in encoded_inputs:
    #     #         if return_tensors == 'pt':
    #     #             encoded_inputs[key] = torch.tensor(encoded_inputs[key], dtype=torch.long)
    #     #         elif return_tensors == 'np':
    #     #             encoded_inputs[key] = np.array(encoded_inputs[key])
    #     #         elif return_tensors == 'tf':
    #     #             import tensorflow as tf
    #     #             encoded_inputs[key] = tf.constant(encoded_inputs[key])
    #     #         else:
    #     #             raise ValueError(f"Unsupported return_tensors type: {return_tensors}")

    #     # return encoded_inputs

# Example usage
midi_path = 'FurElise.mid'
tokens = midi_get_notes(midi_path)
tokenizer = MIDITokenizer()
encoded_tokens = tokenizer.encode(tokens)

from datasets import Dataset, DatasetDict

# Convert the array of token arrays to a Dataset
dataset = Dataset.from_dict({"input_ids": [encoded_tokens]})

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": dataset,
    "test": dataset
})



# train = np.array(encoded_tokens)
# test = np.array(encoded_tokens)



# import torch
# from torch.utils.data import TensorDataset, DataLoader

# Convert the list of token indices to a tensor
# token_tensor = torch.tensor(encoded_tokens * 10000)
# token_tensor = token_tensor.unsqueeze(1)
# Create a TensorDataset
# dataset = TensorDataset(token_tensor)



# from datasets import load_dataset

# eli5 = load_dataset("eli5_category", split="train[:5000]")
# eli5 = eli5.train_test_split(test_size=0.2)

# from transformers import AutoTokenizer
# # https://huggingface.co/docs/transformers/en/model_doc/gpt2
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

# def preprocess_function(examples):
#     return tokenizer([" ".join(x) for x in examples["answers"]])

# tokenized_eli5 = eli5.map(
#     preprocess_function,
#     batched=True,
#     num_proc=4,
#     remove_columns=eli5["train"].column_names,
# )

# block_size = 128
# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#     # customize this part to your needs.
#     if total_length >= block_size:
#         total_length = (total_length // block_size) * block_size
#     # Split by chunks of block_size.
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

# lm_dataset = dataset_dict.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

# tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# Train

training_args = TrainingArguments(
    output_dir="midi-causal-transformer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    # data_collator=data_collator,
)

trainer.train()


import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub("midi-causal-transformer")
tokenizer.push_to_hub("midi-causal-transformer")



 






# Inference

prompt = "Somatic hypermutation allows the immune system to"


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("progz/midi-causal-transformer")
inputs = tokenizer(prompt, return_tensors="pt").input_ids


from transformers import pipeline

generator = pipeline("text-generation", model="progz/midi-causal-transformer")
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generator(prompt)






from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("progz/midi-causal-transformer")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


