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
tokens = midi_get_notes(midi_path)
tokenizer = MIDITokenizer()
encoded_tokens = tokenizer.encode(tokens)


from datasets import load_dataset

eli5 = load_dataset("eli5_category", split="train[:5000]")
eli5 = eli5.train_test_split(test_size=0.2)

from transformers import AutoTokenizer
# https://huggingface.co/docs/transformers/en/model_doc/gpt2
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers"]])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)

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

lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# Train

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()


import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub("my-awesome-model")
tokenizer.push_to_hub("my-awesome-model")



 






# Inference

prompt = "Somatic hypermutation allows the immune system to"


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("progz/my-awesome-model")
inputs = tokenizer(prompt, return_tensors="pt").input_ids


from transformers import pipeline

generator = pipeline("text-generation", model="progz/my-awesome-model")
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generator(prompt)






from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("progz/my-awesome-model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


