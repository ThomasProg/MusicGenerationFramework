# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = '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'

loadFromDisk = False


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load Existing Model

# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

from transformers import AutoModelForCausalLM, GPT2Config

model = None

if (loadFromDisk):
    model = AutoModelForCausalLM.from_pretrained("MIDICausalFinetuning2/")
else:
    # Define the distilgpt2 model configuration
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=400,
        n_layer=6,  # distilgpt2 has fewer layers than GPT-2
        n_head=10,
        bos_token_id=50256,
        eos_token_id=50256
    )

    # Instantiate the model from the configuration
    model = AutoModelForCausalLM.from_config(config)





# Load Dataset

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

from datasets import DatasetDict, Dataset

midi_path = 'FurElise.mid'

from midiTokenizer import MIDITokenizer
from structuredTokenizer import MIDIStructuredTokenizer


midiTokenizer = MIDITokenizer()
structuredTokenizer = MIDIStructuredTokenizer()

import miditoolkit
midiFile = miditoolkit.MidiFile(midi_path)
prev = midiFile.ticks_per_beat 
midiFile.ticks_per_beat = 16 # updated automatically
s = structuredTokenizer.encode(midiFile)
print(s)
midiFile = structuredTokenizer.decode(s)
midiFile.dump("test.mid")


# encodedMIDI = tokenizer.encode(midiTokenizer.encode(midi_path))
encodedMIDI = tokenizer.encode(s)

# train_dataset = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0]], "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]})
train_dataset = Dataset.from_dict({"input_ids": [encodedMIDI], "attention_mask": [[1] * len(encodedMIDI)]})
test_dataset = train_dataset


datasetDict =  DatasetDict({'train': train_dataset, 'test': test_dataset})

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
    output_dir="MIDICausalFinetuning2",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs = 100
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

trainer.push_to_hub("MIDICausalFinetuning2")
# tokenizer.push_to_hub("MIDICausalFinetuning2")



 






# # Inference

# prompt = "Somatic hypermutation allows the immune system to"


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("progz/MIDICausalFinetuning2")
# inputs = tokenizer(prompt, return_tensors="pt").input_ids


# from transformers import pipeline

# generator = pipeline("text-generation", model="progz/MIDICausalFinetuning2")
# # generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
# generator(prompt)






# from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("progz/MIDICausalFinetuning2")
# outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


