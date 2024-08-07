# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = ''

loadFromDisk = False


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load Existing Model

# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")


from structuredTokenizer import MIDIStructuredTokenizer

structuredTokenizer = MIDIStructuredTokenizer()


from transformers import AutoModelForCausalLM, GPT2Config

model = None

if (loadFromDisk):
    model = AutoModelForCausalLM.from_pretrained("MIDICausalFinetuning4/")
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
        vocab_size=structuredTokenizer.nbTokens,#721+128+5000,#50257, # min : 721
        n_positions=1024,
        n_ctx=1024,
        n_embd=400,
        n_layer=6,  # distilgpt2 has fewer layers than GPT-2
        n_head=10,
        n_inner=400*4,
        bos_token_id=structuredTokenizer.bos_token_id,
        eos_token_id=structuredTokenizer.eos_token_id
    )

    # Instantiate the model from the configuration
    model = AutoModelForCausalLM.from_config(config)





# Load Dataset
from datasets import DatasetDict, Dataset

midi_path = "K545 Piano Sonata.mid" # 'FurElise.mid'
midi_path = "Andre_Andante.mid"
# midi_path = "Assets/Datasets/LakhMidiClean/Ludwig_van_Beethoven/5th_Symphony.mid"

import miditoolkit
midiFile = miditoolkit.MidiFile(midi_path)
multSet = set()
for change in midiFile.time_signature_changes:
    multSet.add(change.denominator)
    multSet.add(change.numerator)
mult = 1
for m in multSet:
    mult *= m
# assert(len(midiFile.time_signature_changes) == 1)
# timeSign = midiFile.time_signature_changes[0]
    
print("original:")
newTicksPerBeat = mult # 4*4=16
for note in midiFile.instruments[0].notes:
    note.start = round(note.start * newTicksPerBeat / midiFile.ticks_per_beat)
    note.end = round(note.end * newTicksPerBeat / midiFile.ticks_per_beat)
    # print("%d / %d" % (note.start, note.end))
    print(note.pitch, end=" ")
midiFile.ticks_per_beat = newTicksPerBeat

encodedMIDI = structuredTokenizer.encode(midiFile)
midiFile = structuredTokenizer.decode(encodedMIDI)

print("decoded:")
for note in midiFile.instruments[0].notes:
    print(note.pitch, end=" ")

# encodedMIDI = []
midiFile.ticks_per_beat = newTicksPerBeat
midiFile.dump("test.mid")

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
    output_dir="MIDICausalFinetuning4",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
    num_train_epochs = 500
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

trainer.push_to_hub("MIDICausalFinetuning4")
# tokenizer.push_to_hub("MIDICausalFinetuning4")

