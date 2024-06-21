# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = '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'

# Inference

prompt = "Let me bump! This is nonsense!"


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Progz/MIDICausalFinetuning3")
# inputs = tokenizer(prompt, return_tensors="pt").input_ids

from structuredTokenizer import MIDIStructuredTokenizer
tokenizer = MIDIStructuredTokenizer()

import torch

inputs2 = torch.tensor([[1]])

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Progz/MIDICausalFinetuning3")
outputs = model.generate(inputs2, max_new_tokens=6*30-1, do_sample=True, top_k=50, top_p=0.95)
print(outputs[0])
midiFile = tokenizer.decode(outputs[0].tolist())

midiFile.dump("gentest.mid")
