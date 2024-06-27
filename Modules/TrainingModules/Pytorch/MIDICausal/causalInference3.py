# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = ''

# Inference

from structuredTokenizer import MIDIStructuredTokenizer
tokenizer = MIDIStructuredTokenizer()

import torch

# inputs2 = torch.tensor([[51]])
inputs2 = torch.tensor([[83]])
# inputs2 = torch.tensor([[80]])
# inputs2 = torch.tensor([[65]])

from transformers import AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("Progz/MIDICausalFinetuning3")
model = AutoModelForCausalLM.from_pretrained("Progz/MIDICausalFinetuning3_5thsymphony")
# outputs = model.generate(inputs2, max_new_tokens=6*30-1, do_sample=True, top_k=50, top_p=0.95)
outputs = model.generate(inputs2, max_new_tokens=1024-1, do_sample=True, top_k=50, top_p=0.95, pad_token_id=2000)
torch.set_printoptions(threshold=10_000)
print(outputs[0])
midiFile = tokenizer.decode2(outputs[0].tolist())
midiFile.ticks_per_beat = 3*4
for note in midiFile.instruments[0].notes:
    print("%d / %d / %d" % (note.start, note.end, note.pitch))
midiFile.dump("gentest.mid")
