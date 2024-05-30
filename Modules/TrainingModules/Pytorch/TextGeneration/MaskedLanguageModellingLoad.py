

import os
import pathlib

text = "Hello <mask>"
path = pathlib.Path("my_awesome_eli5_mlm_model/checkpoint-1500")


from transformers import pipeline

mask_filler = pipeline("fill-mask", "distilroberta-base")
mask_filler(text, top_k=3)


from transformers import AutoTokenizer

import torch
 
 
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(path)

inputs = tokenizer(text, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]



from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(path)
logits = model(**inputs).logits
mask_token_logits = logits[0, mask_token_index, :]



top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

for token in top_3_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))










