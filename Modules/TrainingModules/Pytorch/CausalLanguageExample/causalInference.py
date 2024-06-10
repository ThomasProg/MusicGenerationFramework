# https://huggingface.co/docs/transformers/en/tasks/language_modeling


import os
os.environ["HF_TOKEN"] = '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'










# Inference

prompt = "Somatic hypermutation allows the immune system to"


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Progz/my-awesome-model")
inputs = tokenizer(prompt, return_tensors="pt").input_ids


from transformers import pipeline

generator = pipeline("text-generation", model="Progz/my-awesome-model")
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
generator(prompt)






from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Progz/my-awesome-model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


