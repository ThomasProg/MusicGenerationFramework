# from huggingface_hub import notebook_login
# notebook_login()

from datasets import load_dataset

# Load the dataset
eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)

# Split the dataset into train and test
eli5 = eli5.train_test_split(test_size=0.2)

print(eli5["train"][0])

from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", trust_remote_code=True)

# Flatten the dataset
eli5 = eli5.flatten()
print(eli5["train"][0])

block_size = 128

# Define the preprocess function with padding and truncation
def preprocess_function(examples):
    # Join the answers.text field into a single string per example
    texts = [" ".join(answer) for answer in examples["answers.text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=block_size)

# Apply the preprocess function to the dataset
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=eli5["train"].column_names,
)

# Define a function to group texts
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

# Optionally group texts if needed
# lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=1)
lm_dataset = tokenized_eli5

from transformers import DataCollatorForLanguageModeling

# Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

from transformers import AutoModelForMaskedLM

# Load the model
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_eli5_mlm_model",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

import math

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
