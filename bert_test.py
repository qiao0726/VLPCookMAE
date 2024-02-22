from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize and align labels
def tokenize_and_align_labels(texts, labels):
    tokenized_inputs = tokenizer(texts, truncation=True, is_split_into_words=True, padding=True, return_tensors="pt")
    labels = [...] # Adjust your label ids based on tokenization
    return tokenized_inputs, labels

texts = ["This is a sentence.", "This is another one."]
labels = [[...], [...]] # Your POS tags converted to numerical labels
tokenized_inputs, labels = tokenize_and_align_labels(texts, labels)


from transformers import BertForTokenClassification, Trainer, TrainingArguments

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    evaluate_during_training=True,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train the model
trainer.train()


# Evaluate the model
trainer.evaluate()

# Inference
sentence = "Here is a new sentence."
inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt")
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
