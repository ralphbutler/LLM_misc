import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bitnet import replace_linears_in_hf

# Load a model from Hugging Face's Transformers
# model_name = "bert-base-uncased"
model_name = "../../MISTRAL-7B/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_name)
print(model)
print("-"*50)

# Replace Linear layers with BitLinear
replace_linears_in_hf(model)
print(model)
print("-"*50)

# Example text to classify
text = "Replace this with your text"
inputs = tokenizer(
    text, return_tensors="pt", padding=True, truncation=True, max_length=512
)

# Perform inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

# Process predictions
predicted_class_id = predictions.argmax().item()
print(f"Predicted class ID: {predicted_class_id}")

# Optionally, map the predicted class ID to a label, if you know the classification labels
# labels = ["Label 1", "Label 2", ...]  # Define your labels corresponding to the model's classes
# print(f"Predicted label: {labels[predicted_class_id]}")
