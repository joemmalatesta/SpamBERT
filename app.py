import gradio as gr
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load the model and tokenizer for the fine tuned model
model = BertForSequenceClassification.from_pretrained('./SpamBERT')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    # Debug to make sure we are evaluating right
    print("Input: ", text)
    print("Logits:", logits)
    print("Probabilities:", probabilities)  
    prediction = logits.argmax(-1).item()
    return "spam" if prediction == 1 else "ham"

# Gradio Bit 
interface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Spam Detection with BERT",
                          description="Enter a text message to determine if it's spam or not.")
interface.launch()
