# -*- coding: utf-8 -*-
"""
RUTHLESS SUMMARISER - BART Version (Enhanced CLI)
Model: facebook/bart-large-cnn
Trademark © HarshVardhan149 | All Rights Reserved
"""

# ------------------------------------------------------------
# Install necessary libraries
# ------------------------------------------------------------
!pip install --quiet transformers huggingface_hub

# ------------------------------------------------------------
# Import required modules
# ------------------------------------------------------------
from transformers import BartTokenizer, BartForConditionalGeneration
from huggingface_hub import login
import textwrap

# ------------------------------------------------------------
# Authenticate with Hugging Face (Enter your token when prompted)
# ------------------------------------------------------------
login()

# ------------------------------------------------------------
# Load Pre-trained BART Model and Tokenizer
# ------------------------------------------------------------
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# ------------------------------------------------------------
# Ruthless Summarisation Function
# ------------------------------------------------------------
def ruthless_summarise(text):
    """
    Summarises text by stripping all descriptive elements,
    retaining only essential factual content.
    """
    prompt = (
        "Summarise this ruthlessly. Strip away all flowery language, emotion, and description. "
        "Keep only essential factual content:\n\n" + text
    )

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=5,
        max_length=80,
        min_length=15,
        length_penalty=4.0,
        no_repeat_ngram_size=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    if not summary.endswith('.'):
        summary += '.'

    return summary

# ------------------------------------------------------------
# Console-Based Interactive Interface
# ------------------------------------------------------------
print("=" * 60)
print("RUTHLESS SUMMARISER".center(60))
print("Model: facebook/bart-large-cnn".center(60))
print("Version: Enhanced CLI Edition".center(60))
print("=" * 60)
print("\nEnter a piece of writing (story, article, report).")
print("Type 'exit' to quit.\n")

# Input Loop
while True:
    user_input = input("Enter text to summarise:\n\n").strip()

    if user_input.lower() == "exit":
        print("\nExiting summariser. Goodbye.")
        break

    if not user_input:
        print("\n[Warning] Empty input received. Please try again.\n")
        continue

    print("\n" + "-" * 60)
    print("Generating Ruthless Summary... Please wait.\n")

    summary = ruthless_summarise(user_input)

    print("--- RUTHLESS SUMMARY ---\n")
    print(textwrap.fill(summary, width=80))
    print("-" * 60 + "\n")

# ------------------------------------------------------------
# Trademark © HarshVardhan149 | All Rights Reserved | 2025
# ------------------------------------------------------------