# Advanced NLP: Text Generation, Claim Normalization & Multimodal Sarcasm

---

## Project Overview

This project explores three core natural language processing tasks using transformer-based models:

1. **Shakespearean Text Generation:** Implementing a Transformer from scratch to generate Shakespeare-style text.  
2. **Claim Normalization:** Converting noisy social media posts into clear, normalized claims using fine-tuned BART and T5 models.  
3. **Multimodal Sarcasm Explanation:** Combining textual and visual inputs to generate explanations for sarcasm with a TURBO-like model architecture.

Each task demonstrates state-of-the-art NLP techniques tailored for unique challenges in text generation, normalization, and multimodal understanding.

---

## Task 1: Shakespearean Text Generation

- **Goal:** Build a Transformer model from scratch for autoregressive Shakespearean text generation.  
- **Dataset:** Shakespeare train and dev text files.  
- **Key Features:**  
  - Self-attention and multi-head attention  
  - Positional encoding  
  - Causal masking  
  - Residual connections and layer normalization  
- **Evaluation:** Perplexity (excluding padding tokens) and qualitative text generation.  
- **Deliverables:** Trained model, training/validation loss plots, and sample generated text.

---

## Task 2: Claim Normalization

- **Goal:** Normalize noisy social media posts into structured claims.  
- **Dataset:** CLAN dataset of social media posts with normalized claims.  
- **Preprocessing:**  
  - Expand contractions and abbreviations  
  - Clean URLs and special characters  
  - Lowercase normalization  
- **Models:** Fine-tuned BART-base and T5-base.  
- **Evaluation:** ROUGE-L, BLEU-4, and BERTScore metrics to compare model performance.  
- **Deliverables:** Preprocessing and training code, saved best model, and evaluation report.

---

## Task 3: Multimodal Sarcasm Explanation

- **Goal:** Generate explanations for sarcasm using both text and image modalities.  
- **Dataset:** MORE+ dataset with sarcastic posts, images, explanations, and sarcasm targets.  
- **Architecture:**  
  - Text feature extraction with BART  
  - Visual feature extraction with Vision Transformer (ViT)  
  - Gated fusion of modalities  
  - Sequence generation with BART decoder  
- **Training Details:** Batch size 4, learning rate 1e-4, 10 epochs.  
- **Evaluation:** ROUGE, BLEU, METEOR, and BERTScore metrics.  
- **Deliverables:** Trained model checkpoint, training scripts, and qualitative analysis.

---

## Setup and Installation

### Prerequisites

- Python 3.8+  
- PyTorch  
- Transformers (Hugging Face)  
- Tokenizers  
- torchvision, Pillow (for image processing)  
- Other dependencies as listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
