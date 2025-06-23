# Flickr30k-Multilingual-CaptionSuite
This project focuses on multilingual processing, semantic similarity, and cross-lingual paraphrase detection for image captions using NLP and deep learning techniques. It is built on the Flickr30k dataset and performs translation, validation, and classification of captions across multiple languages.

---

## 📌 Project Highlights

### 1.  Caption Preprocessing & Dataset Creation
- Parsed `Flickr30k_captions.csv` (delimiter `|`)
- Cleaned and grouped by image
- Selected the **first caption** per image
- Output: `flickr30k_first_caption.csv`

### 2.  Multilingual Translation (English → Urdu, Spanish, Chinese)
- Used `Helsinki-NLP/opus-mt-*` models via HuggingFace Transformers
- Translated 3,000 captions to:
  - `caption_ur` (Urdu)
  - `caption_es` (Spanish)
  - `caption_zh` (Chinese)
- Output: `flickr30k_translated_3k_sentences.csv`

### 3.  Semantic Validation Using LaBSE
- Loaded `sentence-transformers/LaBSE`
- Calculated cosine similarity between English and translated captions
- Discarded pairs with similarity < 0.75
- Outputs:
  - `filtered_captions_caption_es.csv`
  - `filtered_captions_caption_zh.csv`
  - `filtered_captions_caption_ur.csv`

### 4.  Round-Trip Translation Explainability
- Forward: MarianMT (target → English)
- Backward: mBART (English → target)
- Compared:
  - Token-level edit distance
  - Character-level BLEU score
- Generated an **explainability score** per prompt
- CLI reports highlight mismatched tokens

### 5.  Prompt Paraphrase Discriminator (Binary Classification)
- Created **positive pairs** (same image in different languages)
- Created **negative pairs** (random mismatches)
- Features: `|A - B|` and `A * B` using LaBSE
- Models:
  - `LogisticRegression`
  - `PyTorch Neural Network`

### 6.  Evaluation
- Metrics: **Accuracy**, **ROC-AUC**, **F1**
- Tested on unseen 500-pair dataset
- Evaluated generalization performance of both classifiers

---

## 📁 Directory Structure

```
Multilingual-Caption-Validator/
├── datasets/
│   ├── Flickr30k_captions.csv
│   ├── flickr30k_first_caption.csv
│   ├── flickr30k_translated_3k_sentences.csv
│   ├── filtered_captions_caption_es.csv
│   ├── filtered_captions_caption_zh.csv
│   ├── filtered_captions_caption_ur.csv
├── main.py
├── main.ipynb
├── requirements.txt
└── README.md
```

---

##  Technologies & Models Used

- **Transformers**: MarianMT, mBART
- **Embeddings**: LaBSE (Language-Agnostic BERT Sentence Embedding)
- **Metrics**: BLEU, edit distance, cosine similarity
- **ML Models**: Logistic Regression, PyTorch Neural Net
- **Libraries**: HuggingFace, NLTK, scikit-learn, torch, pandas

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```


```bash
python main.ipynb
```

