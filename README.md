# BART-Based Abstractive Text Summarization

An abstractive text summarization project that uses the **BART Transformer architecture** to generate concise and meaningful summaries from long-form news articles. The system is designed to capture the important information and contextual relationships within an article rather than simply selecting existing sentences.

The model is fine-tuned on the **CNN/DailyMail 3.0.0 dataset** using **PyTorch** and **Hugging Face Transformers**, with summary quality evaluated through both lexical and semantic metrics.

**Dataset:** [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/abisee/cnn_dailymail)

## 📌  Overview

Traditional extractive summarization selects sentences directly from the source document. In contrast, this project uses an **encoder-decoder Transformer architecture** to understand the input article and generate a new summary in natural language.

The project focuses on:

- Generating concise abstractive summaries
- Preserving the main information from the source article
- Maintaining contextual relationships between different parts of the article
- Evaluating generated summaries from both lexical and semantic perspectives
- Studying how continued model fine-tuning affects summarization performance

---

## 🎯 Objectives

- **Abstractive Summary Generation:** Produce new summaries instead of directly copying sentences from the article.
- **Context Preservation:** Use BART's encoder-decoder architecture to capture contextual information from the input.
- **Semantic Similarity:** Use BERTScore to determine how closely the generated summary preserves the meaning of the reference summary.
- **Lexical Similarity:** Apply ROUGE-1, ROUGE-2, and ROUGE-L to measure word and phrase overlap.
- **Generation Control:** Apply beam search and length constraints to improve the quality and conciseness of generated summaries.
- **Training Comparison:** Analyze model performance after different fine-tuning stages.
- **Model Selection:** Select the best checkpoint based on validation ROUGE-1 performance.

---

---

## 📚 Research Publication

This work was published at the **2025 International Conference on Signal Processing, Computation, Electronics, Power and Telecommunication (IConSCEPT)**, IEEE.

**Paper:**  
*Abstractive Text Summarization with Semantic and Contextual Alignment Using BART*

**DOI:** [10.1109/IConSCEPT66142.2025.11436141](https://doi.org/10.1109/IConSCEPT66142.2025.11436141)

**Publisher:** IEEE

---


## 🧠 Model Architecture

The project uses **BART**, a Transformer-based sequence-to-sequence model consisting of an encoder and an autoregressive decoder.

### Workflow

```text
                 ┌──────────────────────┐
                 │     News Article     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   BART Tokenizer     │
                 │   Max: 512 Tokens    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     BART Encoder     │
                 │ Context Representation│
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     BART Decoder     │
                 │ Summary Generation   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Generated Summary  │
                 │   Max: 128 Tokens    │
                 └──────────┬───────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐     ┌───────────────┐
          │    ROUGE     │     │   BERTScore   │
          │ Lexical Match│     │Semantic Match │
          └──────────────┘     └───────────────┘
```

### Base Model

The project uses **BART (Bidirectional and Auto-Regressive Transformer)** as the underlying sequence-to-sequence model.

The pretrained model used in this project is:

```text
facebook/bart-base
```

This pretrained BART model is fine-tuned  to generate abstractive summaries of news articles.

The tokenizer accepts a maximum of **512 input tokens**, while generated summaries are restricted to a maximum of **128 tokens**.

---

## 🔄 Project Workflow

### 1. Dataset Preparation

The project uses the **CNN/DailyMail 3.0.0** dataset. Each sample contains:

- `article` — original news article
- `highlights` — human-written reference summary
- `id` — article identifier

### 2. Text Tokenization

Articles and reference summaries are processed using the BART tokenizer.

```text
Maximum Input Length  = 512 tokens
Maximum Target Length = 128 tokens
```

Longer articles are truncated to fit the configured input size.

The processed data contains:

```text
input_ids
attention_mask
labels
```

### 3. Batch Preparation

`DataCollatorForSeq2Seq` is used to dynamically pad sequences during training, allowing examples with different lengths to be efficiently grouped into batches.

### 4. Model Fine-Tuning

The pretrained `facebook/bart-base` model is fine-tuned using the CNN/DailyMail training data.

The configured training setup includes:

| Parameter | Value |
|---|---:|
| Training Samples | 8,000 |
| Validation Samples | 800 |
| Batch Size | 8 |
| Learning Rate | 5.6e-5 |
| Weight Decay | 0.01 |

The model is evaluated during training, and checkpoints are saved for comparison.

---

## 🧪 Progressive Fine-Tuning

The project investigates the effect of additional fine-tuning by evaluating the model at different training stages.

The model is progressively trained through **3, 6, and 9 epochs**, allowing the changes in summarization performance to be compared using both lexical and semantic evaluation measures.

This also provides insight into how additional training influences model performance and potential overfitting.

---

## ✨ Summary Generation

After fine-tuning, the trained model is used for inference through a Hugging Face summarization pipeline.

The generation configuration includes:

```text
Maximum Summary Length  = 128 tokens
Minimum Summary Length  = 40 tokens
Beam Search             = 4 beams
Early Stopping          = Enabled
No Repeat N-Gram Size   = 3
```

These settings help control summary length and reduce unnecessary repetition.

---

## 📊 Evaluation

The generated summaries are evaluated using **ROUGE** and **BERTScore**.

### ROUGE-1

Measures overlap of individual words between the generated and reference summaries.

### ROUGE-2

Measures two-word sequence overlap and provides a stricter indication of phrase similarity.

### ROUGE-L

Uses the longest common subsequence to assess similarity in sentence structure and word ordering.

### BERTScore F1

Measures semantic similarity using contextual representations, allowing summaries with different wording but similar meaning to receive appropriate similarity scores.

---

## 📈 Experimental Results

The reported validation results across the three training stages are:

| Training Stage | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|
| 3 Epochs | 58.6076 | 37.0028 | 38.9175 | 86.1309 |
| 6 Epochs | 58.0580 | 36.8501 | 38.8543 | 86.4534 |
| 9 Epochs | 58.0705 | 37.0396 | 38.9347 | 86.4515 |

These results allow the effect of additional fine-tuning to be examined from both lexical and semantic perspectives.

---

## 🔍 Qualitative Analysis

In addition to numerical evaluation, generated summaries can be compared with their reference summaries based on:

- Information preservation
- Conciseness
- Contextual coherence
- Redundant content
- Coverage of important information
- Similarity to the reference summary

---

## 🛠️ Technologies Used

### Programming & Deep Learning

- Python
- PyTorch
- Hugging Face Transformers

### NLP & Evaluation

- BART
- Hugging Face Tokenizers
- ROUGE
- BERTScore

### Data & Visualization

- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn

---



## 💻 Training Environment

```text
Platform : Google Colab
GPU      : NVIDIA T4
Framework: PyTorch
Language : Python
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd bart-article-summarization
```

### 2. Install Dependencies

```bash
pip install transformers datasets evaluate torch pandas numpy rouge_score accelerate bert_score matplotlib seaborn
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

Launch the Jupyter/Colab notebook:

```text
BART_Article_Summarization.ipynb
```

Run the cells sequentially to load the dataset, preprocess the text, fine-tune BART, generate summaries, and evaluate the results.

---







