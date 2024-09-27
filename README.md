# NGram Language Model for Code Completion

## Overview

This project implements an [4,3,2,1]-gram language model (unigram, bigram, trigram, and four-gram) for Java code completion. 

### Dataset

The dataset used is the Java subset from the `google/code_x_glue_ct_code_to_text` dataset:
- **Train set:** 165k Java functions.
- **Test set:** 100 samples, with both preprocessed and raw Java code.

### Methodology

- **N-gram Models:** Build unigram to four-gram models by counting token occurrences.
- **Tokenization:** Use CodeLlama's tokenizer to process raw Java code into tokens.
- **Fallback Mechanism:** Predictions start with the 4-gram model and fallback to lower n-grams if necessary.
- **Evaluation Metrics:** Accuracy (exact token match) and Perplexity (predictive quality) are computed on test data.

## Installation

### Prerequisites

Install dependencies:

```bash
pip install nltk transformers tqdm datasets
```

### Setup

Clone the repository:

```bash
git clone https://github.com/{yourusername}/ngram-language-model.git
cd ngram-language-model
```

Load the dataset:

```python
from datasets import load_dataset
ds = load_dataset('google/code_x_glue_ct_code_to_text', split='train')
```

## Usage

### Train Models

Train the N-gram models using preprocessed code tokens or raw Java code:

```python
ngramLM_pre = NGramLM(ds['train']['code_tokens'])
ngramLM_codellama = NGramLM(ds['train']['code'])
```

### Evaluate Models

Evaluate the model on the test set:

```python
avg_acc, avg_ppl = ngramLM_pre.eval_on_corpus(ds['test']['code_tokens'])
print(f"Average Accuracy: {avg_acc}")
```

### Predict Next Token

Predict the next token in a sequence:

```python
next_token = ngramLM_pre.predict_next_word(w1, w2, w3)
print(f"Predicted next token: {next_token}")
```

## Results

- Models are evaluated based on accuracy and perplexity.
- Predictions using consistent tokenization (e.g., [4,3,2,1]-gramPre) outperform mixed-tokenizer approaches.
