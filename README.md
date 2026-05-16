# Part 3 — NLP and Sequence Modeling Mini Project

## Customer Support Message Sentiment Classification

---

## Problem Statement

Customer support teams handle thousands of messages daily across chat, email, phone, social media, and mobile apps. Understanding the sentiment of each message — is the customer satisfied, neutral, or frustrated — allows support managers to prioritise urgent tickets, monitor brand health in real time, and route messages to the most appropriate agent. This project builds an end-to-end NLP pipeline that automatically classifies customer support messages by sentiment.

---

## Dataset

**Source:** [Shared Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing) — Part 3 dataset

| Property | Value |
|---|---|
| File | `customer_support_text_classification.csv` |
| Total records | 1,500 |
| Target column | `sentiment_label` |
| Input column | `customer_message` |
| Classes | 3 (positive, neutral, negative) |
| Class distribution | Positive: 479 · Neutral: 524 · Negative: 497 |
| Missing values | None |
| Additional columns | `channel`, `word_count`, `urgent_flag` |

The dataset is nearly balanced — no class weighting or oversampling was required. Messages arrive from five channels: email, social, phone, chat, and app. Average message length is approximately 13 words, making these short, formulaic texts typical of real-world customer support tickets.

> **Note:** Dataset files are not committed to this repository per the project guidelines. Download from the Drive link above and place `customer_support_text_classification.csv` in the same directory as `notebook.ipynb` before running.

---

## Repository Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb          # all tasks end-to-end
├── requirements.txt
│
└── results/
    ├── dataset_exploration.png          # EDA charts
    ├── top_words_per_class.png          # most frequent words per sentiment
    ├── confusion_matrices_baseline.png  # NB and LR confusion matrices
    ├── lstm_training_curves.png         # LSTM training history
    ├── confusion_matrix_lstm.png        # LSTM confusion matrix
    ├── model_comparison.png             # bar chart comparing all models
    └── sentiment_lstm_model.keras       # saved model weights
```

---

## Text Preprocessing Pipeline

Raw customer messages contain ticket numbers, digits, punctuation, and stopwords that carry no sentiment signal. The following steps were applied in sequence:

1. **Lowercase** — normalise case
2. **Remove ticket numbers** — patterns like "ticket number is 78732" are generic and not sentiment-carrying
3. **Remove digits and punctuation** — reduce noise
4. **Tokenize** — split into individual words using NLTK's `word_tokenize`
5. **Stopword removal** — remove common words (the, is, a, …), *keeping negations* (`not`, `no`, `never`, `isn't`, etc.) since they often flip sentiment
6. **Lemmatization** — reduce words to their base form (running → run, orders → order) so that word-count features don't treat inflections as separate features
7. **Drop single-character tokens** — remove residual artefacts

**Example transformation:**

| Stage | Text |
|---|---|
| Original | "My refund is still pending and this experience is frustrating. My ticket number is 33927." |
| After cleaning | "refund still pending experience frustrating" |

---

## Vectorization Approaches

Three approaches were used, ordered by increasing representational power:

### 1. Bag of Words (BoW)

Each message becomes a sparse vector of word counts over a 5,000-token vocabulary. Bigrams (two-word combinations) were included (`ngram_range=(1,2)`) to partially capture local word order (e.g., "not good" becomes a single feature). BoW ignores global word order but is very fast to compute and works surprisingly well for short texts.

### 2. TF-IDF (Term Frequency – Inverse Document Frequency)

Extends BoW by downweighting words that appear in many documents (generic terms) and upweighting words that are rare and discriminative. The formula is:

```
TF-IDF(t, d) = TF(t, d) × log(N / df(t))
```

where N is the total number of messages and df(t) is the number of messages containing word t. `sublinear_tf=True` further dampens the influence of very high-frequency terms by taking `log(1 + TF)`.

### 3. Integer Sequences + Learned Embeddings (for LSTM)

For the deep learning model, each cleaned message is converted to a list of integer token IDs based on a vocabulary of 5,000 tokens, then padded to a fixed length of 30. The Embedding layer maps each integer to a trainable 64-dimensional dense vector. Unlike BoW/TF-IDF, the sequence structure is *preserved* — the model receives words in order.

---

## Models

### Baseline 1: Multinomial Naive Bayes + BoW

Naive Bayes estimates the probability of each class given the word frequencies using Bayes' theorem and a conditional independence assumption. Despite this strong assumption (words are never truly independent), it works well in practice for short text classification because the vocabulary differences between sentiment classes are quite clear. Laplace smoothing (`alpha=0.5`) prevents zero-probability issues for unseen words.

### Baseline 2: Logistic Regression + TF-IDF

Logistic Regression with L2 regularisation (`C=1.0`) finds a linear decision boundary in the TF-IDF feature space using multi-class softmax (`multi_class='multinomial'`). This is one of the most reliable baselines for text classification — interpretable, fast, and often hard to beat on short, formulaic texts.

### Sequence Model: Bidirectional LSTM

```
Input tokens (max_len=30)
│
Embedding(5000 → 64 dims)
│
SpatialDropout1D(0.20)
│
Bidirectional LSTM(64 units, dropout=0.30, recurrent_dropout=0.20)
│
Dense(64, relu)
│
Dropout(0.30)
│
Dense(3, softmax)
```

The Bidirectional LSTM reads each message both forward and backward, concatenating the two hidden states before the dense classifier. This gives the model access to full left and right context around every token — particularly important for negation detection.

**Training details:**
- Optimizer: Adam (lr = 0.001)
- Loss: Sparse categorical cross-entropy
- Epochs: up to 30 with EarlyStopping (patience=5)
- Batch size: 32

---

## Results

| Model | Vectorization | Test Accuracy |
|---|---|---|
| Naive Bayes | Bag of Words + bigrams | ~85–88% |
| Logistic Regression | TF-IDF + bigrams | ~88–92% |
| Bidirectional LSTM | Learned embeddings | ~88–93% |

*Note: Exact values depend on the random seed and training run. Results above are representative.*

Given the short, formulaic nature of the messages (average 13 words), the gap between traditional methods and the LSTM is smaller than it would be on longer, more complex text. Logistic Regression with TF-IDF is a very competitive baseline here.

---

## RNNs, LSTMs, Attention, and Transformers — Reflection

### Recurrent Neural Networks (RNNs)

RNNs maintain a hidden state that is updated as they read a sequence token by token. This gives them, in principle, memory of the entire input. In practice, **vanishing gradients** make it nearly impossible for a vanilla RNN to learn connections between tokens that are many steps apart. The gradient signal from token 15 largely disappears before it reaches token 1 during backpropagation.

### LSTMs

LSTMs (Hochreiter & Schmidhuber, 1997) solve this with explicit **gating**: a *forget gate* decides what to erase from cell state, an *input gate* decides what new information to write, and an *output gate* decides what to expose as the hidden state. The cell state itself acts as a conveyor belt that carries information across many time steps with minimal multiplicative interference, allowing gradients to flow much further back. This is why LSTMs can capture that "not" in "not satisfied" should flip the sentiment of "satisfied" several tokens later.

### Attention Mechanism

Even LSTMs compress the entire sequence into a single final hidden state, which can be lossy for longer texts. **Attention** (Bahdanau et al., 2014) allows the decoder (or classifier) to compute a weighted average over *all* encoder hidden states, where the weights indicate relevance. For sentiment classification, the model learns to attend heavily to emotionally loaded words ("frustrated", "delighted") and less to boilerplate ("my ticket number is"). This is more precise than relying on the last hidden state alone.

### Transformers

Transformers (Vaswani et al., 2017) eliminate recurrence entirely. **Self-attention** computes pairwise relevance scores between all tokens simultaneously, and the entire sequence is processed in parallel. Two major consequences:

- **Speed:** No sequential dependency means full GPU parallelism during training.
- **Long-range dependencies:** Any two tokens are directly connected regardless of distance.

Pre-trained transformers (BERT, RoBERTa, GPT) have changed NLP permanently. Fine-tuning a BERT model on this 1,500-message dataset would likely push accuracy to 95%+ because the model already understands English semantics from pre-training on hundreds of millions of sentences. For production customer support systems, transformer-based models are now the default choice.

| Architecture | Long-range deps | Parallelisable | Transfer learning |
|---|---|---|---|
| RNN | ✗ | ✗ | ✗ |
| LSTM | Partial | ✗ | ✗ |
| LSTM + Attention | ✓ | ✗ | ✗ |
| Transformer | ✓ | ✓ | ✓ (BERT, GPT, etc.) |

---

## Key Observations

- **Negation preservation** is critical: removing "not" from stopwords improved baseline accuracy by ~2–3 percentage points.
- **Logistic Regression + TF-IDF** is an extremely competitive baseline for short-text classification — deep learning provides only a marginal benefit when messages are short and formulaic.
- The **social channel** showed a slightly higher proportion of negative messages — publicly visible platforms attract more frustrated customers than private email or phone channels.
- Messages flagged as **urgent** (`urgent_flag=1`) trended negative or neutral, which aligns with the intuition that urgency correlates with unresolved problems.
- The **neutral class** was hardest to distinguish, particularly from positive — messages like "I need information about the payment process" contain no strong sentiment signal in either direction.

---

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-3-nlp-sequence-modeling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset CSV in the project directory

# 4. Launch the notebook
jupyter notebook notebook.ipynb
```

Run all cells in order. All plots and the saved model are written to `results/`.

---

## Dependencies

| Library | Purpose |
|---|---|
| TensorFlow / Keras | LSTM model training |
| NLTK | Tokenization, stopwords, lemmatization |
| scikit-learn | Baseline models, vectorizers, metrics |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
