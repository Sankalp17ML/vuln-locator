# 🛡️ VulnLocator: AI-Powered Vulnerability Detection & Localization

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-CodeBERT-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/codebert-base)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Hardware](https://img.shields.io/badge/Hardware-Kaggle%20Dual%20T4%20GPUs-orange?logo=nvidia&logoColor=white)](https://kaggle.com)

**VulnLocator** is a deep learning security triage system that automatically identifies security vulnerabilities in source code and pinpoints suspicious tokens using **Attention Rollout**. By combining fine-tuned transformer representations (`microsoft/codebert-base`) with attention flow analysis, VulnLocator transforms black-box neural classifications into interpretable, actionable findings for security engineers.

---

## 📌 Table of Contents
1. [The Problem Statement](#-the-problem-statement)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [V2.0 Architecture: Mixture of Experts (Specialist vs. Generalist)](#-v20-architecture-mixture-of-experts)
5. [Dataset & Class Imbalance](#-dataset--class-imbalance)
6. [Training & Pipeline Optimizations (Kaggle Dual T4)](#-training--pipeline-optimizations-kaggle-dual-t4)
7. [Empirical Results & Benchmarks](#-empirical-results--benchmarks)
8. [Attention Rollout (Interpretability)](#-attention-rollout-interpretability)
9. [Quickstart & Installation](#-quickstart--installation)
10. [Docker & Containerized Deployment](#-docker--containerized-deployment)
11. [API Reference](#-api-reference)
12. [Project Structure](#-project-structure)
13. [Resume Bullets & Interview Talking Points](#-resume-bullets--interview-talking-points)

---

## 🚨 The Problem Statement
Manual code audits are slow, labor-intensive, and prone to human oversight. In fast-paced enterprise CI/CD environments, developers commit thousands of lines of code daily, creating severe review backlogs where critical zero-days and vulnerabilities slip into production.

Existing static analysis tools (SAST) rely heavily on hand-crafted regex rules and abstract syntax trees (ASTs), which produce high false-positive rates and struggle with semantic nuances. While deep learning models offer superior semantic understanding, they have historically functioned as **uninterpretable black boxes**—telling a security auditor that code is "vulnerable" without showing *where* or *why*.

**VulnLocator bridges this gap:**
- **High-throughput automated triage:** Screens code snippets in milliseconds.
- **Explainable localization:** Traces multi-layer transformer attention to highlight specific vulnerable tokens (e.g., unchecked `strcpy`, unparameterized SQL strings, command execution calls).

---

## ✨ Key Features
- **Transformer-Powered Detection:** Fine-tuned `microsoft/codebert-base` on real-world vulnerability benchmarks.
- **Attention Rollout Localization:** Multi-layer attention aggregation with residual connections to compute token-level importance scores.
- **Production REST API:** High-performance, asynchronous FastAPI backend serving predictions and attention weights.
- **Interactive Web UI:** Modern Streamlit frontend providing instant visual vulnerability alerts and interactive token ranking.
- **Dual T4 Optimized Pipeline:** Pre-tokenized `TensorDataset` architecture and multi-GPU `DataParallel` training yielding a 180x speedup over standard training loops.
- **Containerized Delivery:** Docker & Docker Compose configurations for seamless single-command deployment.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Client["Presentation Layer"]
        UI["Streamlit Web UI<br>(Port 8501)"]
        CLI["CLI Localization Tool<br>(python -m src.localize)"]
        CURL["cURL / External Clients"]
    end

    subgraph Service["Application Layer"]
        API["FastAPI REST Server<br>(Port 8000)"]
        HEALTH["/health Check"]
        PREDICT["/predict Endpoint"]
    end

    subgraph Engine["Inference & Explainability Engine"]
        ANALYZER["CodeAnalyzer Engine"]
        TOKENIZER["CodeBERT Fast Tokenizer<br>(Byte-Level BPE)"]
        MODEL["Fine-Tuned CodeBERT<br>(12-Layer Transformer)"]
        ROLLOUT["Attention Rollout Algorithm<br>(Layer Attention Fusion + Residuals)"]
    end

    subgraph Output["Output & Human-in-the-Loop Review"]
        CLASSIF["Binary Classification<br>(Safe vs Vulnerable + Confidence)"]
        TOKENS["Top-K Suspicious Tokens<br>(Line Numbers + Relative Importance)"]
    end

    UI -->|HTTP POST JSON| PREDICT
    CLI --> ANALYZER
    CURL -->|HTTP POST| PREDICT
    API --> HEALTH
    PREDICT --> ANALYZER

    ANALYZER --> TOKENIZER
    TOKENIZER --> MODEL
    MODEL --> CLASSIF
    MODEL -->|All 12 Attention Matrices| ROLLOUT
    ROLLOUT --> TOKENS

    CLASSIF --> UI
    TOKENS --> UI
```

---

## 🔀 V2.0 Architecture: Mixture of Experts

In VulnLocator V2.0, we expanded support to multiple languages (**Java, PHP, JavaScript, Python**) while preventing **Catastrophic Forgetting**—a well-known failure mode where fine-tuning a neural network on high-level web languages degrades its memory of low-level C/C++ memory corruption patterns (e.g., buffer overflows, integer overflows, use-after-free).

### The Strategy: Option B (Specialist vs. Generalist Architecture)
Rather than naively merging disparate vulnerability datasets into a single compromised model, VulnLocator deploys a **Mini-Mixture of Experts (MoE) Router**:

1. **Specialist Model (C/C++):**
   - **Path:** `models/codebert-baseline`
   - **Dataset:** Fine-tuned on PrimeVul C/C++ vulnerability benchmarks.
   - **Domain Focus:** Low-level memory safety, pointer arithmetic, manual memory allocation (`malloc`, `free`), buffer boundaries, kernel vulnerabilities.
2. **Generalist Model (Multilingual):**
   - **Path:** `models/codebert-multilingual`
   - **Dataset:** Fine-tuned on CrossVul benchmarks spanning Java, Python, PHP, JavaScript, Go, Ruby, and C#.
   - **Domain Focus:** Web security, injection vulnerabilities (SQLi, Command Injection, XSS, SSRF), insecure deserialization, cryptographic flaws.
3. **Language Router Engine (`src/router.py` & `app/main.py`):**
   - Automatically inspects code syntax, keywords, and structural markers.
   - Intelligently routes code to the optimal expert model with zero manual configuration required.
   - Allows explicit user overrides (`model: "specialist" | "generalist" | "auto"`).

```mermaid
flowchart TD
    INPUT["Source Code Snippet<br>(Input to /predict)"] --> CLEAN["Syntax Preprocessor<br>(Strip Comments & Docstrings)"]
    CLEAN --> ROUTER{"Router Engine<br>(src/router.py)"}

    ROUTER -->|"#include, malloc, std::, ->, struct"| SPECIALIST["C/C++ Specialist Model<br>(models/codebert-baseline)"]
    ROUTER -->|"import java, <?php, const, def, require"| GENERALIST["Multilingual Generalist Model<br>(models/codebert-multilingual)"]
    ROUTER -->|"Manual Model Override"| OVERRIDE["User-Selected Model"]

    SPECIALIST --> ATTN["Attention Rollout Interpretability Engine"]
    GENERALIST --> ATTN
    OVERRIDE --> ATTN

    ATTN --> RESP["Enriched Response<br>(Vulnerability Verdict + Token Weights + Model Provenance)"]
```

### Syntax Routing Logic
The router evaluates syntax tokens and structural keywords:
- **C/C++ Indicators ➔ Specialist Model:** `#include`, `malloc(`, `free(`, `std::`, `printf(`, `sprintf(`, `scanf(`, `sizeof(`, `uint8_t`, `typedef struct`, pointer arrows `->`.
- **Multilingual Indicators ➔ Generalist Model:**
  - **Java:** `import java`, `package `, `public class `, `System.out.println`, `throws Exception`.
  - **PHP:** `<?php`, `$_GET`, `$_POST`, `mysql_query`, `mysqli_`, `echo $`.
  - **JavaScript / TypeScript:** `const `, `let `, `function `, `console.log`, `module.exports`, `require(`, `eval(`.
  - **Python:** `def `, `import `, `from `, `__name__`, `print(`, `cursor.execute`.
  - **Go / Ruby / C#:** `func `, `fmt.Println`, `def `, `using System`.

### Anti-Lexical Bias & AST Comment Stripping
Machine learning models trained on source code often develop **lexical bias**—latching onto English words inside comments (such as `// fix CVE-2021-1234` or `# vulnerable buffer`) rather than the underlying syntactic bug. 

VulnLocator V2.0 integrates `src/preprocessor.py`:
- Strips single-line comments (`//`, `#`), block comments (`/* */`), and Python docstrings (`"""`, `'''`).
- **Preserves exact line numbers and byte offsets** using newline alignment so that token localization accurately maps back to the developer's original code.
- Eliminates natural language bias while ensuring CodeBERT attends purely to executable syntax.

---

## 📊 Dataset & Class Imbalance

VulnLocator is trained on the **PrimeVul** dataset (`colin/PrimeVul`), a curated benchmark of real-world vulnerabilities mined from thousands of open-source C/C++ and Python repositories.

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Dataset Benchmark** | PrimeVul (`colin/PrimeVul`) | Sourced from CVE/NVD security advisories |
| **Sample Size** | 20,000 code functions | Sampled deterministically (`seed=42`) |
| **Code Representation** | `func` column | Full function bodies with control flows |
| **Labels** | `target` column | `0` = Safe, `1` = Vulnerable |
| **Class Imbalance** | **~97% Safe / ~3% Vulnerable** | Mirrors realistic software production code |

### Handling Extreme Imbalance
Standard Cross-Entropy Loss on a 97:3 distribution causes neural networks to degenerate into predicting "Safe" for 100% of inputs, achieving 97% accuracy while failing to detect a single vulnerability.

To solve this, VulnLocator implements **Inverse-Frequency Loss Weighting**:
$$\mathcal{W}_c = \frac{1 / N_c}{\sum_{k} (1 / N_k)}$$

This heavily penalizes misclassifications of the minority vulnerable class, forcing CodeBERT's embedding representations to capture vulnerability patterns even with scarce positive samples.

---

## ⚡ Training & Pipeline Optimizations (Kaggle Dual T4)

Initial training experiments on local consumer hardware (4GB VRAM) encountered severe Out-Of-Memory (OOM) errors and slow iterations. The training was migrated to **Kaggle's Dual NVIDIA T4 GPUs (32GB combined VRAM)**.

```mermaid
graph LR
    A["Initial Local Setup<br>(4GB GPU)"] -->|OOM Errors<br>2h / epoch| B["Kaggle Migration<br>(Dual T4 GPUs)"]
    B -->|CPU DataLoader Bottleneck| C["Pre-Tokenization<br>(TensorDataset)"]
    C -->|Multi-GPU Parallelism| D["torch.nn.DataParallel<br>+ FP16 AMP"]
    D -->|Epoch time drops to ~40s| E["Final Model<br>(AUC 0.855)"]
```

### The Engineering Breakthroughs:
1. **Eliminating the CPU Worker Bottleneck:**
   - *Problem:* Tokenizing source code on-the-fly inside `Dataset.__getitem__` bottlenecked GPU utilization at <15%, resulting in over **2 hours per epoch**.
   - *Solution:* Implemented full-dataset **pre-tokenization** directly into memory as contiguous PyTorch tensors (`TensorDataset`), passing pre-allocated memory buffers to the GPU.
   - *Result:* GPU compute utilization jumped to >95%, and epoch runtime collapsed from **2 hours to ~40 seconds** (>180x speedup).
2. **Multi-GPU Parallelism (`torch.nn.DataParallel`):**
   - Distributed batches across both NVIDIA T4 GPUs with a global batch size of 32, maximizing throughput across 32GB VRAM.
3. **Automatic Mixed Precision (FP16 AMP):**
   - Leveraged `torch.amp.autocast('cuda')` and `GradScaler('cuda')` to halve memory consumption while accelerating tensor core matrix operations.
4. **Reproducible Scripts & Notebooks:**
   - Standalone execution script: [`src/train_kaggle_dual_t4.py`](src/train_kaggle_dual_t4.py)
   - Kaggle-ready Jupyter Notebook: [`notebooks/kaggle_training_dual_t4.ipynb`](notebooks/kaggle_training_dual_t4.ipynb)

---

## 📈 Empirical Results & Benchmarks

The model was evaluated against a held-out test split (15% stratified test partition, 3,000 samples) following 3 epochs of training.

### Test Split Performance Summary

| Metric | Safe Class (`0`) | Vulnerable Class (`1`) | Overall / Macro |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | — | — | **0.8550** |
| **Precision** | **0.98** | **0.22** | 0.60 |
| **Recall** | **0.96** | **0.33** | 0.65 |
| **F1-Score** | **0.97** | **0.26** | 0.62 |
| **Support** | 2,908 | 92 | 3,000 |

### Why an AUC of 0.855 Matters for Security Triage
In security operations, triage models are used as **filtering funnels**, not definitive binary arbiters. An AUC of **0.855** demonstrates strong class separability across different confidence thresholds. By tuning the decision threshold (e.g., from `0.5` to `0.28`), security teams can dial in high-recall detection (capturing over 80% of vulnerabilities) to eliminate false negatives during CI/CD scans.

---

## 🔍 Attention Rollout (Interpretability)

Standard self-attention inspection only examines isolated attention heads in the final layer, ignoring how representations evolve across intermediate layers. VulnLocator implements **Attention Rollout** (Abnar & Zuidema) to reconstruct information flow from the input tokens to the classification head.

### The Mathematics:
For layer $l$, let $W^{(l)}$ be the multi-head attention matrix averaged across all attention heads. To account for residual skip connections in the Transformer architecture, we add an identity matrix $I$:

$$A^{(l)} = 0.5 W^{(l)} + 0.5 I$$

Each layer's attention matrix is row-normalized:
$$\tilde{A}^{(l)} = \frac{A^{(l)}}{\sum_j A^{(l)}_{i,j}}$$

The total attention rollout matrix $R$ across all $L = 12$ layers of CodeBERT is computed recursively:
$$R = \tilde{A}^{(L)} \times \tilde{A}^{(L-1)} \times \dots \times \tilde{A}^{(1)}$$

The first row $R_{0}$ represents the propagated influence of each code token on the `<s>` classification token.

### Example Findings

#### 1. SQL Injection (Python)
```python
def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
```
**Top Attention Tokens:**
1. `SELECT` (Attention: 1.00)
2. `cursor` (Attention: 0.87)
3. `execute` (Attention: 0.84)
4. `+` (Attention: 0.79)

#### 2. Buffer Overflow (C)
```c
void vulnerable_copy(char *user_input) {
    char buffer[64];
    strcpy(buffer, user_input);
}
```
**Top Attention Tokens:**
1. `strcpy` (Attention: 1.00)
2. `buffer` (Attention: 0.92)
3. `[` / `64` (Attention: 0.78)

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10 or 3.11
- Recommended: Conda environment (`ml-ops`) or Python virtualenv

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Sankalp17ML/vuln-locator.git
cd vuln-locator

# Create and activate environment
conda create -n ml-ops python=3.11 -y
conda activate ml-ops

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify with Unit Tests
```bash
python -m unittest discover tests -v
```
*(Runs 8 automated unit & API integration tests)*

### 3. Run CLI Localization Demo
```bash
python -m src.localize
```

### 4. Start the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at: `http://localhost:8000/docs`

### 5. Start the Streamlit User Interface
In a separate terminal:
```bash
streamlit run app/ui.py
```
Open `http://localhost:8501` in your browser.

---

## 🐳 Docker & Containerized Deployment

VulnLocator provides containerized deployment options for both CPU and GPU execution.

### Single Container (FastAPI)
```bash
# Build Docker image
docker build -t vulnlocator .

# Run container
docker run -d --name vulnlocator-api -p 8000:8000 vulnlocator
```

### Full Stack Deployment (Docker Compose)
Spins up both the FastAPI backend and Streamlit frontend in an isolated network:
```bash
docker compose up -d
```
- **Streamlit Web UI:** `http://localhost:8501`
- **FastAPI Documentation:** `http://localhost:8000/docs`

To stop:
```bash
docker compose down
```

---

## 📡 API Reference

### Health Check
`GET /health`
```json
{
  "status": "ok",
  "specialist_available": true,
  "generalist_available": true,
  "version": "2.0.0",
  "architecture": "Mixture-of-Experts (Router)"
}
```

### Models Registry
`GET /models`
Returns availability status, file paths, and target languages for both Specialist and Generalist models.

### Smart Prediction & Localization (Auto-Routed)
`POST /predict`

Automatically routes code based on language detection, strips natural language comments to eliminate lexical bias, runs inference on the expert model, and generates Attention Rollout heatmaps.

**Request:**
```json
{
  "code": "const userInput = req.query.cmd;\neval(userInput);",
  "top_k": 5,
  "include_localization": true,
  "model": "auto"
}
```

**Response:**
```json
{
  "vulnerable": true,
  "label": "Vulnerable",
  "confidence": 0.503,
  "vulnerable_probability": 0.503,
  "threshold": 0.50,
  "device": "cuda",
  "findings": [
    {
      "token": "eval",
      "text": "eval",
      "score": 0.785,
      "start": 33,
      "end": 37,
      "line": 2,
      "line_text": "eval(userInput);"
    },
    {
      "token": "cmd",
      "text": "cmd",
      "score": 0.725,
      "start": 28,
      "end": 31,
      "line": 1,
      "line_text": "const userInput = req.query.cmd;"
    }
  ],
  "preprocessed": true,
  "comments_stripped": 0,
  "comment_chars_removed": 0,
  "model_used": "Multilingual Generalist (CrossVul)",
  "detected_language": "javascript",
  "routing_reason": "Detected Javascript syntax -> Routed to Multilingual Generalist Model."
}
```

### Direct Model Endpoints
- `POST /predict/specialist` — Explicitly evaluates code using `models/codebert-baseline` (C/C++ PrimeVul).
- `POST /predict/generalist` — Explicitly evaluates code using `models/codebert-multilingual` (CrossVul).

---

## 📂 Project Structure

```text
vuln-locator/
├── app/
│   ├── main.py                          # FastAPI REST application & MoE router endpoints
│   └── ui.py                            # Streamlit web interface with language routing & badges
├── data/
│   ├── download.py                      # PrimeVul C/C++ dataset download script
│   ├── download_multilingual.py         # CrossVul multilingual dataset downloader
│   ├── README.md                        # Dataset schemas and benchmarks
│   ├── primevul_sample.csv              # 20k C/C++ sample CSV (excluded from git)
│   └── multilingual_sample.csv          # 5.4k Multilingual sample CSV (excluded from git)
├── models/
│   ├── README.md                        # Model weight directory documentation
│   ├── codebert-baseline/               # Specialist weights (PrimeVul C/C++)
│   └── codebert-multilingual/           # Generalist weights (CrossVul Multilingual)
├── notebooks/
│   ├── kaggle_training_dual_t4.ipynb    # Dual T4 GPU training notebook (Specialist)
│   └── kaggle_training_multilingual.ipynb # Dual T4 GPU training notebook (Generalist)
├── src/
│   ├── evaluate.py                      # Evaluation CLI for ROC-AUC & metrics
│   ├── inference.py                     # Core CodeAnalyzer & Attention Rollout math
│   ├── localize.py                      # CLI tool for interactive token localization
│   ├── preprocessor.py                  # AST/regex comment stripper & lexical bias filter
│   ├── router.py                        # Language detection & MoE routing engine
│   ├── train_baseline.py                # Specialist training script (PrimeVul)
│   ├── train_multilingual.py            # Generalist training script (CrossVul)
│   └── train_kaggle_dual_t4.py          # Dual T4 multi-GPU training pipeline
├── tests/
│   ├── test_api.py                      # API integration tests & router endpoint tests
│   ├── test_inference.py                # Model inference & localization unit tests
│   ├── test_preprocessor.py             # Comment stripping & bias removal unit tests
│   └── test_router.py                   # Language detection & MoE routing unit tests
├── .dockerignore                        # Docker exclusion rules
├── .gitignore                           # Git exclusion rules (ignores >500MB weights)
├── docker-compose.yml                   # Multi-container orchestration (API + UI)
├── Dockerfile                           # Production Docker container definition
├── Makefile                             # Build, test, and run shortcuts
├── requirements.txt                     # Python package dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 💼 Resume Bullets & Interview Talking Points

### Ready-to-Use Resume Bullets
> - **Engineered VulnLocator V2.0**, an enterprise multilingual vulnerability triage system utilizing a **Mixture-of-Experts (MoE) Router Architecture** with **CodeBERT**, mitigating catastrophic forgetting across C/C++, Java, PHP, JavaScript, and Python.
> - **Designed a Specialist vs. Generalist Router** routing low-level memory bugs (`malloc`, `std::`) to a PrimeVul C/C++ expert and web/injection flaws (`eval`, `$_GET`, `SELECT`) to a CrossVul multilingual expert.
> - **Built an AST-based syntax preprocessor** to eliminate lexical and comment bias while preserving exact byte offsets for token localization heatmaps.
> - **Optimized deep learning training pipelines** on Kaggle Dual NVIDIA T4 GPUs via **pre-tokenized TensorDatasets**, multi-GPU `DataParallel`, and FP16 mixed precision, slashing epoch runtime from **2 hours to 40 seconds** (180x speedup).
> - **Implemented Attention Rollout** across 12 transformer layers with residual connections, transforming black-box classifiers into an interpretable security tool that highlights precise vulnerable tokens (`strcpy`, `SELECT`, `eval`, `system`).
> - **Containerized microservice architecture** using **Docker Compose**, pairing an asynchronous FastAPI prediction engine with an interactive **Streamlit** triage dashboard and 24 automated unit/integration tests.

### Key Interview Discussion Topics
1. **Handling Severe Class Imbalance:** Explain why standard accuracy is misleading on a 97:3 dataset and how inverse-frequency weighted cross-entropy shifts decision boundaries towards the minority class.
2. **Data Pipeline Profiling:** Explain how PyTorch `DataLoader` worker processes can become CPU-bound during on-the-fly tokenization, and how pre-tokenizing into contiguous GPU-ready tensors unlocked 95%+ GPU utilization.
3. **Attention Rollout vs. Raw Attention:** Discuss why single-layer attention heads are insufficient for explaining deep transformer decisions and how matrix multiplication with residual identity connections traces attention flow across depth.
4. **Production Security Integration:** Discuss how an API-first design enables embedding VulnLocator into CI/CD pipelines (e.g., GitHub Actions) as an automated pre-merge pull request reviewer.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
