# Comparative Study of Neural Machine Translation Paradigms: English-to-Hindi

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20%7C%20Datasets-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-magenta)](https://github.com/huggingface/peft)
[![Dataset](https://img.shields.io/badge/Dataset-cfilt%2Fiitb--english--hindi-green)](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
[![Metric](https://img.shields.io/badge/Metrics-SacreBLEU%20%7C%20chrF-informational)](https://github.com/mjpost/sacrebleu)

An empirical benchmark evaluating five distinct neural machine translation paradigms for English-to-Hindi translation using the benchmark **IIT Bombay English-Hindi Parallel Corpus**. The study contrasts classical sequence-to-sequence training from scratch against multilingual pretrained encoder-decoders, parameter-efficient fine-tuning (LoRA), zero-shot prompting of modern Large Language Models (LLMs), and dedicated NMT production models.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture & Evaluated Paradigms](#system-architecture--evaluated-paradigms)
- [Benchmark Results](#benchmark-results)
- [In-Depth Methodology & Technical Analysis](#in-depth-methodology--technical-analysis)
  - [1. Custom Transformer Encoder-Decoder (From Scratch)](#1-custom-transformer-encoder-decoder-from-scratch)
  - [2. Multilingual Seq2Seq Pretraining: mT5-small](#2-multilingual-seq2seq-pretraining-mt5-small)
  - [3. Small Causal LLM: TinyLlama 1.1B (Zero-Shot vs. LoRA)](#3-small-causal-llm-tinyllama-11b-zero-shot-vs-lora)
  - [4. Instruction-Tuned LLM: Gemma 2 2B (Zero-Shot vs. LoRA)](#4-instruction-tuned-llm-gemma-2-2b-zero-shot-vs-lora)
  - [5. Production Translation Benchmark: OPUS-MT](#5-production-translation-benchmark-opus-mt)
- [Qualitative Translation Showcase](#qualitative-translation-showcase)
- [Engineering Insights & Critical Limitations](#engineering-insights--critical-limitations)
- [Repository Structure](#repository-structure)
- [Reproduction & Execution Guide](#reproduction--execution-guide)
- [References & Acknowledgments](#references--acknowledgments)

---

## Executive Summary

English-to-Hindi translation presents unique challenges due to differences in syntactic typology (Subject-Verb-Object in English vs. Subject-Object-Verb in Hindi), rich morphological inflection, and the Devanagari script's complex conjunct and diacritic system.

This project investigates the trade-offs across model scale, pretraining objectives, fine-tuning techniques, and compute budgets:

1. **Instruction-Tuned LLMs Excel Zero-Shot**: Google's `Gemma-2-2B-IT` achieved the highest translation quality among LLMs (**12.43 BLEU**, **38.84 chrF**) strictly through zero-shot prompting without parameter updates.
2. **LoRA Fine-Tuning Nuance on Small Datasets**: Applying Low-Rank Adaptation (LoRA) to Gemma 2 over 2,000 domain pairs led to a score decrease (**9.05 BLEU**, **35.92 chrF**), demonstrating risk of distribution shift, formatting rigidity, or catastrophic forgetting when adapting powerful zero-shot models on small parallel subsets.
3. **From-Scratch Deep Learning Requires Massive Scale**: A custom 6-layer Transformer trained from scratch on 40,000 sentence pairs yielded **0.15 BLEU** (**4.72 chrF**), highlighting the steep data starvation barrier of neural translation models without pretraining.
4. **FP16 Instability in mT5**: Fine-tuning `google/mt5-small` exposed known numerical instabilities in T5 LayerNorm and vocabulary projection under FP16 mixed precision, resulting in degenerated sentinel tokens (`<extra_id_0>`).
5. **Specialized NMT Remains Highly Competitive**: The lightweight `Helsinki-NLP/opus-mt-en-hi` baseline (76.4M parameters) achieved **9.88 BLEU** and **32.33 chrF** across the full 2,507 test set with sub-second inference (**~0.30s/sentence**).

---

## System Architecture & Evaluated Paradigms

```
                                  ┌────────────────────────────────────────┐
                                  │      cfilt/iitb-english-hindi          │
                                  │   (Official IIT Bombay Parallel Data)  │
                                  └───────────────────┬────────────────────┘
                                                      │
             ┌──────────────────────┬─────────────────┴───────────────┬─────────────────────────┐
             │                      │                                 │                         │
             ▼                      ▼                                 ▼                         ▼
   ┌───────────────────┐  ┌───────────────────┐             ┌───────────────────┐     ┌───────────────────┐
   │ Custom Transformer│  │    mT5-small      │             │ TinyLlama-1.1B    │     │   Gemma-2-2B-IT   │
   │   (From Scratch)  │  │  (Seq2Seq LM)     │             │    (Causal LM)    │     │    (Causal LM)    │
   └─────────┬─────────┘  └─────────┬─────────┘             └─────────┬─────────┘     └─────────┬─────────┘
             │                      │                                 │                         │
      PyTorch nn.Module     HuggingFace Trainer              Prompting vs. LoRA        Prompting vs. LoRA
     Vocab: Unicode NFC      FP16/FP32 Diagnostic               (r=8, alpha=16)           (r=8, alpha=16)
             │                      │                                 │                         │
             ▼                      ▼                                 ▼                         ▼
   ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
   │                                 Standardized Evaluation                                      │
   │               SacreBLEU (Corpus BLEU) + chrF (Character n-gram F-score)                      │
   │                      Latency Profiling (seconds/sentence) & GPU Memory                       │
   └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results

The table below compiles all experimental runs recorded in [`results/model_comparison_results.csv`](results/model_comparison_results.csv) and [`results/gemma_lora_results.json`](results/gemma_lora_results.json).

| Model | Approach | Test Set ($N$) | SacreBLEU | chrF | Avg Latency (s/sent) | Training Time | Total Parameters | Trainable Parameters | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gemma 2 2B** | Zero-Shot Prompting | 100 | **12.43** | **38.84** | 1.97 s | — | 2,614,341,888 | 0 (Inference only) | ✅ Completed (Top Performer) |
| **OPUS-MT (`en-hi`)** | Pretrained NMT Baseline | 2,507 | 9.88 | 32.33 | **0.30 s** | — | 76,381,184 | — | ✅ Completed (Fastest Benchmark) |
| **Gemma 2 2B** | LoRA Adapter ($r=8$) | 100 | 9.05 | 35.92 | 4.38 s | 14.48 min | 2,617,536,768 | 3,194,880 (0.12%) | ⚠️ Completed; scores declined |
| **TinyLlama 1.1B** | LoRA Adapter ($r=8$) | 100 | 0.66 | 12.99 | 3.24 s | 4.20 min | 1,101,174,784 | 1,126,400 (0.10%) | ✅ Completed |
| **TinyLlama 1.1B** | Zero-Shot Prompting | 100 | 0.63 | 13.25 | 3.08 s | — | 1,100,048,384 | 0 (Inference only) | ✅ Completed |
| **Transformer (Scratch)** | 6-Layer Enc-Dec | 2,507 | 0.15 | 4.72 | 0.18 s | 3.85 min | ~12.5M | All (~12.5M) | ✅ Completed (Data-starved) |
| **mT5-small** | Full Seq2Seq Fine-Tuning | — | *Invalid* | *Invalid* | — | 28.06 min | 300,176,768 | 300,176,768 (100%) | ❌ Unstable (FP16 NaN / collapse) |

> [!IMPORTANT]
> **Evaluation Sample Size Notice**: LLM evaluations (Gemma 2 and TinyLlama) were run on the first 100 examples of the official test set due to GPU memory and generation latency constraints. The Transformer-from-scratch and OPUS-MT models were evaluated against the **entire official test split of 2,507 sentences**. Direct comparisons across different test-set sizes should be interpreted with this context.

---

## In-Depth Methodology & Technical Analysis

### 1. Custom Transformer Encoder-Decoder (From Scratch)

- **Architecture**: Standard sequence-to-sequence Transformer following Vaswani et al.
  - Embedding Dimension ($d_{\text{model}}$): 256
  - Attention Heads ($n_{\text{head}}$): 4
  - Encoder / Decoder Layers: 3 encoder layers, 3 decoder layers
  - Feedforward Dimension ($d_{\text{ff}}$): 512
  - Positional Encoding: Sinusoidal
  - Maximum Sequence Length: 40 tokens
- **Data & Preprocessing**:
  - Training subset: 40,000 sentence pairs; Validation subset: 2,000 sentence pairs; Test set: 2,507 pairs.
  - Unicode Normalization: Enforced `unicodedata.normalize("NFC", text)` to ensure compound Devanagari characters (consonants + vowel signs / matras) are unified.
  - Tokenization: Devanagari word tokenization preserving Indic Unicode code blocks and punctuation separation. Minimum token frequency threshold $\ge 2$.
- **Training Protocol**: Adam optimizer ($\text{lr} = 3 \times 10^{-4}$), Cross-Entropy loss with `ignore_index` set to padding token ID, batch size 64, trained for 5 epochs.
- **Outcome Analysis**:
  - The model achieved **0.15 BLEU** and **4.72 chrF**.
  - While it learned basic syntactic alignment (e.g., generating common functional Hindi postpositions such as "के", "में", "को"), training a competitive neural translation model from scratch requires millions of clean parallel pairs. 40,000 examples are insufficient to populate a rich Devanagari vocabulary and capture long-distance reordering.

---

### 2. Multilingual Seq2Seq Pretraining: mT5-small

- **Model Specification**: `google/mt5-small` (300M parameters, 250,100 token multilingual vocabulary).
- **Initial Training**: 2 epochs, batch size 8, learning rate $5 \times 10^{-5}$, FP16 mixed precision via `Seq2SeqTrainer`.
- **Forensic Failure Analysis & Diagnostics**:
  - During evaluation, generated outputs collapsed into repeated sentinel tokens (`<extra_id_0>`).
  - Diagnostic inspection revealed:
    1. **FP16 Underflow / NaN Gradients**: mT5 uses standard LayerNorm without additive bias terms, which causes numerical instability under standard FP16. Gradient norms diverged to `NaN`, leading to zero effective training loss updates.
    2. **Untied Weight Divergence**: Inspection showed `shared.weight` and `lm_head.weight` had a mean absolute difference of **11.47**, indicating the output projection layer drifted completely out of alignment with input embeddings.
    3. **Stability Recovery**: Reducing the learning rate to $1 \times 10^{-5}$, applying `max_grad_norm=1.0`, and disabling FP16 restored finite loss (`14.59`), but the model's vocabulary distribution remained biased toward sentinel generation within the assigned step budget.

---

### 3. Small Causal LLM: TinyLlama 1.1B (Zero-Shot vs. LoRA)

- **Model Specification**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B parameters, 2048 context window).
- **Zero-Shot Prompting**:
  - Chat template with system directive: *"Translate English to natural Hindi in Devanagari script. Output only the translation."*
  - Results: **0.63 BLEU**, **13.25 chrF**.
- **LoRA Adaptation**:
  - Trainable parameters: $1,126,400$ ($0.102\%$ of total weights).
  - Hyperparameters: Rank $r=8$, $\alpha=16$, dropout $0.05$, target modules `["q_proj", "v_proj"]`.
  - Trained on 2,000 parallel pairs in 4.20 minutes on a Tesla T4.
  - Results: **0.66 BLEU**, **12.99 chrF**.
- **Outcome Analysis**:
  - TinyLlama's pretraining corpus consists primarily of English data (SlimPajama and Starcoder). Its vocabulary and internal representations lack adequate coverage of the Devanagari script, preventing meaningful improvement even after adapter tuning.

---

### 4. Instruction-Tuned LLM: Gemma 2 2B (Zero-Shot vs. LoRA)

- **Model Specification**: `google/gemma-2-2b-it` (2.61B parameters, FP16 precision).
- **Zero-Shot Evaluation**:
  - Evaluated on the first 100 test samples using structured chat prompts.
  - Achieved the highest metrics among all generative models: **12.43 BLEU**, **38.84 chrF** with average latency of **1.97 seconds/sentence**.
  - Generated fluent, contextually nuanced Hindi translations with correct SOV structure.
- **LoRA Adaptation**:
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`.
  - Trainable parameters: $3,194,880$ ($0.122\%$ of 2.62B total).
  - Training duration: 14.48 minutes (final training loss: 2.1948).
  - Results: **9.05 BLEU**, **35.92 chrF** (latency: 4.38 s/sentence).
- **Why Did LoRA Scores Decline?**:
  1. **Domain Overfitting**: Adapting on only 2,000 sentence pairs overfitted the adapter to specific sentence constructions and vocabulary in that subset.
  2. **Format Conditioning Shift**: The fine-tuned adapter learned to emit shorter, conservative translations ("कार में एक काले बॉक्स?" vs. "कार में काले बॉक्स है?"), slightly penalizing n-gram overlap against reference test targets.
  3. **Loss of Broad Semantic Generalization**: The pretrained Gemma 2 instruction model already possessed strong cross-lingual reasoning; low-volume gradient updates without a regularization penalty diluted its native zero-shot calibration.

---

### 5. Production Translation Benchmark: OPUS-MT

- **Model Specification**: `Helsinki-NLP/opus-mt-en-hi` (MarianMT architecture, 76.4M parameters).
- **Results on Full Official Test Set ($N=2,507$)**:
  - **SacreBLEU**: 9.88
  - **chrF**: 32.33
  - **Inference Speed**: 0.295s per sentence (total 739.94s for 2,507 sentences).
- **Key Takeaway**: Despite having **34× fewer parameters** than Gemma 2 and **14× fewer** than TinyLlama, OPUS-MT remains an exceptionally effective, cost-efficient baseline for production translation tasks.

---

## Qualitative Translation Showcase

Real outputs from the test set highlighting translation behavior across paradigms:

| Metric / Model | Source English & Translations |
| :--- | :--- |
| **Input Sentence 1** | `A black box in your car?` |
| **Ground Truth (Ref)** | आपकी कार में ब्लैक बॉक्स? |
| **Gemma 2 2B (Zero-Shot)** | कार में काले बॉक्स है? |
| **Gemma 2 2B (LoRA)** | कार में एक काले बॉक्स? |
| **OPUS-MT (Baseline)** | अपनी कार में एक ब्लैक बॉक्स? |
| **Transformer (Scratch)** | क्या आप अपने `<unk>` में `<unk>` |
| **mT5-small (Attempt)** | `<extra_id_0>` |
| **TinyLlama 1.1B (LoRA)** | *Incoherent Roman/Devanagari output* |

<details>
<summary><b>View Complex Example (Compound Clause Translation)</b></summary>

- **Source English**:
  > *"As America's road planners struggle to find the cash to mend a crumbling highway system, many are beginning to see a solution in a little black box that fits neatly by the dashboard of your car."*
- **Reference (IITB)**:
  > *जबकि अमेरिका के सड़क योजनाकार, ध्वस्त होते हुए हाईवे सिस्टम को सुधारने के लिए धन की कमी से जूझ रहे हैं, वहीं बहुत-से लोग इसका समाधान छोटे से ब्लैक बॉक्स में देख रहे हैं, जो आपकी कार के डैशबोर्ड पर सफ़ाई से फिट हो जाता है।*
- **OPUS-MT (`en-hi`)**:
  > *के रूप में अमेरिका के सड़क योजनार एक टूटी हुई सड़क प्रणाली को सुधारने के लिए नकदी को खोजने के लिए संघर्ष कर रहे हैं, कई एक छोटे से काले बॉक्स में एक समाधान देखने के लिए शुरू कर रहे हैं कि अपनी कार के माध्यम से फिट.*
- **Gemma 2 2B (LoRA)**:
  > *अमेरिका के रोड प्लानर इस बात को देखकर कि उनकी सड़कें खराब हो रही हैं, उन्हें एक छोटी सी लाल रंग की बॉक्स में एक समाधान ढूंढना पड़ रहा है जो आपके कार के डैशबोर्ड के पास बैठता है।*
- **Analysis**:
  - OPUS-MT maintains accurate vocabulary ("टूटी हुई सड़क प्रणाली") but struggles slightly with the initial temporal subordinating clause ("के रूप में" literal translation of "As").
  - Gemma 2 exhibits remarkable syntactic fluency, correctly restructuring the entire clause into idiomatic Hindi grammar, though introducing minor lexical divergence (substituting "काले" with "लाल").

</details>

---

## Engineering Insights & Critical Limitations

> [!NOTE]
> **Summary of Lessons Learned in Indic Machine Translation**:
>
> 1. **Script Representation in Pretraining Is Decisive**: Models like TinyLlama fail not because of attention mechanisms, but because non-Latin byte-pair tokens are severely fragmented into multi-byte pieces, destroying semantic cohesion.
> 2. **Indic Tokenization & NFC Normalization**: Devanagari text must always be normalized via `unicodedata.normalize("NFC", text)` before tokenization. Failure to do so splits dependent vowel signs (matras: ा, ि, ी, ु, ू) from their base consonants.
> 3. **T5 / mT5 Architecture Gotcha**: Do not train mT5 with standard `fp16`. Always utilize `bf16` or standard `fp32` to avoid LayerNorm overflow and gradient collapse.
> 4. **LoRA Fine-Tuning Sample Budget**: For 2B+ instruction models, fine-tuning on fewer than 10,000 diverse pairs can degrade zero-shot capabilities. Consider freezing projection layers or mixing in general instruction-following tasks during adapter training.

---

## Repository Structure

```
.
├── English_Hindi_Dataset_Preprocessing.ipynb   # Master notebook (data prep, training, eval)
├── README.md                                    # Comprehensive technical documentation
├── .gitignore                                   # Ignore checkpoints, cache & virtualenvs
└── results/
    ├── gemma_lora_results.json                  # Gemma 2 LoRA evaluation metrics
    └── model_comparison_results.csv             # Unified comparison across all 7 runs
```

---

## Reproduction & Execution Guide

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with at least 8 GB VRAM (Tesla T4 or higher recommended)
- Hugging Face account and user access token (for gated Gemma models)

### Step 1: Clone Repository & Install Dependencies
```bash
git clone https://github.com/your-username/FGAI-Assignment-3-English-Hindi-Translation.git
cd FGAI-Assignment-3-English-Hindi-Translation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft sacrebleu accelerate bitsandbytes
```

### Step 2: Hugging Face Authentication (Gemma 2)
Accept the model license on the [Gemma 2 2B IT page](https://huggingface.co/google/gemma-2-2b-it) and log in:
```bash
huggingface-cli login
```
*(In Google Colab, store your key in the **Secrets** manager as `HF_TOKEN` with notebook access enabled.)*

### Step 3: Run the Pipeline
Open [`English_Hindi_Dataset_Preprocessing.ipynb`](English_Hindi_Dataset_Preprocessing.ipynb) in Jupyter Lab, VS Code, or Google Colab:
- Run cells sequentially from Section 1 (Dataset Preprocessing) through Section 5 (Model Comparison Table).
- Generated predictions and evaluation CSVs are automatically exported to the `results/` directory.

---

## References & Acknowledgments

- **Dataset**: Kunchukuttan et al., *The IIT Bombay English-Hindi Parallel Corpus*, CFILT, IIT Bombay ([Hugging Face Dataset](https://huggingface.co/datasets/cfilt/iitb-english-hindi)).
- **Transformer**: Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017.
- **mT5**: Xue et al., *"mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"*, NAACL 2021.
- **TinyLlama**: Zhang et al., *"TinyLlama: An Open-Source Small Language Model"*, 2024.
- **Gemma 2**: Gemma Team, Google DeepMind, *"Gemma 2: Improving Open Language Models at a Practical Size"*, 2024.
- **OPUS-MT**: Tiedemann et al., *OPUS-MT — Building open translation services for the World*, EAMT 2020.
- **Metrics**: Post, Matt, *"A Call for Clarity in Reporting BLEU Scores"*, WMT 2018 (SacreBLEU).
