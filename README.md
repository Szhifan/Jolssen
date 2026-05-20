# Project Background: Jolssen — Joint Label-Span Encoding for ASAS

## Overview and Motivation

Automatic Short Answer Scoring (ASAS) is a central educational NLP task. Two problems motivate this work:

1. **Label space heterogeneity.** Existing approaches tie model architectures to benchmark-specific label spaces (fixed-head classifiers), making cross-benchmark transfer hard.
2. **Data privacy.** Sending student responses to external LLM APIs violates data-protection regulations (GDPR, UK DfE guidelines), so locally deployable models are required.

The proposed system, **Jolssen** (Joint Label-Span Encoding), addresses both: it encodes rubric descriptions directly alongside the student answer in a single forward pass, avoiding a fixed global label space while keeping inference entirely local.

## Main Modeling Idea

The core idea is **sequence-to-label alignment**:

- Encode the student answer together with task context (question, sample solution) and all rubric-level descriptions as one joint sequence using an instruction-tuned LLM.
- Extract a per-rubric representation by taking the hidden state of the **last token** of each rubric span.
- Extract a global sequence representation from the **final token** of the full sequence (EOS position).
- Align the sequence representation with each rubric representation via a fusion function to produce a score per level.
- Select the highest-scoring level as prediction; optimize with listwise cross-entropy.

This allows scoring of a variable number of rubric levels in a single forward pass without expanding instances into answer-rubric pairs.

## Input Format

Each instance is serialized into a single string using **XML-style field tags**:

```text
<question> ... </question>
<sample solution> ... </sample solution>
<answer> ... </answer>
<rubric> Rubric text level 0 </rubric>
<rubric> Rubric text level 1 </rubric>
<rubric> Rubric text level 2 </rubric>
Instruction (sampled from pool)
```

Rubric span boundaries are tracked in the tokenized sequence. The closing `>` token of each `<rubric>` tag is used as the pooling anchor for the rubric representation.

An instruction is randomly sampled from a pool of M=10 templates per training and test instance, encouraging robustness to instruction phrasing. For multilingual benchmarks (German, Portuguese), field tags and instructions are translated; the `<rubric>` tag is kept in English across all benchmarks. See `scripts_asag/data_processing/benchmark_meta.py` for the full instruction pools.

## Architecture: Jolssen

### Sequence and Rubric Representations

Given encoder hidden states **H** ∈ ℝ^(T×d):

- **Rubric representation** r_k = H[e_k], the hidden state at the last token position of rubric span k (exclusive-end indexing: position e_k - 1 in 0-indexed terms).
- **Sequence representation** z = H[T], the final-token embedding. This carries information from the full sequence under autoregressive attention and is the standard pooling choice for LLM-based classifiers.

### Alignment Fusion Functions

Three fusion variants are compared:

| Name | Expression | Linear head input size |
| --- | --- | --- |
| `concat` | [z; r_k] | 2H |
| `diff` | z − r_k | H |
| `condiff` | [z; r_k; (z − r_k)] | 3H |

Each variant feeds the fused representation through a linear layer to produce a scalar score s_k per rubric level.

Additional experimental variants in `SpanFuser` include bilinear (`p-bl`), gated (`p-gate`), text-span-based (`t-concat`, `t-diff`), and no-alignment (`p-only`, `l-only`) modes.

### Training Objective

Scores s_k over all K rubric levels are normalized with softmax into a probability distribution:

```text
p_k = exp(s_k) / sum_k' exp(s_k')
```

Optimized with cross-entropy against the gold label. Batches may mix instances with different label cardinalities: padding slots are filled with −∞ before softmax via a validity mask, so they do not contribute to the distribution.

### Rubric-Independent Attention (RIA)

Under standard causal attention, the hidden state of rubric k attends to all tokens from rubrics 1,...,k−1, making the rubric representation dependent on the ordering and wording of preceding rubrics. This degrades zero-shot transfer to unseen benchmarks that have different rubric sets.

**RIA** is a block-sparse attention masking strategy that removes this dependency:

- The input is split into a **context block** (question + sample solution + answer) and K **rubric blocks**.
- Each rubric block attends only to the context block and its own tokens, never to other rubric blocks.
- **Position ID reindexing**: each rubric block's position IDs are reset to start from the last context token position, so positional embeddings are also independent across rubrics.

Under RIA, the scoring head uses the rubric representation r_k directly (`l-only` fusion) without sequence alignment, because the sequence-level interaction is already baked into the context-conditioned rubric representation. Instruction tokens are excluded from RIA inputs since full-sequence conditioning is not needed.

RIA substantially improves zero-shot transfer to unseen and cross-lingual benchmarks while retaining single-pass efficiency.

## Model Variants and Baselines

### Jolssen (main model, `span`)

`SpanAlignmentModel` in `modelling/modelling_span.py`. Encodes the full joint sequence, extracts per-rubric span representations, and aligns them with the sequence embedding via a fusion function. Two sub-variants:

- **Standard** (causal attention): competitive for mono-benchmark training.
- **RIA** (`--rubric-independent-attn --reindex-rub`): substantially better for zero-shot and cross-lingual transfer.

### Baselines

**No Alignment (`cls`)**
Standard sequence classifier (without or with rubric text in the input, denoted +rubric / -rubric). Uses a fixed-label linear head trained with cross-entropy; excludes ASAP-SAS because its per-question label counts vary. Implemented via `--model-class span --span-fuse-type p-only` (no span extraction required) or by dropping rubrics with `--drop-all-rubrics`.

**Rubric Retrieval (`xnet`)**
`AsagXnet` in `modelling/modelling_xnet.py`. Each (answer, rubric) pair is encoded independently as a separate sequence; a linear head produces a scalar score per pair. During training, per-pair logits are regrouped and normalized with softmax for listwise cross-entropy. At inference, the highest-scoring rubric is selected. Cannot model inter-rubric interactions; training cost scales linearly with the number of rubric levels. Not evaluated with 7B/8B models due to prohibitive cost.

**LLM Generation (`gen`)**
A causal LLM fine-tuned to generate the label string autoregressively (e.g. "Correct", "Partially Correct", "Incorrect"). No classification head; prediction is the highest-likelihood generated token(s). Implemented via `scripts_asag/train_gen.py`.

## Backbone Models

Experiments use four instruction-tuned LLMs as encoders:

| Model | Size | Notes |
| --- | --- | --- |
| Llama-3.2-1B-Instruct | 1B | Smallest; used for all baselines |
| Llama-3.2-3B-Instruct | 3B | Balanced size |
| Mistral-7B-v0.1 | 7B | Rubric Retrieval excluded |
| Llama-3.1-8B-Instruct | 8B | Rubric Retrieval excluded |

All models are fine-tuned with LoRA (rank 64, α=64, target="all-linear"). Learning rate: 2×10⁻⁴ for 1B/3B, 5×10⁻⁵ for 7B/8B. Optimizer: paged AdamW for LLMs, AdamW otherwise.

## ASAS Benchmarks

Six public ASAS benchmarks are used. Native rubric annotations exist only for `alice_lp`; all others are augmented with LLM-synthesized rubrics using `gpt-4o-mini`.

| Benchmark | Key | Lang | Levels | Original Context | Augmented | Test Splits |
| --- | --- | --- | --- | --- | --- | --- |
| **ALICE-LP** | `alice_lp` | DE | 3 | q, s, r (native) | — | UA, UQ |
| **ASAP-SAS** | `asap_sas` | EN | 3–4 | q, r (native) | s | UA |
| **iStudio** | `istudio` | EN | 3 | q, qc, s | r | UA |
| **PT-ASAG** | `pt_asag` | PT | 4 | q, s | r | UA |
| **SciEntsBank** | `scientsbank` | EN | 3 | q, s | r | UA, UQ, UD |
| **Beetle** | `beetle` | EN | 3 | q, s | r | UA, UQ |

*q = question; s = sample solution; r = rubrics; qc = question context. Test splits: UA (unseen answers), UQ (unseen questions), UD (unseen domains).*

Note: SciEntsBank uses correct/partially\_correct/incorrect (not the original correct/incorrect/contradictory) to follow the convention of other benchmarks.

The benchmark_meta also registers `alice_ke` (knowledge elements) and `alice_sk` (scientific skills) as additional ALICE subtasks, though these are not reported in the main paper results.

### LLM-Based Benchmark Augmentation

For datasets without native rubric annotations, `gpt-4o-mini` generates per-level rubric descriptions given the question, available reference answers, and up to 10 student answer examples per level. ASAP-SAS also lacks sample solutions; these are generated separately from the question alone. Details and prompts are in `generate_rubrics_workflow.ipynb` and `generate_reference_answers.ipynb`.

## Experimental Protocol

### Mono-benchmark Training

Train and evaluate on a single benchmark. Evaluation is on the held-out test split(s) of that benchmark (UA/UQ/UD where applicable). Used for ablation studies and cross-lingual zero-shot evaluation (train EN, eval DE/PT).

### Joint Training and Cross-benchmark Evaluation

Train jointly on the English ASAS benchmarks (ASAP-SAS, SciEntsBank, Beetle), with iStudio held out as an unseen-benchmark validation set. Evaluate:

1. On English training benchmarks — measures positive transfer from joint training.
2. On non-English benchmarks (Alice-LP, PT-ASAG) under zero-shot and few-shot settings — measures cross-lingual and domain transfer.

## Cross-Benchmark transfer

The framework supports training on dataset X and testing on Y


## Training Tasks

Three distinct training scenarios are supported.

### 1. Mono-dataset ASAS Training (`scripts_asag/train.py`)

Train and evaluate on a single ASAG benchmark via `TaskArguments` + `main()`. Entry point for mono-benchmark fine-tuning, ablation studies, and cross-lingual zero-shot evaluation.

### 2. Multi-dataset ASAG Training (`scripts_asag/train_multi.py`)

Train jointly on multiple ASAG benchmarks via `MultiTaskArguments` and `MultiTaskDataPipeline`. A single model learns shared representations across heterogeneous label spaces and languages. Per-benchmark metrics and aggregate scores are reported.

### 3. Mono-dataset Training for Non-ASAS Benchmarks (`scripts_others/train.py`)

Train and evaluate on a single non-ASAS benchmark using the same `TaskArguments` / `main()` pattern but routing through `DataPipelineOther`. Tests whether the alignment framework generalizes beyond ASAS.

## Codebase Map

### Training Entry Points

| File | Purpose |
| --- | --- |
| `scripts_asag/train.py` | Single-benchmark ASAG training. `TaskArguments` + `main()`. |
| `scripts_asag/train_gen.py` | LLM generation baseline training. |
| `scripts_asag/train_multi.py` | Multi-benchmark ASAG training. `MultiTaskArguments`, `MultiTaskDataPipeline`, `main()`. |
| `scripts_others/train.py` | Training on non-ASAS benchmarks (same `TaskArguments` / `main()` pattern). |
| `trainer.py` | Shared training infrastructure: `AsagTrainingArguments`, `ModelLoader` (loads backbone + LoRA + optional BnB quantization), `LoraAwareTrainer` (handles LoRA checkpoint loading), `AsagTrainer` (custom training loop), `compute_metrics()`. |

### Data Processing — ASAG (`scripts_asag/data_processing/`)

| File | Purpose |
| --- | --- |
| `benchmark_meta.py` | Metadata registry for all ASAG benchmarks: paths, label maps, rubric configs, instruction pools per language. |
| `general_asag_loader.py` | Base class `ASAG_Data_Loader` — shared loading and preprocessing logic. |
| `alice_asag_loader.py` | `Alice_Loader(ASAG_Data_Loader)` — ALICE-specific loading (LP/KE/SK subtasks). |
| `lasa_format.py` | `build_lasa_llm_input()` — serializes one instance into the XML-tagged input string and returns character-level span boundaries for the answer and each rubric. |
| `data_prep.py` | `DataPipeline` — tokenises and assembles model inputs for ASAG; `MultiTaskDataPipeline` for multi-benchmark training; helpers `is_llm_model()`, `get_tokenizer()`. |

### Data Processing — Other Benchmarks (`scripts_others/data_processing/`)

| File | Purpose |
| --- | --- |
| `benchmark_meta.py` | Metadata registry for non-ASAS benchmarks: format function names, label configs, language tags, suffix lists. |
| `data_loader.py` | Per-benchmark format functions: `format_piqa`, `format_xstance`, `format_semeval2016`, `format_figqa`, `format_ag_news`, `format_imdb`, `format_eic`. |
| `data_prep.py` | `OtherDataLoader`, `DataPipelineOther` — loads and tokenises non-ASAS data; mirrors the ASAG `DataPipeline` interface. |

### Modelling (`modelling/`)

| File | Purpose |
| --- | --- |
| `modelling_utils.py` | `BaseAsagModel` (base class with LoRA/BnB support, `listwise_loss`, `freeze_lm_backbone_layers`), `Pooler` (avg / weightedavg / cls / last), `BackwardSupportedArguments` (model config dataclass), `build_rubric_block_attention_mask()` and `build_rubric_block_position_ids()` (RIA helpers), attention-mask helpers, `flip_tensor`. |
| `modelling_span.py` | `SpanFuser` — fuses span representations (concat/diff/condiff/bilinear/gate/p-only/l-only); `SpanAlignmentModel(BaseAsagModel)` — the main Jolssen model. |
| `modelling_xnet.py` | `AsagXnet(BaseAsagModel)` — rubric-retrieval / cross-encoder baseline. |
| `custom_llama.py` | Patched HF Llama classes (`LlamaDecoderLayer`, `LlamaModel`, `LlamaForSequenceClassification`) with residual-connection and bidirectionality support. |
| `custom_mistral.py` | Same patches for Mistral (`MistralDecoderLayer`, `MistralModel`, `MistralForSequenceClassification`). |

### Utilities

| File | Purpose |
| --- | --- |
| `utils.py` | `evaluate()`, `metrics_calc()`, `eval_report()`, `save_report()`, `save_prediction()`, `get_label_weights()`, `per_qid_metrics()`, misc helpers (`set_seed`, `configure_logging`, `batch_to_device`, `clear_gpu_memory`). |
