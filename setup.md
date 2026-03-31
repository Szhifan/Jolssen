# Project Background: Aligning Sequence to Labels for Unified ASAG and Beyon

## Main modeling idea
The core idea is **sequence-to-label alignment**:
- encode the student answer (and possibly task context such as question/sample solution),
- encode label descriptions / rubric levels,
- compute alignment between sequence representation and label representations,
- predict the score from those alignments.

This should allow:
- support for varying numbers of labels,
- a single forward pass prediction design,

## Architecture exploration planned in the proposal
The proposal highlights several axes of experimentation.

### 1. Sequence representation choice
Test how to represent the answer sequence:
- **span embedding** variants,
- **mean pooling**,
- **last token / EOS** representation,
- **CLS** representation where applicable.

A recurring question is whether, especially for unidirectional LLMs, the last token might better capture overall sequence meaning than span-style aggregation.

### 2. Label alignment mechanism
Directly align rubric/label embeddings with sequence embeddings.
Possible representations or feature-combination functions include:
- **difference (Diff)**
- **concatenation (Concat)**
- **bilinear attention**

The design space here is a major experimental variable.

### 3. Model variants and baselines

#### Span model (`span`)
The main proposed architecture. Encodes the full student answer (together with task context and all rubric descriptions) as **a single joint sequence**. Span pooling is used to extract per-rubric representations directly from this shared contextualised encoding. Alignment scores are computed from the resulting spans and predictions are made in a single forward pass.

#### Cross-encoder / xnet (`xnet`)
A **cross-encoder baseline** for comparison with the span model. Each (answer, rubric) pair is encoded **independently** as a separate sequence, yielding one scalar score per pair. During training, the individual per-pair logits for all rubrics belonging to the same answer are **regrouped** and passed through a softmax together, so the loss is still computed jointly over all rubric options (listwise cross-entropy). At inference the predicted rubric is the one with the highest score.

This design makes no assumptions about how many rubrics exist per question and requires no span extraction, but it cannot model interactions between rubric descriptions within the same forward pass.


## Dataset and benchmark strategy
A major part of the project is not just architecture, but also **benchmark unification / augmentation**.

### Rubric-based and reference-based ASAG
The proposed architecture is naturally suited to **rubric-based** scoring, but rubric-based benchmarks are limited.
So the proposal suggests augmenting conventional ASAG datasets with richer task metadata.

### Planned augmentation ideas
Use LLMs to synthesize or normalize:
- per-level rubrics,
- multiple reference/sample solutions,
- task instructions,
- multilingual variants.


## Universal ASAG objective
The longer-term goal is a **single cohesive model** trained across multiple benchmarks.
All benchmarks are registered in `src/data_processing_asag/benchmark_meta.py`.
Label semantics are rubric descriptions; default 3-level scale: *Incorrect / Partially Correct / Correct* (`LEVEL2LABEL`).

| Benchmark | Key | Lang | Labels | Context | Notes |
|---|---|---|---|---|---|
| **ALICE-LP** | `alice_lp` | DE | 3 | question, sample\_solution | ALICE learning-progress subtask; German student answers |
| **ALICE-KE** | `alice_ke` | DE | 3 | question, sample\_solution | ALICE knowledge-elements subtask; rubrics focus on concept usage |
| **ALICE-SK** | `alice_sk` | DE | 3 | question, sample\_solution | ALICE scientific-skills subtask; rubrics focus on cognitive/reasoning skills |
| **ASAP-SAS** | `asap_sas` | EN | varies | question, sample\_solution | Kaggle ASAP short-answer scoring; per-question label count |
| **SciEntsBank** | `scientsbank` | EN | 3 | question, sample\_solution | Science Entailment Bank; UA / UQ / UD test splits |
| **Beetle** | `beetle` | EN | 3 | question, sample\_solution | Electronics tutoring dialogues; UA / UQ test splits |
| **iSTudio** | `istudio` | EN | 3 | question\_context, question, sample\_solution | Includes broader question context field |
| **PT-ASAG** | `pt_asag` | PT | 4 | question, sample\_solution | Portuguese ASAG; cross-lingual zero-shot transfer target |

Key research questions include:
- Can a single model learn across heterogeneous benchmarks?
- Can rubric/reference normalization reduce benchmark mismatch?
- Can an English-trained model transfer to German/Portuguese in few-shot or zero-shot settings?

## Cross-lingual angle
Cross-lingual transfer is a defined part of the proposal.
Examples:
Train on all the english-language benchamrk and perform zero-shot evaluatio on 
german and portuguese datasets.

## Connection to broader NLP
The proposal explicitly states that the framework may generalize beyond ASAG to **sequence classification with label semantics**.

The analogy is:
- **answer** -> input text
- **question / sample solution / instruction** -> task-specific context
- **rubrics** -> label descriptions

## Non-ASAG benchmarks (`other_benchmarks/`)

Used to test whether the framework generalises beyond ASAG to other classification tasks with meaningful label semantics.
All benchmarks are registered in `src/data_processing_other/benchmark_meta.py`.

| Benchmark | Lang | Task | Labels |
|---|---|---|---|
| **Winogrande** | EN | Commonsense: choose the word that best completes a sentence | 2 (free-text options) |
| **PIQA** | EN | Physical commonsense: choose the more plausible solution to a goal | 2 (free-text options) |
| **FiQA** | EN | Figurative language: choose the ending matching the figurative meaning | 2 (free-text options) |
| **xStance** | EN/DE/FR/IT | Political stance of a comment toward a question | 2 Against/Favor |
| **SemEval-2016 Task 6** | EN | Stance of a tweet toward a named target | 3 Against/Favor/None |
| **C-STANCE** | ZH | Chinese zero-shot stance detection (~48 k pairs, ACL 2023) | 3 反对/支持/中立 |
| **Yelp Polarity** | EN | Binary sentiment of reviews (560 k train) | 2 Negative/Positive |
| **EIC** | EN | Edit intent in scientific paper revisions | 5 Claim/Clarity/Fact/Grammar/Other |

xStance is the primary cross-lingual benchmark; C-STANCE is the Chinese zero-shot transfer target.

## Training tasks

Three distinct training scenarios are supported, each with its own entry point.

### 1. Mono-dataset ASAG training (`train.py`)
Train and evaluate on a **single ASAG benchmark**.
- One benchmark is selected via `TaskArguments` (e.g. `alice_lp`, `beetle`, `asap_sas`).
- The model learns to align student-answer representations to the label/rubric descriptions for that benchmark only.
- Evaluation is performed on the held-out split(s) of the same benchmark (standard train / dev / test or UA / UQ / UD splits where applicable).
- Use case: benchmark-specific fine-tuning, ablation studies, cross-lingual zero-shot evaluation (train EN → eval DE/PT).

### 2. Multi-dataset ASAG training (`train_multitask.py`)
Train jointly on **multiple ASAG benchmarks** and evaluate on all of them.
- Benchmarks are specified via `MultiTaskArguments`; data is interleaved by `MultiTaskDataPipeline`.
- A single model learns shared representations across heterogeneous label spaces and languages.
- Per-benchmark metrics are reported alongside aggregate scores.
- Use case: universal ASAG model, studying cross-benchmark transfer, reducing per-task data requirements.

### 3. Mono-dataset training for non-ASAG benchmarks (`train_other.py`)
Train and evaluate on a **single non-ASAG benchmark** (stance, sentiment, commonsense, etc.).
- Uses the same `TaskArguments` / `main()` pattern as `train.py` but routes through `DataPipelineOther`.
- Tests whether the sequence-to-label alignment framework generalises beyond ASAG to other classification tasks with meaningful label semantics.
- Use case: out-of-domain generalisation experiments, validating the universality of the alignment approach.

## Codebase map (`src/`)

### Training entry points

| File | Purpose |
|---|---|
| `train.py` | Single-benchmark ASAG training. Defines `TaskArguments` + `main()`. |
| `train_multitask.py` | Multi-benchmark ASAG training. `MultiTaskArguments`, `MultiTaskDataPipeline`, `main()`. |
| `train_other.py` | Training on non-ASAG benchmarks (same `TaskArguments` / `main()` pattern). |
| `trainer.py` | Shared training infrastructure: `AsagTrainingArguments`, `ModelLoader` (loads backbone + LoRA), `AsagTrainer` (custom training loop), `compute_metrics()`. |

### Data processing — ASAG (`data_processing_asag/`)

| File | Purpose |
|---|---|
| `benchmark_meta.py` | Metadata registry for all ASAG benchmarks (paths, label maps, rubric configs). |
| `general_asag_loader.py` | Base class `ASAG_Data_Loader` — shared loading / preprocessing logic for all ASAG datasets. |
| `alice_asag_loader.py` | `Alice_Loader(ASAG_Data_Loader)` — ALICE-specific loading. |
| `data_prep.py` | `DataPipeline` — tokenises and assembles model inputs for ASAG; helpers `is_llm_model()`, `get_tokenizer()`. |

### Data processing — other benchmarks (`data_processing_other/`)

| File | Purpose |
|---|---|
| `benchmark_meta.py` | Metadata registry for non-ASAG benchmarks (format function names, label configs, language tags, suffix lists). |
| `data_loader.py` | Per-benchmark format functions: `format_piqa`, `format_xstance`, `format_semeval2016`, `format_cstance`, `format_yelp`, `format_eic`, `format_figqa`, `format_winogrande`. |
| `data_prep.py` | `OtherDataLoader`, `DataPipelineOther` — loads and tokenises non-ASAG data; mirrors the ASAG `DataPipeline` interface. |

### Modelling (`modelling/`)

| File | Purpose |
|---|---|
| `modelling_utils.py` | `BaseAsagModel` (base class), `Pooler` (span / mean / CLS / last-token), `BackwardSupportedArguments` (model config dataclass), attention-mask helpers (`get_noncausal_attention_mask`, `get_backward_attention_mask`), `flip_tensor`. |
| `modelling_span.py` | `SpanFuser` — fuses span representations; `SpanAlignmentModel(BaseAsagModel)` — the main sequence-to-label alignment model. |
| `modelling_xnet.py` | `AsagXnet(BaseAsagModel)` — cross-encoder / xnet variant of the alignment model. |
| `custom_llama.py` | Patched HF Llama classes (`LlamaDecoderLayer`, `LlamaModel`, `LlamaForSequenceClassification`) with residual-connection and bidirectionality support. |
| `custom_mistral.py` | Same patches for Mistral (`MistralDecoderLayer`, `MistralModel`, `MistralForSequenceClassification`). |

### Utilities

| File | Purpose |
|---|---|
| `utils.py` | `evaluate()`, `metrics_calc()`, `eval_report()`, `save_report()`, `save_prediction()`, `get_label_weights()`, `per_qid_metrics()`, `extract_llama_attention()`, misc helpers (`set_seed`, `configure_logging`, `batch_to_device`, `clear_gpu_memory`). |


