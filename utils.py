import json
import logging
import random
import numpy as np
import torch

import re 
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc 
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, confusion_matrix
from collections import defaultdict
import pandas as pd
from collections import deque

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(filename=None, level=logging.INFO):
    logging.basicConfig(
        filename=filename,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

def batch_to_device(batch, device):
    """
    Move the batch to the specified device.
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device) 
    return batch 


def mean_dequeue(deque):
    """
    Calculate the mean of the last N elements in a deque.
    """
    if len(deque) == 0:
        return 0
    return sum(deque) / len(deque)


def _restore_logits_to_canonical_order(logit_row, rubric_index_map):
    if rubric_index_map is None:
        return list(logit_row)
    if len(rubric_index_map) == 0:
        return []

    canonical_size = max(int(idx) for idx in rubric_index_map) + 1
    restored = [None] * canonical_size
    for compact_idx, canonical_idx in enumerate(rubric_index_map):
        restored[int(canonical_idx)] = float(logit_row[compact_idx])
    return restored


def get_optimizer_step(optimizer):
    try:
        for params in optimizer.param_groups[0]["params"]:
            params_state = optimizer.state[params]
            if "step" in params_state:
                return params_state["step"]

        return -1
    except KeyError:
        return -1 
    

   
def metrics_calc(labels, pred_id):
    """
    Calculate the metrics for the predictions, including Quadratic Weighted Kappa (QWK), F1 score, and accuracy.
    """
    
    qwk = cohen_kappa_score(labels, pred_id, weights="quadratic")
    f1 = f1_score(labels, pred_id, average='macro')
    acc = accuracy_score(labels, pred_id)
    conf_matrix = confusion_matrix(labels, pred_id)
    
    metrics = {
        "qwk": qwk,
        "f1": f1,
        "accuracy": acc,
        "confusion_matrix": conf_matrix.tolist()
    }
    return metrics

        
def eval_report(pred_df, group_by=None):
    """
    Report the evaluation result, print the overall F1 and accuracy to the logger.
    Additionally, create a dictionary that stores the results, sorted by the code of the datapoint,
    along with the overall metrics.
    """
    results = {}

    # Calculate overall metrics
    metrics = metrics_calc(pred_df["labels"].values, pred_df["pred_id"].values)
    results["qwk"] = metrics["qwk"]
    results["f1"] = metrics["f1"]
    results["accuracy"] = metrics["accuracy"]
    return results

def save_report(metrics, path):
    """
    Save the metrics to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4) 
def save_prediction(pred_df,id2label,path):
    """
    conver the predictions to the original labels and save them to a CSV file.
    """

    pred_df["pred_label"] = [id2label[pred] for pred in pred_df["pred_id"].values]
    with open(path, "w") as f:
        pred_df.to_csv(f, index=False) 


def get_label_weights(dataset,label_field="labels"):
    """
    Calculate label weights for the optimizer based on the label distribution in the dataset.
    """
    label_ids = np.array(dataset[label_field])
    unique_labels, counts = torch.unique(torch.tensor(label_ids), return_counts=True)
    total_count = len(label_ids)
    w = 1 / (counts / total_count)
    return w 

def transform_for_inference(pred_df, other_filds=None):
    pred_df["logit_label"] = pred_df['logits'].apply(lambda x: float(x[1])) if len(pred_df['logits'].iloc[0]) > 1 else pred_df['logits']
    final_fields = ["id", "rubric_level", "level", "logit_label", "question_id"] + (other_filds if other_filds else [])
    final_df = pred_df.loc[pred_df.groupby('id')['logit_label'].idxmax()][final_fields]
    final_df = final_df.rename(columns={'rubric_level': 'pred_id', 'level': 'labels'})
    return final_df 

import torch

def _find_self_attention_layers(model):
    """Return the transformer layer stack after unwrapping common PEFT/HF wrappers."""
    seen = set()
    stack = [model]

    while stack:
        module = stack.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))

        layers = getattr(module, "layers", None)
        if layers is not None and len(layers) > 0 and hasattr(layers[0], "self_attn"):
            return layers

        for attr in ("model", "base_model", "encoder"):
            child = getattr(module, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)

        get_base_model = getattr(module, "get_base_model", None)
        if callable(get_base_model):
            try:
                child = get_base_model()
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                stack.append(child)

    raise RuntimeError(f"Could not find a LLaMA-style layer stack in {type(model).__name__}.")


def extract_llama_attention(
    model,
    input_ids,
    layer_idx,
    attention_mask=None,
):
    """
    Extract attention weights from a specific LLaMA layer via a forward hook.

    Requires eager attention backend (not FlashAttention/SDPA) so that
    attn_weights are actually computed. Load the model with
    attn_implementation="eager" if weights are not captured.

    Args:
        model: LlamaForSequenceClassification or similar wrapper
        input_ids: (batch, seq_len)
        layer_idx: int, which transformer layer to tap
        attention_mask: optional (batch, seq_len)

    Returns:
        attn_weights: Tensor of shape (batch, num_heads, seq_len, seq_len)
    """
    model.eval()
    captured = {}

    def attn_hook(module, input, output):
        # LlamaAttention.forward returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured["attn"] = output[1].detach().cpu()

    layers = _find_self_attention_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx={layer_idx} is outside the available range 0..{len(layers) - 1}")

    handle = layers[layer_idx].self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    handle.remove()

    if "attn" not in captured:
        raise RuntimeError(
            "Attention weights not captured. "
            "Ensure the model was loaded with attn_implementation='eager' "
            "(FlashAttention/SDPA do not materialise attn_weights) "
            "and that the layer index is valid."
        )

    return captured["attn"]  # (batch, num_heads, seq_len, seq_len)


@torch.no_grad() 
def evaluate(
    model,
    dataset,
    batch_size,
    collate_fn=None,
    save_attweights=False,
    layer_idx=-1,
    attn_max_examples=0,
): 
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False) 
    data_iterator = tqdm(dataloader, desc="Evaluating", position=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    eval_loss = []
    acc_history = deque(maxlen=10)
    predictions = defaultdict(list)
    model = model.to(torch.float32)
    attention_weights = []
    captured_attention_examples = 0
    for step, (batch, meta) in enumerate(data_iterator):
        meta = meta or {}
        batch = batch_to_device(batch, device)
        
        # Extract attention weights if requested
        if save_attweights and (attn_max_examples <= 0 or captured_attention_examples < attn_max_examples):
            attn_batch_size = batch["input_ids"].shape[0]
            if attn_max_examples > 0:
                attn_batch_size = min(attn_batch_size, attn_max_examples - captured_attention_examples)

            attn_input_ids = batch["input_ids"][:attn_batch_size]
            attn_attention_mask = batch.get("attention_mask", None)
            if attn_attention_mask is not None:
                attn_attention_mask = attn_attention_mask[:attn_batch_size]

            # For SpanAlignmentModel, we need to access the encoder
            base_model = model.encoder if hasattr(model, 'encoder') else model
            try:
                attn_weights = extract_llama_attention(
                    base_model,
                    attn_input_ids,
                    layer_idx=layer_idx,
                    attention_mask=attn_attention_mask,
                )
                attn_meta = {
                    key: value[:attn_batch_size]
                    for key, value in meta.items()
                    if isinstance(value, list)
                }
                attention_weights.append(
                    {
                        "weights": attn_weights,
                        "input_ids": attn_input_ids.detach().cpu(),
                        "attention_mask": attn_attention_mask.detach().cpu() if attn_attention_mask is not None else None,
                        "meta": attn_meta,
                        "layer_idx": layer_idx,
                    }
                )
                captured_attention_examples += attn_batch_size
            except Exception as e:
                print(f"Warning: Could not extract attention weights: {e}")
        
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu()
        eval_loss.append(loss.item())
        logits_list = logits.tolist()
        compact_pred_id = np.argmax(logits, axis=1).tolist()

        restored_pred_id = []
        restored_logits = []
        rubric_index_maps = meta.get("rubric_index_map")
        for idx, pred in enumerate(compact_pred_id):
            if rubric_index_maps is None:
                restored_pred_id.append(pred)
                restored_logits.append(logits_list[idx])
            else:
                restore_map = rubric_index_maps[idx]
                restored_pred_id.append(int(restore_map[pred]))
                restored_logits.append(_restore_logits_to_canonical_order(logits_list[idx], restore_map))

        original_labels = meta.get("original_label")
        if original_labels is None:
            restored_labels = batch["labels"].detach().cpu().numpy().tolist()
        else:
            restored_labels = [int(label) for label in original_labels]

        # collect data to put in the prediction dict
        predictions["pred_id"].extend(restored_pred_id)
        predictions["labels"].extend(restored_labels)
        predictions["logits"].extend(restored_logits)
        acc = accuracy_score(restored_labels, restored_pred_id)
        acc_history.append(acc)
        data_iterator.set_description(
            "Evaluating: loss {:.4f} acc {:.4f} ≈".format(
                mean_dequeue(eval_loss),
                mean_dequeue(acc_history),
            )
        )
        for key, value in meta.items():
            if key in {"original_label", "rubric_index_map"}:
                continue
            predictions[key].extend(value)
    pred_df = pd.DataFrame(predictions)
    eval_loss = np.mean(eval_loss)
    
    if save_attweights and attention_weights:
        return pred_df, eval_loss, attention_weights
    else:
        return pred_df, eval_loss 


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    # if torch.distributed.is_available() and torch.distributed.is_initialized():
    #     torch.distributed.barrier()
    #     torch.distributed.destroy_process_group()
    print("GPU memory cleared")

def per_qid_metrics(file_path_or_df):
    # Accept either a file path (str) or a DataFrame
    if isinstance(file_path_or_df, pd.DataFrame):
        df = file_path_or_df
    else:
        try:
            df = pd.read_csv(file_path_or_df)
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    # Check for required columns
    required_columns = ['pred_id', 'labels', 'question_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        return None

    print(f"{'Question ID':<12} | {'QWK':<10} | {'F1 (Macro)':<10} | {'Accuracy':<10} | {'Count':<10}")
    print("-" * 65)

    metrics_per_question = []

    # Group by question_id
    grouped = df.groupby('question_id')

    for q_id, group in grouped:
        y_true = group['labels']
        y_pred = group['pred_id']
        
        # Calculate metrics
        # QWK
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        # F1 Score (Macro)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        
        # Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        print(f"{q_id:<12} | {qwk:<10.4f} | {f1:<10.4f} | {acc:<10.4f} | {len(group):<10}")
        
        metrics_per_question.append({
            'question_id': q_id,
            'qwk': qwk,
            'f1': f1,
            'accuracy': acc
        })

    print("-" * 65)
    
    # Calculate Macro Average across questions
    if metrics_per_question:
        avg_qwk = np.mean([m['qwk'] for m in metrics_per_question])
        avg_f1 = np.mean([m['f1'] for m in metrics_per_question])
        avg_acc = np.mean([m['accuracy'] for m in metrics_per_question])
        
        print(f"{'Average':<12} | {avg_qwk:<10.4f} | {avg_f1:<10.4f} | {avg_acc:<10.4f} | {len(df):<10}")
        metrics_per_question.append({
            'question_id': 'average',
            'qwk': avg_qwk,
            'f1': avg_f1,
            'accuracy': avg_acc
        })
    return metrics_per_question
