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

def extract_llama_attention(
    model,
    input_ids,
    layer_idx,
    attention_mask=None,
):
    """
    Extract attention weights from a specific LLaMA layer during inference.

    Args:
        model: LlamaModel or LlamaForCausalLM
        input_ids: (batch, seq_len)
        layer_idx: int, which transformer layer
        attention_mask: optional (batch, seq_len)

    Returns:
        attn_weights: Tensor of shape
            (batch, num_heads, seq_len, seq_len)
    """

    model.eval()
    captured = []

    def attn_hook(module, input, output):
        """
        LLaMA self_attn forward returns:
        attn_output, attn_weights, past_key_value
        """
        # output is a tuple

        if isinstance(output, tuple) and len(output) > 1:
            for i in range(output[1].size(0)):  # Iterate over batch size
                captured.append(output[1][i].detach().cpu())

    # register hook
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(attn_hook)

    with torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,  # we rely on hook
            return_dict=True,
        )

    handle.remove()

    if "attn" not in captured:
        raise RuntimeError(
            "Attention not captured. "
            "Check that FlashAttention is disabled and layer index is valid."
        )

    return captured["attn"]


@torch.no_grad() 
def evaluate(model, dataset, batch_size, collate_fn=None, save_attweights=False, layer_idx=-1): 
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
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, device)
        
        # Extract attention weights if requested
        if save_attweights:
            # For SpanAlignmentModel, we need to access the encoder
            base_model = model.encoder if hasattr(model, 'encoder') else model
            try:
                attn_weights = extract_llama_attention(
                    base_model,
                    batch['input_ids'],
                    layer_idx=layer_idx if layer_idx >= 0 else base_model.config.num_hidden_layers - 1,
                    attention_mask=batch.get('attention_mask', None)
                )
                attention_weights.append(attn_weights)
            except Exception as e:
                print(f"Warning: Could not extract attention weights: {e}")
        
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu()
        eval_loss.append(loss.item())
        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id.tolist())
        predictions["labels"].extend(batch["labels"].detach().cpu().numpy().tolist())
        predictions["logits"].extend(logits.tolist())
        acc = accuracy_score(batch["labels"].detach().cpu().numpy(), pred_id)
        acc_history.append(acc)
        data_iterator.set_description(
            "Evaluating: loss {:.4f} acc {:.4f} ≈".format(
                mean_dequeue(eval_loss),
                mean_dequeue(acc_history),
            )
        )
        for key, value in meta.items():
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

