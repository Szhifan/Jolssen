#!/usr/bin/env python3
"""
Web interface version of experiment parameter configuration tool
Uses Flask to provide a simple Web UI
"""
import os
if not os.getenv('FLASK_PORT'):
    os.environ['FLASK_PORT'] = '8080'
try:
    from flask import Flask, render_template_string, request, jsonify
except ImportError:
    print("❌ Flask not installed, please use CLI version: python config_ui.py")
    print("\nTo install Flask:")
    print("  pip install flask")
    exit(1)

import json
import subprocess
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import config_web_multitask as multitask_web
MODEL2SHORTNAME = {
    "markussagen/xlm-roberta-longformer-base-4096": "xlm-roberta-long",
    "jhu-clsp/mmBERT-base": "mmBERT-base",
    "meta-llama/Llama-3.2-1B-instruct": "llama3.2-1B-instruct",
    "meta-llama/Llama-3.2-3B-instruct": "llama3.2-3B-instruct",
    "meta-llama/Llama-3.2-1B": "llama3.2-1B",
    "meta-llama/Llama-3.2-3B": "llama3.2-3B",
    "mistralai/Mistral-7B-v0.1": "mistral-7B-v0.1",
    "nvidia/NV-Embed-v2": "nv-embed-v2",
}

ASAG_BENCHMARK_OPTIONS = [
    ("alice_lp", "ALICE LP"),
    ("alice_ke", "ALICE KE"),
    ("alice_sk", "ALICE SK"),
    ("asap_sas", "ASAP SAS"),
    ("beetle", "BEETLE"),
    ("istudio", "iStudio"),
    ("pt_asag", "PT ASAG"),
    ("scientsbank", "Scientsbank"),
]
ASAG_BENCHMARK_SET = {key for key, _ in ASAG_BENCHMARK_OPTIONS}

# Legacy defaults keep old "no manual input needed" behavior.
LEGACY_WANDB_API_KEY = "REDACTED_WANDB_KEY"
LEGACY_HF_TOKEN = "REDACTED_HF_TOKEN"

@dataclass
class ExperimentConfig:
    """Data class to hold experiment configuration parameterss"""
    base_model: str
    benchmark: str
    model_class: str = "span"  # New field for model class
    span_fuse_type: str = "p-concat"  # Only used for span model
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    train_frac: float = 1.0
    lr: float = 2e-4
    max_epoch: int = 4
    use_lora: bool = True
    use_bnb: bool = True
    add_suffix: bool = True
    add_context: bool = True
    random_suffix: bool = True
    use_translated_prompts: bool = True
    random_solution: bool = False
    random_drop_rub: float = 0.0
    bf16: bool = True
    log_wandb: bool = True
    seed: int = 114514
    exp_name: Optional[str] = None
    
    # 新增字段
    pool_type: str = "last"
    span_pool_type: str = "last"  # Only used for span model
    num_bidir_layers: float = 0
    num_prune_layers: float = 0
    num_fuse_layers: float = 0
    layer_fuse_type: str = "avg"
    num_unsink_layers: float = 0
    pairwise_margin: float = 0.1  # For xnet-pwr
    
    def generate_exp_name(self) -> str:
        if self.exp_name:
            return self.exp_name
        parts = [
            self.benchmark,
            MODEL2SHORTNAME.get(self.base_model, self.base_model.split("/")[-1]).lower(),
        ]
        # Add model_class if not default span
        if self.model_class != "span":
            parts.append(self.model_class)
        # Add fusion type only for span model
        if self.model_class == "span":
            parts.append(self.span_fuse_type)
        if self.random_solution:
            parts.append("randsolu")
        return "-".join(parts)


def _task_signature(tasks: list[str], max_items: int = 3) -> str:
    if not tasks:
        return "none"
    visible = tasks[:max_items]
    suffix = "-etc" if len(tasks) > max_items else ""
    return "+".join(visible) + suffix


@dataclass
class MultiExperimentConfig:
    """Data class for train_multi.py configuration."""
    base_model: str
    train_tasks: list[str]
    eval_tasks: list[str] = field(default_factory=list)
    test_tasks: list[str] = field(default_factory=list)
    model_class: str = "span"
    span_fuse_type: str = "p-concat"
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    train_frac: float = 1.0
    lr: float = 2e-4
    max_epoch: int = 4
    use_lora: bool = True
    use_bnb: bool = True
    add_suffix: bool = True
    add_context: bool = True
    random_suffix: bool = True
    use_translated_prompts: bool = True
    random_solution: bool = False
    random_drop_rub: float = 0.0
    bf16: bool = True
    log_wandb: bool = True
    seed: int = 114514
    exp_name: Optional[str] = None
    pool_type: str = "last"
    span_pool_type: str = "last"
    num_bidir_layers: float = 0
    num_prune_layers: float = 0
    num_fuse_layers: float = 0
    layer_fuse_type: str = "avg"
    num_unsink_layers: float = 0

    def generate_exp_name(self) -> str:
        if self.exp_name:
            return self.exp_name
        parts = [
            "multi",
            _task_signature(self.train_tasks),
            MODEL2SHORTNAME.get(self.base_model, self.base_model.split("/")[-1]).lower(),
        ]
        if self.model_class != "span":
            parts.append(self.model_class)
        else:
            parts.append(self.span_fuse_type)
        if self.eval_tasks and self.eval_tasks != self.train_tasks:
            parts.append(f"eval-{_task_signature(self.eval_tasks, max_items=2)}")
        if self.test_tasks and self.test_tasks != self.train_tasks:
            parts.append(f"test-{_task_signature(self.test_tasks, max_items=2)}")
        if self.random_solution:
            parts.append("randsolu")
        return "-".join(parts)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASAG Experiment Configuration Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 16px;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-section {
            margin-bottom: 40px;
        }
        
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        
        select, input[type="text"], input[type="number"], input[type="range"] {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        select:focus, input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .checkbox-group {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
        }
        
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            margin-right: 8px;
            cursor: pointer;
        }
        
        .checkbox-item label {
            margin: 0;
            cursor: pointer;
            font-weight: 400;
        }
        
        .preview-box {
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .preview-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 12px;
        }
        
        .preview-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }
        
        .preview-item:last-child {
            border-bottom: none;
        }
        
        .preview-label {
            font-weight: 500;
            color: #555;
        }
        
        .preview-value {
            color: #667eea;
            font-family: monospace;
        }
        
        .exp-name-highlight {
            background: #fff3cd;
            padding: 12px;
            border-left: 4px solid #ffc107;
            border-radius: 4px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 14px;
        }
        
        .button-group {
            display: flex;
            gap: 12px;
            margin-top: 30px;
            justify-content: center;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .alert {
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        .alert-info {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            color: #0c5aa0;
        }
        
        .alert-success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .alert-error {
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }
        
        .command-box {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            overflow-x: auto;
            margin-top: 20px;
            line-height: 1.6;
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #999;
        }
        
        .batch-summary {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        }
        
        .experiment-name-item {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        .experiment-name-item .exp-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
            font-size: 14px;
        }
        
        .experiment-name-item input {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
            font-family: monospace;
        }
        
        .experiment-name-item input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
        }
        
        /* Autocomplete styles */
        .form-group {
            position: relative;
        }
        
        .autocomplete-suggestions {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 6px 6px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }
        
        .autocomplete-item {
            padding: 10px 12px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }
        
        .autocomplete-item:hover,
        .autocomplete-item.selected {
            background: #f5f5f5;
        }
        
        .autocomplete-item:last-child {
            border-bottom: none;
        }
        
        /* Progress bar styles */
        .progress-container {
            margin: 20px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s;
        }
        
        .progress-text {
            text-align: center;
            margin-top: 10px;
            font-weight: 500;
        }
        
        .batch-log {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .model-specific {
            font-size: 12px;
            color: #999;
            font-weight: 400;
            font-style: italic;
        }
        
        .disabled-field {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .btn-delete {
            background: #dc3545;
            color: white;
            padding: 4px 12px;
            font-size: 12px;
            border-radius: 4px;
            cursor: pointer;
            border: none;
            margin-left: 10px;
        }
        
        .btn-delete:hover {
            background: #c82333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ASAG Experiment Configuration</h1>
            <p>Quick configuration and batch experiment launcher</p>
            <p><a href="/multi" style="color:#fff; border:1px solid rgba(255,255,255,0.55); border-radius:6px; padding:6px 10px; text-decoration:none; display:inline-block; margin-top:10px; font-weight:600;">Open train_multi.py Page</a></p>
        </div>
        
        <div class="content">
            <form id="configForm">
                <!-- Benchmark Arguments -->
                <div class="form-section">
                    <div class="section-title">📊 Benchmark Arguments</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="benchmark">Benchmark/Dataset *</label>
                            <select id="benchmark" name="benchmark" required onchange="updateExperimentNames()" multiple>
                                <option value="alice_lp" selected>ALICE LP</option>
                                <option value="alice_ke">ALICE KE</option>
                                <option value="alice_sk">ALICE SK</option>
                                <option value="asap_sas">ASAP SAS</option>
                                <option value="beetle">BEETLE</option>
                                <option value="istudio">iStudio</option>
                                <option value="pt_asag">PT ASAG</option>
                                <option value="scientsbank">Scientsbank</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="trainFrac">Training Data Fraction</label>
                            <select id="trainFrac" name="train_frac">
                                <option value="0.1">10% (0.1)</option>
                                <option value="0.2">20% (0.2)</option>
                                <option value="0.3">30% (0.3)</option>
                                <option value="0.5">50% (0.5)</option>
                                <option value="0.8">80% (0.8)</option>
                                <option value="1.0" selected>100% (1.0)</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="addSuffix">Add Suffix</label>
                            <select id="addSuffix" name="add_suffix" onchange="updateExperimentNames()" multiple>
                                <option value="false">No</option>
                                <option value="true" selected>Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="addContext">Add Context</label>
                            <select id="addContext" name="add_context" onchange="updateExperimentNames()" multiple>
                                <option value="false">No</option>
                                <option value="true" selected>Yes</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="randomSuffix">Random Suffix</label>
                            <select id="randomSuffix" name="random_suffix" onchange="updateExperimentNames()" multiple>
                                <option value="false">No</option>
                                <option value="true" selected>Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="randomSolution">Random Solution</label>
                            <select id="randomSolution" name="random_solution" onchange="updateExperimentNames()" multiple>
                                <option value="false" selected>No</option>
                                <option value="true">Yes</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="useTranslated">Use Translated Prompts</label>
                        <select id="useTranslated" name="use_translated_prompts" onchange="updateExperimentNames()" multiple>
                            <option value="false">No</option>
                            <option value="true" selected>Yes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="randomDropRub">Random Drop Rubric Probability</label>
                        <input type="number" id="randomDropRub" name="random_drop_rub" value="0.0" min="0" max="1" step="0.1" placeholder="0.0 to 1.0">
                    </div>
                </div>
                
                <!-- Modelling Arguments -->
                <div class="form-section">
                    <div class="section-title">🤖 Modelling Arguments</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="modelClass">Model Class *</label>
                            <select id="modelClass" name="model_class" required onchange="updateModelClassUI(); updateExperimentNames();" multiple>
                                <option value="span" selected>Span Alignment (span)</option>
                                <option value="xnet">Cross-Network (xnet)</option>
                                <option value="xnet-pwr">Cross-Network with Pairwise Ranking (xnet-pwr)</option>

                            </select>
                        </div>
                        <div class="form-group">
                            <label for="baseModel">Base Model *</label>
                            <select id="baseModel" name="base_model" required onchange="updateExperimentNames()" multiple>
                                <option value="markussagen/xlm-roberta-longformer-base-4096" selected>XLM-Roberta Long (markussagen/xlm-roberta-longformer-base-4096)</option>
                                <option value="jhu-clsp/mmBERT-base">mmBERT-base (jhu-clsp/mmBERT-base)</option>
                                <option value="meta-llama/Llama-3.2-1B-instruct">Llama 3.2 1B Instruct</option>
                                <option value="meta-llama/Llama-3.2-3B-instruct">Llama 3.2 3B Instruct</option>
                                <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
                                <option value="meta-llama/Llama-3.2-3B">Llama 3.2 3B</option>
                                <option value="mistralai/Mistral-7B-v0.1">Mistral 7B v0.1</option>
                                <option value="nvidia/NV-Embed-v2">NVIDIA NV-Embed v2</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row" id="spanFuseTypeRow">
                        <div class="form-group">
                            <label for="spanFuseType">Span Fusion Type * <span class="model-specific">(span only)</span></label>
                    <select id="spanFuseType" name="span_fuse_type" onchange="updateExperimentNames()" multiple>
                        <option value="p-concat" selected>Pooled-Label Concat (p-concat)</option>
                        <option value="p-diff">Pooled-Label Difference (p-diff)</option>
                        <option value="p-gate">Pooled-Label Gate (p-gate)</option>
                        <option value="p-condiff">Pooled-Label Concat + Difference (p-condiff)</option>
                        <option value="p-bl">Pooled-Label Bilinear (p-bl)</option>
                        <option value="p-only">Pooled Only (p-only)</option>
                        <option value="l-only">Label Only (l-only)</option>
                        <option value="t-bl">Text-Label Bilinear (t-bl)</option>
                        <option value="t-concat">Text-Label Concat (t-concat)</option>
                        <option value="t-diff">Text-Label Difference (t-diff)</option>
                        <option value="tpl-concat">Text-Pooled-Label Concat (tpl-concat)</option>
                    </select>
                        </div>
                        <div class="form-group">
                            <label for="spanPoolType">Span Pooling Type <span class="model-specific">(span only)</span></label>
                            <select id="spanPoolType" name="span_pool_type">
                                <option value="mean">Mean Pooling</option>
                                <option value="last" selected>Last Token</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="poolType">Pooling Type</label>
                            <select id="poolType" name="pool_type">
                                <option value="avg">Average Pooling</option>
                                <option value="weightedavg">Weighted Average Pooling</option>
                                <option value="cls">CLS Token</option>
                                <option value="last" selected>Last Token</option>
                            </select>
                        </div>
                        <div class="form-group" id="pairwiseMarginGroup" style="display: none;">
                            <label for="pairwiseMargin">Pairwise Margin <span class="model-specific">(xnet-pwr only)</span></label>
                            <input type="number" id="pairwiseMargin" name="pairwise_margin" value="0.1" step="0.01" min="0">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="numBidirLayers">Number of Bidirectional Layers</label>
                            <input type="number" id="numBidirLayers" name="num_bidir_layers" value="0" min="0" step="1">
                        </div>
                        <div class="form-group">
                            <label for="numPruneLayers">Number of Pruned Layers</label>
                            <input type="number" id="numPruneLayers" name="num_prune_layers" value="0" min="0" step="1">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="numFuseLayers">Number of Fused Layers</label>
                            <input type="number" id="numFuseLayers" name="num_fuse_layers" value="0" min="0" step="1">
                        </div>
                        <div class="form-group">
                            <label for="layerFuseType">Layer Fusion Type</label>
                            <select id="layerFuseType" name="layer_fuse_type">
                                <option value="avg" selected>Average</option>
                                <option value="weighted">Weighted</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="numUnsinkLayers">Number of Unsink Layers</label>
                        <input type="number" id="numUnsinkLayers" name="num_unsink_layers" value="0" min="0" step="1">
                    </div>
                </div>
                
                <!-- Training Arguments -->
                <div class="form-section">
                    <div class="section-title">🏃 Training Arguments</div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="batchSize">Batch Size</label>
                            <select id="batchSize" name="batch_size">
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="4">4</option>
                                <option value="8" selected>8</option>
                                <option value="16">16</option>
                                <option value="32">32</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="gradAccumSteps">Gradient Accumulation Steps</label>
                            <select id="gradAccumSteps" name="gradient_accumulation_steps">
                                <option value="1">1</option>
                                <option value="2" selected>2</option>
                                <option value="4">4</option>
                                <option value="8">8</option>
                                <option value="16">16</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="lr">Learning Rate</label>
                            <select id="lr" name="lr">
                                <option value="1e-5">1e-5</option>
                                <option value="2e-5">2e-5</option>
                                <option value="5e-5">5e-5</option>
                                <option value="1e-4">1e-4</option>
                                <option value="2e-4" selected>2e-4</option>
                                <option value="5e-4">5e-4</option>
                                <option value="1e-3">1e-3</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="maxEpoch">Maximum Epochs</label>
                            <select id="maxEpoch" name="max_epoch">
                                <option value="1">1</option>
                                <option value="2">2</option>
                                <option value="3">3</option>
                                <option value="4">4</option>
                                <option value="5">5</option>
                                <option value="6" selected>6</option>
                                <option value="8">8</option>
                                <option value="10">10</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="seed">Random Seed</label>
                            <input type="text" id="seed" name="seed" value="114514" pattern="\d+" title="enter random seed">
                        </div>
                        <div class="form-group">
                            <label for="logWandb">Logging</label>
                            <div class="checkbox-item">
                                <input type="checkbox" id="logWandb" name="log_wandb" checked>
                                <label for="logWandb">Log to Weights & Biases</label>
                            </div>
                        </div>
                    </div>
                    <div class="checkbox-group">
                        <div class="checkbox-item">
                            <input type="checkbox" id="useLora" name="use_lora" checked>
                            <label for="useLora">Use LoRA Fine-tuning</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="useBnb" name="use_bnb" checked>
                            <label for="useBnb">Use 4-bit Quantization</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="bf16" name="bf16" checked>
                            <label for="bf16">Use BF16 Mixed Precision</label>
                        </div>
                    </div>
                </div>
                
                <!-- Experiment Names -->
                <div class="form-section">
                    <div class="section-title">📝 Experiment Names</div>
                    <div class="alert alert-info">
                        ℹ️ Each experiment gets its own name field. Press Tab for auto-completion. Empty fields will use auto-generated names.
                    </div>
                    <div id="experimentNamesContainer">
                        <p>Select models, benchmarks, and fusion types to see experiment name fields</p>
                    </div>
                </div>
                
                <!-- Batch Experiment Summary -->
                <div class="form-section">
                    <div class="section-title">📊 Experiment Summary</div>
                    <div class="batch-summary" id="batchSummary">
                        <p>Total experiments will be calculated based on selected combinations</p>
                    </div>
                </div>
                
                <!-- Buttons -->
                <div class="button-group">
                    <button type="button" class="btn-secondary" onclick="resetForm()">Reset</button>
                    <button type="button" class="btn-primary" onclick="startBatchExperiments()">Start Experiments</button>
                </div>
            </form>
            
            <!-- Command Output -->
            <div id="commandOutput" style="display: none; margin-top: 40px;">
                <div class="section-title">Command to Execute</div>
                <div class="command-box" id="commandText"></div>
            </div>
        </div>
        
        <div class="footer">
            © 2024 ASAG Experiment Configuration Tool | Quick configuration and batch experiment launcher
        </div>
    </div>
    
    <script>
        let experimentConfigs = [];
        
        function updateModelClassUI() {
            const modelClasses = Array.from(document.getElementById('modelClass').selectedOptions).map(o => o.value);
            const hasSpan = modelClasses.includes('span');
            const hasXnetPwr = modelClasses.some(mc => mc === 'xnet-pwr');
            
            // Enable/disable span-specific fields
            const spanFuseTypeRow = document.getElementById('spanFuseTypeRow');
            const spanFuseType = document.getElementById('spanFuseType');
            const spanPoolType = document.getElementById('spanPoolType');
            const pairwiseMarginGroup = document.getElementById('pairwiseMarginGroup');
            
            if (hasSpan) {
                spanFuseTypeRow.classList.remove('disabled-field');
                spanFuseType.disabled = false;
                spanPoolType.disabled = false;
            } else {
                spanFuseTypeRow.classList.add('disabled-field');
                spanFuseType.disabled = true;
                spanPoolType.disabled = true;
            }
            
            // Show/hide pairwise margin for xnet-pwr
            if (hasXnetPwr) {
                pairwiseMarginGroup.style.display = 'block';
            } else {
                pairwiseMarginGroup.style.display = 'none';
            }
        }
        
        function updateExperimentNames() {
            // Generate all possible experiment combinations
            const models = Array.from(document.getElementById('baseModel').selectedOptions).map(o => o.value);
            const benchmarks = Array.from(document.getElementById('benchmark').selectedOptions).map(o => o.value);
            const modelClasses = Array.from(document.getElementById('modelClass').selectedOptions).map(o => o.value);
            const fuseTypes = Array.from(document.getElementById('spanFuseType').selectedOptions).map(o => o.value);
            
            // Get dataset-related parameters
            const addSuffixOpts = Array.from(document.getElementById('addSuffix').selectedOptions).map(o => o.value === 'true');
            const addContextOpts = Array.from(document.getElementById('addContext').selectedOptions).map(o => o.value === 'true');
            const randomSuffixOpts = Array.from(document.getElementById('randomSuffix').selectedOptions).map(o => o.value === 'true');
            const randomSolutionOpts = Array.from(document.getElementById('randomSolution').selectedOptions).map(o => o.value === 'true');
            const useTranslatedOpts = Array.from(document.getElementById('useTranslated').selectedOptions).map(o => o.value === 'true');
            
            // Use actual selected values (don't override with defaults)
            const finalAddSuffix = addSuffixOpts.length > 0 ? addSuffixOpts : [true];
            const finalAddContext = addContextOpts.length > 0 ? addContextOpts : [true];
            const finalRandomSuffix = randomSuffixOpts.length > 0 ? randomSuffixOpts : [true];
            const finalRandomSolution = randomSolutionOpts.length > 0 ? randomSolutionOpts : [false];
            const finalUseTranslated = useTranslatedOpts.length > 0 ? useTranslatedOpts : [true];
            
            experimentConfigs = [];
            
            if (models.length === 0 || benchmarks.length === 0 || modelClasses.length === 0) {
                document.getElementById('experimentNamesContainer').innerHTML = '<p>Select models, benchmarks, and model classes to see experiment name fields</p>';
                updateBatchSummary();
                return;
            }
            
            let index = 0;
            models.forEach(model => {
                benchmarks.forEach(benchmark => {
                    modelClasses.forEach(modelClass => {
                        finalAddSuffix.forEach(addSuffix => {
                            finalAddContext.forEach(addContext => {
                                finalRandomSuffix.forEach(randomSuffix => {
                                    finalRandomSolution.forEach(randomSolution => {
                                        finalUseTranslated.forEach(useTranslated => {
                                            if (modelClass === 'span') {
                                                // For span model, iterate over fusion types
                                                if (fuseTypes.length === 0) {
                                                    return; // Skip if no fusion type selected for span
                                                }
                                                fuseTypes.forEach(fuseType => {
                                                    const config = {
                                                        id: index++,
                                                        model,
                                                        benchmark,
                                                        modelClass,
                                                        fuseType,
                                                        addSuffix,
                                                        addContext,
                                                        randomSuffix,
                                                        randomSolution,
                                                        useTranslated,
                                                        autoName: generateAutoName(model, benchmark, modelClass, fuseType, addSuffix, addContext, randomSuffix, randomSolution, useTranslated),
                                                        customName: ''
                                                    };
                                                    experimentConfigs.push(config);
                                                });
                                            } else {
                                                // For xnet models, no fusion type needed
                                                const config = {
                                                    id: index++,
                                                    model,
                                                    benchmark,
                                                    modelClass,
                                                    fuseType: null,
                                                    addSuffix,
                                                    addContext,
                                                    randomSuffix,
                                                    randomSolution,
                                                    useTranslated,
                                                    autoName: generateAutoName(model, benchmark, modelClass, null, addSuffix, addContext, randomSuffix, randomSolution, useTranslated),
                                                    customName: ''
                                                };
                                                experimentConfigs.push(config);
                                            }
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
            
            renderExperimentNameFields();
            updateBatchSummary();
        }
        
        function generateAutoName(model, benchmark, modelClass, fuseType, addSuffix, addContext, randomSuffix, randomSolution, useTranslated) {
            let simplifiedModel = model.split('/').pop().toLowerCase();
            simplifiedModel = simplifiedModel.replace('llama-', 'llama');
            simplifiedModel = simplifiedModel.replace('llama-3.2', 'llama3.2');
            
            let parts = [benchmark, simplifiedModel];
            if (modelClass !== 'span') {
                parts.push(modelClass);
            }
            if (fuseType) {
                parts.push(fuseType);
            }
            
            if (!addSuffix) parts.push('nosuffix');
            if (!addContext) parts.push('nocontext');
            if (randomSuffix === false) parts.push('fixins');
            if (randomSolution) parts.push('randsolu');
            if (!useTranslated) parts.push('notranslate');
            
            return parts.join('-');
        }
        
        function renderExperimentNameFields() {
            const container = document.getElementById('experimentNamesContainer');
            
            if (experimentConfigs.length === 0) {
                container.innerHTML = '<p>Select models, benchmarks, and model classes to see experiment name fields</p>';
                return;
            }
            
            let html = '';
            experimentConfigs.forEach(config => {
                const displayParts = [
                    config.benchmark,
                    config.model.split('/').pop(),
                    config.modelClass
                ];
                if (config.fuseType) {
                    displayParts.push(config.fuseType);
                }
                
                // Add dataset config indicators
                const datasetOpts = [];
                if (!config.addSuffix) datasetOpts.push('nosuffix');
                if (!config.addContext) datasetOpts.push('nocontext');
                if (config.randomSuffix === false) datasetOpts.push('fixins');
                if (config.randomSolution) datasetOpts.push('randsolu');
                if (!config.useTranslated) datasetOpts.push('notranslate');
                
                if (datasetOpts.length > 0) {
                    displayParts.push(`[${datasetOpts.join(',')}]`);
                }
                
                const displayText = displayParts.join(' + ');
                
                html += `
                    <div class="experiment-name-item" id="exp-item-${config.id}">
                        <div class="exp-label">
                            ${displayText}
                            <button class="btn-delete" onclick="deleteExperiment(${config.id})" title="delete this experiment">delete</button>
                        </div>
                        <input type="text" 
                               id="expName${config.id}" 
                               placeholder="${config.autoName}" 
                               value="${config.customName}"
                               onkeydown="handleTabComplete(event, ${config.id})"
                               onchange="updateCustomName(${config.id}, this.value)">
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        function updateCustomName(id, value) {
            const config = experimentConfigs.find(c => c.id === id);
            if (config) {
                config.customName = value;
            }
        }
        
        function handleTabComplete(event, id) {
            if (event.key === 'Tab') {
                event.preventDefault();
                const config = experimentConfigs.find(c => c.id === id);
                if (config) {
                    document.getElementById(`expName${id}`).value = config.autoName;
                    config.customName = config.autoName;
                }
            }
        }
        
        function deleteExperiment(id) {
            if (confirm('Are you sure you want to delete this experiment configuration?')) {
                // Remove from experimentConfigs
                experimentConfigs = experimentConfigs.filter(c => c.id !== id);
                
                // Remove from DOM
                const element = document.getElementById(`exp-item-${id}`);
                if (element) {
                    element.remove();
                }
                
                // Update summary
                updateBatchSummary();
            }
        }
        
        function updateBatchSummary() {
            const total = experimentConfigs.length;
            
            if (total === 0) {
                document.getElementById('batchSummary').innerHTML = '<p>Total experiments will be calculated based on selected combinations</p>';
                return;
            }
            
            const models = [...new Set(experimentConfigs.map(c => c.model))];
            const benchmarks = [...new Set(experimentConfigs.map(c => c.benchmark))];
            const modelClasses = [...new Set(experimentConfigs.map(c => c.modelClass))];
            const fuseTypes = [...new Set(experimentConfigs.map(c => c.fuseType).filter(f => f))];
            
            let summaryHTML = `
                <p><strong>Selected Configurations:</strong></p>
                <ul>
                    <li>Models: ${models.length} (${models.map(m => m.split('/').pop()).join(', ')})</li>
                    <li>Benchmarks: ${benchmarks.length} (${benchmarks.join(', ')})</li>
                    <li>Model Classes: ${modelClasses.length} (${modelClasses.join(', ')})</li>`;
            
            if (fuseTypes.length > 0) {
                summaryHTML += `<li>Fusion Types: ${fuseTypes.length} (${fuseTypes.join(', ')})</li>`;
            }
            
            summaryHTML += `
                </ul>
                <p><strong>Total Experiments: ${total}</strong></p>
                ${total > 0 ? `<p><strong>Preview:</strong> ${experimentConfigs.slice(0, 3).map(c => c.customName || c.autoName).join(', ')}${total > 3 ? '...' : ''}</p>` : ''}
            `;
            
            document.getElementById('batchSummary').innerHTML = summaryHTML;
        }
        
        
        function startBatchExperiments() {
            if (experimentConfigs.length === 0) {
                alert('Please select at least one option for model, benchmark, and fusion type');
                return;
            }
            
            const configs = getBatchConfigs();
            
            fetch('/api/start-batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({configs: configs})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(`✓ Batch script generated!\\nTotal experiments: ${data.total_experiments}\\nScript: ${data.run_script}`);
                } else {
                    alert('✗ Batch start failed: ' + data.error);
                }
            })
            .catch(error => {
                alert('Batch submission failed: ' + error);
            });
        }
        
        function getBatchConfigs() {
            if (experimentConfigs.length === 0) {
                return [];
            }
            
            const baseConfig = getFormConfig();
            
            return experimentConfigs.map(expConfig => {
                const config = {...baseConfig};
                config.base_model = expConfig.model;
                config.benchmark = expConfig.benchmark;
                config.model_class = expConfig.modelClass;
                if (expConfig.fuseType) {
                    config.span_fuse_type = expConfig.fuseType;
                }
                // Set dataset-related parameters
                config.add_suffix = expConfig.addSuffix;
                config.add_context = expConfig.addContext;
                config.random_suffix = expConfig.randomSuffix;
                config.random_solution = expConfig.randomSolution;
                config.use_translated_prompts = expConfig.useTranslated;
                config.exp_name = expConfig.customName || expConfig.autoName;
                return config;
            });
        }
        
        function getFormConfig() {
            return {
                span_pool_type: document.getElementById('spanPoolType').value,
                pool_type: document.getElementById('poolType').value,
                layer_fuse_type: document.getElementById('layerFuseType').value,
                pairwise_margin: document.getElementById('pairwiseMargin').value,
                num_bidir_layers: document.getElementById('numBidirLayers').value,
                num_prune_layers: document.getElementById('numPruneLayers').value,
                num_fuse_layers: document.getElementById('numFuseLayers').value,
                num_unsink_layers: document.getElementById('numUnsinkLayers').value,
                batch_size: document.getElementById('batchSize').value,
                gradient_accumulation_steps: document.getElementById('gradAccumSteps').value,
                train_frac: document.getElementById('trainFrac').value,
                lr: document.getElementById('lr').value,
                max_epoch: document.getElementById('maxEpoch').value,
                seed: document.getElementById('seed').value,
                random_drop_rub: document.getElementById('randomDropRub').value,
                use_lora: document.getElementById('useLora').checked,
                use_bnb: document.getElementById('useBnb').checked,
                bf16: document.getElementById('bf16').checked,
                log_wandb: document.getElementById('logWandb').checked,
            };
        }

        function resetForm() {
            document.getElementById('configForm').reset();
            document.getElementById('commandOutput').style.display = 'none';
            experimentConfigs = [];
            updateModelClassUI();
            updateExperimentNames();
        }
        
        // Event listeners
        document.getElementById('benchmark').addEventListener('change', updateExperimentNames);
        document.getElementById('baseModel').addEventListener('change', updateExperimentNames);
        document.getElementById('spanFuseType').addEventListener('change', updateExperimentNames);
        document.getElementById('addSuffix').addEventListener('change', updateExperimentNames);
        document.getElementById('addContext').addEventListener('change', updateExperimentNames);
        document.getElementById('randomSuffix').addEventListener('change', updateExperimentNames);
        document.getElementById('randomSolution').addEventListener('change', updateExperimentNames);
        document.getElementById('useTranslated').addEventListener('change', updateExperimentNames);
        document.getElementById('modelClass').addEventListener('change', function() {
            updateModelClassUI();
            updateExperimentNames();
        });
        
        // Initialize
        updateModelClassUI();
        updateExperimentNames();
    </script>
    
</body>
</html>
"""


MULTI_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASAG Multi-Task Configuration</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 980px;
            margin: 0 auto;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 28px;
            text-align: center;
        }
        .header h1 { font-size: 30px; margin-bottom: 10px; }
        .header p { opacity: 0.92; }
        .page-nav { margin-top: 12px; }
        .page-nav a {
            color: #fff;
            font-weight: 600;
            text-decoration: none;
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 6px;
            padding: 6px 10px;
            display: inline-block;
        }
        .content { padding: 30px; }
        .section { margin-bottom: 28px; }
        .section h2 {
            font-size: 19px;
            margin-bottom: 14px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .form-group { margin-bottom: 14px; }
        label { display: block; margin-bottom: 6px; font-weight: 600; color: #444; }
        select, input[type="text"], input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #d9d9d9;
            border-radius: 6px;
            font-size: 14px;
        }
        select[multiple] {
            min-height: 100px;
        }
        .select-tall {
            min-height: 170px !important;
        }
        .help-text {
            color: #666;
            font-size: 12px;
            margin-top: 4px;
        }
        .checkbox-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(180px, 1fr));
            gap: 10px;
        }
        .check-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .check-item input { width: 16px; height: 16px; }
        .button-row {
            display: flex;
            gap: 12px;
            justify-content: center;
            margin-top: 20px;
        }
        button {
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
        }
        .btn-primary {
            color: #fff;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .btn-secondary {
            background: #eceff3;
            color: #333;
        }
        .output {
            margin-top: 24px;
            background: #1e1e1e;
            color: #d4d4d4;
            border-radius: 6px;
            padding: 14px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            display: none;
        }
        .status {
            margin-top: 16px;
            font-size: 14px;
            color: #2f4f8f;
        }
        .disabled {
            opacity: 0.55;
            pointer-events: none;
        }
        .experiment-name-item {
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 8px;
            background: #fafafa;
        }
        .experiment-name-label {
            font-size: 13px;
            color: #444;
            margin-bottom: 8px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .exp-name-input {
            width: 100%;
            padding: 8px 10px;
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .btn-delete {
            background: #dc3545;
            color: #fff;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-delete:hover { background: #c82333; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ASAG Multi-Task Configuration</h1>
            <p>Configure <code>src/train_multi.py</code> with dedicated train/eval/test benchmark lists</p>
            <div class="page-nav">
                <a href="/">Open Single-Benchmark Page</a>
            </div>
        </div>
        <div class="content">
            <form id="multiForm">
                <div class="section">
                    <h2>Benchmark Splits</h2>
                    <div class="grid">
                        <div class="form-group">
                            <label for="trainTasks">Train Benchmarks *</label>
                            <select id="trainTasks" name="train_tasks" multiple required class="select-tall">
                                <option value="alice_lp" selected>ALICE LP</option>
                                <option value="alice_ke">ALICE KE</option>
                                <option value="alice_sk">ALICE SK</option>
                                <option value="asap_sas">ASAP SAS</option>
                                <option value="beetle">BEETLE</option>
                                <option value="istudio">iStudio</option>
                                <option value="pt_asag">PT ASAG</option>
                                <option value="scientsbank">Scientsbank</option>
                            </select>
                            <div class="help-text">Select one or more datasets for joint training.</div>
                        </div>
                        <div class="form-group">
                            <label for="evalTasks">Eval Benchmarks</label>
                            <select id="evalTasks" name="eval_tasks" multiple class="select-tall">
                                <option value="alice_lp">ALICE LP</option>
                                <option value="alice_ke">ALICE KE</option>
                                <option value="alice_sk">ALICE SK</option>
                                <option value="asap_sas">ASAP SAS</option>
                                <option value="beetle">BEETLE</option>
                                <option value="istudio">iStudio</option>
                                <option value="pt_asag">PT ASAG</option>
                                <option value="scientsbank">Scientsbank</option>
                            </select>
                            <div class="help-text">Leave empty to default to train benchmarks.</div>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="testTasks">Test Benchmarks</label>
                        <select id="testTasks" name="test_tasks" multiple class="select-tall">
                            <option value="alice_lp">ALICE LP</option>
                            <option value="alice_ke">ALICE KE</option>
                            <option value="alice_sk">ALICE SK</option>
                            <option value="asap_sas">ASAP SAS</option>
                            <option value="beetle">BEETLE</option>
                            <option value="istudio">iStudio</option>
                            <option value="pt_asag">PT ASAG</option>
                            <option value="scientsbank">Scientsbank</option>
                        </select>
                        <div class="help-text">Leave empty to default to train benchmarks.</div>
                    </div>
                </div>

                <div class="section">
                    <h2>Model</h2>
                    <div class="grid">
                        <div class="form-group">
                            <label for="modelClass">Model Class</label>
                            <select id="modelClass" name="model_class" onchange="updateSpanUI()" multiple>
                                <option value="span" selected>span</option>
                                <option value="xnet">xnet</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="baseModel">Base Model</label>
                            <select id="baseModel" name="base_model" multiple>
                                <option value="markussagen/xlm-roberta-longformer-base-4096" selected>XLM-Roberta Long</option>
                                <option value="jhu-clsp/mmBERT-base">mmBERT-base</option>
                                <option value="meta-llama/Llama-3.2-1B-instruct">Llama 3.2 1B Instruct</option>
                                <option value="meta-llama/Llama-3.2-3B-instruct">Llama 3.2 3B Instruct</option>
                                <option value="meta-llama/Llama-3.2-1B">Llama 3.2 1B</option>
                                <option value="meta-llama/Llama-3.2-3B">Llama 3.2 3B</option>
                                <option value="mistralai/Mistral-7B-v0.1">Mistral 7B v0.1</option>
                                <option value="nvidia/NV-Embed-v2">NVIDIA NV-Embed v2</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid" id="spanControls">
                        <div class="form-group">
                            <label for="spanFuseType">Span Fuse Type</label>
                            <select id="spanFuseType" name="span_fuse_type" multiple>
                                <option value="p-concat" selected>p-concat</option>
                                <option value="p-diff">p-diff</option>
                                <option value="p-gate">p-gate</option>
                                <option value="p-condiff">p-condiff</option>
                                <option value="p-bl">p-bl</option>
                                <option value="p-only">p-only</option>
                                <option value="l-only">l-only</option>
                                <option value="t-bl">t-bl</option>
                                <option value="t-concat">t-concat</option>
                                <option value="t-diff">t-diff</option>
                                <option value="tpl-concat">tpl-concat</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="spanPoolType">Span Pool Type</label>
                            <select id="spanPoolType" name="span_pool_type">
                                <option value="mean">mean</option>
                                <option value="last" selected>last</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group">
                            <label for="poolType">Pool Type</label>
                            <select id="poolType" name="pool_type">
                                <option value="avg">avg</option>
                                <option value="weightedavg">weightedavg</option>
                                <option value="cls">cls</option>
                                <option value="last" selected>last</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="layerFuseType">Layer Fuse Type</label>
                            <select id="layerFuseType" name="layer_fuse_type">
                                <option value="avg" selected>avg</option>
                                <option value="weighted">weighted</option>
                            </select>
                        </div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="numBidirLayers">Num Bidirectional Layers</label><input type="number" id="numBidirLayers" name="num_bidir_layers" value="0" min="0" step="1"></div>
                        <div class="form-group"><label for="numPruneLayers">Num Pruned Layers</label><input type="number" id="numPruneLayers" name="num_prune_layers" value="0" min="0" step="1"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="numFuseLayers">Num Fused Layers</label><input type="number" id="numFuseLayers" name="num_fuse_layers" value="0" min="0" step="1"></div>
                        <div class="form-group"><label for="numUnsinkLayers">Num Unsink Layers</label><input type="number" id="numUnsinkLayers" name="num_unsink_layers" value="0" min="0" step="1"></div>
                    </div>
                </div>

                <div class="section">
                    <h2>Training</h2>
                    <div class="grid">
                        <div class="form-group"><label for="batchSize">Batch Size</label><input type="number" id="batchSize" name="batch_size" value="8" min="1"></div>
                        <div class="form-group"><label for="gradAccum">Gradient Accumulation Steps</label><input type="number" id="gradAccum" name="gradient_accumulation_steps" value="2" min="1"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="trainFrac">Train Fraction</label><input type="number" id="trainFrac" name="train_frac" value="1.0" min="0.1" max="1.0" step="0.1"></div>
                        <div class="form-group"><label for="lr">Learning Rate</label><input type="text" id="lr" name="lr" value="2e-4"></div>
                    </div>
                    <div class="grid">
                        <div class="form-group"><label for="maxEpoch">Max Epoch</label><input type="number" id="maxEpoch" name="max_epoch" value="4" min="1"></div>
                        <div class="form-group"><label for="seed">Seed</label><input type="number" id="seed" name="seed" value="114514"></div>
                    </div>
                    <div class="form-group">
                        <label for="randomDropRub">Random Drop Rubric Probability</label>
                        <input type="number" id="randomDropRub" name="random_drop_rub" value="0.0" min="0" max="1" step="0.1">
                    </div>
                    <div class="checkbox-row">
                        <div class="check-item"><input type="checkbox" id="useLora" checked><label for="useLora">use_lora</label></div>
                        <div class="check-item"><input type="checkbox" id="useBnb" checked><label for="useBnb">use_bnb</label></div>
                        <div class="check-item"><input type="checkbox" id="bf16" checked><label for="bf16">bf16</label></div>
                        <div class="check-item"><input type="checkbox" id="logWandb" checked><label for="logWandb">log_wandb</label></div>
                        <div class="check-item"><input type="checkbox" id="addSuffix" checked><label for="addSuffix">add_suffix</label></div>
                        <div class="check-item"><input type="checkbox" id="addContext" checked><label for="addContext">add_context</label></div>
                        <div class="check-item"><input type="checkbox" id="randomSuffix" checked><label for="randomSuffix">random_suffix</label></div>
                        <div class="check-item"><input type="checkbox" id="useTranslated" checked><label for="useTranslated">use_translated_prompts</label></div>
                        <div class="check-item"><input type="checkbox" id="randomSolution"><label for="randomSolution">random_solution</label></div>
                    </div>
                </div>

                <div class="section">
                    <h2>Experiment Names</h2>
                    <div class="help-text">Each generated combination gets a name field. Press Tab to copy auto-name into a field.</div>
                    <div class="help-text">Auto-generated preview: <code id="autoExpName"></code></div>
                    <div id="experimentNamesContainer" style="margin-top:10px;">
                        <p>Select model parameters to generate experiment names.</p>
                    </div>
                </div>

                <div class="section">
                    <h2>Experiment Summary</h2>
                    <div id="batchSummary">
                        <p>Total experiments will be calculated based on selected combinations.</p>
                    </div>
                </div>

                <div class="button-row">
                    <button type="button" class="btn-secondary" onclick="resetForm()">Reset</button>
                    <button type="button" class="btn-primary" onclick="startBatchExperiments()">Start Experiments</button>
                </div>
                <div class="status" id="statusText"></div>
                <pre class="output" id="commandOutput"></pre>
            </form>
        </div>
    </div>

    <script>
        // Hard fallback: keep mono-style global entrypoints alive even if main script fails.
        (function () {
            function selectedValues(id) {
                var el = document.getElementById(id);
                if (!el || !el.options) return [];
                var opts = Array.prototype.slice.call(el.options);
                var out = opts.filter(function (o) { return !!o.selected; }).map(function (o) { return o.value; });
                if (!out.length && opts.length) {
                    opts[0].selected = true;
                    out = [opts[0].value];
                }
                return out;
            }

            function valueOf(id, fallback) {
                var el = document.getElementById(id);
                if (!el) return fallback;
                var v = el.value;
                return (v === undefined || v === null || v === '') ? fallback : v;
            }

            function checkedOf(id, fallback) {
                var el = document.getElementById(id);
                if (!el) return !!fallback;
                return !!el.checked;
            }

            function setStatus(message, isError) {
                var status = document.getElementById('statusText');
                if (status) {
                    status.textContent = String(message || '');
                    status.style.color = isError ? '#b00020' : '#2f4f8f';
                }
            }

            function cartesianConfigs() {
                var trainTasks = selectedValues('trainTasks');
                var evalTasks = selectedValues('evalTasks');
                var testTasks = selectedValues('testTasks');
                if (trainTasks.length && !evalTasks.length) evalTasks = trainTasks.slice();
                if (trainTasks.length && !testTasks.length) testTasks = trainTasks.slice();

                var models = selectedValues('baseModel');
                var classes = selectedValues('modelClass');
                var fuseTypes = selectedValues('spanFuseType');
                if (!fuseTypes.length) fuseTypes = ['p-concat'];

                var base = {
                    train_tasks: trainTasks,
                    eval_tasks: evalTasks,
                    test_tasks: testTasks,
                    span_pool_type: valueOf('spanPoolType', 'last'),
                    pool_type: valueOf('poolType', 'last'),
                    layer_fuse_type: valueOf('layerFuseType', 'avg'),
                    num_bidir_layers: Number(valueOf('numBidirLayers', '0')),
                    num_prune_layers: Number(valueOf('numPruneLayers', '0')),
                    num_fuse_layers: Number(valueOf('numFuseLayers', '0')),
                    num_unsink_layers: Number(valueOf('numUnsinkLayers', '0')),
                    batch_size: Number(valueOf('batchSize', '8')),
                    gradient_accumulation_steps: Number(valueOf('gradAccum', '2')),
                    train_frac: Number(valueOf('trainFrac', '1.0')),
                    lr: valueOf('lr', '2e-4'),
                    max_epoch: Number(valueOf('maxEpoch', '4')),
                    seed: Number(valueOf('seed', '114514')),
                    random_drop_rub: Number(valueOf('randomDropRub', '0.0')),
                    use_lora: checkedOf('useLora', true),
                    use_bnb: checkedOf('useBnb', true),
                    bf16: checkedOf('bf16', true),
                    log_wandb: checkedOf('logWandb', true),
                    add_suffix: checkedOf('addSuffix', true),
                    add_context: checkedOf('addContext', true),
                    random_suffix: checkedOf('randomSuffix', true),
                    use_translated_prompts: checkedOf('useTranslated', true),
                    random_solution: checkedOf('randomSolution', false)
                };

                var out = [];
                for (var i = 0; i < models.length; i++) {
                    for (var j = 0; j < classes.length; j++) {
                        if (classes[j] === 'span') {
                            for (var k = 0; k < fuseTypes.length; k++) {
                                var cfgSpan = Object.assign({}, base, {
                                    base_model: models[i],
                                    model_class: classes[j],
                                    span_fuse_type: fuseTypes[k]
                                });
                                out.push(cfgSpan);
                            }
                        } else {
                            var cfgXnet = Object.assign({}, base, {
                                base_model: models[i],
                                model_class: classes[j]
                            });
                            out.push(cfgXnet);
                        }
                    }
                }
                return out;
            }

            // fallback global for inline button onclick
            window.startBatchExperiments = window.startBatchExperiments || function () {
                if (typeof window.startBatch === 'function' && window.startBatch !== window.startBatchExperiments) {
                    return window.startBatch();
                }
                var configs = cartesianConfigs();
                if (!configs.length) {
                    setStatus('Please select at least one base model and one model class.', true);
                    alert('Please select at least one base model and one model class.');
                    return;
                }
                if (!configs[0].train_tasks || !configs[0].train_tasks.length) {
                    setStatus('Please select at least one train benchmark.', true);
                    alert('Please select at least one train benchmark.');
                    return;
                }

                setStatus('Submitting ' + configs.length + ' experiment(s)...');
                var xhr = new XMLHttpRequest();
                xhr.open('POST', '/api/start-batch-multi', true);
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.onreadystatechange = function () {
                    if (xhr.readyState !== 4) return;
                    try {
                        var data = JSON.parse(xhr.responseText || '{}');
                        if (xhr.status >= 200 && xhr.status < 300 && data.success) {
                            setStatus('run script: ' + data.run_script + ' (' + configs.length + ' experiments)');
                            alert('Batch script generated!\nTotal experiments: ' + configs.length + '\nScript: ' + data.run_script);
                        } else {
                            var err = data.error || ('HTTP ' + xhr.status);
                            setStatus('Failed to create run script: ' + err, true);
                            alert('Failed to create run script: ' + err);
                        }
                    } catch (e) {
                        setStatus('Create run script failed: ' + e, true);
                        alert('Create run script failed: ' + e);
                    }
                };
                xhr.send(JSON.stringify({ configs: configs }));
            };

            window.resetForm = window.resetForm || function () {
                var form = document.getElementById('multiForm');
                if (form) form.reset();
                setStatus('Form reset.');
            };
        })();
    </script>
    <script>
        const MODEL2SHORTNAME = {
            "markussagen/xlm-roberta-longformer-base-4096": "xlm-roberta-long",
            "jhu-clsp/mmBERT-base": "mmBERT-base",
            "meta-llama/Llama-3.2-1B-instruct": "llama3.2-1B-instruct",
            "meta-llama/Llama-3.2-3B-instruct": "llama3.2-3B-instruct",
            "meta-llama/Llama-3.2-1B": "llama3.2-1B",
            "meta-llama/Llama-3.2-3B": "llama3.2-3B",
            "mistralai/Mistral-7B-v0.1": "mistral-7B-v0.1",
            "nvidia/NV-Embed-v2": "nv-embed-v2",
        };

        let experimentConfigs = [];

        window.addEventListener('error', (ev) => {
            const msg = `JS error: ${ev.message} @ ${ev.filename || 'inline'}:${ev.lineno || 0}`;
            const status = document.getElementById('statusText');
            if (status) {
                status.textContent = msg;
                status.style.color = '#b00020';
            }
            const out = document.getElementById('commandOutput');
            if (out) {
                out.style.display = 'block';
                out.textContent = msg;
            }
        });

        function getSelectedValues(id) {
            const el = document.getElementById(id);
            if (!el) {
                return [];
            }
            const selectedOpts = el.selectedOptions
                ? Array.from(el.selectedOptions)
                : Array.from(el.options || []).filter((opt) => opt.selected);
            return selectedOpts.map((opt) => opt.value);
        }

        function setStatus(message, isError = false) {
            const status = document.getElementById('statusText');
            if (status) {
                status.textContent = message;
                status.style.color = isError ? '#b00020' : '#2f4f8f';
            }
            if (isError) {
                const out = document.getElementById('commandOutput');
                if (out) {
                    out.style.display = 'block';
                    out.textContent = String(message);
                }
            }
        }

        function getSelectedOrDefault(id) {
            const values = getSelectedValues(id);
            if (values.length > 0) {
                return values;
            }
            const el = document.getElementById(id);
            if (el && el.options && el.options.length > 0) {
                el.options[0].selected = true;
                return [el.options[0].value];
            }
            return [];
        }

        function ensureDefaultSelections() {
            getSelectedOrDefault('trainTasks');
            getSelectedOrDefault('baseModel');
            getSelectedOrDefault('modelClass');
            getSelectedOrDefault('spanFuseType');
        }

        function shortModelName(model) {
            return (MODEL2SHORTNAME[model] || model.split('/').pop()).toLowerCase();
        }

        function taskSignature(tasks, maxItems = 3) {
            if (!tasks || tasks.length === 0) {
                return 'none';
            }
            const visible = tasks.slice(0, maxItems);
            const suffix = tasks.length > maxItems ? '-etc' : '';
            return `${visible.join('+')}${suffix}`;
        }

        function computeAutoName(config) {
            const trainTasks = config.train_tasks || [];
            const evalTasks = (config.eval_tasks && config.eval_tasks.length > 0) ? config.eval_tasks : trainTasks;
            const testTasks = (config.test_tasks && config.test_tasks.length > 0) ? config.test_tasks : trainTasks;

            const parts = [
                'multi',
                taskSignature(trainTasks),
                shortModelName(config.base_model || ''),
            ];

            if (config.model_class !== 'span') {
                parts.push(config.model_class);
            } else {
                parts.push(config.span_fuse_type || 'p-concat');
            }

            if (evalTasks.join(',') !== trainTasks.join(',')) {
                parts.push(`eval-${taskSignature(evalTasks, 2)}`);
            }
            if (testTasks.join(',') !== trainTasks.join(',')) {
                parts.push(`test-${taskSignature(testTasks, 2)}`);
            }
            if (config.random_solution) {
                parts.push('randsolu');
            }
            return parts.join('-');
        }

        function updateSpanUI() {
            const hasSpan = getSelectedOrDefault('modelClass').includes('span');
            const controls = document.getElementById('spanControls');
            controls.classList.toggle('disabled', !hasSpan);
            document.getElementById('spanFuseType').disabled = !hasSpan;
            document.getElementById('spanPoolType').disabled = !hasSpan;
        }

        function getBaseConfig() {
            const formData = new FormData(document.getElementById('multiForm'));
            const config = {};

            for (const [key, value] of formData.entries()) {
                if (!['train_tasks', 'eval_tasks', 'test_tasks', 'base_model', 'model_class', 'span_fuse_type', 'exp_name'].includes(key)) {
                    config[key] = value;
                }
            }

            config.train_tasks = getSelectedValues('trainTasks');
            config.eval_tasks = getSelectedValues('evalTasks');
            config.test_tasks = getSelectedValues('testTasks');
            if (config.train_tasks.length > 0) {
                if (config.eval_tasks.length === 0) {
                    config.eval_tasks = [...config.train_tasks];
                }
                if (config.test_tasks.length === 0) {
                    config.test_tasks = [...config.train_tasks];
                }
            }

            config.use_lora = document.getElementById('useLora').checked;
            config.use_bnb = document.getElementById('useBnb').checked;
            config.bf16 = document.getElementById('bf16').checked;
            config.log_wandb = document.getElementById('logWandb').checked;
            config.add_suffix = document.getElementById('addSuffix').checked;
            config.add_context = document.getElementById('addContext').checked;
            config.random_suffix = document.getElementById('randomSuffix').checked;
            config.use_translated_prompts = document.getElementById('useTranslated').checked;
            config.random_solution = document.getElementById('randomSolution').checked;

            return config;
        }

        function buildExperimentConfigs() {
            try {
                const baseConfig = getBaseConfig();
                const baseModels = getSelectedOrDefault('baseModel');
                const modelClasses = getSelectedOrDefault('modelClass');
                const spanFuseTypes = getSelectedOrDefault('spanFuseType');

                const prevCustom = new Map(experimentConfigs.map((cfg) => [cfg.key, cfg.customName || '']));

                const next = [];
                let idx = 0;

                baseModels.forEach((baseModel) => {
                    modelClasses.forEach((modelClass) => {
                        if (modelClass === 'span') {
                            const fuseTypes = spanFuseTypes.length > 0 ? spanFuseTypes : ['p-concat'];
                            fuseTypes.forEach((spanFuseType) => {
                                const cfg = {
                                    ...baseConfig,
                                    id: idx++,
                                    base_model: baseModel,
                                    model_class: modelClass,
                                    span_fuse_type: spanFuseType,
                                };
                                cfg.autoName = computeAutoName(cfg);
                                cfg.key = `${cfg.base_model}|${cfg.model_class}|${cfg.span_fuse_type}`;
                                cfg.customName = prevCustom.get(cfg.key) || '';
                                next.push(cfg);
                            });
                        } else {
                            const cfg = {
                                ...baseConfig,
                                id: idx++,
                                base_model: baseModel,
                                model_class: modelClass,
                            };
                            cfg.autoName = computeAutoName(cfg);
                            cfg.key = `${cfg.base_model}|${cfg.model_class}|_`;
                            cfg.customName = prevCustom.get(cfg.key) || '';
                            next.push(cfg);
                        }
                    });
                });

                experimentConfigs = next;
                renderExperimentNameFields();
                updateAutoExpNamePreview();
                updateBatchSummary();
                if (next.length > 0) {
                    setStatus(`Prepared ${next.length} experiment name(s).`);
                } else {
                    setStatus('No valid experiment combinations with current selections.', true);
                }
            } catch (err) {
                setStatus(`buildExperimentConfigs failed: ${err}`, true);
                throw err;
            }
        }

        function renderExperimentNameFields() {
            const container = document.getElementById('experimentNamesContainer');
            if (experimentConfigs.length === 0) {
                container.innerHTML = '<p>Select model parameters to generate experiment names.</p>';
                return;
            }

            let html = '';
            experimentConfigs.forEach((cfg) => {
                const labelParts = [
                    cfg.base_model.split('/').pop(),
                    cfg.model_class,
                ];
                if (cfg.model_class === 'span') {
                    labelParts.push(cfg.span_fuse_type || 'p-concat');
                }
                html += `
                    <div class="experiment-name-item" id="exp-item-${cfg.id}">
                        <div class="experiment-name-label">
                            <span>${labelParts.join(' + ')}</span>
                            <button type="button" class="btn-delete" onclick="deleteExperiment(${cfg.id})">delete</button>
                        </div>
                        <input
                            class="exp-name-input"
                            id="expName${cfg.id}"
                            type="text"
                            value="${cfg.customName || ''}"
                            placeholder="${cfg.autoName}"
                            onkeydown="handleTabComplete(event, ${cfg.id})"
                            oninput="updateCustomName(${cfg.id}, this.value)"
                        >
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function updateCustomName(id, value) {
            const cfg = experimentConfigs.find((c) => c.id === id);
            if (cfg) {
                cfg.customName = value;
            }
        }

        function handleTabComplete(event, id) {
            if (event.key !== 'Tab') {
                return;
            }
            event.preventDefault();
            const cfg = experimentConfigs.find((c) => c.id === id);
            if (!cfg) {
                return;
            }
            const input = document.getElementById(`expName${id}`);
            if (input) {
                input.value = cfg.autoName;
                cfg.customName = cfg.autoName;
            }
        }

        function deleteExperiment(id) {
            experimentConfigs = experimentConfigs.filter((cfg) => cfg.id !== id);
            renderExperimentNameFields();
            updateAutoExpNamePreview();
            updateBatchSummary();
        }

        function updateAutoExpNamePreview() {
            const node = document.getElementById('autoExpName');
            if (!node) {
                return;
            }
            if (experimentConfigs.length === 0) {
                node.textContent = '';
                return;
            }
            const names = experimentConfigs.map((cfg) => cfg.autoName);
            node.textContent = names.length === 1 ? names[0] : `${names[0]} (+${names.length - 1} more)`;
        }

        function getBatchConfigs() {
            return experimentConfigs.map((cfg) => {
                const out = {
                    ...cfg,
                    exp_name: (cfg.customName && cfg.customName.trim()) ? cfg.customName.trim() : cfg.autoName,
                };
                delete out.id;
                delete out.key;
                delete out.autoName;
                delete out.customName;
                return out;
            });
        }

        function updateBatchSummary() {
            const summary = document.getElementById('batchSummary');
            if (!summary) {
                return;
            }

            const total = experimentConfigs.length;
            if (total === 0) {
                summary.innerHTML = '<p>Total experiments will be calculated based on selected combinations.</p>';
                return;
            }

            const models = [...new Set(experimentConfigs.map((cfg) => cfg.base_model.split('/').pop()))];
            const trainSets = [...new Set(experimentConfigs.map((cfg) => (cfg.train_tasks || []).join('+')))].filter(Boolean);
            const evalSets = [...new Set(experimentConfigs.map((cfg) => (cfg.eval_tasks || []).join('+')))].filter(Boolean);
            const testSets = [...new Set(experimentConfigs.map((cfg) => (cfg.test_tasks || []).join('+')))].filter(Boolean);
            const modelClasses = [...new Set(experimentConfigs.map((cfg) => cfg.model_class))];
            const fuseTypes = [...new Set(experimentConfigs.map((cfg) => cfg.span_fuse_type).filter(Boolean))];

            let html = `
                <p><strong>Selected Configurations:</strong></p>
                <ul>
                    <li>Models: ${models.length} (${models.join(', ')})</li>
                    <li>Train Benchmarks: ${trainSets.length} (${trainSets.join(' | ')})</li>
                    <li>Eval Benchmarks: ${evalSets.length} (${evalSets.join(' | ')})</li>
                    <li>Test Benchmarks: ${testSets.length} (${testSets.join(' | ')})</li>
                    <li>Model Classes: ${modelClasses.length} (${modelClasses.join(', ')})</li>`;

            if (fuseTypes.length > 0) {
                html += `<li>Fusion Types: ${fuseTypes.length} (${fuseTypes.join(', ')})</li>`;
            }

            html += `
                </ul>
                <p><strong>Total Experiments: ${total}</strong></p>
                <p><strong>Preview:</strong> ${experimentConfigs.slice(0, 3).map((cfg) => cfg.customName || cfg.autoName).join(', ')}${total > 3 ? '...' : ''}</p>
            `;
            summary.innerHTML = html;
        }

        function validateConfigs(configs) {
            if (!configs || configs.length === 0) {
                setStatus('Please select at least one base model and one model class.', true);
                alert('Please select at least one base model and one model class.');
                return false;
            }
            if (!configs[0].train_tasks || configs[0].train_tasks.length === 0) {
                setStatus('Please select at least one train benchmark.', true);
                alert('Please select at least one train benchmark.');
                return false;
            }
            return true;
        }

        async function generateCommand() {
            setStatus('Generate Command clicked...');
            const configs = getBatchConfigs();
            if (!validateConfigs(configs)) return;

            setStatus(`Generating ${configs.length} command(s)...`);
            try {
                const generated = [];
                for (let i = 0; i < configs.length; i++) {
                    const response = await fetch('/api/generate-command-multi', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(configs[i]),
                    });
                    const data = await response.json();
                    if (!data.success) {
                        setStatus(`Failed on config ${i + 1}: ${data.error}`, true);
                        alert(`Failed on config ${i + 1}: ${data.error}`);
                        return;
                    }
                    generated.push(data);
                }

                const output = document.getElementById('commandOutput');
                output.style.display = 'block';
                output.textContent = generated.map((item, idx) => `# Experiment ${idx + 1}: ${item.exp_name}\n${item.command}`).join('\n\n');
                setStatus(`Prepared ${generated.length} command(s).`);
            } catch (err) {
                setStatus(`Failed to generate commands: ${err}`, true);
                alert(`Failed to generate commands: ${err}`);
            }
        }

        async function startBatch() {
            const configs = getBatchConfigs();
            if (!validateConfigs(configs)) return;

            setStatus(`Submitting ${configs.length} experiment(s) to /api/start-batch-multi ...`);
            try {
                const response = await fetch('/api/start-batch-multi', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ configs }),
                });
                const data = await response.json();
                if (!data.success) {
                    setStatus('Failed to create run script: ' + data.error, true);
                    alert('Failed to create run script: ' + data.error);
                    return;
                }
                setStatus(`run script: ${data.run_script} (${configs.length} experiments)`);
                alert(`Batch script generated!\nTotal experiments: ${configs.length}\nScript: ${data.run_script}`);
            } catch (err) {
                setStatus(`Create run script failed: ${err}`, true);
                alert(`Create run script failed: ${err}`);
            }
        }

        async function startBatchExperiments() {
            return startBatch();
        }

        function resetForm() {
            document.getElementById('multiForm').reset();
            experimentConfigs = [];
            document.getElementById('commandOutput').style.display = 'none';
            ensureDefaultSelections();
            updateSpanUI();
            buildExperimentConfigs();
            setStatus('Form reset.');
        }

        document.querySelectorAll('#multiForm select, #multiForm input').forEach((el) => {
            el.addEventListener('change', () => {
                updateSpanUI();
                buildExperimentConfigs();
            });
            if (el.type === 'text' || el.type === 'number') {
                el.addEventListener('input', () => {
                    updateSpanUI();
                    buildExperimentConfigs();
                });
            }
        });

        function updateExperimentNames() {
            buildExperimentConfigs();
        }

        window.generateCommand = generateCommand;
        window.startBatch = startBatch;
        window.startBatchExperiments = startBatchExperiments;
        window.resetForm = resetForm;
        window.updateExperimentNames = updateExperimentNames;
        window.setStatus = setStatus;

        try {
            ensureDefaultSelections();
            updateSpanUI();
            updateExperimentNames();
            setStatus('Ready.');
        } catch (err) {
            setStatus(`Initialization failed: ${err}`, true);
        }
    </script>
</body>
</html>
"""

# 创建Flask应用
app = Flask(__name__)

# 添加安全配置
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
    SESSION_COOKIE_SECURE=False,  # 在生产环境中设为True
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    JSON_SORT_KEYS=False,
)

workspace_root = Path(__file__).parent


def _build_embedded_multi_template() -> str:
    template = multitask_web.HTML_TEMPLATE
    template = template.replace(
        "<p>Standalone config page for <code>src/train_multi.py</code></p>",
        '<p>Configure <code>src/train_multi.py</code> from the main web UI</p>'
        '<p><a href="/" style="color:#fff; border:1px solid rgba(255,255,255,0.55); '
        'border-radius:6px; padding:6px 10px; text-decoration:none; display:inline-block; '
        'margin-top:10px; font-weight:600;">Open Single-Benchmark Page</a></p>',
    )
    template = template.replace(
        "fetch('/api/generate-command', {",
        "fetch('/api/generate-command-multi', {",
    )
    template = template.replace(
        "fetch('/api/start-batch', {",
        "fetch('/api/start-batch-multi', {",
    )
    return template

# 简单的安全中间件（可选）
@app.before_request
def security_headers():
    """添加安全头和 CORS 支持"""
    pass

@app.after_request
def after_request(response):
    """添加安全响应头和 CORS 头"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Disable cache to avoid stale JS/HTML after rapid UI edits.
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    # 添加 CORS 支持
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response



def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _parse_task_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        raw_values = [part for part in stripped.replace(',', ' ').split(' ') if part]
    else:
        raise ValueError(f"Task list must be list or string, got {type(value)}")
    normalized = [str(item).strip() for item in raw_values if str(item).strip()]
    return _dedupe_keep_order(normalized)


def _coerce_numeric_fields(config_dict, int_fields, float_fields):
    for field_name in int_fields:
        if field_name in config_dict and isinstance(config_dict[field_name], str):
            config_dict[field_name] = int(config_dict[field_name])
    for field_name in float_fields:
        if field_name in config_dict and isinstance(config_dict[field_name], str):
            config_dict[field_name] = float(config_dict[field_name])


def _shell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _extract_api_tokens(payload) -> tuple[str, str]:
    wandb_api_key = ""
    hf_token = ""
    if isinstance(payload, dict):
        wandb_api_key = str(payload.get('wandb_api_key') or '').strip()
        hf_token = str(payload.get('hf_token') or '').strip()

    if not wandb_api_key:
        wandb_api_key = str(os.getenv('WANDB_API_KEY', '')).strip()
    if not hf_token:
        hf_token = str(os.getenv('HF_TOKEN', '')).strip()

    if not wandb_api_key:
        wandb_api_key = LEGACY_WANDB_API_KEY
    if not hf_token:
        hf_token = LEGACY_HF_TOKEN

    return wandb_api_key, hf_token


def _write_api_token_exports(file_obj, wandb_api_key: str, hf_token: str):
    if wandb_api_key:
        file_obj.write(f"export WANDB_API_KEY={_shell_single_quote(wandb_api_key)}\n")
    if hf_token:
        file_obj.write(f"export HF_TOKEN={_shell_single_quote(hf_token)}\n")
    if wandb_api_key or hf_token:
        file_obj.write("\n")


def _normalize_multitask_config(config_dict):
    train_tasks = _parse_task_list(config_dict.get('train_tasks'))
    eval_tasks = _parse_task_list(config_dict.get('eval_tasks'))
    test_tasks = _parse_task_list(config_dict.get('test_tasks'))

    if not train_tasks:
        raise ValueError('train_tasks must contain at least one benchmark')

    all_tasks = train_tasks + eval_tasks + test_tasks
    invalid = [task for task in all_tasks if task not in ASAG_BENCHMARK_SET]
    if invalid:
        raise ValueError(
            f"Unknown benchmark(s): {invalid}. Valid options: {', '.join(sorted(ASAG_BENCHMARK_SET))}"
        )

    config_dict['train_tasks'] = train_tasks
    config_dict['eval_tasks'] = eval_tasks if eval_tasks else list(train_tasks)
    config_dict['test_tasks'] = test_tasks if test_tasks else list(train_tasks)


def _finalize_cmd_parts(cmd_parts):
    if cmd_parts and cmd_parts[-1].endswith(' \\'):
        cmd_parts[-1] = cmd_parts[-1].rstrip(' \\')
    return cmd_parts


def _build_train_multi_cmd_parts(config: MultiExperimentConfig, save_dir: str):
    cmd_parts = [
        'accelerate launch \\',
        '    src/train_multi.py \\',
        f'    --save-dir {save_dir} \\',
        f"    --train-tasks {' '.join(config.train_tasks)} \\",
        f"    --eval-tasks {' '.join(config.eval_tasks)} \\",
        f"    --test-tasks {' '.join(config.test_tasks)} \\",
        f'    --base-model "{config.base_model}" \\',
        f'    --model-class {config.model_class} \\',
        f'    --batch-size {config.batch_size} \\',
        f'    --gradient-accumulation-steps {config.gradient_accumulation_steps} \\',
        f'    --train-frac {config.train_frac} \\',
        f'    --lr {config.lr} \\',
        f'    --max-epoch {config.max_epoch} \\',
        f'    --seed {config.seed} \\',
    ]

    if config.model_class == 'span':
        cmd_parts.append(f'    --span-fuse-type {config.span_fuse_type} \\')
        if config.span_pool_type != 'last':
            cmd_parts.append(f'    --span-pool-type {config.span_pool_type} \\')

    if config.num_bidir_layers > 0:
        cmd_parts.append(f'    --num-bidir-layers {config.num_bidir_layers} \\')
    if config.num_prune_layers > 0:
        cmd_parts.append(f'    --num-prune-layers {config.num_prune_layers} \\')
    if config.num_fuse_layers > 0:
        cmd_parts.append(f'    --num-fuse-layers {config.num_fuse_layers} \\')
        cmd_parts.append(f'    --fuse-type {config.layer_fuse_type} \\')
    if config.num_unsink_layers > 0:
        cmd_parts.append(f'    --num-unsink-layers {config.num_unsink_layers} \\')
    if config.pool_type != 'last':
        cmd_parts.append(f'    --pool-type {config.pool_type} \\')

    if config.random_solution:
        cmd_parts.append('    --random-solution \\')
    if config.use_lora:
        cmd_parts.append('    --use-lora \\')
    if config.use_bnb:
        cmd_parts.append('    --use-bnb \\')
    if config.add_suffix:
        cmd_parts.append('    --add-suffix \\')
    if config.add_context:
        cmd_parts.append('    --add-context \\')
    if config.random_suffix:
        cmd_parts.append('    --random-suffix \\')
    if config.use_translated_prompts:
        cmd_parts.append('    --use_translated_prompts \\')
    if config.random_drop_rub > 0:
        cmd_parts.append(f'    --random-drop-rub {config.random_drop_rub} \\')
    if config.bf16:
        cmd_parts.append('    --bf16 \\')
    if config.log_wandb:
        cmd_parts.append('    --log-wandb')

    return _finalize_cmd_parts(cmd_parts)


@app.route('/multi')
def multi_index():
    """train_multi.py configuration page"""
    return render_template_string(_build_embedded_multi_template())


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate-command', methods=['POST'])
def generate_command():
    """Generate command"""
    try:
        if not request.json:
            return jsonify({"success": False, "error": "No JSON data received"})
        
        config_dict = request.json
        print(f"📝 Generating command, configuration: {config_dict}")  # Debug info
        
        # Validate required fields
        required_fields = ['base_model', 'benchmark', 'model_class']
        missing_fields = [field for field in required_fields if not config_dict.get(field)]
        
        # For span model, also require span_fuse_type
        if config_dict.get('model_class') == 'span' and not config_dict.get('span_fuse_type'):
            missing_fields.append('span_fuse_type')
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            })
        
        # Convert string numbers to int/float
        int_fields = ['batch_size', 'gradient_accumulation_steps', 'max_epoch', 'seed']
        float_fields = ['train_frac', 'lr', 'pairwise_margin', 'num_bidir_layers', 'num_prune_layers', 'num_fuse_layers', 'num_unsink_layers', 'random_drop_rub']
        
        for field in int_fields:
            if field in config_dict and isinstance(config_dict[field], str):
                config_dict[field] = int(config_dict[field])
        
        for field in float_fields:
            if field in config_dict and isinstance(config_dict[field], str):
                config_dict[field] = float(config_dict[field])

        
        config = ExperimentConfig(**{k: v for k, v in config_dict.items() if k in ExperimentConfig.__dataclass_fields__})
        
        exp_name = config.generate_exp_name()
        exp_root = workspace_root / f"results_{config.benchmark}" / exp_name;
        
                # 在 generate_command 函数中，替换这部分代码 (1028-1096行)：
        
        cmd_parts = [
            "accelerate launch \\",
            "    src/train_asag.py \\",
            f"    --save-dir {exp_root} \\",
            f"    --benchmark {config.benchmark} \\",
            f"    --base-model \"{config.base_model}\" \\",
            f"    --model-class {config.model_class} \\",
            f"    --batch-size {config.batch_size} \\",
            f"    --gradient-accumulation-steps {config.gradient_accumulation_steps} \\",
            f"    --train-frac {config.train_frac} \\",
            f"    --lr {config.lr} \\",
            f"    --max-epoch {config.max_epoch} \\",
            f"    --seed {config.seed} \\",
        ]
        

        if config.model_class == "span":
            cmd_parts.append(f"    --span-fuse-type {config.span_fuse_type} \\")
            if config.span_pool_type != "last":
                cmd_parts.append(f"    --span-pool-type {config.span_pool_type} \\")
        

        if config.model_class == "xnet-pwr":
            cmd_parts.append(f"    --pairwise-margin {config.pairwise_margin} \\")
        

        if config.num_bidir_layers > 0:
            cmd_parts.append(f"    --num-bidir-layers {config.num_bidir_layers} \\")
        
        if config.num_prune_layers > 0:
            cmd_parts.append(f"    --num-prune-layers {config.num_prune_layers} \\")
            
        if config.num_fuse_layers > 0:
            cmd_parts.append(f"    --num-fuse-layers {config.num_fuse_layers} \\")
            cmd_parts.append(f"    --fuse-type {config.layer_fuse_type} \\")
            
        if config.num_unsink_layers > 0:
            cmd_parts.append(f"    --num-unsink-layers {config.num_unsink_layers} \\")
            
        if config.pool_type != "last":
            cmd_parts.append(f"    --pool-type {config.pool_type} \\")
            
        if config.random_solution:
            cmd_parts.append("    --random-solution \\")
        
        if config.use_lora:
            cmd_parts.append("    --use-lora \\")
        
        if config.use_bnb:
            cmd_parts.append("    --use-bnb \\")
        
        if config.add_suffix:
            cmd_parts.append("    --add-suffix \\")
        
        if config.add_context:
            cmd_parts.append("    --add-context \\")
        
        if config.random_suffix:
            cmd_parts.append("    --random-suffix \\")
        
        if config.use_translated_prompts:
            cmd_parts.append("    --use_translated_prompts \\")
        
        if config.random_drop_rub > 0:
            cmd_parts.append(f"    --random-drop-rub {config.random_drop_rub} \\")
        
        if config.bf16:
            cmd_parts.append("    --bf16 \\")
        
        if config.log_wandb:
            cmd_parts.append("    --log-wandb")
        
        # Remove trailing backslash from last line
        if cmd_parts and cmd_parts[-1].endswith(" \\"):
            cmd_parts[-1] = cmd_parts[-1].rstrip(" \\")
        
        command = "\n".join(cmd_parts);
        print(f"✅ Command generated successfully");
        return jsonify({"success": True, "command": command});
        
    except Exception as e :
        import traceback
        error_msg = f"Command generation failed: {str(e)}";
        print(f"❌ {error_msg}");
        traceback.print_exc();  # 打印完整错误栈
        return jsonify({"success": False, "error": error_msg});
@app.route('/api/start-batch', methods=['POST', 'OPTIONS'])
def start_batch():
    """Start batch experiments"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        payload = request.json
        if not payload or 'configs' not in payload:
            return jsonify({"success": False, "error": "No configurations provided"})
        
        configs = payload['configs'];
        wandb_api_key, hf_token = _extract_api_tokens(payload)
        print(f"📥 Received {len(configs)} batch configurations");
        
        batch_id = int(time.time());
        batch_results = [];
        
        # 收集所有命令
        commands = [];
        
        for i, config_dict in enumerate(configs):
            try:
                # Convert string numbers to int/float
                int_fields = ['batch_size', 'gradient_accumulation_steps', 'max_epoch', 'seed']
                float_fields = ['train_frac', 'lr', 'pairwise_margin', 'num_bidir_layers', 'num_prune_layers', 'num_fuse_layers', 'num_unsink_layers', 'random_drop_rub']
                
                for field in int_fields:
                    if field in config_dict and isinstance(config_dict[field], str):
                        config_dict[field] = int(config_dict[field])
                
                for field in float_fields:
                    if field in config_dict and isinstance(config_dict[field], str):
                        config_dict[field] = float(config_dict[field])
                
                config = ExperimentConfig(**{k: v for k, v in config_dict.items() if k in ExperimentConfig.__dataclass_fields__})
                
                exp_name = config.generate_exp_name()
                exp_root = workspace_root / f"results_{config.benchmark}" / exp_name
        
                # 生成命令
                cmd_parts = [
                    f"# Experiment {i+1}: {exp_name}",
                    f"EXP_ROOT=\"{exp_root}\"",
                    f"mkdir -p ${{EXP_ROOT}}",
                    f"export WANDB_NAME=\"{exp_name}\"",
                    "accelerate launch \\",
                    "    src/train_asag.py \\",
                    f"    --save-dir ${{EXP_ROOT}} \\",
                    f"    --benchmark {config.benchmark} \\",
                    f"    --base-model \"{config.base_model}\" \\",
                    f"    --model-class {config.model_class} \\",
                    f"    --batch-size {config.batch_size} \\",
                    f"    --gradient-accumulation-steps {config.gradient_accumulation_steps} \\",
                    f"    --train-frac {config.train_frac} \\",
                    f"    --lr {config.lr} \\",
                    f"    --max-epoch {config.max_epoch} \\",
                    f"    --seed {config.seed} \\",
                ]
                
                # Add span-specific parameters
                if config.model_class == "span":
                    cmd_parts.append(f"    --span-fuse-type {config.span_fuse_type} \\")
                    if config.span_pool_type != "last":
                        cmd_parts.append(f"    --span-pool-type {config.span_pool_type} \\")
                
                # Add xnet-pwr specific parameters
                if config.model_class == "xnet-pwr":
                    cmd_parts.append(f"    --pairwise-margin {config.pairwise_margin} \\")
                
                # Add optional parameters
                if config.num_bidir_layers > 0:
                    cmd_parts.append(f"    --num-bidir-layers {config.num_bidir_layers} \\")
                
                if config.num_prune_layers > 0:
                    cmd_parts.append(f"    --num-prune-layers {config.num_prune_layers} \\")
                
                if config.num_fuse_layers > 0:
                    cmd_parts.append(f"    --num-fuse-layers {config.num_fuse_layers} \\")
                    cmd_parts.append(f"    --fuse-type {config.layer_fuse_type} \\")
                
                if config.num_unsink_layers > 0:
                    cmd_parts.append(f"    --num-unsink-layers {config.num_unsink_layers} \\")
                
                if config.pool_type != "last":
                    cmd_parts.append(f"    --pool-type {config.pool_type} \\")
                
                if config.random_solution:
                    cmd_parts.append("    --random-solution \\")
                
                if config.use_lora:
                    cmd_parts.append("    --use-lora \\")
                
                if config.use_bnb:
                    cmd_parts.append("    --use-bnb \\")
                
                if config.add_suffix:
                    cmd_parts.append("    --add-suffix \\")
                
                if config.add_context:
                    cmd_parts.append("    --add-context \\")
                
                if config.random_suffix:
                    cmd_parts.append("    --random-suffix \\")
                
                if config.use_translated_prompts:
                    cmd_parts.append("    --use_translated_prompts \\")
                
                if config.random_drop_rub > 0:
                    cmd_parts.append(f"    --random-drop-rub {config.random_drop_rub} \\")
                
                if config.bf16:
                    cmd_parts.append("    --bf16 \\")
                
                if config.log_wandb:
                    cmd_parts.append("    --log-wandb")
                
                # Remove trailing backslash from last line and add log redirection
                if cmd_parts and cmd_parts[-1].endswith(" \\"):
                    cmd_parts[-1] = cmd_parts[-1].rstrip(" \\")
                cmd_parts[-1] += f" 2>&1 | tee ${{EXP_ROOT}}/out.log"
                cmd_parts.append("")  # Empty line between experiments
                
                commands.append("\n".join(cmd_parts))
                
                batch_results.append({
                    "exp_name": exp_name,
                    "status": "configured"
                })
                
                print(f"✅ [{i+1}/{len(configs)}] Command prepared: {exp_name}")
                
            except Exception as e:
                error_msg = f"Failed to prepare experiment {i+1}: {str(e)}"
                print(f"❌ {error_msg}")
                batch_results.append({
                    "exp_name": f"experiment_{i+1}",
                    "error": error_msg,
                    "status": "failed"
                })
        
        # 生成 run.sh
        run_sh_path = workspace_root / f"run_batch_{batch_id}.sh"
        with open(run_sh_path, "w", encoding='utf-8') as f:
            f.write("#!/usr/bin/env bash\n")
            _write_api_token_exports(f, wandb_api_key, hf_token)
            f.write(f"# Batch ID: {batch_id}\n")
            f.write(f"# Total experiments: {len(configs)}\n")
            f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("\n".join(commands))
        
        # Make executable
        os.chmod(run_sh_path, 0o755)
        
        print(f"✅ Generated run script: {run_sh_path}")
        
        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "total_experiments": len(configs),
            "results": batch_results,
            "run_script": str(run_sh_path),
            "message": f"Batch script generated: {run_sh_path}"
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Batch configuration failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({"success": False, "error": error_msg})


@app.route('/api/generate-command-multi', methods=['POST', 'OPTIONS'])
def generate_command_multi():
    """Generate command for train_multi.py"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "expected JSON body"})

        config = multitask_web._to_config(request.get_json(silent=True) or {})
        exp_name = config.generate_exp_name()
        exp_root = workspace_root / 'results_multi' / exp_name
        command = "\n".join(multitask_web._build_command_lines(config, str(exp_root)))

        return jsonify({
            "success": True,
            "command": command,
            "exp_name": exp_name,
            "save_dir": str(exp_root),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})


@app.route('/api/start-batch-multi', methods=['POST', 'OPTIONS'])
def start_batch_multi():
    """Create run script for one or more train_multi.py configs"""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "expected JSON body"})

        payload = request.get_json(silent=True) or {}
        raw_configs = payload.get('configs')
        if not isinstance(raw_configs, list) or not raw_configs:
            return jsonify({"success": False, "error": "configs must be a non-empty list"})

        wandb_api_key, hf_token = _extract_api_tokens(payload)

        parsed_configs = []
        parse_errors = []
        for idx, raw in enumerate(raw_configs, 1):
            try:
                parsed_configs.append(multitask_web._to_config(raw))
            except Exception as exc:
                parse_errors.append(f'config #{idx}: {exc}')

        if not parsed_configs:
            return jsonify({"success": False, "error": '; '.join(parse_errors) or 'no valid configs'})

        run_script_path, results, command_blocks = multitask_web._build_run_script(
            parsed_configs,
            wandb_api_key,
            hf_token,
        )

        return jsonify({
            "success": True,
            "run_script": str(run_script_path),
            "total_experiments": len(parsed_configs),
            "results": results,
            "errors": parse_errors,
            "commands_preview": "\n".join(command_blocks[:3]),
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)})



if __name__ == '__main__':
    import os
    import socket
    
    # 获取本机IP地址 - 使用更稳健的方法
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                return local_ip
            except:
                return "127.0.0.1"
    
    local_ip = get_local_ip()
    
    # 从环境变量获取配置，提供默认值
    host = os.getenv('FLASK_HOST', '0.0.0.0')  # 监听所有网卡
    port = int(os.getenv('FLASK_PORT', '5000'))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'  # 生产环境默认关闭
    
    print("\n" + "="*60)
    print("    🌐 ASAG Experiment Configuration - Web Interface")
    print("="*60)
    print(f"\n🌐 Server started, accessible via:")
    print(f"   • Local access: http://localhost:{port}")
    print(f"   • LAN access: http://{local_ip}:{port}")
    if host == '0.0.0.0':
        print(f"   • Remote access: http://[your-server-ip]:{port}")
    print(f"\n⚙️  Configuration:")
    print(f"   • Host: {host}")
    print(f"   • Port: {port}")
    print(f"   • Debug: {debug}")
    print(f"\n📝 Usage for remote access:")
    print(f"   export FLASK_HOST=0.0.0.0")
    print(f"   export FLASK_PORT=5000")
    print(f"   python config_web_ui.py")
    print(f"\n💡 Tip: Press Ctrl+C to exit")
    print("="*60 + "\n")
    
    try:
        app.run(debug=debug, host=host, port=port, threaded=True)
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        print("\n💡 Possible solutions:")
        print("  1. Check if port is already in use: lsof -i :{port}")
        print("  2. Try using different port: FLASK_PORT=8080 python config_web_ui.py")
        print("  3. Check firewall: sudo ufw allow {port}")
        print("  4. Verify permissions for workspace directory")
