"""
通用Embedding分析函数

提供适用于任何embedding层的通用分析函数。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


def analyze_jacobian_numeric(
    embedding_layer: nn.Module,
    sample_inputs: np.ndarray,
    device: torch.device,
    epsilon: float = 1e-5,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    通用的Jacobian数值分析方法
    
    使用有限差分法计算Jacobian矩阵，适用于任何embedding层。
    
    Args:
        embedding_layer: 要分析的embedding层
        sample_inputs: 样本输入 [n_samples, ...]
        device: 计算设备
        epsilon: 扰动幅度
        feature_names: 特征名称列表
        
    Returns:
        Jacobian分析结果字典
    """
    n_samples = min(100, len(sample_inputs))
    sample_inputs = sample_inputs[:n_samples]
    
    # 确定输入维度
    if sample_inputs.ndim == 3:
        # 个股因子: [batch, seq_len, features]
        n_features = sample_inputs.shape[2]
    elif sample_inputs.ndim == 2:
        # 市场因子: [batch, features]
        n_features = sample_inputs.shape[1]
    else:
        n_features = sample_inputs.shape[-1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 获取输出维度
    with torch.no_grad():
        x_tensor = torch.tensor(sample_inputs[0:1], dtype=torch.float32, device=device)
        output = embedding_layer(x_tensor)
        output_dim = output.shape[-1]
    
    results = {
        'mean_jacobian_norm': [],
        'per_input_sensitivity': [],
        'per_output_sensitivity': []
    }
    
    for i in range(n_samples):
        x = sample_inputs[i:i+1]
        
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            base_output = embedding_layer(x_tensor)
        
        # 计算Jacobian
        jacobian = np.zeros((n_features, output_dim))
        
        for j in range(n_features):
            x_plus = x.copy()
            x_minus = x.copy()
            
            if x.ndim == 3:
                x_plus[0, :, j] += epsilon
                x_minus[0, :, j] -= epsilon
            else:
                x_plus[0, j] += epsilon
                x_minus[0, j] -= epsilon
            
            with torch.no_grad():
                x_plus_tensor = torch.tensor(x_plus, dtype=torch.float32, device=device)
                x_minus_tensor = torch.tensor(x_minus, dtype=torch.float32, device=device)
                out_plus = embedding_layer(x_plus_tensor)
                out_minus = embedding_layer(x_minus_tensor)
            
            grad = (out_plus - out_minus).cpu().numpy() / (2 * epsilon)
            jacobian[j, :] = np.abs(grad).mean(axis=tuple(range(grad.ndim - 1)))
        
        results['mean_jacobian_norm'].append(np.linalg.norm(jacobian))
        results['per_input_sensitivity'].append(np.linalg.norm(jacobian, axis=1))
        results['per_output_sensitivity'].append(np.linalg.norm(jacobian, axis=0))
    
    avg_input_sens = np.mean(results['per_input_sensitivity'], axis=0)
    avg_output_sens = np.mean(results['per_output_sensitivity'], axis=0)
    
    return {
        'mean_jacobian_norm': float(np.mean(results['mean_jacobian_norm'])),
        'input_sensitivity': {name: float(avg_input_sens[i]) for i, name in enumerate(feature_names)},
        'output_sensitivity_range': [float(np.min(avg_output_sens)), float(np.max(avg_output_sens))],
        'jacobian_matrices': results.get('jacobian_matrices', [])
    }


def analyze_local_sensitivity(
    embedding_layer: nn.Module,
    sample_inputs: np.ndarray,
    device: torch.device,
    epsilon: float = 1e-4,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    局部敏感性分析
    
    分析输入微小扰动时，embedding输出的变化。
    
    Args:
        embedding_layer: 要分析的embedding层
        sample_inputs: 样本输入
        device: 计算设备
        epsilon: 扰动幅度
        feature_names: 特征名称列表
        
    Returns:
        局部敏感性分析结果
    """
    n_samples = min(50, len(sample_inputs))
    sample_inputs = sample_inputs[:n_samples]
    
    # 确定输入维度
    if sample_inputs.ndim == 3:
        n_features = sample_inputs.shape[2]
    elif sample_inputs.ndim == 2:
        n_features = sample_inputs.shape[1]
    else:
        n_features = sample_inputs.shape[-1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    results = {name: [] for name in feature_names}
    results['overall'] = []
    
    with torch.no_grad():
        for i in range(n_samples):
            x = sample_inputs[i:i+1]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
            base_output = embedding_layer(x_tensor)
            
            # 各维度单独扰动
            for j, name in enumerate(feature_names):
                x_perturbed = x.copy()
                if x.ndim == 3:
                    x_perturbed[0, :, j] += epsilon
                else:
                    x_perturbed[0, j] += epsilon
                
                x_perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32, device=device)
                perturbed_output = embedding_layer(x_perturbed_tensor)
                
                diff = torch.norm(perturbed_output - base_output).item()
                results[name].append(diff)
            
            # 全维度同时扰动
            x_all_perturbed = x.copy() + epsilon
            x_all_perturbed_tensor = torch.tensor(x_all_perturbed, dtype=torch.float32, device=device)
            all_perturbed_output = embedding_layer(x_all_perturbed_tensor)
            overall_diff = torch.norm(all_perturbed_output - base_output).item()
            results['overall'].append(overall_diff)
    
    # 汇总统计
    summary = {}
    for name in feature_names:
        summary[name] = {
            'mean': float(np.mean(results[name])),
            'std': float(np.std(results[name]))
        }
    summary['overall'] = {
        'mean': float(np.mean(results['overall'])),
        'std': float(np.std(results['overall']))
    }
    
    return summary


def analyze_output_diversity(
    embedding_layer: nn.Module,
    sample_inputs: np.ndarray,
    device: torch.device
) -> Dict[str, Any]:
    """
    输出多样性分析
    
    分析不同输入产生的embedding输出是否足够分散。
    
    Args:
        embedding_layer: 要分析的embedding层
        sample_inputs: 样本输入
        device: 计算设备
        
    Returns:
        输出多样性分析结果
    """
    n_samples = min(500, len(sample_inputs))
    sample_inputs = sample_inputs[:n_samples]
    
    with torch.no_grad():
        sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=device)
        outputs = embedding_layer(sample_inputs_tensor)
        
        # 展平输出
        outputs_flat = outputs.reshape(len(outputs), -1).cpu().numpy()
        inputs_flat = sample_inputs.reshape(len(sample_inputs), -1)
        
        # 计算输入-输出距离相关性
        n_pairs = min(1000, len(sample_inputs) * (len(sample_inputs) - 1) // 2)
        input_dists = []
        output_dists = []
        
        indices = np.random.choice(len(sample_inputs), min(100, len(sample_inputs)), replace=False)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i, idx_j = indices[i], indices[j]
                input_dist = np.linalg.norm(inputs_flat[idx_i] - inputs_flat[idx_j])
                output_dist = np.linalg.norm(outputs_flat[idx_i] - outputs_flat[idx_j])
                input_dists.append(input_dist)
                output_dists.append(output_dist)
        
        input_dists = np.array(input_dists)
        output_dists = np.array(output_dists)
        
        correlation, _ = pearsonr(input_dists, output_dists)
        
        # 计算输出向量余弦相似度
        output_norms = np.linalg.norm(outputs_flat, axis=1)
        output_cosine_sims = []
        for i in range(min(200, len(outputs_flat))):
            for j in range(i + 1, min(200, len(outputs_flat))):
                cos_sim = 1 - cosine(outputs_flat[i], outputs_flat[j])
                output_cosine_sims.append(cos_sim)
    
    return {
        'input_output_correlation': float(correlation),
        'output_cosine_similarity': float(np.mean(output_cosine_sims)),
        'output_norm_mean': float(np.mean(output_norms)),
        'output_norm_std': float(np.std(output_norms)),
        'output_norm_range': [float(np.min(output_norms)), float(np.max(output_norms))]
    }


def analyze_saturation(
    embedding_layer: nn.Module,
    sample_inputs: np.ndarray,
    device: torch.device
) -> Dict[str, Any]:
    """
    饱和度分析
    
    分析embedding输出的分布特征，检测饱和和死神经元。
    
    Args:
        embedding_layer: 要分析的embedding层
        sample_inputs: 样本输入
        device: 计算设备
        
    Returns:
        饱和度分析结果
    """
    n_samples = min(100, len(sample_inputs))
    sample_inputs = sample_inputs[:n_samples]
    
    with torch.no_grad():
        x_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=device)
        hidden = embedding_layer(x_tensor)
        
        hidden_flat = hidden.cpu().numpy().flatten()
        
        # 饱和比例 (|x| > 3)
        saturation_ratio = np.mean(np.abs(hidden_flat) > 3)
        
        # 死神经元比例 (|x| < 0.01)
        dead_ratio = np.mean(np.abs(hidden_flat) < 0.01)
    
    return {
        'hidden_mean': float(np.mean(hidden_flat)),
        'hidden_std': float(np.std(hidden_flat)),
        'hidden_min': float(np.min(hidden_flat)),
        'hidden_max': float(np.max(hidden_flat)),
        'saturation_ratio': float(saturation_ratio),
        'dead_neuron_ratio': float(dead_ratio)
    }


def analyze_global_sensitivity(
    embedding_layer: nn.Module,
    sample_inputs: np.ndarray,
    device: torch.device,
    feature_names: Optional[List[str]] = None,
    value_ranges: Optional[Dict[str, tuple]] = None
) -> Dict[str, Any]:
    """
    全局敏感性分析
    
    分析不同输入范围，embedding输出的变化幅度。
    
    Args:
        embedding_layer: 要分析的embedding层
        sample_inputs: 样本输入
        device: 计算设备
        feature_names: 特征名称列表
        value_ranges: 各特征的值范围，如 {'Open': (-0.1, 0.1), 'Volume': (0, 1)}
        
    Returns:
        全局敏感性分析结果
    """
    # 确定输入维度
    if sample_inputs.ndim == 3:
        n_features = sample_inputs.shape[2]
    elif sample_inputs.ndim == 2:
        n_features = sample_inputs.shape[1]
    else:
        n_features = sample_inputs.shape[-1]
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    if value_ranges is None:
        # 默认范围
        value_ranges = {name: (-0.1, 0.1) for name in feature_names}
    
    base_input = np.array(sample_inputs[:1])
    results = {}
    
    with torch.no_grad():
        for j, name in enumerate(feature_names):
            outputs = []
            
            min_val, max_val = value_ranges.get(name, (-0.1, 0.1))
            values = np.linspace(min_val, max_val, 21)
            
            for v in values:
                x = base_input.copy()
                if x.ndim == 3:
                    x[0, :, j] = v
                else:
                    x[0, j] = v
                
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                output = embedding_layer(x_tensor)
                outputs.append(output[0, ...].cpu().numpy().flatten())
            
            outputs = np.array(outputs)
            output_range = outputs.max(axis=0) - outputs.min(axis=0)
            
            results[name] = {
                'output_range': output_range.tolist(),
                'output_std': float(np.std(outputs, axis=0).mean()),
                'value_range': (min_val, max_val)
            }
    
    return results
