"""
Embedding模块评估脚本

核心思路：评估"Embedding模块"（细处理 + Embedding层）的映射性质
评估对象：粗处理后的数据 → 细处理 → Embedding层 → d_model维输出

评估维度：
1. 局部敏感性分析：输入微小扰动时，输出如何变化
2. 全局敏感性分析: 不同输入范围，输出变化幅度
3. Jacobian分析: 每个输入维度对每个输出维度的影响（数值方法）
4. 表示多样性: 不同输入产生的输出是否足够分散
5. 饱和度分析: 输出分布特征
6. 扰动传播: 各输入维度的扰动如何传播到输出
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import argparse

from model import create_model
from data import load_and_preprocess_data,FeatureNormalizer
from config import DataConfig, ModelConfig


class EmbeddingModule(nn.Module):
    """
    统一的 Embedding 模块
    
    将细处理和 Embedding 层封装为一个整体：
    - 细处理：FeatureNormalizer（训练时学到的变换）
    - Embedding 层：nn.Linear（训练时学到的映射）
    
    注意：粗处理（OHLE变换）在数据加载阶段完成，
    本模块接收粗处理后的数据作为输入。
    """
    
    def __init__(self, embedding_layer, feature_normalizer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.feature_normalizer = feature_normalizer
    
    def forward(self, x):
        """
        Args:
            x: 粗处理后的数据 [batch, seq_len, 6]
               范围：OHLE [-0.1, 0.1], Volume [0, 1], Exchange [0, 1]
        
        Returns:
            embedded: [batch, seq_len, d_model]
        """
        if self.feature_normalizer is not None:
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            if x_np.ndim == 3:
                batch_size, seq_len, n_features = x_np.shape
                x_normalized = np.empty_like(x_np, dtype=np.float32)
                for b in range(batch_size):
                    x_normalized[b] = self.feature_normalizer.transform(x_np[b])
                x = torch.tensor(x_normalized, dtype=torch.float32, device=self.embedding_layer.weight.device)
            else:
                x_normalized = self.feature_normalizer.transform(x_np)
                x = torch.tensor(x_normalized, dtype=torch.float32, device=self.embedding_layer.weight.device)
        
        return self.embedding_layer(x)
    
    def transform_numpy(self, x_np):
        """
        对 numpy 数组应用细处理（不包含 Embedding 层）
        
        Args:
            x_np: 粗处理后的 numpy 数组 [batch, seq_len, 6] 或 [seq_len, 6]
        
        Returns:
            normalized: 细处理后的 numpy 数组
        """
        if self.feature_normalizer is None:
            return x_np
        
        if x_np.ndim == 3:
            batch_size, seq_len, n_features = x_np.shape
            x_normalized = np.empty_like(x_np, dtype=np.float32)
            for b in range(batch_size):
                x_normalized[b] = self.feature_normalizer.transform(x_np[b])
            return x_normalized
        else:
            return self.feature_normalizer.transform(x_np)


class EmbeddingModuleAnalyzer:
    """Embedding模块分析器 - 评估细处理+Embedding层的映射性质"""
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embedding_module = None
        self.feature_normalizer = None
        
    def load_model(self, model_path=None, feature_normalizer=None):
        """
        加载模型并创建 EmbeddingModule
        
        Args:
            model_path: 模型文件路径
            feature_normalizer: 特征归一化器实例
        """
        self.feature_normalizer = feature_normalizer
        
        if model_path:
            self.model_path = model_path
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_arch = checkpoint.get('model_arch', {})
                self.model = create_model(model_arch=model_arch).to(self.device)
                
                current_state = self.model.state_dict()
                loaded_state = checkpoint['state_dict']
                
                matched_state = {}
                for key in current_state:
                    if key in loaded_state:
                        if current_state[key].shape == loaded_state[key].shape:
                            matched_state[key] = loaded_state[key]
                        else:
                            print(f"  跳过不匹配的权重: {key}")
                    else:
                        print(f"  跳过缺失的权重: {key}")
                
                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型: {self.model_path}")
            else:
                self.model = create_model().to(self.device)
                current_state = self.model.state_dict()
                
                matched_state = {}
                for key in current_state:
                    if key in checkpoint:
                        if current_state[key].shape == checkpoint[key].shape:
                            matched_state[key] = checkpoint[key]
                
                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型(旧格式): {self.model_path}")
        else:
            self.model = create_model().to(self.device)
            print("使用随机初始化模型")
        self.model.eval()
        
        if hasattr(self.model, 'embedding'):
            embedding_layer = self.model.embedding
            self.embedding_module = EmbeddingModule(embedding_layer, self.feature_normalizer)
        else:
            raise AttributeError("模型不包含embedding层")
        
        return self.model
    
    def analyze_jacobian(self, sample_inputs, n_samples=100, store_matrices=False):
        """
        Jacobian矩阵分析: 使用数值方法计算 ∂output/∂input
        
        由于 FeatureNormalizer 不是 PyTorch 模块，使用有限差分法计算 Jacobian
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
            store_matrices: 是否存储完整的Jacobian矩阵
        """
        print("\n[Jacobian矩阵分析 - 数值方法]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        results = {
            'mean_jacobian_norm': [],
            'per_input_sensitivity': [],
            'per_output_sensitivity': []
        }
        
        if store_matrices:
            results['jacobian_matrices'] = []
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        epsilon = 1e-5
        
        for i in range(min(n_samples, len(sample_inputs))):
            x = sample_inputs[i:i+1]
            
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                base_output = self.embedding_module(x_tensor)
            
            jacobian = np.zeros((6, 48))
            
            for j in range(6):
                x_plus = x.copy()
                x_plus[0, :, j] += epsilon
                x_minus = x.copy()
                x_minus[0, :, j] -= epsilon
                
                with torch.no_grad():
                    x_plus_tensor = torch.tensor(x_plus, dtype=torch.float32, device=self.device)
                    x_minus_tensor = torch.tensor(x_minus, dtype=torch.float32, device=self.device)
                    
                    out_plus = self.embedding_module(x_plus_tensor)
                    out_minus = self.embedding_module(x_minus_tensor)
                
                grad = (out_plus - out_minus).cpu().numpy() / (2 * epsilon)
                jacobian[j, :] = np.abs(grad).mean(axis=1)
            
            if store_matrices:
                results['jacobian_matrices'].append(jacobian)
            results['mean_jacobian_norm'].append(np.linalg.norm(jacobian))
            
            input_sensitivity = np.linalg.norm(jacobian, axis=1)
            results['per_input_sensitivity'].append(input_sensitivity)
            
            output_sensitivity = np.linalg.norm(jacobian, axis=0)
            results['per_output_sensitivity'].append(output_sensitivity)
        
        avg_input_sens = np.mean(results['per_input_sensitivity'], axis=0)
        avg_output_sens = np.mean(results['per_output_sensitivity'], axis=0)
        
        print(f"  平均Jacobian范数: {np.mean(results['mean_jacobian_norm']):.4f}")
        print(f"  各输入维度敏感性:")
        for i, name in enumerate(feature_names):
            print(f"    {name}: {avg_input_sens[i]:.4f}")
        
        return_dict = {
            'mean_jacobian_norm': float(np.mean(results['mean_jacobian_norm'])),
            'input_sensitivity': {name: float(avg_input_sens[i]) for i, name in enumerate(feature_names)},
            'output_sensitivity_range': [float(np.min(avg_output_sens)), float(np.max(avg_output_sens))]
        }
        
        if store_matrices:
            return_dict['jacobian_matrices'] = results['jacobian_matrices']
            
        return return_dict
    
    def analyze_local_sensitivity(self, sample_inputs, n_samples=50, epsilon=1e-4):
        """
        局部敏感性分析: 输入微小扰动时，Embedding模块输出如何变化
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
            epsilon: 扰动幅度
        """
        print("\n[局部敏感性分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        results = {name: [] for name in feature_names}
        results['overall'] = []
        
        with torch.no_grad():
            for i in range(min(n_samples, len(sample_inputs))):
                x = sample_inputs[i:i+1]
                x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                base_output = self.embedding_module(x_tensor)
                
                for j, name in enumerate(feature_names):
                    x_perturbed = x.copy()
                    x_perturbed[0, :, j] += epsilon
                    
                    x_perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32, device=self.device)
                    perturbed_output = self.embedding_module(x_perturbed_tensor)
                    
                    diff = torch.norm(perturbed_output - base_output).item()
                    results[name].append(diff)
                
                x_all_perturbed = x.copy()
                x_all_perturbed += epsilon
                x_all_perturbed_tensor = torch.tensor(x_all_perturbed, dtype=torch.float32, device=self.device)
                all_perturbed_output = self.embedding_module(x_all_perturbed_tensor)
                overall_diff = torch.norm(all_perturbed_output - base_output).item()
                results['overall'].append(overall_diff)
        
        summary = {}
        print(f"  扰动幅度 ε = {epsilon}")
        print(f"  各维度扰动导致的输出变化:")
        for name in feature_names:
            mean_diff = np.mean(results[name])
            std_diff = np.std(results[name])
            summary[name] = {'mean': float(mean_diff), 'std': float(std_diff)}
            print(f"    {name}: {mean_diff:.6f} +/- {std_diff:.6f}")
        
        print(f"  全维度同时扰动: {np.mean(results['overall']):.6f}")
        summary['overall'] = {'mean': float(np.mean(results['overall'])), 'std': float(np.std(results['overall']))}
        
        return summary
    
    def analyze_global_sensitivity(self, sample_inputs, n_samples=100):
        """
        全局敏感性分析: 不同输入范围，Embedding模块输出变化幅度
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
        """
        print("\n[全局敏感性分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")

        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        results = {name: {'output_range': [], 'output_std': []} for name in feature_names}

        base_input = np.array(sample_inputs[:1])
        
        with torch.no_grad():
            for j, name in enumerate(feature_names):
                outputs = []
                
                values = np.linspace(-0.1, 0.1, 21) if name in ['Open', 'High', 'Low', 'Close'] else np.linspace(0, 1, 21)
                
                for v in values:
                    x = base_input.copy()
                    x[0, :, j] = v
                    x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                    output = self.embedding_module(x_tensor)
                    outputs.append(output[0, 0, :].cpu().numpy())
                
                outputs = np.array(outputs)
                output_range = outputs.max(axis=0) - outputs.min(axis=0)
                results[name]['output_range'] = output_range.tolist()
                results[name]['output_std'] = float(np.std(outputs, axis=0).mean())
        
        print(f"  各输入维度变化时，输出变化范围(平均):")
        for name in feature_names:
            avg_range = np.mean(results[name]['output_range'])
            print(f"    {name}: 平均变化范围 = {avg_range:.4f}, 输出std = {results[name]['output_std']:.4f}")
        
        return results
    
    def analyze_input_output_diversity(self, sample_inputs, n_samples=500):
        """
        表示多样性分析: 不同输入产生的Embedding模块输出是否足够分散
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
        """
        print("\n[表示多样性分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            outputs = self.embedding_module(sample_inputs_tensor)
            
            inputs_flat = sample_inputs.reshape(len(sample_inputs), -1)
            outputs_flat = outputs.reshape(len(outputs), -1).cpu().numpy()
            
            n_pairs = min(1000, len(sample_inputs) * (len(sample_inputs) - 1) // 2)
            input_dists = []
            output_dists = []
            
            indices = np.random.choice(len(sample_inputs), min(100, len(sample_inputs)), replace=False)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    input_dist = np.linalg.norm(inputs_flat[idx_i] - inputs_flat[idx_j])
                    output_dist = np.linalg.norm(outputs_flat[idx_i] - outputs_flat[idx_j])
                    input_dists.append(input_dist)
                    output_dists.append(output_dist)
            
            input_dists = np.array(input_dists)
            output_dists = np.array(output_dists)
            
            correlation, _ = pearsonr(input_dists, output_dists)
            
            output_norms = np.linalg.norm(outputs_flat, axis=1)
            output_cosine_sims = []
            for i in range(min(200, len(outputs_flat))):
                for j in range(i+1, min(200, len(outputs_flat))):
                    cos_sim = 1 - cosine(outputs_flat[i], outputs_flat[j])
                    output_cosine_sims.append(cos_sim)
            
        print(f"  输入-输出距离相关性: {correlation:.4f}")
        print(f"  输出向量平均余弦相似度: {np.mean(output_cosine_sims):.4f}")
        print(f"  输出向量范数: {np.mean(output_norms):.4f} +/- {np.std(output_norms):.4f}")
        print(f"  输出向量范数范围: [{np.min(output_norms):.4f}, {np.max(output_norms):.4f}]")
        
        return {
            'input_output_correlation': float(correlation),
            'output_cosine_similarity': float(np.mean(output_cosine_sims)),
            'output_norm_mean': float(np.mean(output_norms)),
            'output_norm_std': float(np.std(output_norms)),
            'output_norm_range': [float(np.min(output_norms)), float(np.max(output_norms))]
        }
    
    def analyze_saturation(self, sample_inputs, n_samples=100):
        """
        饱和度分析: 分析Embedding模块输出的分布特征
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
        """
        print("\n[饱和度分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        with torch.no_grad():
            x_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            hidden = self.embedding_module(x_tensor)
            
            hidden_flat = hidden.cpu().numpy().flatten()
            
            saturation_ratio = np.mean(np.abs(hidden_flat) > 3)
            
            dead_ratio = np.mean(np.abs(hidden_flat) < 0.01)
        
        print(f"  Embedding模块输出:")
        print(f"    均值: {np.mean(hidden_flat):.4f}")
        print(f"    标准差: {np.std(hidden_flat):.4f}")
        print(f"    范围: [{np.min(hidden_flat):.4f}, {np.max(hidden_flat):.4f}]")
        print(f"    饱和比例(|x|>3): {saturation_ratio*100:.2f}%")
        print(f"    死神经元比例(|output|<0.01): {dead_ratio*100:.2f}%")
        
        return {
            'hidden_mean': float(np.mean(hidden_flat)),
            'hidden_std': float(np.std(hidden_flat)),
            'hidden_min': float(np.min(hidden_flat)),
            'hidden_max': float(np.max(hidden_flat)),
            'saturation_ratio': float(saturation_ratio),
            'dead_neuron_ratio': float(dead_ratio)
        }
    
    def analyze_critical_points(self, sample_inputs, n_samples=50):
        """
        临界点分析: 找出哪些输入区域会导致Embedding模块输出剧烈变化
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
        """
        print("\n[临界点分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        
        second_order_sensitivity = {name: [] for name in feature_names}
        
        epsilon = 1e-3
        
        for i in range(min(n_samples, len(sample_inputs))):
            x = sample_inputs[i:i+1]
            
            for j, name in enumerate(feature_names):
                x_plus = x.copy()
                x_plus[0, :, j] += epsilon
                x_minus = x.copy()
                x_minus[0, :, j] -= epsilon
                
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                    x_plus_tensor = torch.tensor(x_plus, dtype=torch.float32, device=self.device)
                    x_minus_tensor = torch.tensor(x_minus, dtype=torch.float32, device=self.device)
                    
                    out_plus = self.embedding_module(x_plus_tensor)
                    out_minus = self.embedding_module(x_minus_tensor)
                    out_base = self.embedding_module(x_tensor)
                
                second_deriv = torch.norm(out_plus - 2*out_base + out_minus) / (epsilon ** 2)
                second_order_sensitivity[name].append(second_deriv.item())
        
        print(f"  各输入维度的二阶敏感性(曲率):")
        results = {}
        for name in feature_names:
            mean_sens = np.mean(second_order_sensitivity[name])
            max_sens = np.max(second_order_sensitivity[name])
            results[name] = {'mean': float(mean_sens), 'max': float(max_sens)}
            print(f"    {name}: 平均={mean_sens:.4f}, 最大={max_sens:.4f}")
        
        return results
    
    def analyze_dimension_contribution(self, sample_inputs, n_samples=100):
        """
        特征重要性分析（消融实验）
        
        分析各特征对Embedding模块输出的影响。
        使用语义正确的"零值"进行消融：
        - OHLC: 0.0（中间值，涨跌为0）
        - Volume: 0.5（变化率为0，表示维持不变）
        - Exchange: 0.01（1%换手率，普通股票日常水平）
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            n_samples: 样本数量
        """
        print("\n[特征重要性分析（Embedding模块）]")

        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行评估!")
        
        sample_inputs = np.array(sample_inputs[:n_samples])

        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']

        zero_values = {
            'Open': 0.0,
            'High': 0.0,
            'Low': 0.0,
            'Close': 0.0,
            'Volume': 0.5,
            'Exchange': 0.01
        }

        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            base_output = self.embedding_module(sample_inputs_tensor)
            base_norm = torch.norm(base_output).item()

            importance_scores = []

            for j, name in enumerate(feature_names):
                masked_input = sample_inputs.copy()
                masked_input[:, :, j] = zero_values[name]

                masked_input_tensor = torch.tensor(masked_input, dtype=torch.float32, device=self.device)
                masked_output = self.embedding_module(masked_input_tensor)
                masked_norm = torch.norm(masked_output).item()

                relative_change = abs(masked_norm - base_norm) / (base_norm + 1e-6)
                importance_scores.append(relative_change)

        print(f"  各特征对Embedding模块输出的影响（消融实验）:")
        for i, name in enumerate(feature_names):
            print(f"    {name}: 相对变化={importance_scores[i]:.4f} ({importance_scores[i]*100:.2f}%)")

        sorted_indices = np.argsort(importance_scores)[::-1]
        print(f"\n  特征重要性排序:")
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"    {rank}. {feature_names[idx]}: {importance_scores[idx]*100:.2f}%")

        return {
            'feature_names': feature_names,
            'importance_scores': importance_scores,
            'sorted_indices': sorted_indices.tolist()
        }
    
    def visualize_sensitivity(self, sample_inputs, save_dir='out_eval_results', dimension_contribution_results=None):
        """可视化敏感性分析结果
        
        Args:
            sample_inputs: 测试样本（粗处理后的数据）
            save_dir: 保存目录
            dimension_contribution_results: 特征重要性分析结果
        """
        if sample_inputs is None:
            raise ValueError("必须提供真实数据样本(sample_inputs)进行可视化!")

        # 确保正确的字体和减号显示设置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        os.makedirs(save_dir, exist_ok=True)

        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']

        n_samples = min(100, len(sample_inputs))
        sample_inputs = np.array(sample_inputs[:n_samples])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        print("\n[生成可视化图表...]")
        
        ax = axes[0, 0]
        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            base_output = self.embedding_module(sample_inputs_tensor)
            
            sensitivities = []
            for j, name in enumerate(feature_names):
                diffs = []
                for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
                    x_perturbed = sample_inputs.copy()
                    x_perturbed[:, :, j] += eps
                    x_perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32, device=self.device)
                    perturbed_output = self.embedding_module(x_perturbed_tensor)
                    diff = torch.norm(perturbed_output - base_output).item()
                    diffs.append(diff)
                sensitivities.append(diffs)
            
            sensitivities = np.array(sensitivities)
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            
            for i, name in enumerate(feature_names):
                ax.loglog(epsilons, sensitivities[i], 'o-', label=name)
            ax.set_xlabel('Perturbation Size (ε)')
            ax.set_ylabel('Output Change')
            ax.set_title('Local Sensitivity vs Perturbation Size\n(Embedding Module)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        with torch.no_grad():
            base_input = np.zeros((1, 30, 6), dtype=np.float32)
            
            feature_indices = {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4, 'Exchange': 5}
            
            for name in ['Open', 'Close', 'Volume']:
                outputs = []
                j = feature_indices[name]
                values = np.linspace(-0.1, 0.1, 21) if name != 'Volume' else np.linspace(0, 1, 21)
                
                for v in values:
                    x = base_input.copy()
                    x[0, :, j] = v
                    x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                    output = self.embedding_module(x_tensor)
                    norm = torch.norm(output).item()
                    outputs.append(norm)
                
                ax.plot(values, outputs, 'o-', label=name)
            
            ax.set_xlabel('Input Value')
            ax.set_ylabel('Output Norm')
            ax.set_title('Output Norm vs Input Value\n(Embedding Module)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        with torch.no_grad():
            n_pca_samples = min(500, len(sample_inputs))
            test_inputs = sample_inputs[:n_pca_samples]
            test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32, device=self.device)
            outputs = self.embedding_module(test_inputs_tensor)
            outputs_flat = outputs.reshape(-1, 48).cpu().numpy()

            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            outputs_2d = pca.fit_transform(outputs_flat)

            ax.scatter(outputs_2d[:, 0], outputs_2d[:, 1], alpha=0.5, s=5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Output Space Distribution (PCA)\n(Embedding Module)')
        
        ax = axes[1, 0]
        with torch.no_grad():
            x = sample_inputs[:50]
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            
            hidden = self.embedding_module(x_tensor)
            
            hidden_flat = hidden.cpu().numpy().flatten()
            
            ax.hist(hidden_flat, bins=50, alpha=0.7, edgecolor='black', label='Embedding Module Output')
            ax.axvline(x=-3, color='orange', linestyle='--', label='x=+/-3')
            ax.axvline(x=3, color='orange', linestyle='--')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Embedding Module Output Distribution')
            ax.legend()
        
        ax = axes[1, 1]
        if dimension_contribution_results is not None:
            importance_scores = dimension_contribution_results['importance_scores']
            feature_names_result = dimension_contribution_results['feature_names']

            bars = ax.bar(feature_names_result, [s*100 for s in importance_scores],
                        alpha=0.7, color='steelblue')
            ax.set_ylabel('Relative Change (%)')
            ax.set_title('Feature Importance (Ablation Study)\nEmbedding Module Output Change')
            ax.grid(axis='y', alpha=0.3)

            for bar, score in zip(bars, importance_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_scores)*100*0.01,
                       f'{score*100:.2f}%', ha='center', va='bottom', fontsize=9)
        else:
            zero_values = {
                'Open': 0.0,
                'High': 0.0,
                'Low': 0.0,
                'Close': 0.0,
                'Volume': 0.5,
                'Exchange': 0.01
            }
            with torch.no_grad():
                sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
                base_output = self.embedding_module(sample_inputs_tensor)
                base_norm = torch.norm(base_output).item()

                contributions = []
                for j, name in enumerate(feature_names):
                    masked = sample_inputs.copy()
                    masked[:, :, j] = zero_values[name]
                    masked_tensor = torch.tensor(masked, dtype=torch.float32, device=self.device)
                    masked_output = self.embedding_module(masked_tensor)
                    masked_norm = torch.norm(masked_output).item()
                    relative_change = abs(masked_norm - base_norm) / (base_norm + 1e-6)
                    contributions.append(relative_change)

                bars = ax.bar(feature_names, [c*100 for c in contributions], alpha=0.7, color='steelblue')
                ax.set_ylabel('Relative Change (%)')
                ax.set_title('Feature Importance (Ablation Study)\nEmbedding Module Output Change')
                ax.grid(axis='y', alpha=0.3)

                for bar, contrib in zip(bars, contributions):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{contrib*100:.2f}%', ha='center', va='bottom', fontsize=9)

        ax = axes[1, 2]
        with torch.no_grad():
            n_norm_samples = min(200, len(sample_inputs))
            test_inputs = sample_inputs[:n_norm_samples]
            test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32, device=self.device)
            outputs = self.embedding_module(test_inputs_tensor)

            input_norms = np.linalg.norm(test_inputs.reshape(n_norm_samples, -1), axis=1)
            output_norms = torch.norm(outputs.reshape(n_norm_samples, -1), dim=1).cpu().numpy()

            ax.scatter(input_norms, output_norms, alpha=0.5)
            ax.set_xlabel('Input Norm (Coarse Processed)')
            ax.set_ylabel('Output Norm')
            ax.set_title('Input-Output Norm Relationship\n(Embedding Module)')

            z = np.polyfit(input_norms, output_norms, 1)
            p = np.poly1d(z)
            ax.plot(input_norms, p(input_norms), "r--", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'embedding_module_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化图表已保存到: {save_dir}/embedding_module_analysis.png")
    
    def print_summary(self, results):
        """生成分析总结"""
        summary = {
            'key_findings': [],
            'potential_issues': [],
            'recommendations': []
        }
        
        if 'local_sensitivity' in results:
            ls = results['local_sensitivity']
            sensitivities = [(name, ls[name]['mean']) for name in ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']]
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            
            max_sens = sensitivities[0]
            min_sens = sensitivities[-1]
            
            summary['key_findings'].append(f"最敏感输入维度: {max_sens[0]} (变化={max_sens[1]:.4f})")
            summary['key_findings'].append(f"最不敏感输入维度: {min_sens[0]} (变化={min_sens[1]:.4f})")
            
            if max_sens[1] / (min_sens[1] + 1e-10) > 10:
                summary['potential_issues'].append(f"输入维度敏感性差异过大 ({max_sens[0]}比{min_sens[0]}敏感{max_sens[1]/(min_sens[1]+1e-10):.1f}倍)")
                summary['recommendations'].append("考虑对输入特征进行归一化，使各维度敏感性均衡")
        
        if 'diversity' in results:
            div = results['diversity']
            summary['key_findings'].append(f"输出向量平均余弦相似度: {div['output_cosine_similarity']:.4f}")
            
            if div['output_cosine_similarity'] > 0.9:
                summary['potential_issues'].append("输出向量过于相似，表示多样性不足")
                summary['recommendations'].append("考虑增加embedding维度或添加正则化")
        
        if 'saturation' in results:
            sat = results['saturation']
            summary['key_findings'].append(f"饱和比例: {sat['saturation_ratio']*100:.2f}%")
            summary['key_findings'].append(f"死神经元比例: {sat['dead_neuron_ratio']*100:.2f}%")
            
            if sat['saturation_ratio'] > 0.1:
                summary['potential_issues'].append(f"饱和比例较高 ({sat['saturation_ratio']*100:.1f}%)")
                summary['recommendations'].append("考虑添加LayerNorm或调整权重初始化")
        
        """打印分析总结到屏幕"""
        print("\n" + "="*70)
        print("分析总结")

        print("\n关键发现:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")

        if summary['potential_issues']:
            print("\n潜在问题:")
            for issue in summary['potential_issues']:
                print(f"  ⚠ {issue}")

        if summary['recommendations']:
            print("\n优化建议:")
            for rec in summary['recommendations']:
                print(f"  💡 {rec}")
        return

def main():
    parser = argparse.ArgumentParser(description='Embedding模块评估')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要分析的模型文件路径（例如: ./out/modelB_xxx.pth）。如果不指定，将自动使用最新的模型')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用的模型文件并退出')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Embedding模块评估...")
    print("评估对象: 细处理(FeatureNormalizer) + Embedding层")

    out_dir = './out'
    model_files = []
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith('.pth'):
                model_files.append(os.path.join(out_dir, f))

    if args.list_models:
        print("\n可用的模型文件:")
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, mf in enumerate(model_files, 1):
                mtime = datetime.fromtimestamp(os.path.getmtime(mf)).strftime('%Y-%m-%d %H:%M:%S')
                size = os.path.getsize(mf) / (1024 * 1024)
                print(f"  {i}. {os.path.basename(mf)} ({mtime}, {size:.2f} MB)")
        else:
            print("  未找到任何模型文件")
        return None, None

    if args.model:
        if os.path.sep not in args.model and '/' not in args.model:
            potential_path = os.path.join(out_dir, args.model)
            if os.path.exists(potential_path):
                args.model = potential_path

        if not os.path.exists(args.model):
            print(f"\n错误: 指定的模型文件不存在: {args.model}")
            print("\n可用的模型文件:")
            if model_files:
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                for i, mf in enumerate(model_files, 1):
                    print(f"  {i}. {os.path.basename(mf)}")
            else:
                print("  未找到任何模型文件")
            return None, None
        model_path = args.model
        print(f"\n使用指定的模型: {os.path.basename(model_path)}")
    else:
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"\n找到 {len(model_files)} 个模型文件，使用最新的: {os.path.basename(model_files[0])}")
            model_path = model_files[0]
        else:
            print("\n未找到模型文件，将使用随机初始化模型")
            model_path = None
    
    save_dir = './out_eval_results'
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n⚠ 未找到归一化器文件: {DataConfig.NORMALIZER_PATH}")
        sys.exit()

    print("\n"+"="*60)
    print("[步骤1] 加载数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()

    print("\n[步骤2] 准备测试样本...")
    from data import coarse_normalize_context_window
    all_inputs = []
    for stock in test_stock_info[:50]:
        data = stock['data']
        test_split = stock['test_split_point']
        for i in range(test_split, min(test_split + 10, len(data) - 33)):
            input_seq = coarse_normalize_context_window(
                data, i, 30, check_limit_up=False, required_length=33
            )
            if input_seq is not None:
                all_inputs.append(input_seq)
    
    sample_inputs = np.array(all_inputs[:500])
    print(f"  准备了 {len(sample_inputs)} 个测试样本（粗处理后的数据）")

    analyzer = EmbeddingModuleAnalyzer(model_path=model_path)
    analyzer.load_model(feature_normalizer=feature_normalizer)
    
    print("\n[步骤3] 执行分析...")
    results = {}
    
    results['jacobian'] = analyzer.analyze_jacobian(sample_inputs, n_samples=50)
    results['local_sensitivity'] = analyzer.analyze_local_sensitivity(sample_inputs, n_samples=50)
    results['global_sensitivity'] = analyzer.analyze_global_sensitivity(sample_inputs, n_samples=50)
    results['diversity'] = analyzer.analyze_input_output_diversity(sample_inputs, n_samples=200)
    results['saturation'] = analyzer.analyze_saturation(sample_inputs, n_samples=100)
    results['critical_points'] = analyzer.analyze_critical_points(sample_inputs, n_samples=30)
    results['dimension_contribution'] = analyzer.analyze_dimension_contribution(sample_inputs, n_samples=100)

    print("\n[步骤4] 生成可视化...")
    analyzer.visualize_sensitivity(sample_inputs, save_dir, dimension_contribution_results=results['dimension_contribution'])

    print("\n[步骤5] 生成总结...")
    analyzer.print_summary(results)

    print(f"分析完成！可视化图表已保存到: {save_dir}/embedding_module_analysis.png")

    return results


if __name__ == "__main__":
    results = main()
