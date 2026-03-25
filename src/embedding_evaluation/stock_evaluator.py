"""
个股因子Embedding评估器

评估个股特征(OHLCV+Exchange)的Embedding模块（细处理 + Embedding层）。
"""

import os
import sys

_current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional

from .base import FactorEmbeddingEvaluator
from .registry import register_evaluator
from . import analyzers
from .analyzers import EmbeddingModuleWrapper

from src.config import ModelConfig, DataConfig
from src.data import coarse_normalize_context_window


@register_evaluator
class StockFactorEvaluator(FactorEmbeddingEvaluator):
    """
    个股特征Embedding评估器
    
    评估对象: EmbeddingModule = 细处理(FeatureNormalizer) + Embedding层
    输入: 粗处理后的数据 [batch, seq_len, 6]
    输出: Embedding层输出 [batch, seq_len, d_model]
    """
    
    FACTOR_NAME = "stock"
    INPUT_DIM = 6
    FEATURE_NAMES = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
    
    @property
    def factor_name(self) -> str:
        return self.FACTOR_NAME
    
    @property
    def input_dim(self) -> int:
        return self.INPUT_DIM
    
    @property
    def output_dim(self) -> int:
        return ModelConfig.D_MODEL
    
    def get_embedding_module(self, model: nn.Module, feature_normalizer: Optional[Any] = None) -> nn.Module:
        """
        获取Embedding模块（细处理+Embedding层的组合）
        
        Args:
            model: StockTransformer模型
            feature_normalizer: 特征归一化器，可选
            
        Returns:
            EmbeddingModuleWrapper 实例
        """
        return EmbeddingModuleWrapper(model.embedding, feature_normalizer)
    
    def prepare_sample_data(self, stock_info_list: List[dict], n_samples: int = 500) -> np.ndarray:
        """
        准备个股特征样本数据（只做粗处理，不做细处理）
        
        细处理在评估时由 EmbeddingModuleWrapper 完成。
        
        Args:
            stock_info_list: 股票信息列表
            n_samples: 需要的样本数量
            
        Returns:
            样本输入数据 [n_samples, seq_len, 6] 粗处理后的数据
        """
        all_inputs = []
        
        for stock in stock_info_list[:50]:
            data = stock['data']
            test_split = stock.get('test_split_point', 0)
            
            for i in range(test_split, min(test_split + 10, len(data) - 33)):
                input_seq = coarse_normalize_context_window(
                    data, i, 
                    DataConfig.CONTEXT_LENGTH, 
                    check_limit_up=False, 
                    required_length=DataConfig.REQUIRED_LENGTH
                )
                
                if input_seq is not None:
                    all_inputs.append(input_seq)
                    
                if len(all_inputs) >= n_samples:
                    break
            
            if len(all_inputs) >= n_samples:
                break
        
        return np.array(all_inputs[:n_samples])
    
    def evaluate(self, model: nn.Module, sample_data: np.ndarray, device: torch.device, 
                 feature_normalizer: Optional[Any] = None) -> Dict[str, Any]:
        """
        执行个股Embedding模块评估
        
        包括Jacobian分析、局部敏感性、全局敏感性、输出多样性和饱和度分析。
        
        Args:
            model: 模型实例
            sample_data: 样本数据（粗处理后的数据）
            device: 计算设备
            feature_normalizer: 特征归一化器，可选
            
        Returns:
            评估结果字典
        """
        embedding_module = self.get_embedding_module(model, feature_normalizer)
        results = {}
        
        print(f"  执行Jacobian分析...")
        results['jacobian'] = analyzers.analyze_jacobian_numeric(
            embedding_module, sample_data, device,
            feature_names=self.FEATURE_NAMES
        )
        
        print(f"  执行局部敏感性分析...")
        results['local_sensitivity'] = analyzers.analyze_local_sensitivity(
            embedding_module, sample_data, device,
            feature_names=self.FEATURE_NAMES
        )
        
        print(f"  执行全局敏感性分析...")
        results['global_sensitivity'] = analyzers.analyze_global_sensitivity(
            embedding_module, sample_data, device,
            feature_names=self.FEATURE_NAMES
        )
        
        print(f"  执行输出多样性分析...")
        results['diversity'] = analyzers.analyze_output_diversity(
            embedding_module, sample_data, device
        )
        
        print(f"  执行饱和度分析...")
        results['saturation'] = analyzers.analyze_saturation(
            embedding_module, sample_data, device
        )
        
        return results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        打印个股因子评估结果摘要
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"STOCK Factor Embedding Evaluation Summary")
        print(f"{'='*60}")
        
        if 'jacobian' in results:
            jac = results['jacobian']
            print(f"\nJacobian Analysis:")
            print(f"  Mean Norm: {jac.get('mean_jacobian_norm', 'N/A'):.6f}")
            if 'input_sensitivity' in jac:
                print(f"  Input Sensitivity:")
                for name, value in jac['input_sensitivity'].items():
                    print(f"    {name}: {value:.6f}")
        
        if 'local_sensitivity' in results:
            sens = results['local_sensitivity']
            print(f"\nLocal Sensitivity:")
            for name in self.FEATURE_NAMES:
                if name in sens and isinstance(sens[name], dict):
                    print(f"  {name}: mean={sens[name]['mean']:.6f}, std={sens[name]['std']:.6f}")
            if 'overall' in sens:
                print(f"  Overall: mean={sens['overall']['mean']:.6f}, std={sens['overall']['std']:.6f}")
        
        if 'diversity' in results:
            div = results['diversity']
            print(f"\nOutput Diversity:")
            print(f"  Input-Output Correlation: {div.get('input_output_correlation', 'N/A'):.6f}")
            print(f"  Cosine Similarity: {div.get('output_cosine_similarity', 'N/A'):.6f}")
            print(f"  Output Norm: {div.get('output_norm_mean', 'N/A'):.6f} ± {div.get('output_norm_std', 'N/A'):.6f}")
        
        if 'saturation' in results:
            sat = results['saturation']
            print(f"\nSaturation:")
            print(f"  Mean: {sat.get('hidden_mean', 'N/A'):.6f}")
            print(f"  Std: {sat.get('hidden_std', 'N/A'):.6f}")
            print(f"  Saturation Ratio: {sat.get('saturation_ratio', 'N/A'):.4f}")
            print(f"  Dead Neuron Ratio: {sat.get('dead_neuron_ratio', 'N/A'):.4f}")
