"""
市场因子Embedding评估器

评估市场Token(MarketTokenEncoder)的Embedding模块（归一化 + MarketTokenEncoder）。
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
from .analyzers import MarketEmbeddingWrapper

from src.config import ModelConfig, DataConfig
from src.data import GlobalDataManager


@register_evaluator
class MarketFactorEvaluator(FactorEmbeddingEvaluator):
    """
    市场Token Embedding评估器
    
    评估对象: MarketEmbeddingWrapper = transform_market + MarketTokenEncoder
    输入: 原始大盘涨跌序列 [batch, market_context_length]
    输出: 市场token embedding [batch, 1, d_model]
    """
    
    FACTOR_NAME = "market"
    
    @property
    def factor_name(self) -> str:
        return self.FACTOR_NAME
    
    @property
    def input_dim(self) -> int:
        return DataConfig.MARKET_CONTEXT_LENGTH
    
    @property
    def output_dim(self) -> int:
        return ModelConfig.D_MODEL
    
    @property
    def feature_names(self) -> List[str]:
        return [f'Market_Day_{i+1}' for i in range(DataConfig.MARKET_CONTEXT_LENGTH)]
    
    def get_embedding_module(self, model: nn.Module, feature_normalizer: Optional[Any] = None) -> nn.Module:
        """
        获取Embedding模块（归一化+MarketTokenEncoder的组合）
        
        Args:
            model: StockTransformer模型
            feature_normalizer: 特征归一化器，可选
            
        Returns:
            MarketEmbeddingWrapper 实例
        """
        return MarketEmbeddingWrapper(model.market_encoder, feature_normalizer)
    
    def prepare_sample_data(self, stock_info_list: List[dict], n_samples: int = 500) -> np.ndarray:
        """
        准备市场数据样本（原始大盘涨跌序列，不做归一化）
        
        归一化在评估时由 MarketEmbeddingWrapper 完成。
        
        Args:
            stock_info_list: 股票信息列表
            n_samples: 需要的样本数量
            
        Returns:
            市场序列数据 [n_samples, market_context_length] 原始大盘涨跌序列
            
        Raises:
            RuntimeError: 如果GlobalDataManager未加载大盘数据
        """
        gdm = GlobalDataManager.get_instance()
        if not gdm.is_market_data_loaded():
            raise RuntimeError(
                "GlobalDataManager未加载大盘数据。"
                "请先调用gdm.load_market_data()加载数据。"
            )
        
        market_samples = []
        
        for stock in stock_info_list[:50]:
            times = stock.get('times', None)
            if times is None:
                continue
            
            test_split = stock.get('test_split_point', 0)
            
            for i in range(test_split, min(test_split + 10, len(times))):
                target_date = int(times[i])
                market_seq = gdm.get_market_context(target_date)
                
                if market_seq is not None:
                    market_samples.append(market_seq)
                
                if len(market_samples) >= n_samples:
                    break
            
            if len(market_samples) >= n_samples:
                break
        
        if len(market_samples) == 0:
            raise RuntimeError("无法获取任何市场数据样本。请检查大盘数据是否正确加载。")
        
        return np.array(market_samples[:n_samples])
    
    def evaluate(self, model: nn.Module, sample_data: np.ndarray, device: torch.device,
                 feature_normalizer: Optional[Any] = None) -> Dict[str, Any]:
        """
        执行市场Embedding模块评估
        
        包括Jacobian分析、局部敏感性、输出多样性、饱和度分析，
        以及市场特定的时间衰减分析。
        
        Args:
            model: 模型实例
            sample_data: 样本数据（原始大盘涨跌序列）
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
            feature_names=self.feature_names
        )
        
        print(f"  执行局部敏感性分析...")
        results['local_sensitivity'] = analyzers.analyze_local_sensitivity(
            embedding_module, sample_data, device,
            feature_names=self.feature_names
        )
        
        print(f"  执行输出多样性分析...")
        results['diversity'] = analyzers.analyze_output_diversity(
            embedding_module, sample_data, device
        )
        
        print(f"  执行饱和度分析...")
        results['saturation'] = analyzers.analyze_saturation(
            embedding_module, sample_data, device
        )
        
        print(f"  执行时间衰减分析...")
        results['temporal_decay'] = self._analyze_temporal_decay(
            embedding_module, sample_data, device
        )
        
        return results
    
    def _analyze_temporal_decay(
        self, 
        embedding_module: nn.Module, 
        sample_data: np.ndarray, 
        device: torch.device
    ) -> Dict[str, Any]:
        """
        分析市场embedding对历史数据的敏感度随时间的衰减
        
        即：越久远的数据（序列中靠前的元素）对当前embedding的影响是否越小。
        
        Args:
            embedding_module: Embedding模块
            sample_data: 样本数据
            device: 计算设备
            
        Returns:
            时间衰减分析结果
        """
        n_samples = min(100, len(sample_data))
        sample_data = sample_data[:n_samples]
        context_length = sample_data.shape[1]
        
        decay_scores = []
        
        with torch.no_grad():
            for i in range(n_samples):
                x = sample_data[i:i+1]
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
                base_output = embedding_module(x_tensor)
                
                step_impacts = []
                epsilon = 1e-3
                
                for step in range(context_length):
                    x_perturbed = x.copy()
                    x_perturbed[0, step] += epsilon
                    
                    x_perturbed_tensor = torch.tensor(x_perturbed, dtype=torch.float32, device=device)
                    perturbed_output = embedding_module(x_perturbed_tensor)
                    
                    impact = torch.norm(perturbed_output - base_output).item()
                    step_impacts.append(impact)
                
                decay_scores.append(step_impacts)
        
        decay_scores = np.array(decay_scores)
        avg_decay = decay_scores.mean(axis=0)
        
        recent_impact = avg_decay[-5:].mean()
        old_impact = avg_decay[:5].mean()
        decay_ratio = recent_impact / (old_impact + 1e-10)
        
        return {
            'temporal_impacts': avg_decay.tolist(),
            'recent_impact': float(recent_impact),
            'old_impact': float(old_impact),
            'decay_ratio': float(decay_ratio),
            'has_decay': bool(decay_ratio > 1.2)
        }
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        打印市场因子评估结果摘要
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"MARKET Factor Embedding Evaluation Summary")
        print(f"{'='*60}")
        
        if 'jacobian' in results:
            jac = results['jacobian']
            print(f"\nJacobian Analysis:")
            print(f"  Mean Norm: {jac.get('mean_jacobian_norm', 'N/A'):.6f}")
            if 'input_sensitivity' in jac:
                sens = jac['input_sensitivity']
                keys = list(sens.keys())
                print(f"  Input Sensitivity (showing first/last 5):")
                for name in keys[:5]:
                    print(f"    {name}: {sens[name]:.6f}")
                if len(keys) > 10:
                    print(f"    ... ({len(keys) - 10} more) ...")
                for name in keys[-5:]:
                    print(f"    {name}: {sens[name]:.6f}")
        
        if 'local_sensitivity' in results:
            sens = results['local_sensitivity']
            print(f"\nLocal Sensitivity:")
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
        
        if 'temporal_decay' in results:
            td = results['temporal_decay']
            print(f"\nTemporal Decay Analysis:")
            print(f"  Recent Impact (last 5 days): {td.get('recent_impact', 'N/A'):.6f}")
            print(f"  Old Impact (first 5 days): {td.get('old_impact', 'N/A'):.6f}")
            print(f"  Decay Ratio: {td.get('decay_ratio', 'N/A'):.4f}")
            print(f"  Has Temporal Decay: {td.get('has_decay', 'N/A')}")
