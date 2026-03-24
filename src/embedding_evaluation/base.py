"""
因子Embedding评估器基类

定义所有因子评估器的统一接口规范。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class FactorEmbeddingEvaluator(ABC):
    """
    因子Embedding评估器基类
    
    所有特定因子的评估器都应继承此类，实现以下抽象方法：
    - factor_name: 返回因子名称
    - input_dim: 返回输入维度
    - output_dim: 返回输出维度
    - get_embedding_layer: 从模型中提取对应的embedding层
    - prepare_sample_data: 准备该因子的样本数据
    - evaluate: 执行评估
    
    示例:
        @register_evaluator
        class MyFactorEvaluator(FactorEmbeddingEvaluator):
            @property
            def factor_name(self) -> str:
                return "my_factor"
            
            def get_embedding_layer(self, model) -> nn.Module:
                return model.my_factor_encoder
            
            ...
    """
    
    @property
    @abstractmethod
    def factor_name(self) -> str:
        """
        因子名称，如 'stock', 'market', 'sector' 等
        
        Returns:
            因子的唯一标识名称
        """
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        输入维度
        
        Returns:
            该因子输入特征的维度
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        输出维度（通常是d_model）
        
        Returns:
            embedding输出维度
        """
        pass
    
    @abstractmethod
    def get_embedding_layer(self, model: nn.Module) -> nn.Module:
        """
        从模型中提取对应的embedding层
        
        Args:
            model: StockTransformer模型实例
            
        Returns:
            对应的embedding层（如 model.embedding, model.market_encoder 等）
        """
        pass
    
    @abstractmethod
    def prepare_sample_data(self, stock_info_list: list, n_samples: int = 500) -> np.ndarray:
        """
        准备该因子的样本数据
        
        Args:
            stock_info_list: 股票信息列表
            n_samples: 需要的样本数量
            
        Returns:
            该因子的输入数据，形状取决于因子类型
            - 个股因子: [n_samples, seq_len, 6]
            - 市场因子: [n_samples, market_context_length]
            - 其他因子: 根据具体因子定义
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, sample_data: np.ndarray, device: torch.device) -> Dict[str, Any]:
        """
        执行评估
        
        Args:
            model: 模型实例
            sample_data: 样本数据（由prepare_sample_data生成）
            device: 计算设备
            
        Returns:
            评估结果字典，应包含以下标准键：
            - 'jacobian': Jacobian分析结果
            - 'local_sensitivity': 局部敏感性分析结果
            - 'diversity': 输出多样性分析结果
            - 'saturation': 饱和度分析结果
            因子特定的额外键也可以包含
        """
        pass
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        打印评估结果摘要
        
        子类可以重写此方法以提供因子特定的摘要格式。
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"{self.factor_name.upper()} Factor Evaluation Summary")
        print(f"{'='*60}")
        
        if 'jacobian' in results:
            jac = results['jacobian']
            print(f"\nJacobian Analysis:")
            print(f"  Mean Norm: {jac.get('mean_jacobian_norm', 'N/A')}")
            if 'input_sensitivity' in jac:
                print(f"  Input Sensitivity: {jac['input_sensitivity']}")
        
        if 'local_sensitivity' in results:
            sens = results['local_sensitivity']
            print(f"\nLocal Sensitivity:")
            for key, value in sens.items():
                if isinstance(value, dict) and 'mean' in value:
                    print(f"  {key}: {value['mean']:.6f}")
        
        if 'diversity' in results:
            div = results['diversity']
            print(f"\nOutput Diversity:")
            print(f"  Cosine Similarity: {div.get('output_cosine_similarity', 'N/A')}")
            print(f"  Output Norm Mean: {div.get('output_norm_mean', 'N/A')}")
        
        if 'saturation' in results:
            sat = results['saturation']
            print(f"\nSaturation:")
            print(f"  Saturation Ratio: {sat.get('saturation_ratio', 'N/A')}")
            print(f"  Dead Neuron Ratio: {sat.get('dead_neuron_ratio', 'N/A')}")
