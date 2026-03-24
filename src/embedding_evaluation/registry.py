"""
因子评估器注册中心

管理所有可用的因子评估器，支持动态注册和获取。
"""

from typing import Dict, Type, Optional, List
from .base import FactorEmbeddingEvaluator


class FactorEvaluatorRegistry:
    """
    因子评估器注册中心
    
    管理所有可用的因子评估器，支持动态注册和获取。
    使用类变量存储注册信息，确保全局唯一。
    
    使用示例:
        # 注册评估器
        @register_evaluator
        class MyFactorEvaluator(FactorEmbeddingEvaluator):
            ...
        
        # 获取评估器
        evaluator_class = FactorEvaluatorRegistry.get('my_factor')
        evaluator = FactorEvaluatorRegistry.create_evaluator('my_factor')
        
        # 列出所有因子
        factors = FactorEvaluatorRegistry.list_factors()
    """
    
    _evaluators: Dict[str, Type[FactorEmbeddingEvaluator]] = {}
    
    @classmethod
    def register(cls, evaluator_class: Type[FactorEmbeddingEvaluator]) -> Type[FactorEmbeddingEvaluator]:
        """
        注册评估器
        
        Args:
            evaluator_class: 评估器类，必须继承自FactorEmbeddingEvaluator
            
        Returns:
            传入的评估器类（便于装饰器使用）
            
        Raises:
            TypeError: 如果传入的类不继承自FactorEmbeddingEvaluator
            ValueError: 如果因子名称已存在
        """
        if not issubclass(evaluator_class, FactorEmbeddingEvaluator):
            raise TypeError(f"评估器必须继承自FactorEmbeddingEvaluator: {evaluator_class}")
        
        # 创建临时实例获取factor_name
        temp_instance = evaluator_class()
        factor_name = temp_instance.factor_name
        
        if factor_name in cls._evaluators:
            raise ValueError(f"因子 '{factor_name}' 已被注册: {cls._evaluators[factor_name]}")
        
        cls._evaluators[factor_name] = evaluator_class
        return evaluator_class
    
    @classmethod
    def get(cls, factor_name: str) -> Optional[Type[FactorEmbeddingEvaluator]]:
        """
        获取评估器类
        
        Args:
            factor_name: 因子名称
            
        Returns:
            评估器类，如果不存在则返回None
        """
        return cls._evaluators.get(factor_name)
    
    @classmethod
    def list_factors(cls) -> List[str]:
        """
        列出所有已注册的因子
        
        Returns:
            因子名称列表
        """
        return list(cls._evaluators.keys())
    
    @classmethod
    def create_evaluator(cls, factor_name: str) -> FactorEmbeddingEvaluator:
        """
        创建评估器实例
        
        Args:
            factor_name: 因子名称
            
        Returns:
            评估器实例
            
        Raises:
            ValueError: 如果因子不存在
        """
        evaluator_class = cls.get(factor_name)
        if evaluator_class is None:
            available = ', '.join(cls.list_factors())
            raise ValueError(f"未知的因子: '{factor_name}'。可用因子: [{available}]")
        return evaluator_class()
    
    @classmethod
    def unregister(cls, factor_name: str) -> bool:
        """
        注销评估器
        
        Args:
            factor_name: 因子名称
            
        Returns:
            是否成功注销
        """
        if factor_name in cls._evaluators:
            del cls._evaluators[factor_name]
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """清空所有注册的评估器"""
        cls._evaluators.clear()
    
    @classmethod
    def is_registered(cls, factor_name: str) -> bool:
        """
        检查因子是否已注册
        
        Args:
            factor_name: 因子名称
            
        Returns:
            是否已注册
        """
        return factor_name in cls._evaluators


def register_evaluator(evaluator_class: Type[FactorEmbeddingEvaluator]) -> Type[FactorEmbeddingEvaluator]:
    """
    装饰器：自动注册评估器
    
    使用示例:
        @register_evaluator
        class MyFactorEvaluator(FactorEmbeddingEvaluator):
            @property
            def factor_name(self) -> str:
                return "my_factor"
            ...
    
    Args:
        evaluator_class: 评估器类
        
    Returns:
        传入的评估器类
    """
    return FactorEvaluatorRegistry.register(evaluator_class)
