"""
多因子Embedding评估模块

提供插件化的embedding评估架构，支持多种量化因子的独立评估和对比分析。

使用示例:
    from embedding_evaluation import (
        MultiFactorEmbeddingAnalyzer,
        FactorEvaluatorRegistry,
        FactorEmbeddingEvaluator
    )
    
    # 查看所有可用因子
    print(FactorEvaluatorRegistry.list_factors())
    
    # 创建分析器
    analyzer = MultiFactorEmbeddingAnalyzer(
        model_path='./out/model.pth',
        factors=['stock', 'market']
    )
    
    # 执行评估
    results = analyzer.analyze_all_factors(stock_info_list)
"""

from .base import FactorEmbeddingEvaluator
from .registry import FactorEvaluatorRegistry, register_evaluator
from .multi_factor_analyzer import MultiFactorEmbeddingAnalyzer

__all__ = [
    'FactorEmbeddingEvaluator',
    'FactorEvaluatorRegistry',
    'register_evaluator',
    'MultiFactorEmbeddingAnalyzer',
]

# 自动导入并注册所有评估器
from . import stock_evaluator
from . import market_evaluator
