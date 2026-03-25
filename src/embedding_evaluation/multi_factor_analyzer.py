"""
多因子Embedding统一分析器

协调多个因子评估器的执行，汇总结果，进行对比分析和可视化。
"""

import os
import sys

# 确保能正确导入其他模块（无论从哪里运行）
_current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

from .registry import FactorEvaluatorRegistry
from .base import FactorEmbeddingEvaluator

from src.model import create_model
from src.config import ModelConfig


class MultiFactorEmbeddingAnalyzer:
    """
    多因子Embedding统一分析器
    
    协调多个因子评估器的执行，汇总结果，进行对比分析和可视化。
    
    使用示例:
        analyzer = MultiFactorEmbeddingAnalyzer(
            model_path='./src/out/model.pth',
            factors=['stock', 'market']
        )
        analyzer.load_model(feature_normalizer=feature_normalizer)
        results = analyzer.analyze_all_factors(stock_info_list)
        analyzer.visualize_all_factors(results)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None,
                 factors: Optional[List[str]] = None):
        """
        初始化多因子分析器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备，None则自动选择
            factors: 要评估的因子列表，None则评估所有已注册因子
        """
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.feature_normalizer = None
        
        # 确定要评估的因子
        if factors is None or factors == ['all']:
            self.factors = FactorEvaluatorRegistry.list_factors()
        else:
            # 验证因子名称
            available = FactorEvaluatorRegistry.list_factors()
            invalid = [f for f in factors if f not in available]
            if invalid:
                raise ValueError(f"未知因子: {invalid}。可用因子: {available}")
            self.factors = factors
        
        # 创建评估器实例
        self.evaluators: Dict[str, FactorEmbeddingEvaluator] = {}
        for factor_name in self.factors:
            try:
                self.evaluators[factor_name] = FactorEvaluatorRegistry.create_evaluator(factor_name)
            except ValueError as e:
                print(f"警告: 无法创建因子 '{factor_name}' 的评估器: {e}")
        
        print(f"已配置的因子评估器: {list(self.evaluators.keys())}")
    
    def load_model(self, model_path: Optional[str] = None, 
                   feature_normalizer: Optional[Any] = None) -> None:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径，None则使用初始化时的路径
            feature_normalizer: 特征归一化器
        """
        if model_path is not None:
            self.model_path = model_path
        
        if self.model_path is None:
            raise ValueError("必须提供模型路径")
        
        self.feature_normalizer = feature_normalizer
        
        print(f"\n正在加载模型: {self.model_path}")
        self.model = create_model()
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
    
    def analyze_all_factors(self, stock_info_list: List[dict], 
                           n_samples: int = 500) -> Dict[str, Dict[str, Any]]:
        """
        评估所有配置的因子
        
        Args:
            stock_info_list: 股票信息列表
            n_samples: 每个因子的样本数量
            
        Returns:
            各因子的评估结果 {factor_name: results}
        """
        if self.model is None:
            raise RuntimeError("请先调用load_model()加载模型")
        
        results = {}
        
        for factor_name, evaluator in self.evaluators.items():
            print(f"\n{'='*60}")
            print(f"评估因子: {factor_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # 准备样本数据（只做粗处理）
                print(f"  准备样本数据...")
                sample_data = evaluator.prepare_sample_data(stock_info_list, n_samples)
                print(f"  获取 {len(sample_data)} 个样本")
                
                # 执行评估（细处理在评估时由 EmbeddingModuleWrapper 完成）
                factor_results = evaluator.evaluate(self.model, sample_data, self.device, self.feature_normalizer)
                results[factor_name] = factor_results
                
                # 打印摘要
                evaluator.print_summary(factor_results)
                
            except Exception as e:
                print(f"  错误: 评估因子 '{factor_name}' 失败: {e}")
                import traceback
                traceback.print_exc()
                results[factor_name] = {"error": str(e)}
        
        return results
    
    def compare_factors(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        对比不同因子的embedding特性
        
        Args:
            results: 各因子的评估结果
            
        Returns:
            对比分析结果
        """
        comparison = {
            'sensitivity_comparison': {},
            'diversity_comparison': {},
            'saturation_comparison': {},
            'recommendations': []
        }
        
        # 对比各因子的局部敏感性
        for factor_name, result in results.items():
            if 'error' in result:
                continue
                
            if 'local_sensitivity' in result and 'overall' in result['local_sensitivity']:
                comparison['sensitivity_comparison'][factor_name] = \
                    result['local_sensitivity']['overall']['mean']
            
            if 'diversity' in result:
                comparison['diversity_comparison'][factor_name] = {
                    'correlation': result['diversity'].get('input_output_correlation', 0),
                    'cosine_sim': result['diversity'].get('output_cosine_similarity', 0)
                }
            
            if 'saturation' in result:
                comparison['saturation_comparison'][factor_name] = {
                    'saturation_ratio': result['saturation'].get('saturation_ratio', 0),
                    'dead_ratio': result['saturation'].get('dead_neuron_ratio', 0)
                }
        
        # 生成对比建议
        self._generate_recommendations(comparison)
        
        return comparison
    
    def _generate_recommendations(self, comparison: Dict[str, Any]) -> None:
        """生成优化建议"""
        recommendations = []
        
        # 敏感性分析建议
        sens = comparison.get('sensitivity_comparison', {})
        if sens:
            max_sens = max(sens.values())
            min_sens = min(sens.values())
            if max_sens > min_sens * 2:
                max_factor = max(sens, key=sens.get)
                recommendations.append(
                    f"{max_factor}因子的敏感性过高({max_sens:.4f})，"
                    f"建议检查是否存在梯度爆炸风险"
                )
        
        # 多样性分析建议
        div = comparison.get('diversity_comparison', {})
        for factor, metrics in div.items():
            if metrics.get('cosine_sim', 0) > 0.9:
                recommendations.append(
                    f"{factor}因子的输出余弦相似度过高({metrics['cosine_sim']:.4f})，"
                    f"建议增加embedding维度或调整网络结构"
                )
        
        # 饱和度分析建议
        sat = comparison.get('saturation_comparison', {})
        for factor, metrics in sat.items():
            if metrics.get('saturation_ratio', 0) > 0.3:
                recommendations.append(
                    f"{factor}因子的饱和比例过高({metrics['saturation_ratio']:.4f})，"
                    f"建议使用LayerNorm或调整激活函数"
                )
            if metrics.get('dead_ratio', 0) > 0.1:
                recommendations.append(
                    f"{factor}因子的死神经元比例过高({metrics['dead_ratio']:.4f})，"
                    f"建议使用LeakyReLU或调整学习率"
                )
        
        comparison['recommendations'] = recommendations
    
    def visualize_all_factors(self, results: Dict[str, Dict[str, Any]], 
                              save_dir: str = 'out_eval_results') -> None:
        """
        可视化所有因子的评估结果
        
        Args:
            results: 各因子的评估结果
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 为每个因子生成独立图表
        for factor_name, result in results.items():
            if 'error' not in result:
                self._visualize_single_factor(factor_name, result, save_dir)
        
        # 生成对比图表
        if len(results) > 1:
            self._visualize_factor_comparison(results, save_dir)
        
        print(f"\n可视化结果已保存到: {save_dir}")
    
    def _visualize_single_factor(self, factor_name: str, 
                                  result: Dict[str, Any], 
                                  save_dir: str) -> None:
        """可视化单个因子的结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{factor_name.upper()} Factor Embedding Analysis', fontsize=14)
        
        # 1. 局部敏感性
        if 'local_sensitivity' in result:
            ax = axes[0, 0]
            sens = result['local_sensitivity']
            features = [k for k in sens.keys() if k != 'overall']
            values = [sens[k]['mean'] for k in features]
            ax.bar(range(len(features)), values)
            ax.set_title('Local Sensitivity')
            ax.set_xlabel('Feature')
            ax.set_ylabel('Sensitivity')
            if len(features) <= 10:
                ax.set_xticks(range(len(features)))
                ax.set_xticklabels(features, rotation=45, ha='right')
            else:
                ax.set_xticks([])
        
        # 2. Jacobian范数分布
        if 'jacobian' in result and 'mean_jacobian_norm' in result['jacobian']:
            ax = axes[0, 1]
            ax.text(0.5, 0.5, f"Mean Jacobian Norm:\n{result['jacobian']['mean_jacobian_norm']:.4f}",
                   ha='center', va='center', fontsize=12)
            ax.set_title('Jacobian Analysis')
            ax.axis('off')
        
        # 3. 输出多样性
        if 'diversity' in result:
            ax = axes[0, 2]
            div = result['diversity']
            metrics = ['correlation', 'cosine_sim']
            values = [div.get('input_output_correlation', 0), 
                     div.get('output_cosine_similarity', 0)]
            ax.bar(metrics, values)
            ax.set_title('Output Diversity')
            ax.set_ylim(0, 1)
        
        # 4. 饱和度
        if 'saturation' in result:
            ax = axes[1, 0]
            sat = result['saturation']
            metrics = ['Saturation', 'Dead Neurons']
            values = [sat.get('saturation_ratio', 0), 
                     sat.get('dead_neuron_ratio', 0)]
            ax.bar(metrics, values)
            ax.set_title('Saturation Analysis')
            ax.set_ylim(0, 1)
        
        # 5. 隐藏层分布
        if 'saturation' in result:
            ax = axes[1, 1]
            sat = result['saturation']
            ax.text(0.5, 0.5, 
                   f"Hidden Mean: {sat.get('hidden_mean', 0):.4f}\n"
                   f"Hidden Std: {sat.get('hidden_std', 0):.4f}\n"
                   f"Range: [{sat.get('hidden_min', 0):.4f}, {sat.get('hidden_max', 0):.4f}]",
                   ha='center', va='center', fontsize=10)
            ax.set_title('Hidden Stats')
            ax.axis('off')
        
        # 6. 时间衰减（仅市场因子）
        if 'temporal_decay' in result:
            ax = axes[1, 2]
            td = result['temporal_decay']
            if 'temporal_impacts' in td:
                impacts = td['temporal_impacts']
                ax.plot(range(len(impacts)), impacts, marker='o')
                ax.set_title('Temporal Decay')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Impact')
                ax.invert_xaxis()  # 最近的时间在右边
        else:
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{factor_name}_embedding_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_factor_comparison(self, results: Dict[str, Dict[str, Any]], 
                                     save_dir: str) -> None:
        """可视化因子对比结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Multi-Factor Embedding Comparison', fontsize=14)
        
        factors = [f for f in results.keys() if 'error' not in results[f]]
        
        # 1. 敏感性对比
        ax = axes[0, 0]
        sensitivities = []
        for f in factors:
            if 'local_sensitivity' in results[f] and 'overall' in results[f]['local_sensitivity']:
                sensitivities.append(results[f]['local_sensitivity']['overall']['mean'])
            else:
                sensitivities.append(0)
        ax.bar(factors, sensitivities)
        ax.set_title('Sensitivity Comparison')
        ax.set_ylabel('Mean Sensitivity')
        
        # 2. 多样性对比
        ax = axes[0, 1]
        correlations = []
        for f in factors:
            if 'diversity' in results[f]:
                correlations.append(results[f]['diversity'].get('input_output_correlation', 0))
            else:
                correlations.append(0)
        ax.bar(factors, correlations)
        ax.set_title('Input-Output Correlation')
        ax.set_ylabel('Correlation')
        ax.set_ylim(0, 1)
        
        # 3. 饱和度对比
        ax = axes[1, 0]
        sat_ratios = []
        for f in factors:
            if 'saturation' in results[f]:
                sat_ratios.append(results[f]['saturation'].get('saturation_ratio', 0))
            else:
                sat_ratios.append(0)
        ax.bar(factors, sat_ratios)
        ax.set_title('Saturation Ratio')
        ax.set_ylabel('Ratio')
        ax.set_ylim(0, 1)
        
        # 4. 死神经元对比
        ax = axes[1, 1]
        dead_ratios = []
        for f in factors:
            if 'saturation' in results[f]:
                dead_ratios.append(results[f]['saturation'].get('dead_neuron_ratio', 0))
            else:
                dead_ratios.append(0)
        ax.bar(factors, dead_ratios)
        ax.set_title('Dead Neuron Ratio')
        ax.set_ylabel('Ratio')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'factor_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Dict[str, Any]], 
                     filepath: str) -> None:
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果
            filepath: 保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"评估结果已保存到: {filepath}")
    
    @staticmethod
    def load_results(filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        从文件加载评估结果
        
        Args:
            filepath: 文件路径
            
        Returns:
            评估结果
        """
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        print(f"评估结果已从 {filepath} 加载")
        return results
