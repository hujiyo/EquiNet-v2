"""
Embedding层评估工具 - 主入口

基于多因子架构的统一embedding评估工具。
支持个股因子(stock)和市场因子(market)的评估，并可扩展支持更多因子。

使用示例:
    # 评估所有因子（自动使用最新的模型）
    python embedding_evaluator.py
    
    # 评估指定模型
    python embedding_evaluator.py --model model.pth
    
    # 评估指定因子
    python embedding_evaluator.py --model model.pth --factors stock,market
    
    # 仅评估市场因子
    python embedding_evaluator.py --model model.pth --factors market
    
    # 列出新架构支持的所有因子
    python embedding_evaluator.py --list-factors
"""

import os
import sys

# 确保能正确导入其他模块（无论从哪里运行）
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import pickle
import numpy as np
import torch

# 导入新的多因子评估架构
from src.embedding_evaluation import (
    MultiFactorEmbeddingAnalyzer,
    FactorEvaluatorRegistry
)

from src.data import load_and_preprocess_data, GlobalDataManager, FeatureNormalizer
from src.config import DataConfig


def find_latest_model():
    """
    查找最新的模型文件
    
    搜索 DataConfig.OUTPUT_DIR 目录下的 .pth 文件，
    按修改时间排序，返回最新的模型路径。
    
    Returns:
        str: 最新模型文件的路径，如果没有找到则返回 None
    """
    out_dir = DataConfig.OUTPUT_DIR
    model_files = []
    
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith('.pth'):
                model_files.append(os.path.join(out_dir, f))
    
    if model_files:
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return model_files[0]
    
    return None


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Embedding层评估工具 - 支持多因子架构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估所有因子（使用默认模型）
  python embedding_evaluator.py
  
  # 评估指定模型
  python embedding_evaluator.py --model model.pth
  
  # 评估指定因子
  python embedding_evaluator.py --model model.pth --factors stock,market
  
  # 仅评估市场因子
  python embedding_evaluator.py --model model.pth --factors market
  
  # 列出新架构支持的所有因子
  python embedding_evaluator.py --list-factors
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='模型文件路径，不指定则自动查找最新的模型')
    parser.add_argument('--factors', type=str, default='all',
                       help='要评估的因子，逗号分隔，"all"表示所有因子 (默认: all)')
    parser.add_argument('--list-factors', action='store_true',
                       help='列出所有可用的因子')
    parser.add_argument('--output-dir', type=str, default='./src/out_eval_results',
                       help=f'输出目录 (默认: ./src/out_eval_results)')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='每个因子的样本数量 (默认: 500)')
    parser.add_argument('--feature-normalizer', type=str, default=None,
                       help='特征归一化器文件路径')
    parser.add_argument('--save-results', action='store_true',
                       help='保存评估结果到文件')
    parser.add_argument('--no-visualization', action='store_true',
                       help='不生成可视化图表')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 处理模型路径
    if args.model is None:
        # 未指定模型，自动查找最新的模型
        model_path = find_latest_model()
        if model_path:
            print(f"找到模型文件，使用最新的: {os.path.basename(model_path)}")
        else:
            print("未找到模型文件，将使用随机初始化模型")
            model_path = None
    else:
        # 自动拼接模型路径（如果传入的是文件名而非完整路径）
        if not os.path.dirname(args.model):
            model_path = os.path.join(DataConfig.OUTPUT_DIR, args.model)
        elif not os.path.isabs(args.model) and not args.model.startswith('./src/'):
            # 如果传入的是相对路径但不是以 ./src/ 开头，也拼接
            model_path = os.path.join(DataConfig.OUTPUT_DIR, os.path.basename(args.model))
        else:
            model_path = args.model
    
    # 列出所有可用因子
    if args.list_factors:
        print("\n可用的因子评估器:")
        print("=" * 60)
        factors = FactorEvaluatorRegistry.list_factors()
        if factors:
            for i, factor in enumerate(factors, 1):
                evaluator_class = FactorEvaluatorRegistry.get(factor)
                temp_instance = evaluator_class()
                print(f"{i}. {factor}")
                print(f"   输入维度: {temp_instance.input_dim}")
                print(f"   输出维度: {temp_instance.output_dim}")
        else:
            print("暂无注册的因子评估器")
        print("=" * 60)
        return
    
    # 解析因子列表
    if args.factors.lower() == 'all':
        factors = None  # None表示所有因子
    else:
        factors = [f.strip() for f in args.factors.split(',')]
    
    print("\n" + "=" * 60)
    print("Embedding层评估工具 - 多因子架构")
    print("=" * 60)
    
    # 加载特征归一化器
    feature_normalizer = None
    normalizer_path = DataConfig.NORMALIZER_PATH
    
    if args.feature_normalizer and os.path.exists(args.feature_normalizer):
        print(f"\n正在加载特征归一化器: {args.feature_normalizer}")
        feature_normalizer = FeatureNormalizer.load(args.feature_normalizer)
        print("特征归一化器加载完成")
    elif os.path.exists(normalizer_path):
        print(f"\n正在加载特征归一化器: {normalizer_path}")
        feature_normalizer = FeatureNormalizer.load(normalizer_path)
        print("特征归一化器加载完成")
    else:
        print(f"\n⚠ 未找到归一化器文件: {normalizer_path}")
        print("  评估将只使用粗处理，不应用细归一化")
    
    # 加载股票数据
    print("\n正在加载股票数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    
    # 加载大盘数据（如果评估市场因子）
    if factors is None or 'market' in factors:
        print("\n正在加载大盘数据...")
        gdm = GlobalDataManager.get_instance()
        try:
            gdm.load_market_data()
            print("✓ 大盘数据加载成功")
        except FileNotFoundError as e:
            print(f"⚠ 大盘数据加载失败: {e}")
            if factors and 'market' in factors:
                print("错误: 需要大盘数据但无法加载")
                return
            # 如果评估所有因子，则跳过市场因子
            if factors is None:
                print("将跳过市场因子评估")
    
    # 创建多因子分析器
    print(f"\n正在初始化多因子分析器...")
    try:
        analyzer = MultiFactorEmbeddingAnalyzer(
            model_path=model_path,
            factors=factors
        )
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    # 加载模型
    analyzer.load_model(feature_normalizer=feature_normalizer)
    
    # 执行评估
    print("\n" + "=" * 60)
    print("开始执行因子评估")
    print("=" * 60)
    
    results = analyzer.analyze_all_factors(test_stock_info, n_samples=args.n_samples)
    
    # 对比分析
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("因子对比分析")
        print("=" * 60)
        comparison = analyzer.compare_factors(results)
        
        if comparison.get('recommendations'):
            print("\n优化建议:")
            for i, rec in enumerate(comparison['recommendations'], 1):
                print(f"{i}. {rec}")
    
    # 可视化
    if not args.no_visualization:
        print("\n正在生成可视化图表...")
        analyzer.visualize_all_factors(results, save_dir=args.output_dir)
    
    # 保存结果
    if args.save_results:
        results_path = os.path.join(args.output_dir, 'evaluation_results.pkl')
        analyzer.save_results(results, results_path)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)
    
    if not args.no_visualization:
        print(f"可视化结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
