# 多因子Embedding评估架构

> 本文档介绍 `embedding_evaluation` 模块的设计理念和使用方法。

---

## 背景

### 问题

原有的 `embedding_evaluator.py` 只能评估个股特征embedding层，存在以下问题：

1. 新增了 `market_token` 市场因子后，评估工具没有跟进
2. 未来还会引入更多量化因子（行业因子、宏观因子等），每个因子都有独立的embedding层
3. 缺乏统一的评估框架，新增因子需要大量修改代码

### 解决方案

采用**插件化架构**重构评估模块：

- 每个因子的embedding层作为独立的评估插件
- 通过装饰器自动注册，无需修改核心代码
- 统一的分析器协调多因子评估和对比

---

## 架构设计

### 文件结构

```
src/embedding_evaluation/
├── __init__.py              # 模块入口，自动导入评估器
├── base.py                  # FactorEmbeddingEvaluator 抽象基类
├── registry.py              # 注册中心 + @register_evaluator 装饰器
├── analyzers.py             # 通用分析函数
├── stock_evaluator.py       # 个股因子评估器
├── market_evaluator.py      # 市场因子评估器
└── multi_factor_analyzer.py # 统一分析器

src/embedding_evaluator.py   # 命令行入口（简化版）
```

### 核心类

#### FactorEmbeddingEvaluator（抽象基类）

定义所有因子评估器的统一接口：

```python
class FactorEmbeddingEvaluator(ABC):
    @property
    @abstractmethod
    def factor_name(self) -> str:
        """因子名称，如 'stock', 'market'"""
        pass
    
    @abstractmethod
    def get_embedding_layer(self, model) -> nn.Module:
        """从模型中提取对应的embedding层"""
        pass
    
    @abstractmethod
    def prepare_sample_data(self, stock_info_list, n_samples) -> np.ndarray:
        """准备该因子的样本数据"""
        pass
    
    @abstractmethod
    def evaluate(self, model, sample_data, device) -> Dict[str, Any]:
        """执行评估"""
        pass
```

#### FactorEvaluatorRegistry（注册中心）

管理所有可用的因子评估器：

```python
# 注册
FactorEvaluatorRegistry.register(MyFactorEvaluator)

# 获取
evaluator = FactorEvaluatorRegistry.create_evaluator('my_factor')

# 列出所有因子
FactorEvaluatorRegistry.list_factors()
```

#### @register_evaluator（装饰器）

简化注册流程：

```python
@register_evaluator
class MyFactorEvaluator(FactorEmbeddingEvaluator):
    ...
```

---

## 已实现的评估器

### StockFactorEvaluator（个股因子）

评估 `StockTransformer.embedding` 层，输入为6维股票特征。

**输入维度**: 6 (Open, High, Low, Close, Volume, Exchange)

**评估内容**:
- Jacobian分析
- 局部/全局敏感性分析
- 输出多样性分析
- 饱和度分析

### MarketFactorEvaluator（市场因子）

评估 `StockTransformer.market_encoder` 层，输入为大盘涨跌序列。

**输入维度**: `DataConfig.MARKET_CONTEXT_LENGTH`

**评估内容**:
- 上述所有分析
- **时间衰减分析**（特有）：检测历史数据对当前embedding的影响是否随时间衰减

---

## 使用方法

### 命令行

```bash
# 列出所有可用因子
python embedding_evaluator.py --list-factors

# 评估所有因子
python embedding_evaluator.py --model ./out/model.pth

# 评估指定因子
python embedding_evaluator.py --model ./out/model.pth --factors stock,market

# 仅评估市场因子
python embedding_evaluator.py --model ./out/model.pth --factors market

# 保存评估结果
python embedding_evaluator.py --model ./out/model.pth --save-results
```

### Python API

```python
from embedding_evaluation import (
    MultiFactorEmbeddingAnalyzer,
    FactorEvaluatorRegistry
)

# 查看可用因子
print(FactorEvaluatorRegistry.list_factors())

# 创建分析器
analyzer = MultiFactorEmbeddingAnalyzer(
    model_path='./out/model.pth',
    factors=['stock', 'market']
)

# 加载模型
analyzer.load_model(feature_normalizer=feature_normalizer)

# 执行评估
results = analyzer.analyze_all_factors(stock_info_list)

# 对比分析
comparison = analyzer.compare_factors(results)

# 可视化
analyzer.visualize_all_factors(results, save_dir='./out_eval_results')
```

---

## 扩展指南

### 添加新因子

创建新的评估器类，继承 `FactorEmbeddingEvaluator` 并添加装饰器：

```python
# src/embedding_evaluation/sector_evaluator.py

from .base import FactorEmbeddingEvaluator
from .registry import register_evaluator

@register_evaluator
class SectorFactorEvaluator(FactorEmbeddingEvaluator):
    """行业因子Embedding评估器"""
    
    @property
    def factor_name(self) -> str:
        return "sector"
    
    @property
    def input_dim(self) -> int:
        return 10  # 假设有10个行业
    
    @property
    def output_dim(self) -> int:
        return ModelConfig.D_MODEL
    
    def get_embedding_layer(self, model) -> nn.Module:
        return model.sector_encoder
    
    def prepare_sample_data(self, stock_info_list, n_samples=500) -> np.ndarray:
        # 准备行业数据...
        pass
    
    def evaluate(self, model, sample_data, device) -> Dict[str, Any]:
        # 执行评估...
        pass
```

然后在 `__init__.py` 中导入：

```python
from . import sector_evaluator
```

### 添加新的分析指标

在 `analyzers.py` 中添加新的分析函数：

```python
def analyze_new_metric(embedding_layer, sample_inputs, device):
    """新的分析指标"""
    # 实现分析逻辑...
    return results
```

然后在各评估器的 `evaluate` 方法中调用。

---

## 评估指标说明

详见 [embedding-evaluation-guide.md](./embedding-evaluation-guide.md)

### 通用指标

| 指标 | 说明 | 健康范围 |
|------|------|----------|
| Jacobian范数 | 输入对输出的总体响应强度 | 5000-15000 |
| 局部敏感性 | 微小扰动导致的输出变化 | 适中 |
| 输出多样性 | 不同输入产生不同输出的能力 | 余弦相似度 0.2-0.4 |
| 饱和度 | 输出落在激活函数饱和区的比例 | < 1% |
| 死神经元比例 | 输出始终接近0的神经元比例 | < 5% |

### 市场因子特有指标

| 指标 | 说明 |
|------|------|
| 时间衰减 | 历史数据对当前embedding的影响是否随时间衰减 |
| decay_ratio | 最近5天影响 / 最远5天影响，>1.2 表示有衰减 |

---

## 输出结果

### 文件结构

```
out_eval_results/
├── stock_embedding_analysis.png    # 个股因子评估图表
├── market_embedding_analysis.png   # 市场因子评估图表
├── factor_comparison.png           # 因子对比图表
└── evaluation_results.pkl          # 评估结果（可选保存）
```

### 对比分析

当评估多个因子时，会自动生成对比分析和优化建议：

```
因子对比分析
============================================================

优化建议:
1. market因子的敏感性过高(0.0234)，建议检查是否存在梯度爆炸风险
2. stock因子的输出余弦相似度过高(0.92)，建议增加embedding维度或调整网络结构
```

---

## 注意事项

1. **大盘数据依赖**: 评估 `market` 因子前，需要先加载大盘数据
2. **模型兼容性**: 确保模型包含对应的embedding层
3. **样本数量**: 默认每个因子采样500个样本，可通过 `--n-samples` 调整

---

> 文档版本：v2.0
> 最后更新：2026-03-24
> 适用项目：EquiNet股票预测模型
