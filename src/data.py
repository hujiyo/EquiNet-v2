"""
EquiNet 数据处理模块

包含所有数据相关的功能：
- 数据加载和预处理
- 样本生成
- 时间顺序采样器
- 评估数据集创建
- 预测函数
- 特征归一化模块
- 大盘数据加载模块
"""

import os
import sys
import random
import argparse
import pickle
import numpy as np
import pandas as pd
from config import DataConfig, generate_label, calculate_returns
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from typing import Dict, List, Tuple, Optional


class FeatureNormalizer:
    """
    特征归一化器 - 两阶段归一化

    阶段1: QuantileTransformer → 处理偏态和集中度问题
    阶段2: StandardScaler → 确保均值0方差1

    使用方法：
        # 训练阶段
        normalizer = FeatureNormalizer()
        normalizer.fit(train_stock_info)
        normalizer.save('normalizer.pkl')

        # 推理阶段
        normalizer = FeatureNormalizer.load('normalizer.pkl')
        normalized_data = normalizer.transform(raw_data)
    """

    def __init__(self,
                 output_distribution='normal',
                 n_quantiles=1000,
                 random_state=42):
        """
        Args:
            output_distribution: 'normal' 或 'uniform'
                - 'normal': 输出符合标准正态分布（推荐）
                - 'uniform': 输出符合 [0, 1] 均匀分布
            n_quantiles: 分位数数量，越多越精确但越慢
            random_state: 随机种子
        """
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        # 为每个特征组创建独立的 pipeline
        self.ohl_pipeline = self._create_pipeline()
        self.volume_pipeline = self._create_pipeline()
        self.exchange_pipeline = self._create_pipeline()
        self.market_pipeline = self._create_pipeline()

        self.is_fitted = False

    def _create_pipeline(self):
        """
        创建两阶段归一化 pipeline

        为什么需要 StandardScaler？
        - QuantileTransformer 的输出虽然是正态分布，但均值和方差可能不是 0 和 1
        - StandardScaler 确保最终输出严格满足：均值=0，标准差=1
        """
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ('quantile', QuantileTransformer(
                output_distribution=self.output_distribution,
                n_quantiles=self.n_quantiles,
                random_state=self.random_state,
                subsample=100000
            )),
            ('scaler', StandardScaler())
        ])

    def _collect_training_features(self, train_stock_info: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从训练集收集所有特征值（避免数据泄漏）

        关键：只使用每只股票的训练集部分（train_end_idx 之前）
        
        使用 data.py 中的 coarse_normalize_context_window() 进行粗处理，
        确保与训练时的数据处理逻辑完全一致。

        Returns:
            ohl_data: OHLC 特征 [N_samples * 30 * 4]
            volume_data: Volume 特征 [N_samples * 30]
            exchange_data: Exchange 特征 [N_samples * 30]
        """
        from data import coarse_normalize_context_window, DataConfig
        
        ohl_data = []
        volume_data = []
        exchange_data = []
        
        context_length = DataConfig.CONTEXT_LENGTH

        for stock in train_stock_info:
            data = stock['data']
            train_end_idx = stock.get('train_end_idx', len(data))

            for i in range(1, train_end_idx - context_length):
                # 使用统一的粗处理函数
                input_seq = coarse_normalize_context_window(
                    data, i, context_length,
                    check_limit_up=False,  # 拟合归一化器时不过滤涨停，使用更多数据
                    required_length=context_length
                )
                
                if input_seq is None:
                    continue
                
                ohl_data.append(input_seq[:, :4].flatten())
                volume_data.append(input_seq[:, 4].flatten())
                exchange_data.append(input_seq[:, 5].flatten())

        ohl_data = np.concatenate(ohl_data) if ohl_data else np.array([])
        volume_data = np.concatenate(volume_data) if volume_data else np.array([])
        exchange_data = np.concatenate(exchange_data) if exchange_data else np.array([])

        print(f"[FeatureNormalizer] 收集到的训练数据:")
        print(f"  OHLC: {len(ohl_data)} 个值")
        print(f"  Volume: {len(volume_data)} 个值")
        print(f"  Exchange: {len(exchange_data)} 个值")

        return ohl_data, volume_data, exchange_data

    def _collect_market_features(self, market_data: Dict) -> np.ndarray:
        """
        从大盘数据收集特征值用于拟合归一化器

        Args:
            market_data: load_market_data() 返回的数据字典

        Returns:
            market_data_flat: 大盘涨跌幅数据 [N]
        """
        changes = market_data['changes']
        
        print(f"[FeatureNormalizer] 收集到的大盘数据:")
        print(f"  Market: {len(changes)} 个值")
        print(f"  范围: [{changes.min():.4f}, {changes.max():.4f}]")

        return changes

    def fit(self, train_stock_info: List[Dict]):
        """
        在训练集上拟合归一化器

        Args:
            train_stock_info: 训练集股票信息列表
        """
        print("\n[FeatureNormalizer] 开始拟合归一化器...")
        print(f"  输出分布: {self.output_distribution}")
        print(f"  分位数数量: {self.n_quantiles}")

        ohl_data, volume_data, exchange_data = self._collect_training_features(train_stock_info)

        print("\n[FeatureNormalizer] 拟合 OHLC 特征...")
        self.ohl_pipeline.fit(ohl_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Volume 特征...")
        self.volume_pipeline.fit(volume_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Exchange 特征...")
        self.exchange_pipeline.fit(exchange_data.reshape(-1, 1))

        gdm = GlobalDataManager.get_instance()
        if gdm.is_market_data_loaded():
            print("[FeatureNormalizer] 拟合 Market 特征...")
            market_features = self._collect_market_features(gdm.get_market_data())
            self.market_pipeline.fit(market_features.reshape(-1, 1))

        self.is_fitted = True

        self._print_transform_stats(ohl_data, volume_data, exchange_data)

        print("\n[FeatureNormalizer] ✓ 拟合完成！")

    def _print_transform_stats(self, ohl_data, volume_data, exchange_data):
        """
        打印变换后的统计信息，验证归一化效果
        """
        print("\n[FeatureNormalizer] 变换后的统计信息:")

        ohl_transformed = self.ohl_pipeline.transform(ohl_data.reshape(-1, 1)).flatten()
        print(f"  OHLC:")
        print(f"    均值: {ohl_transformed.mean():.6f}")
        print(f"    标准差: {ohl_transformed.std():.6f}")
        print(f"    范围: [{ohl_transformed.min():.6f}, {ohl_transformed.max():.6f}]")

        volume_transformed = self.volume_pipeline.transform(volume_data.reshape(-1, 1)).flatten()
        print(f"  Volume:")
        print(f"    均值: {volume_transformed.mean():.6f}")
        print(f"    标准差: {volume_transformed.std():.6f}")
        print(f"    范围: [{volume_transformed.min():.6f}, {volume_transformed.max():.6f}]")

        exchange_transformed = self.exchange_pipeline.transform(exchange_data.reshape(-1, 1)).flatten()
        print(f"  Exchange:")
        print(f"    均值: {exchange_transformed.mean():.6f}")
        print(f"    标准差: {exchange_transformed.std():.6f}")
        print(f"    范围: [{exchange_transformed.min():.6f}, {exchange_transformed.max():.6f}]")

        gdm = GlobalDataManager.get_instance()
        if gdm.is_market_data_loaded():
            market_features = gdm.get_market_data()['changes']
            market_transformed = self.market_pipeline.transform(market_features.reshape(-1, 1)).flatten()
            print(f"  Market:")
            print(f"    均值: {market_transformed.mean():.6f}")
            print(f"    标准差: {market_transformed.std():.6f}")
            print(f"    范围: [{market_transformed.min():.6f}, {market_transformed.max():.6f}]")

    def transform(self, input_seq: np.ndarray) -> np.ndarray:
        """
        对单个样本应用归一化

        ⚠️ 重要：此函数可以在训练集、验证集、测试集上调用
        因为它只使用 fit() 时学到的参数，不会产生数据泄漏

        Args:
            input_seq: [context_length, 6] 原始输入序列

        Returns:
            normalized_seq: [context_length, 6] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        normalized = np.empty_like(input_seq, dtype=np.float32)

        # 展平以便转换
        ohl_flat = input_seq[:, :4].flatten()  # [context_length * 4]
        volume_flat = input_seq[:, 4].flatten()  # [context_length]
        exchange_flat = input_seq[:, 5].flatten()  # [context_length]

        # 转换每个特征组
        normalized_ohl = self.ohl_pipeline.transform(
            ohl_flat.reshape(-1, 1)
        ).flatten()
        normalized_volume = self.volume_pipeline.transform(
            volume_flat.reshape(-1, 1)
        ).flatten()
        normalized_exchange = self.exchange_pipeline.transform(
            exchange_flat.reshape(-1, 1)
        ).flatten()

        # 重塑回原始形状
        normalized[:, :4] = normalized_ohl.reshape(input_seq[:, :4].shape)
        normalized[:, 4] = normalized_volume
        normalized[:, 5] = normalized_exchange

        return normalized

    def transform_market(self, market_seq: np.ndarray) -> np.ndarray:
        """
        对市场数据序列应用归一化

        Args:
            market_seq: [market_context_length] 大盘涨跌幅序列

        Returns:
            normalized_market: [market_context_length] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        normalized = self.market_pipeline.transform(
            market_seq.reshape(-1, 1)
        ).flatten().astype(np.float32)

        return normalized

    def fit_transform(self, train_stock_info: List[Dict]) -> 'FeatureNormalizer':
        """
        拟合并返回归一化器（链式调用）

        Args:
            train_stock_info: 训练集股票信息列表

        Returns:
            self: 拟合后的归一化器
        """
        self.fit(train_stock_info)
        return self

    def save(self, path: str):
        """
        保存归一化器到文件

        Args:
            path: 保存路径（例如: './normalizer.pkl'）
        """
        if not self.is_fitted:
            raise RuntimeError("无法保存未拟合的归一化器")

        with open(path, 'wb') as f:
            pickle.dump({
                'ohl_pipeline': self.ohl_pipeline,
                'volume_pipeline': self.volume_pipeline,
                'exchange_pipeline': self.exchange_pipeline,
                'market_pipeline': self.market_pipeline,
                'is_fitted': self.is_fitted,
                'output_distribution': self.output_distribution,
                'n_quantiles': self.n_quantiles,
                'random_state': self.random_state
            }, f)

        print(f"[FeatureNormalizer] ✓ 归一化器已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """
        从文件加载归一化器

        Args:
            path: 归一化器文件路径

        Returns:
            加载的归一化器实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"归一化器文件不存在: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 创建新实例
        normalizer = cls(
            output_distribution=data['output_distribution'],
            n_quantiles=data['n_quantiles'],
            random_state=data['random_state']
        )

        # 恢复状态
        normalizer.ohl_pipeline = data['ohl_pipeline']
        normalizer.volume_pipeline = data['volume_pipeline']
        normalizer.exchange_pipeline = data['exchange_pipeline']
        normalizer.market_pipeline = data.get('market_pipeline', normalizer._create_pipeline())
        normalizer.is_fitted = data['is_fitted']

        print(f" ✓ 归一化器已从 {path} 加载")

        return normalizer


class GlobalDataManager:
    """
    全局数据管理器（单例模式）
    
    统一管理大盘数据等全局性数据，避免在函数间频繁传递参数。
    
    使用方法：
        # 初始化（通常在训练/评估开始前调用一次）
        gdm = GlobalDataManager.get_instance()
        gdm.load_market_data()
        
        # 在任何需要的地方获取市场数据
        market_seq = gdm.get_market_context(target_date)
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'GlobalDataManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._market_data = None
        self._date_to_idx = {}
        GlobalDataManager._initialized = True
    
    def load_market_data(self, market_data_path: str = None) -> 'GlobalDataManager':
        """
        加载大盘数据并构建索引
        
        Args:
            market_data_path: 大盘数据文件路径，默认使用 DataConfig.MARKET_DATA_PATH
            
        Returns:
            self: 支持链式调用
        """
        if self._market_data is not None:
            return self
        
        if market_data_path is None:
            market_data_path = DataConfig.MARKET_DATA_PATH
        
        if not os.path.exists(market_data_path):
            raise FileNotFoundError(f"大盘数据文件不存在: {market_data_path}")
        
        df = pd.read_csv(market_data_path)
        df = df.iloc[::-1].reset_index(drop=True)
        
        if 'time' in df.columns:
            dates = df['time'].values
        elif 'date' in df.columns:
            dates = df['date'].values
        else:
            raise ValueError("大盘数据文件缺少日期列 ('time' 或 'date')")
        
        if 'end' in df.columns:
            closes = df['end'].values.astype(np.float32)
        elif 'close' in df.columns:
            closes = df['close'].values.astype(np.float32)
        else:
            raise ValueError("大盘数据文件缺少收盘价列 ('end' 或 'close')")
        
        changes = np.zeros(len(closes), dtype=np.float32)
        changes[1:] = (closes[1:] - closes[:-1]) / closes[:-1]
        changes[0] = 0.0
        np.clip(changes, -0.1, 0.1, out=changes)
        
        self._market_data = {
            'dates': dates,
            'changes': changes,
            'closes': closes,
        }
        
        self._date_to_idx = {int(d): i for i, d in enumerate(dates)}
        
        print(f"[GlobalDataManager] 大盘数据已加载: {len(dates)} 天")
        print(f"  日期范围: {dates[0]} - {dates[-1]}")
        print(f"  涨跌幅范围: [{changes.min():.4f}, {changes.max():.4f}]")
        
        return self
    
    def is_market_data_loaded(self) -> bool:
        """检查大盘数据是否已加载"""
        return self._market_data is not None
    
    def get_market_context(self, target_date: int) -> Optional[np.ndarray]:
        """
        获取指定日期前N天的大盘涨跌序列
        
        Args:
            target_date: 目标日期 (YYYYMMDD 格式的整数)
            
        Returns:
            np.ndarray: [MARKET_CONTEXT_LENGTH] 涨跌幅序列，如果数据不足返回 None
        """
        if not self.is_market_data_loaded():
            return None
        
        context_length = DataConfig.MARKET_CONTEXT_LENGTH
        
        if target_date not in self._date_to_idx:
            return None
        
        target_idx = self._date_to_idx[target_date]
        start_idx = target_idx - context_length
        
        if start_idx < 0:
            return None
        
        return self._market_data['changes'][start_idx:target_idx].copy()
    
    def get_market_context_batch(
        self, 
        target_dates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量获取大盘涨跌序列
        
        Args:
            target_dates: 目标日期数组 [batch_size]
            
        Returns:
            market_seqs: [batch_size, MARKET_CONTEXT_LENGTH] 涨跌幅序列
            valid_mask: [batch_size] bool数组，标记哪些样本有效
        """
        if not self.is_market_data_loaded():
            batch_size = len(target_dates)
            return np.zeros((batch_size, DataConfig.MARKET_CONTEXT_LENGTH), dtype=np.float32), np.zeros(batch_size, dtype=bool)
        
        context_length = DataConfig.MARKET_CONTEXT_LENGTH
        
        batch_size = len(target_dates)
        market_seqs = np.zeros((batch_size, context_length), dtype=np.float32)
        valid_mask = np.ones(batch_size, dtype=bool)
        
        for i, target_date in enumerate(target_dates):
            target_date_int = int(target_date)
            
            if target_date_int not in self._date_to_idx:
                valid_mask[i] = False
                continue
            
            target_idx = self._date_to_idx[target_date_int]
            start_idx = target_idx - context_length
            
            if start_idx < 0:
                valid_mask[i] = False
                continue
            
            market_seqs[i] = self._market_data['changes'][start_idx:target_idx]
        
        return market_seqs, valid_mask
    
    def get_market_data(self) -> Optional[Dict]:
        """
        获取原始市场数据字典（向后兼容）
        
        Returns:
            dict: {'dates': ..., 'changes': ..., 'closes': ...}
        """
        return self._market_data


def process_single_file(args):
    file_path, file_name, test_days, train_start_year = args
    """
    处理单个股票CSV文件，返回包含训练和测试数据的字典
    
    数据处理流程：
    1. 读取CSV并反转时间顺序
    2. 提取OHLCV数据：['start', 'max', 'min', 'end', 'volume', 'exchange']
    3. 验证数据长度是否满足最低要求
    
    数据分割策略（确保训练集和测试集严格分离）：
    - 测试集：最后 test_days 天的数据，完全冻结用于评估
    - 训练集：从 train_start_year 开始到 train_end_idx 结束
    - 缓冲区：训练集结束后有 REQUIRED_LENGTH 天的缓冲区，防止数据泄露
    
    关键索引计算：
    - train_end_idx: 训练集最后一个可用位置 = data_length - test_days - required_length
    - test_split_point: 测试集起始位置 = data_length - test_days
    - train_start_idx: 根据 train_start_year 找到的实际起始位置
    
    返回数据包含完整的训练和测试数据副本，使用时需根据索引切片访问。
    
    Args:
        file_path: CSV文件路径
        file_name: 文件名
        test_days: 测试集天数
        train_start_year: 训练开始年份
    
    Returns:
        dict or None: 包含股票信息的字典，数据不足时返回None
    """
    try:
        df = pd.read_csv(file_path)
        df = df.iloc[::-1].reset_index(drop=True)
                
        data = df[['start', 'max', 'min', 'end', 'volume', 'exchange']].values
        times = df['time'].values
        
        data_length = len(data)
        required_length = DataConfig.REQUIRED_LENGTH
        
        min_required_length = required_length + test_days
        if data_length < min_required_length:
            return None
        
        train_end_idx = data_length - test_days - required_length
        
        test_split_point = data_length - test_days
        
        train_start_date = train_start_year * 10000 + 101
        valid_indices = np.where(times >= train_start_date)[0]
        
        if len(valid_indices) > 0:
            train_start_idx = valid_indices[0]
        else:
            train_start_idx = 0
        
        if train_start_idx >= train_end_idx:
            return None
        
        train_length = train_end_idx - train_start_idx
        if train_length < required_length:
            return None
        
        train_data = data.copy()
        test_data = data.copy()
        
        stock_info = {
            'file_name': file_name,              # 股票文件名，用于识别不同股票
            'data_length': data_length,          # 总数据长度（天数），用于验证数据充足性
            'train_data': train_data,            # 完整数据副本，训练时根据[train_start_idx:train_end_idx]切片访问
            'test_data': test_data,              # 完整数据副本，测试时根据[test_split_point:]切片访问
            'train_start_idx': train_start_idx,  # 训练集起始索引，根据train_start_year计算得出
            'train_end_idx': train_end_idx,      # 训练集结束索引，确保与测试集有缓冲区
            'train_length': train_length,        # 可用训练数据长度，用于验证训练集是否充足
            'test_split_point': test_split_point, # 测试集起始索引，固定为最后test_days天的开始位置
            'times': times.copy(),
        }
        
        return stock_info
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return None


def load_and_preprocess_data(data_dir=DataConfig.DATA_DIR, test_days=DataConfig.TEST_DAYS, train_start_year=DataConfig.TRAIN_START_YEAR):
    """
    数据加载和预处理，使用多进程并行加载
    
    采样边界设计：
    - 训练集：从TRAIN_START_YEAR年（或上市日）到 总长度-test_days-REQUIRED_LENGTH
    - 测试集：最近test_days天
    - 最低数据要求：test_days + REQUIRED_LENGTH
    """
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_files.sort()
    
    print(f"总共 {len(all_files)} 只股票文件")
    print(f"划分策略:")
    print(f"  - 训练集: {train_start_year}年起（或上市日）到 总长度-{test_days}-{DataConfig.REQUIRED_LENGTH}")
    print(f"  - 测试集: 最近 {test_days} 天")
    print(f"  - 最低数据要求: {test_days + DataConfig.REQUIRED_LENGTH} 天（测试集{test_days}天 + 训练样本{DataConfig.REQUIRED_LENGTH}天）")
    
    file_args = [(os.path.join(data_dir, f), f, test_days, train_start_year) for f in all_files]
    num_workers = min(cpu_count(), 8)
    
    with Pool(num_workers) as pool:
        all_stock_info = [r for r in pool.map(process_single_file, file_args) if r is not None]
    
    discarded_count = len(all_files) - len(all_stock_info)
    print(f"有效股票: {len(all_stock_info)} 只，丢弃: {discarded_count} 只")
    
    train_stock_info = []
    test_stock_info = []
    
    for stock_info in all_stock_info:
        train_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['train_data'],
            'data_length': stock_info['data_length'],
            'train_start_idx': stock_info['train_start_idx'],
            'train_end_idx': stock_info['train_end_idx'],
            'times': stock_info['times'],
        })
        
        test_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['test_data'],
            'data_length': stock_info['data_length'],
            'test_split_point': stock_info['test_split_point'],
            'times': stock_info['times'],
        })
    
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    
    return train_stock_info, test_stock_info


class TemporalSampler:
    """
    时间顺序采样器：采样头在多个股票上同步向前移动，不回头
    
    采样边界设计：
    - 每只股票的指针初始位置 = train_start_idx（TRAIN_START_YEAR年起始位置+1，或上市第一天+1）
    - 每只股票的指针末位置 = train_end_idx（总长度 - TEST_DAYS - REQUIRED_LENGTH）
    
    关键设计：start_pos = max(1, train_start_idx + 1)
    原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
    因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
    
    核心算法：
    1. 计算总样本数和每个epoch需要的样本数
    2. 将总样本数均匀分配到各个epoch
    3. 每轮从所有股票当前位置各取一个样本，然后指针前进
    """
    def __init__(self, stock_info_list):
        self.stock_info_list = stock_info_list
        self.required_length = DataConfig.REQUIRED_LENGTH

        self.stock_positions = []
        self.stock_start_positions = []
        self.stock_max_positions = []
        self.can_loop = []
        self.loop_counts = [0] * len(stock_info_list)
        
        for stock_info in stock_info_list:
            train_start_idx = stock_info.get('train_start_idx', 0)
            train_end_idx = stock_info.get('train_end_idx', len(stock_info['data']))
            data_length = stock_info.get('data_length', 0)
            
            # 关键设计：start_pos = max(1, train_start_idx + 1)
            # 原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
            # 因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
            start_pos = max(1, train_start_idx + 1)
            max_pos = train_end_idx
            
            if start_pos > max_pos:
                start_pos = max_pos + 1
            
            self.stock_positions.append(start_pos)
            self.stock_start_positions.append(start_pos)
            self.stock_max_positions.append(max_pos)
            self.can_loop.append(data_length > 600)

        valid_stocks = sum(1 for i in range(len(stock_info_list)) 
                         if self.stock_positions[i] <= self.stock_max_positions[i])
        total_samples = sum(max(0, self.stock_max_positions[i] - self.stock_positions[i] + 1) 
                          for i in range(len(stock_info_list)))
        
        if valid_stocks == 0:
            raise ValueError(
                f"没有有效的训练股票！\n"
                f"  总股票数: {len(stock_info_list)}\n"
                f"  请检查数据质量或调整参数"
            )
        
        print(f"  初始化采样器: {valid_stocks}只有效股票, 总样本数={total_samples}")
        print(f"  采样策略: 时间顺序前进，支持循环采样（数据长度>600的股票）")

    def sample_batch_rounds(self, num_rounds):
        """
        批量采样多轮：一次性生成多轮的样本索引，提高效率

        参数:
            num_rounds: 要采样的轮数

        返回: [(stock_idx, start_idx), ...] 所有轮次的样本索引列表
        """
        all_samples = []

        for _ in range(num_rounds):
            for stock_idx in range(len(self.stock_info_list)):
                current_pos = self.stock_positions[stock_idx]
                max_pos = self.stock_max_positions[stock_idx]

                if current_pos > max_pos and self.can_loop[stock_idx]:
                    current_pos = self.stock_start_positions[stock_idx]
                    self.stock_positions[stock_idx] = current_pos
                    self.loop_counts[stock_idx] += 1

                if current_pos <= max_pos:
                    all_samples.append((stock_idx, current_pos))
                    self.stock_positions[stock_idx] += 1

            if not any(self.stock_positions[i] <= self.stock_max_positions[i] 
                      for i in range(len(self.stock_info_list))):
                break

        return all_samples
    
    def get_progress(self):
        """获取当前采样进度"""
        total_samples = 0
        current_samples = 0
        for start_pos, pos, max_pos in zip(self.stock_start_positions, self.stock_positions, self.stock_max_positions):
            if start_pos <= max_pos:
                total_samples += max_pos - start_pos + 1
                current_samples += max(0, min(pos, max_pos + 1) - start_pos)
        return current_samples, total_samples

    def get_loop_stats(self):
        """获取循环统计信息"""
        looped_stocks_count = sum(1 for c in self.loop_counts if c > 0)
        total_loops = sum(self.loop_counts)
        return looped_stocks_count, total_loops


class RandomSampler:
    """
    随机采样器：每次随机选择股票和位置进行采样
    
    与TemporalSampler的区别：
    - TemporalSampler: 时间顺序前进，指针不回头（除非循环）
    - RandomSampler: 每次完全随机选择样本，无时间顺序
    
    适用场景：
    - 对比实验：评估时间顺序采样对模型效果的影响
    - 数据增强：打破时间依赖，增加样本多样性
    """
    def __init__(self, stock_info_list):
        self.stock_info_list = stock_info_list
        self.required_length = DataConfig.REQUIRED_LENGTH
        
        self.valid_stock_indices = []
        self.stock_sample_ranges = []
        
        for stock_idx, stock_info in enumerate(stock_info_list):
            train_start_idx = stock_info.get('train_start_idx', 0)
            train_end_idx = stock_info.get('train_end_idx', len(stock_info['data']))
            
            # 关键设计：start_pos = max(1, train_start_idx + 1)
            # 原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
            # 因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
            start_pos = max(1, train_start_idx + 1)
            max_pos = train_end_idx
            
            if start_pos <= max_pos:
                self.valid_stock_indices.append(stock_idx)
                self.stock_sample_ranges.append((start_pos, max_pos))
        
        valid_stocks = len(self.valid_stock_indices)
        total_samples = sum(max_pos - start_pos + 1 
                          for start_pos, max_pos in self.stock_sample_ranges)
        
        if valid_stocks == 0:
            raise ValueError(
                f"没有有效的训练股票！\n"
                f"  总股票数: {len(stock_info_list)}\n"
                f"  请检查数据质量或调整参数"
            )
        
        print(f"  初始化随机采样器: {valid_stocks}只有效股票, 总样本数={total_samples}")
        print(f"  采样策略: 完全随机采样，每次随机选择股票和位置")

    def sample_batch_rounds(self, num_rounds, rng=None):
        """
        随机采样多轮：按样本数量加权随机采样

        参数:
            num_rounds: 要采样的轮数（每轮采样 valid_stocks 个样本）
            rng: 随机数生成器（用于可复现性）

        返回: [(stock_idx, start_idx), ...] 所有轮次的样本索引列表
        
        加权策略：
            - 每只股票被选中的概率与其样本数量成正比
            - 样本多的股票被采样次数多，样本少的股票被采样次数少
            - 确保每个样本被采样的期望概率相等
        """
        if rng is None:
            rng = random.Random()
        
        all_samples = []
        num_samples_per_round = len(self.valid_stock_indices)
        total_samples_to_generate = num_rounds * num_samples_per_round
        
        sample_weights = [max_pos - start_pos + 1 
                         for start_pos, max_pos in self.stock_sample_ranges]
        
        stock_to_range = {
            stock_idx: self.stock_sample_ranges[i]
            for i, stock_idx in enumerate(self.valid_stock_indices)
        }
        
        for _ in range(total_samples_to_generate):
            stock_idx = rng.choices(
                self.valid_stock_indices,
                weights=sample_weights,
                k=1
            )[0]
            
            start_pos, max_pos = stock_to_range[stock_idx]
            start_idx = rng.randint(start_pos, max_pos)
            all_samples.append((stock_idx, start_idx))
        
        return all_samples
    
    def get_progress(self):
        """随机采样器无进度概念，返回 (0, 1) 表示无限采样"""
        return 0, 1

    def get_loop_stats(self):
        """随机采样器无循环概念"""
        return 0, 0


def create_sampler(stock_info_list, strategy=None):
    """
    根据配置创建采样器
    
    参数:
        stock_info_list: 股票信息列表
        strategy: 采样策略，可选 'temporal' 或 'random'，默认使用 DataConfig.SAMPLING_STRATEGY
    
    返回:
        sampler: TemporalSampler 或 RandomSampler 实例
    """
    if strategy is None:
        strategy = DataConfig.SAMPLING_STRATEGY
    
    if strategy == 'random':
        print("使用随机采样策略")
        return RandomSampler(stock_info_list)
    else:
        print("使用时间顺序采样策略")
        return TemporalSampler(stock_info_list)


def generate_sample_from_index(stock_info_list, stock_idx, start_idx, feature_normalizer=None):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引
        feature_normalizer: 可选的特征归一化器实例

    返回: (input_seq, target, cumulative_return, daily_returns, market_seq) 或 None（如果样本无效）
        market_seq: [market_context_length] 大盘涨跌序列，如果 GlobalDataManager 未加载市场数据则返回 None
    
    核心概念区分：
        【涨跌幅】用于标签生成，判断股票走势强弱
            - 基准是前一日收盘价
            - Day1涨跌幅 = (T+1收盘 - T日收盘) / T日收盘
            - Day2涨跌幅 = (T+2收盘 - T+1收盘) / T+1收盘
            - Day3涨跌幅 = (T+3收盘 - T+2收盘) / T+2收盘

        【收益率】用于计算投资回报，评估模型表现，支持智能止损：
            - 基准是买入价（T+1开盘价）
            - Day1 ≤ -3% → 第二天开盘止损
            - Day1+Day2 < -2% 或 Day1,Day2都<1% → 第二天收盘止损
            - 否则持有满3天，第三天收盘卖出
            - cumulative_return 为实际累计收益率（调用方应优先使用此值）       
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    stock_times = stock_info.get('times', None)
    context_length = DataConfig.CONTEXT_LENGTH
    required_length = DataConfig.REQUIRED_LENGTH

    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer
    )
    
    if input_seq is None:
        return None

    t1_open = stock_data[start_idx + context_length, 0]
    t1_close = stock_data[start_idx + context_length, 3]
    t2_open = stock_data[start_idx + context_length + 1, 0]
    t2_close = stock_data[start_idx + context_length + 1, 3]
    t3_close = stock_data[start_idx + context_length + 2, 3]

    if t1_open == 0 or t1_close == 0 or t2_open == 0 or t2_close == 0 or t3_close == 0:
        return None

    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    closes = input_seq_raw[:, 3]

    daily_price_changes = []
    day1_price_change = (t1_close - closes[-1]) / closes[-1]
    daily_price_changes.append(day1_price_change)
    day2_price_change = (t2_close - t1_close) / t1_close
    daily_price_changes.append(day2_price_change)
    day3_price_change = (t3_close - t2_close) / t2_close
    daily_price_changes.append(day3_price_change)

    cumulative_return, daily_returns = calculate_returns(
        t1_open=t1_open,
        t1_close=t1_close,
        t2_open=t2_open,
        t2_close=t2_close,
        t3_close=t3_close,
        day1_change=daily_price_changes[0],
        day2_change=daily_price_changes[1],
        day3_change=daily_price_changes[2]
    )

    target = float(generate_label(
        day1_change=daily_price_changes[0],
        day2_change=daily_price_changes[1],
        day3_change=daily_price_changes[2]
    ))

    market_seq = None
    gdm = GlobalDataManager.get_instance()
    if gdm.is_market_data_loaded() and stock_times is not None:
        sample_end_idx = start_idx + context_length
        target_date = stock_times[sample_end_idx]
        market_seq = gdm.get_market_context(target_date)
        if market_seq is not None and feature_normalizer is not None:
            market_seq = feature_normalizer.transform_market(market_seq)

    return input_seq, target, cumulative_return, daily_returns, market_seq


def generate_sample_from_index_partial(stock_info_list, stock_idx, start_idx, feature_normalizer=None):
    """
    生成样本，支持不完整的未来数据（用于最近几天的临时评估）

    与generate_sample_from_index的区别：
    - 由run.py使用，与模型训练阶段脚本无关
    - 允许未来数据不足3天
    - 返回可用天数信息
    - 不生成标签（仅用于推理展示）

    返回: (input_seq, cumulative_return, daily_returns, available_days, market_seq) 或 None
        available_days: 可用的未来天数 (1, 2, 或 3)
        market_seq: [market_context_length] 大盘涨跌序列，如果 GlobalDataManager 未加载市场数据则返回 None
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    stock_times = stock_info.get('times', None)
    context_length = DataConfig.CONTEXT_LENGTH
    data_length = len(stock_data)

    required_length = min(DataConfig.REQUIRED_LENGTH, data_length - start_idx)
    
    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer
    )
    
    if input_seq is None:
        return None

    t1_idx = start_idx + context_length
    t2_idx = start_idx + context_length + 1
    t3_idx = start_idx + context_length + 2
    
    available_days = 0
    if t1_idx < data_length:
        available_days = 1
    if t2_idx < data_length:
        available_days = 2
    if t3_idx < data_length:
        available_days = 3
    
    if available_days == 0:
        return None
    
    t1_open = stock_data[t1_idx, 0]
    t1_close = stock_data[t1_idx, 3]

    if t1_open == 0 or t1_close == 0:
        return None

    t_day_close = stock_data[start_idx + context_length - 1, 3]

    day1_change = None
    day2_change = None
    day3_change = None

    day1_change = (t1_close - t_day_close) / t_day_close

    t2_open = None
    t2_close = None
    t3_close = None

    if available_days >= 2:
        t2_open = stock_data[t2_idx, 0]
        t2_close = stock_data[t2_idx, 3]
        if t2_open == 0 or t2_close == 0:
            return None
        day2_change = (t2_close - t1_close) / t1_close

    if available_days >= 3:
        t3_close = stock_data[t3_idx, 3]
        if t3_close == 0:
            return None
        day3_change = (t3_close - t2_close) / t2_close

    cumulative_return, daily_returns = calculate_returns(
        t1_open=t1_open,
        t1_close=t1_close,
        t2_open=t2_open,
        t2_close=t2_close,
        t3_close=t3_close,
        day1_change=day1_change,
        day2_change=day2_change,
        day3_change=day3_change
    )

    market_seq = None
    gdm = GlobalDataManager.get_instance()
    if gdm.is_market_data_loaded() and stock_times is not None:
        sample_end_idx = start_idx + context_length
        target_date = stock_times[sample_end_idx]
        market_seq = gdm.get_market_context(target_date)
        if market_seq is not None and feature_normalizer is not None:
            market_seq = feature_normalizer.transform_market(market_seq)

    return input_seq, cumulative_return, daily_returns, available_days, market_seq


def sample_with_pools(sampler, stock_info_list, batch_size, batches_per_epoch, rng, feature_normalizer=None):
    """
    使用样本池机制采样（流式处理版）：
    1. 按时间顺序遍历样本索引
    2. 实时填充正负样本池
    3. 一旦正样本达到配额且负样本足够，立即生成Batch并清空负样本池
    4. 确保Batch之间的时间有序性，严格防止未来数据泄露到过去的Batch中
    5. 支持循环采样：数据到达末尾后自动循环回起点
    6. 动态生成索引：按需生成，直到batch数量满足要求

    Args:
        sampler: 采样器实例
        stock_info_list: 股票信息列表
        batch_size: 批次大小
        batches_per_epoch: 每个epoch的batch数量
        rng: 随机数生成器
        feature_normalizer: 可选的特征归一化器实例
    """
    positive_ratio = 0.25
    pos_quota = max(1, int(batch_size * positive_ratio))
    neg_quota = batch_size - pos_quota

    pos_pool_inputs = []
    pos_pool_targets = []
    pos_pool_returns = []
    pos_pool_market = []
    neg_pool_inputs = []
    neg_pool_targets = []
    neg_pool_returns = []
    neg_pool_market = []

    all_batch_inputs = []
    all_batch_targets = []
    all_batch_returns = []
    all_batch_market = []

    batches_generated = 0
    
    initial_rounds = 50
    total_rounds_generated = 0
    total_indices_generated = 0
    
    print(f"    动态采样策略：按需生成索引，直到满足{batches_per_epoch}个batch...")
    
    while batches_generated < batches_per_epoch:
        if isinstance(sampler, RandomSampler):
            sample_indices = sampler.sample_batch_rounds(initial_rounds, rng)
        else:
            sample_indices = sampler.sample_batch_rounds(initial_rounds)
        
        if len(sample_indices) == 0:
            print(f"\n    ⚠ 警告：采样头已到达所有股票终点且无法循环，停止采样")
            break
        
        total_rounds_generated += initial_rounds
        total_indices_generated += len(sample_indices)
        
        for stock_idx, start_idx in sample_indices:
            if batches_generated >= batches_per_epoch:
                break

            sample = generate_sample_from_index(stock_info_list, stock_idx, start_idx, feature_normalizer)
            if sample is None:
                continue

            input_seq, target, cumulative_return, _, market_seq = sample

            if target >= 0.5:
                pos_pool_inputs.append(input_seq)
                pos_pool_targets.append(target)
                pos_pool_returns.append(cumulative_return)
                if market_seq is not None:
                    pos_pool_market.append(market_seq)
            else:
                neg_pool_inputs.append(input_seq)
                neg_pool_targets.append(target)
                neg_pool_returns.append(cumulative_return)
                if market_seq is not None:
                    neg_pool_market.append(market_seq)
            
            if len(pos_pool_inputs) >= pos_quota and len(neg_pool_inputs) >= neg_quota:
                batch_pos_inputs = pos_pool_inputs[:pos_quota]
                batch_pos_targets = pos_pool_targets[:pos_quota]
                batch_pos_returns = pos_pool_returns[:pos_quota]
                batch_pos_market = pos_pool_market[:pos_quota] if pos_pool_market else []
                
                neg_indices = rng.sample(range(len(neg_pool_inputs)), neg_quota)
                batch_neg_inputs = [neg_pool_inputs[i] for i in neg_indices]
                batch_neg_targets = [neg_pool_targets[i] for i in neg_indices]
                batch_neg_returns = [neg_pool_returns[i] for i in neg_indices]
                batch_neg_market = [neg_pool_market[i] for i in neg_indices] if neg_pool_market else []
                
                batch_inputs = batch_pos_inputs + batch_neg_inputs
                batch_targets = batch_pos_targets + batch_neg_targets
                batch_returns = batch_pos_returns + batch_neg_returns
                batch_market = batch_pos_market + batch_neg_market if (batch_pos_market or batch_neg_market) else []
                
                combined = list(zip(batch_inputs, batch_targets, batch_returns))
                rng.shuffle(combined)
                b_inputs, b_targets, b_returns = zip(*combined)
                
                if batch_market:
                    market_combined = list(zip(batch_market, batch_targets))
                    rng.shuffle(market_combined)
                    b_market = [m for m, _ in market_combined]
                else:
                    b_market = []
                
                all_batch_inputs.extend(b_inputs)
                all_batch_targets.extend(b_targets)
                all_batch_returns.extend(b_returns)
                if b_market:
                    all_batch_market.extend(b_market)
                
                batches_generated += 1
                
                pos_pool_inputs = pos_pool_inputs[pos_quota:]
                pos_pool_targets = pos_pool_targets[pos_quota:]
                pos_pool_returns = pos_pool_returns[pos_quota:]
                pos_pool_market = pos_pool_market[pos_quota:] if pos_pool_market else []
                neg_pool_inputs = []
                neg_pool_targets = []
                neg_pool_returns = []
                neg_pool_market = []
        
        print(f"    已生成 {batches_generated}/{batches_per_epoch} 个Batch (已采样{total_rounds_generated}轮)", end='\r', flush=True)
        
        if batches_generated < batches_per_epoch:
            remaining_batches = batches_per_epoch - batches_generated
            if batches_generated > 0:
                estimated_rounds = max(20, int(remaining_batches / batches_generated * total_rounds_generated * 1.2))
                initial_rounds = min(estimated_rounds, 100)
            else:
                initial_rounds = 100

    print(f"\n    已生成 {batches_generated}/{batches_per_epoch} 个batch (总共采样{total_rounds_generated}轮, {total_indices_generated}个索引)")
    
    if batches_generated < batches_per_epoch:
        print(f"    ⚠ 警告：样本不足，仅生成 {batches_generated} 个Batch (目标: {batches_per_epoch})")
        if batches_generated == 0:
             raise ValueError(f"样本严重不足：无法生成任何Batch")

    if all_batch_market:
        return np.asarray(all_batch_inputs), np.asarray(all_batch_targets), np.asarray(all_batch_returns), np.asarray(all_batch_market)
    else:
        return np.asarray(all_batch_inputs), np.asarray(all_batch_targets), np.asarray(all_batch_returns), None


def create_fixed_evaluation_dataset(test_stock_info, feature_normalizer=None):
    """
    创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）
    
    只包含完整样本（available_days == 3），用于模型评估
    
    Args:
        test_stock_info: 测试集股票信息列表
        feature_normalizer: 可选的特征归一化器实例
    
    返回:
        eval_inputs: 输入序列
        eval_targets: 标签
        eval_cumulative_returns: 累计收益率 = (T+3收盘 - T+1开盘) / T+1开盘
        eval_day_indices: 每个样本对应的预测日在测试集中的相对偏移量（用于实战收益率按天分组）
        eval_daily_returns: 每日收益率列表 [[r1, r2, r3], ...]
            - 基准是买入价（T+1开盘价），用于计算投资回报
            - r1 = (T+1收盘 - T+1开盘) / T+1开盘（Day1日内收益）
            - r2 = (T+2收盘 - T+1收盘) / T+1开盘（Day2收益贡献）
            - r3 = (T+3收盘 - T+2收盘) / T+1开盘（Day3收益贡献）
            - r1 + r2 + r3 = 累计收益率
        eval_market_seqs: 大盘涨跌序列 [N, market_context_length]，如果 GlobalDataManager 未加载市场数据则返回 None
    """
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []
    eval_day_indices = []
    eval_daily_returns = []
    eval_market_seqs = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))

        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.REQUIRED_LENGTH
        
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx, feature_normalizer)
            if sample is None:
                continue

            input_seq, target, cumulative_return, daily_returns, market_seq = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)
            eval_cumulative_returns.append(float(cumulative_return))
            eval_daily_returns.append(daily_returns)
            if market_seq is not None:
                eval_market_seqs.append(market_seq)
            
            predict_day_idx = start_idx + DataConfig.CONTEXT_LENGTH
            day_index = predict_day_idx - test_split_point
            eval_day_indices.append(day_index)

    if len(eval_inputs) == 0:
        raise ValueError("固定评估集为空：test_stock_info中没有可用样本")

    if eval_market_seqs:
        return (np.asarray(eval_inputs), np.asarray(eval_targets), 
                np.asarray(eval_cumulative_returns), np.asarray(eval_day_indices),
                eval_daily_returns, np.asarray(eval_market_seqs))
    else:
        return (np.asarray(eval_inputs), np.asarray(eval_targets), 
                np.asarray(eval_cumulative_returns), np.asarray(eval_day_indices),
                eval_daily_returns, None)


def create_recent_days_dataset(test_stock_info, feature_normalizer=None):
    """
    创建最近几天的临时评估数据集（用于run.py中显示最近几天的实战收益率）
    
    包含完整样本和临时样本，用于展示最近几天的选股情况
    - 完整样本（available_days == 3）：与 create_fixed_evaluation_dataset 一致
    - 临时样本（available_days < 3）：仅用于展示，方便用户决策

    Args:
        test_stock_info: 测试集股票信息列表
        feature_normalizer: 可选的特征归一化器实例
    
    返回:
        recent_inputs: 输入序列
        recent_cumulative_returns: 累计收益率（可能不完整）
        recent_day_indices: 预测日索引
        recent_available_days: 可用天数 (1, 2, 或 3)
        recent_market_seqs: 大盘涨跌序列 [N, market_context_length]，如果 GlobalDataManager 未加载市场数据则返回 None
    """
    recent_inputs = []
    recent_cumulative_returns = []
    recent_day_indices = []
    recent_available_days = []
    recent_market_seqs = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))
        
        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.CONTEXT_LENGTH - 1
        
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index_partial([stock_info], 0, start_idx, feature_normalizer)
            if sample is None:
                continue

            input_seq, cumulative_return, daily_returns, available_days, market_seq = sample
            
            predict_day_idx = start_idx + DataConfig.CONTEXT_LENGTH
            
            recent_inputs.append(input_seq)
            recent_cumulative_returns.append(float(cumulative_return))
            recent_available_days.append(available_days)
            if market_seq is not None:
                recent_market_seqs.append(market_seq)
            
            day_index = predict_day_idx - test_split_point
            recent_day_indices.append(day_index)

    if len(recent_inputs) == 0:
        return None, None, None, None, None

    if recent_market_seqs:
        return (np.asarray(recent_inputs), np.asarray(recent_cumulative_returns), 
                np.asarray(recent_day_indices), np.asarray(recent_available_days),
                np.asarray(recent_market_seqs))
    else:
        return (np.asarray(recent_inputs), np.asarray(recent_cumulative_returns), 
                np.asarray(recent_day_indices), np.asarray(recent_available_days), None)


def normalize_and_validate_context_window(stock_data, start_idx, context_length,
                                          check_limit_up=True, required_length=None,
                                          feature_normalizer=None,
                                          apply_fine_normalization=True):
    """
    统一的上下文窗口归一化和验证函数

    用于消除 run.py 和 data.py 中的代码重复。
    执行完整的数据验证和归一化流程，与 generate_sample_from_index 保持一致。

    数据处理分两阶段：
        - 粗处理：CSV → OHLE 格式（涨跌幅 -0.1~0.1，Volume 0~1，Exchange 0~1）
        - 细处理：OHLE → 标准化数据（均值≈0，方差≈1）

    Args:
        stock_data: 股票原始数据 [N, 6]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口
        feature_normalizer: 可选的特征归一化器实例，用于细处理阶段
        apply_fine_normalization: 是否应用细处理（默认 True）。设为 False 时只执行粗处理。

    Returns:
        input_seq: [context_length, 6] 归一化后的输入序列，或 None（如果验证失败）
            - 粗处理后：OHLE: -0.1~0.1, Volume: 0~1, Exchange: 0~1
            - 细处理后：均值≈0，方差≈1

    验证项：
        1. 基准日（start_idx-1）的 OHLC 和 volume 非零
        2. 上下文窗口的 close 和 volume 非零
        3. 涨停过滤：窗口内任何一天涨跌幅不超过 11%
        4. 上下文最后一天涨停过滤（可选，通过 DataConfig 控制）
        5. 归一化后无 nan/inf
    """
    if start_idx < 1:
        return None
    
    if required_length is None:
        required_length = context_length
    
    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    prev_day_data = stock_data[start_idx - 1]

    prev_close = prev_day_data[3]
    prev_volume = prev_day_data[4]
    if prev_close == 0 or prev_volume == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    closes = input_seq_raw[:, 3]
    volumes = input_seq_raw[:, 4]
    if np.any(closes == 0) or np.any(volumes == 0):
        return None

    if check_limit_up:
        sample_window_start = start_idx - 1
        sample_window_end = start_idx + required_length
        sample_data = stock_data[sample_window_start:sample_window_end]

        limit_threshold = 0.11
        for day_idx in range(1, len(sample_data)):
            today_close = sample_data[day_idx, 3]
            yesterday_close = sample_data[day_idx - 1, 3]

            if yesterday_close == 0:
                return None
            daily_return = (today_close - yesterday_close) / yesterday_close
            if abs(daily_return) > limit_threshold:
                return None

    # 上下文最后一天涨停过滤（独立于 check_limit_up，仅受 DataConfig 控制）
    if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP:
        last_day_idx = start_idx + context_length - 1
        prev_day_idx = start_idx + context_length - 2
        prev_day_close = stock_data[prev_day_idx, 3]
        last_day_close = stock_data[last_day_idx, 3]

        if prev_day_close == 0:
            return None
        last_day_return = (last_day_close - prev_day_close) / prev_day_close
        if last_day_return >= 0.095:
            return None

    input_seq = np.empty((context_length, 6), dtype=np.float32)
    
    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]
    
    input_seq[0, 4] = (volumes[0] - prev_volume) / prev_volume
    if context_length > 1:
        input_seq[1:, 4] = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    
    input_seq[:, 5] = input_seq_raw[:, 5] / 100.0

    # ========== 粗处理阶段 ==========
    # OHLE: 涨跌幅，范围 -0.1 ~ 0.1
    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])
    # Volume: 变化率缩放，范围 0 ~ 1
    np.clip(input_seq[:, 4], -5.0, 5.0, out=input_seq[:, 4])
    input_seq[:, 4] = input_seq[:, 4] / 10.0 + 0.5
    np.clip(input_seq[:, 4:6], 0.0, 1.0, out=input_seq[:, 4:6])

    # ========== 细处理阶段（可选）==========
    # 应用高级特征归一化，将粗处理结果转换为均值≈0、方差≈1的标准化数据
    if apply_fine_normalization and feature_normalizer is not None:
        input_seq = feature_normalizer.transform(input_seq)

    if np.any(~np.isfinite(input_seq)):
        return None

    return input_seq


def coarse_normalize_context_window(stock_data, start_idx, context_length,
                                     check_limit_up=True, required_length=None):
    """
    粗处理：CSV → OHLE 格式

    只执行粗处理阶段，不应用细处理（特征归一化器）。
    输出数据范围：
        - OHLE: -0.1 ~ 0.1（涨跌幅）
        - Volume: 0 ~ 1（成交量变化率）
        - Exchange: 0 ~ 1（换手率）

    Args:
        stock_data: 股票原始数据 [N, 6]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口

    Returns:
        input_seq: [context_length, 6] 粗处理后的输入序列，或 None（如果验证失败）
    """
    return normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=check_limit_up,
        required_length=required_length,
        feature_normalizer=None,
        apply_fine_normalization=False
    )


def fine_normalize_batch(input_seq, feature_normalizer):
    """
    细处理：OHLE → 标准化数据

    将粗处理后的数据送入细处理阶段，应用特征归一化器。
    输出数据特性：均值≈0，方差≈1

    Args:
        input_seq: 粗处理后的数据
            - 单个样本: [seq_len, 6]
            - 批量样本: [batch_size, seq_len, 6]
        feature_normalizer: 特征归一化器实例

    Returns:
        normalized_seq: 标准化后的数据，形状与输入相同
    """
    return feature_normalizer.transform(input_seq)


def fit_feature_normalizer(output_path='./normalizer.pkl', output_distribution='normal', n_quantiles=1000):
    """
    在训练集上拟合特征归一化器并保存到文件

    Args:
        output_path: 归一化器输出文件路径
        output_distribution: 输出分布类型 ('normal' 或 'uniform')
        n_quantiles: 分位数数量

    Returns:
        normalizer: 拟合后的 FeatureNormalizer 实例
    """
    if os.path.exists(output_path):
        print(f"归一化器文件已存在: {output_path}")
        response = input("是否重新训练？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("\n[步骤1] 加载训练集数据...")

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print(f"训练集股票数: {len(train_stock_info)}")
    print(f"测试集股票数: {len(test_stock_info)}")

    print("\n[步骤1.5] 加载大盘数据...")
    gdm = GlobalDataManager.get_instance()
    try:
        gdm.load_market_data()
        print("✓ 大盘数据加载成功")
    except FileNotFoundError as e:
        print(f"⚠ 大盘数据加载失败: {e}")
        print("  市场数据归一化器将不会被拟合")

    print("\n[步骤2] 创建特征归一化器...")
    print(f"  输出分布: {output_distribution}")
    print(f"  分位数数量: {n_quantiles}")

    normalizer = FeatureNormalizer(output_distribution=output_distribution,n_quantiles=n_quantiles)

    print("\n[步骤3] 在训练集上拟合归一化器...")
    normalizer.fit(train_stock_info)

    print("\n[步骤4] 保存归一化器...")
    normalizer.save(output_path)

    return normalizer


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='数据处理模块 兼 拟合特征归一化器训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
用法示例：
  python data.py                                           # 使用默认参数拟合归一化器
  python data.py --output-distribution uniform             # 使用均匀分布拟合
  python data.py --n-quantiles 500                         # 使用500个分位数拟合
  python data.py --output ./my_normalizer.pkl              # 指定输出文件路径
        '''
    )
    parser.add_argument('--output-distribution', type=str, default='normal',choices=['normal', 'uniform'],
                        help='输出分布类型: normal (标准正态) 或 uniform (均匀分布)，默认 normal')
    parser.add_argument('--n-quantiles', type=int, default=1000,help='分位数数量（默认1000，越大越精确但越慢）')
    parser.add_argument('--output', type=str, default='./normalizer.pkl',
                        help='归一化器输出文件路径，默认 ./normalizer.pkl')

    args = parser.parse_args()
    fit_feature_normalizer(output_path=args.output,
    output_distribution=args.output_distribution,n_quantiles=args.n_quantiles)
    print(f"✓ 特征归一化器训练完成！已保存到: {args.output}")


if __name__ == "__main__":
    main()
