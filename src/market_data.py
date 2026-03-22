"""
大盘数据加载模块

加载大盘指数数据，计算涨跌幅，为市场token提供输入。

数据格式要求:
- CSV文件，包含日期和收盘价列
- 日期格式: YYYYMMDD (整数)
- 收盘价列名: 'end' 或 'close'

使用方法:
    market_data = load_market_data('./market_data.csv')
    market_seq = get_market_context(market_data, target_date, context_length=10)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import DataConfig


def load_market_data(market_data_path: str = None) -> Dict:
    """
    加载大盘数据并计算涨跌幅

    Args:
        market_data_path: 大盘数据文件路径，默认使用 DataConfig.MARKET_DATA_PATH

    Returns:
        dict: {
            'dates': np.ndarray,      # 日期数组 [N]
            'changes': np.ndarray,    # 涨跌幅数组 [N], 范围 [-0.1, 0.1]
            'closes': np.ndarray,     # 收盘价数组 [N]
        }
    """
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

    print(f"[MarketData] 加载大盘数据: {len(dates)} 天")
    print(f"  日期范围: {dates[0]} - {dates[-1]}")
    print(f"  涨跌幅范围: [{changes.min():.4f}, {changes.max():.4f}]")

    return {
        'dates': dates,
        'changes': changes,
        'closes': closes,
    }


def find_date_index(dates: np.ndarray, target_date: int) -> int:
    """
    在日期数组中查找目标日期的索引

    Args:
        dates: 日期数组
        target_date: 目标日期 (YYYYMMDD 格式的整数)

    Returns:
        索引，如果未找到返回 -1
    """
    indices = np.where(dates == target_date)[0]
    if len(indices) > 0:
        return indices[0]
    return -1


def get_market_context(
    market_data: Dict,
    target_date: int,
    context_length: int = None
) -> Optional[np.ndarray]:
    """
    获取指定日期前N天的大盘涨跌序列

    Args:
        market_data: load_market_data() 返回的数据字典
        target_date: 目标日期 (YYYYMMDD 格式的整数)
        context_length: 观察窗口长度，默认使用 DataConfig.MARKET_CONTEXT_LENGTH

    Returns:
        np.ndarray: [context_length] 涨跌幅序列，如果数据不足返回 None
    """
    if context_length is None:
        context_length = DataConfig.MARKET_CONTEXT_LENGTH

    dates = market_data['dates']
    changes = market_data['changes']

    target_idx = find_date_index(dates, target_date)
    if target_idx < 0:
        return None

    start_idx = target_idx - context_length
    if start_idx < 0:
        return None

    return changes[start_idx:target_idx].copy()


def get_market_context_batch(
    market_data: Dict,
    target_dates: np.ndarray,
    context_length: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量获取大盘涨跌序列

    Args:
        market_data: load_market_data() 返回的数据字典
        target_dates: 目标日期数组 [batch_size]
        context_length: 观察窗口长度

    Returns:
        market_seqs: [batch_size, context_length] 涨跌幅序列
        valid_mask: [batch_size] bool数组，标记哪些样本有效
    """
    if context_length is None:
        context_length = DataConfig.MARKET_CONTEXT_LENGTH

    batch_size = len(target_dates)
    market_seqs = np.zeros((batch_size, context_length), dtype=np.float32)
    valid_mask = np.ones(batch_size, dtype=bool)

    dates = market_data['dates']
    changes = market_data['changes']

    date_to_idx = {int(d): i for i, d in enumerate(dates)}

    for i, target_date in enumerate(target_dates):
        target_date_int = int(target_date)

        if target_date_int not in date_to_idx:
            valid_mask[i] = False
            continue

        target_idx = date_to_idx[target_date_int]
        start_idx = target_idx - context_length

        if start_idx < 0:
            valid_mask[i] = False
            continue

        market_seqs[i] = changes[start_idx:target_idx]

    return market_seqs, valid_mask


def get_market_context_for_sample(
    market_data: Dict,
    stock_data: np.ndarray,
    stock_times: np.ndarray,
    sample_end_idx: int,
    context_length: int = None
) -> Optional[np.ndarray]:
    """
    根据样本的结束索引获取对应的大盘涨跌序列

    Args:
        market_data: load_market_data() 返回的数据字典
        stock_data: 股票数据数组
        stock_times: 股票日期数组
        sample_end_idx: 样本在股票数据中的结束索引
        context_length: 大盘观察窗口长度

    Returns:
        np.ndarray: [context_length] 涨跌幅序列，如果数据不足返回 None
    """
    if context_length is None:
        context_length = DataConfig.MARKET_CONTEXT_LENGTH

    target_date = stock_times[sample_end_idx]

    return get_market_context(market_data, target_date, context_length)
