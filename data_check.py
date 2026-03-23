"""
股票数据质量检查脚本

基于 Baostock 验证本地数据的完整性和准确性（使用不复权数据）：
- 检查最近 100 天数据是否最新
- 检查是否有遗漏的交易日
- 检查数据是否准确（OHLC 逻辑、价格范围等）
- 假设更早的数据是正确的，只检查近 100 天

使用方法:    
python data_check.py # 检查所有股票（默认检查最近 100 天）    
python data_check.py --days 50 # 指定检查天数   
python data_check.py --stocks 000001 600000 # 检查指定股票    
python data_check.py --no-backup # 禁用备份
python data_check.py --verbose # 详细输出模式
"""

import time
import datetime
import pandas as pd
import baostock as bs
import shutil
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import argparse
from src.config import DataConfig

class CheckStatus(Enum):
    """检查状态枚举"""
    PASS = "✓"
    FAIL = "✗ 失败"
    WARNING = "⚠ 警告"
    SKIP = "○ 跳过"


@dataclass
class CheckResult:
    """检查结果数据类"""
    stock_code: str
    status: CheckStatus
    message: str
    details: Optional[Dict] = None


class DataChecker:
    def __init__(self, data_dir: str, check_days: int = 100, backup: bool = True):
        """
        Args:
            data_dir: 数据存储目录
            check_days: 检查最近多少天的数据（默认 100 天）
            backup: 是否启用备份（默认启用）
        """
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir.parent / "data_backup"
        self.enable_backup = backup
        self.check_days = check_days
        self.login_success = False

        self.stats = {
            'total': 0,
            'pass': 0,
            'fail': 0,
            'warning': 0,
            'skip': 0
        }
        
    def login_baostock(self) -> bool:
        try:
            lg = bs.login()
            if lg.error_code == '0':
                self.login_success = True
                return True
            else:
                print(f"✗ Baostock 登录失败：{lg.error_msg}")
                return False
        except Exception as e:
            print(f"✗ Baostock 登录异常：{e}")
            return False
    
    def logout_baostock(self):
        if self.login_success:
            bs.logout()
            self.login_success = False
    
    def backup_data(self):
        """备份现有数据"""
        if not self.enable_backup:
            return
        
        try:
            if self.data_dir.exists():
                backup_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / backup_timestamp
                backup_path.mkdir(parents=True, exist_ok=True)
                
                for csv_file in self.data_dir.glob("*.csv"):
                    shutil.copy2(csv_file, backup_path / csv_file.name)
                
                print(f"✓ 数据已备份到：{backup_path}")
        except Exception as e:
            print(f"✗ 备份失败：{e}")
    
    def get_all_csv_files(self) -> List[Path]:
        """获取所有 CSV 文件（排除大盘指数文件）"""
        return sorted([f for f in self.data_dir.glob("*.csv") if f.name != DataConfig.MARKET_DATA_FILE])
    
    def get_last_date_in_file(self, file_path: Path) -> Optional[str]:
        """获取文件中最新日期"""
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                latest_date = df.iloc[0]['time']
                if isinstance(latest_date, (int, float)):
                    latest_date = str(int(latest_date))
                else:
                    latest_date = str(latest_date)
                return latest_date if len(latest_date) == 8 else None
            return None
        except Exception as e:
            print(f"读取文件 {file_path.name} 失败：{e}")
            return None
    
    def get_recent_local_data(self, file_path: Path, days: int = 100) -> Optional[pd.DataFrame]:
        """
        获取本地文件最近 N 天的数据
        
        Args:
            file_path: 文件路径
            days: 获取最近多少天
        Returns:
            DataFrame 或 None
        """
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return None
            
            df = df.sort_values('time', ascending=False)
            recent_data = df.head(days).copy()
            
            recent_data['time'] = recent_data['time'].astype(str)
            return recent_data
        except Exception as e:
            print(f"读取本地数据失败：{e}")
            return None
    
    def fetch_baostock_data(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从 Baostock 获取指定日期范围的数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        Returns:
            DataFrame 或 None
        """
        try:
            code_with_prefix = self._format_stock_code(stock_code)
            if code_with_prefix is None:
                return None
            
            rs = bs.query_history_k_data_plus(
                code_with_prefix,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                return None
            
            data_list = []
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                data_list.append(row)
            
            if not data_list:
                return None
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            df = df.rename(columns={
                'date': 'time',
                'open': 'start',
                'high': 'max',
                'low': 'min',
                'close': 'end'
            })
            
            df['time'] = df['time'].str.replace('-', '')
            df = df[df['time'] != '']
            
            if len(df) == 0:
                return None
            
            df['time'] = df['time'].astype(str)
            df['start'] = pd.to_numeric(df['start'], errors='coerce')
            df['max'] = pd.to_numeric(df['max'], errors='coerce')
            df['min'] = pd.to_numeric(df['min'], errors='coerce')
            df['end'] = pd.to_numeric(df['end'], errors='coerce')
            
            # 本地数据的 volume 字段存储的是成交额（单位：千元）,Baostock中的amount 单位是元
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df['volume'] = (df['amount'] / 1000.0).fillna(0.0)
            else:
                print(f"⚠ {stock_code} 警告：Baostock 未返回 amount 字段，使用原始 volume 字段")
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
            
            df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
            
            df = df.dropna(subset=['start', 'max', 'min', 'end'])
            
            return df[['time', 'start', 'max', 'min', 'end', 'volume', 'turn']]
            
        except Exception as e:
            print(f"获取 Baostock 数据失败：{e}")
            return None

    def fetch_index_baostock_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        从 Baostock 获取上证指数指定日期范围的数据

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        Returns:
            DataFrame 或 None
        """
        try:
            rs = bs.query_history_k_data_plus(
                DataConfig.MARKET_CODE,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )

            if rs.error_code != '0':
                return None

            data_list = []
            while rs.error_code == '0' and rs.next():
                row = rs.get_row_data()
                data_list.append(row)

            if not data_list:
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)

            df = df.rename(columns={
                'date': 'time',
                'open': 'start',
                'high': 'max',
                'low': 'min',
                'close': 'end'
            })

            df['time'] = df['time'].str.replace('-', '')
            df = df[df['time'] != '']

            if len(df) == 0:
                return None

            df['time'] = df['time'].astype(str)
            df['start'] = pd.to_numeric(df['start'], errors='coerce')
            df['max'] = pd.to_numeric(df['max'], errors='coerce')
            df['min'] = pd.to_numeric(df['min'], errors='coerce')
            df['end'] = pd.to_numeric(df['end'], errors='coerce')

            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                df['volume'] = (df['amount'] / 1000.0).fillna(0.0)
            else:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)

            df['turn'] = pd.to_numeric(df['turn'], errors='coerce')

            df = df.dropna(subset=['start', 'max', 'min', 'end'])

            return df[['time', 'start', 'max', 'min', 'end', 'volume', 'turn']]

        except Exception as e:
            print(f"获取上证指数 Baostock 数据失败：{e}")
            return None

    def check_index_data(self, verbose: bool = False) -> CheckResult:
        """
        检查上证指数数据的完整性

        Args:
            verbose: 是否详细输出
        Returns:
            CheckResult
        """
        file_path = self.data_dir / DataConfig.MARKET_DATA_FILE

        if not file_path.exists():
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.FAIL,
                message="大盘数据文件不存在"
            )

        local_latest = self.get_last_date_in_file(file_path)

        if not local_latest:
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.FAIL,
                message="无法读取大盘数据"
            )

        today = datetime.datetime.now()
        local_date_obj = datetime.datetime.strptime(local_latest, "%Y%m%d")
        days_diff = (today - local_date_obj).days

        if days_diff > 10:
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.WARNING,
                message=f"数据滞后 {days_diff} 天 (最新：{local_latest})",
                details={'latest_date': local_latest, 'days_behind': days_diff}
            )

        check_start_date = self._get_previous_date(local_latest, self.check_days)
        local_recent = self.get_recent_local_data(file_path, self.check_days)

        if local_recent is None or len(local_recent) == 0:
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.FAIL,
                message="本地数据为空"
            )

        baostock_start = self._date_to_baostock_format(check_start_date)
        baostock_end = self._date_to_baostock_format(local_latest)

        bs_data = self.fetch_index_baostock_data(baostock_start, baostock_end)

        if bs_data is None or len(bs_data) == 0:
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.SKIP,
                message="无法获取 Baostock 数据",
                details={'latest_date': local_latest}
            )

        local_dates = set(local_recent['time'])
        bs_dates = set(bs_data['time'])

        missing_dates = bs_dates - local_dates

        if missing_dates:
            return CheckResult(
                stock_code=DataConfig.MARKET_DATA_FILE,
                status=CheckStatus.FAIL,
                message=f"缺失 {len(missing_dates)} 个交易日",
                details={'missing_dates': sorted(list(missing_dates))}
            )

        for _, bs_row in bs_data.iterrows():
            bs_time = bs_row['time']
            local_row = local_recent[local_recent['time'] == bs_time]

            if len(local_row) == 0:
                continue

            local_row = local_row.iloc[0]

            if abs(local_row['start'] - bs_row['start']) > 0.01:
                return CheckResult(
                    stock_code=DataConfig.MARKET_DATA_FILE,
                    status=CheckStatus.FAIL,
                    message=f"开盘价不匹配 ({bs_time}): 本地={local_row['start']}, Baostock={bs_row['start']}",
                    details={'date': bs_time, 'field': 'start'}
                )

            if abs(local_row['end'] - bs_row['end']) > 0.01:
                return CheckResult(
                    stock_code=DataConfig.MARKET_DATA_FILE,
                    status=CheckStatus.FAIL,
                    message=f"收盘价不匹配 ({bs_time}): 本地={local_row['end']}, Baostock={bs_row['end']}",
                    details={'date': bs_time, 'field': 'end'}
                )

        return CheckResult(
            stock_code=DataConfig.MARKET_DATA_FILE,
            status=CheckStatus.PASS,
            message=f"数据完整且准确 (最新：{local_latest}, 检查 {len(bs_data)} 个交易日)",
            details={'latest_date': local_latest, 'checked_days': len(bs_data)}
        )

    def _date_to_baostock_format(self, date_str: str) -> str:
        """转换日期格式从 YYYYMMDD 到 YYYY-MM-DD"""
        if len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str
    
    def _get_previous_date(self, date_str: str, days: int) -> str:
        """获取 N 天前的日期"""
        try:
            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d")
            prev_date = date_obj - datetime.timedelta(days=days)
            return prev_date.strftime("%Y%m%d")
        except:
            return date_str
    
    def check_data_integrity(self, stock_code: str, file_path: Path, verbose: bool = False) -> CheckResult:
        """
        检查单只股票数据的完整性
        
        Args:
            stock_code: 股票代码
            file_path: 文件路径
            verbose: 是否详细输出
        Returns:
            CheckResult
        """
        local_latest = self.get_last_date_in_file(file_path)
        
        if not local_latest:
            return CheckResult(
                stock_code=stock_code,
                status=CheckStatus.FAIL,
                message="无法读取本地数据"
            )
        
        today = datetime.datetime.now()
        local_date_obj = datetime.datetime.strptime(local_latest, "%Y%m%d")
        days_diff = (today - local_date_obj).days
        
        if days_diff > 10:
            return CheckResult(
                stock_code=stock_code,
                status=CheckStatus.WARNING,
                message=f"数据滞后 {days_diff} 天 (最新：{local_latest})",
                details={'latest_date': local_latest, 'days_behind': days_diff}
            )
        
        check_start_date = self._get_previous_date(local_latest, self.check_days)
        local_recent = self.get_recent_local_data(file_path, self.check_days)
        
        if local_recent is None or len(local_recent) == 0:
            return CheckResult(
                stock_code=stock_code,
                status=CheckStatus.FAIL,
                message="本地数据为空"
            )
        
        baostock_start = self._date_to_baostock_format(check_start_date)
        baostock_end = self._date_to_baostock_format(local_latest)
        
        bs_data = self.fetch_baostock_data(stock_code, baostock_start, baostock_end)
        
        if bs_data is None or len(bs_data) == 0:
            return CheckResult(
                stock_code=stock_code,
                status=CheckStatus.SKIP,
                message="无法获取 Baostock 数据（可能停牌）",
                details={'latest_date': local_latest}
            )
        
        local_dates = set(local_recent['time'])
        bs_dates = set(bs_data['time'])
        
        missing_dates = bs_dates - local_dates
        extra_dates = local_dates - bs_dates
        
        if missing_dates:
            return CheckResult(
                stock_code=stock_code,
                status=CheckStatus.FAIL,
                message=f"缺失 {len(missing_dates)} 个交易日",
                details={'missing_dates': sorted(list(missing_dates))}
            )
        
        if len(extra_dates) > 0:
            if verbose:
                print(f"  ⚠ {stock_code} 多余 {len(extra_dates)} 个日期（可能是未来数据）")
        
        for _, bs_row in bs_data.iterrows():
            bs_time = bs_row['time']
            local_row = local_recent[local_recent['time'] == bs_time]
            
            if len(local_row) == 0:
                continue
            
            local_row = local_row.iloc[0]
            
            if abs(local_row['start'] - bs_row['start']) > 0.01:
                return CheckResult(
                    stock_code=stock_code,
                    status=CheckStatus.FAIL,
                    message=f"开盘价不匹配 ({bs_time}): 本地={local_row['start']}, Baostock={bs_row['start']}",
                    details={'date': bs_time, 'field': 'start'}
                )
            
            if abs(local_row['end'] - bs_row['end']) > 0.01:
                return CheckResult(
                    stock_code=stock_code,
                    status=CheckStatus.FAIL,
                    message=f"收盘价不匹配 ({bs_time}): 本地={local_row['end']}, Baostock={bs_row['end']}",
                    details={'date': bs_time, 'field': 'end'}
                )
            
            if abs(local_row['volume'] - bs_row['volume']) / max(bs_row['volume'], 1) > 0.05:
                return CheckResult(
                    stock_code=stock_code,
                    status=CheckStatus.WARNING,
                    message=f"成交量差异较大 ({bs_time}): 本地={local_row['volume']}, Baostock={bs_row['volume']}",
                    details={'date': bs_time, 'field': 'volume'}
                )
        
        return CheckResult(
            stock_code=stock_code,
            status=CheckStatus.PASS,
            message=f"数据完整且准确 (最新：{local_latest}, 检查 {len(bs_data)} 个交易日)",
            details={'latest_date': local_latest, 'checked_days': len(bs_data)}
        )
    
    def check_ohlc_logic(self, file_path: Path, check_days: int = 100) -> CheckResult:
        """
        检查 OHLC 逻辑是否正确
        
        Args:
            file_path: 文件路径
            check_days: 检查最近多少天
        Returns:
            CheckResult
        """
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return CheckResult(
                    stock_code=file_path.stem,
                    status=CheckStatus.FAIL,
                    message="数据为空"
                )
            
            df = df.sort_values('time', ascending=False).head(check_days)
            
            for idx, row in df.iterrows():
                if row['max'] < row['min']:
                    return CheckResult(
                        stock_code=file_path.stem,
                        status=CheckStatus.FAIL,
                        message=f"最高价 < 最低价 ({row['time']})",
                        details={'date': row['time'], 'max': row['max'], 'min': row['min']}
                    )
                
                if row['start'] < row['min'] or row['start'] > row['max']:
                    return CheckResult(
                        stock_code=file_path.stem,
                        status=CheckStatus.FAIL,
                        message=f"开盘价超出范围 ({row['time']})",
                        details={'date': row['time'], 'start': row['start'], 'max': row['max'], 'min': row['min']}
                    )
                
                if row['end'] < row['min'] or row['end'] > row['max']:
                    return CheckResult(
                        stock_code=file_path.stem,
                        status=CheckStatus.FAIL,
                        message=f"收盘价超出范围 ({row['time']})",
                        details={'date': row['time'], 'end': row['end'], 'max': row['max'], 'min': row['min']}
                    )
            
            return CheckResult(
                stock_code=file_path.stem,
                status=CheckStatus.PASS,
                message=f"OHLC 逻辑正确 (检查 {len(df)} 天)"
            )
            
        except Exception as e:
            return CheckResult(
                stock_code=file_path.stem,
                status=CheckStatus.FAIL,
                message=f"检查异常：{e}"
            )
    
    def check_price_changes(self, file_path: Path, check_days: int = 100) -> CheckResult:
        """
        检查涨跌幅是否合理（无异常暴涨暴跌）
        
        Args:
            file_path: 文件路径
            check_days: 检查最近多少天
        Returns:
            CheckResult
        """
        try:
            df = pd.read_csv(file_path)
            if len(df) < 2:
                return CheckResult(
                    stock_code=file_path.stem,
                    status=CheckStatus.SKIP,
                    message="数据不足"
                )
            
            df = df.sort_values('time', ascending=False).head(check_days)
            
            df = df.reset_index(drop=True)
            
            for i in range(1, len(df)):
                prev_close = df.iloc[i-1]['end']
                curr_close = df.iloc[i]['end']
                
                if prev_close > 0:
                    change_pct = (curr_close - prev_close) / prev_close
                    
                    if abs(change_pct) > 0.20:
                        return CheckResult(
                            stock_code=file_path.stem,
                            status=CheckStatus.WARNING,
                            message=f"涨跌幅异常 ({df.iloc[i]['time']}): {change_pct*100:.2f}%",
                            details={'date': df.iloc[i]['time'], 'change': change_pct}
                        )
            
            return CheckResult(
                stock_code=file_path.stem,
                status=CheckStatus.PASS,
                message=f"涨跌幅正常 (检查 {len(df)-1} 次)"
            )
            
        except Exception as e:
            return CheckResult(
                stock_code=file_path.stem,
                status=CheckStatus.FAIL,
                message=f"检查异常：{e}"
            )
    
    def repair_stock_data(self, stock_code: str, file_path: Path, result: CheckResult) -> bool:
        """
        尝试修复检测到的数据错误

        修复策略：
        - 缺失交易日：补拉缺失日期数据并合并写入
        - 价格不匹配 / OHLC 逻辑错误：重新全量拉取覆盖
        - 数据滞后（WARNING）：增量补更到最新

        Args:
            stock_code: 股票代码
            file_path: 本地文件路径
            result: 对应的 CheckResult
        Returns:
            修复是否成功
        """
        msg = result.message

        # ── 情况1：缺失交易日 ──────────────────────────────────────────
        if "缺失" in msg and result.details and 'missing_dates' in result.details:
            missing_dates = result.details['missing_dates']
            print(f"  → 修复：补拉 {len(missing_dates)} 个缺失交易日")
            start_bs = f"{missing_dates[0][:4]}-{missing_dates[0][4:6]}-{missing_dates[0][6:]}"
            end_bs   = f"{missing_dates[-1][:4]}-{missing_dates[-1][4:6]}-{missing_dates[-1][6:]}"
            patch_df = self.fetch_baostock_data(stock_code, start_bs, end_bs)
            if patch_df is None or len(patch_df) == 0:
                print(f"  ✗ 无法获取补丁数据，跳过")
                return False
            try:
                old_df = pd.read_csv(file_path)
                old_df['time'] = old_df['time'].astype(str)
                patch_df['time'] = patch_df['time'].astype(str)
                patch_df = patch_df.rename(columns={'turn': 'exchange'})
                patch_df = patch_df[['time', 'start', 'max', 'min', 'end', 'volume', 'exchange']]
                existing = set(old_df['time'])
                patch_df = patch_df[~patch_df['time'].isin(existing)]
                if len(patch_df) == 0:
                    print(f"  ⚠ 补丁数据已存在，无需写入")
                    return True
                combined = pd.concat([old_df, patch_df], ignore_index=True)
                combined = combined.sort_values('time', ascending=False).reset_index(drop=True)
                combined.to_csv(file_path, index=False)
                print(f"  ✓ 补入 {len(patch_df)} 条，总计 {len(combined)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        # ── 情况2：价格不匹配 / OHLC 逻辑错误 / 无法读取 ───────────────
        if any(k in msg for k in ["不匹配", "OHLC", "超出范围", "无法读取", "最高价"]):
            print(f"  → 修复：重新全量拉取 {stock_code}")
            rs = bs.query_history_k_data_plus(
                self._format_stock_code(stock_code),
                "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,"
                "pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date="2010-01-01",
                end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3"
            )
            if rs.error_code != '0':
                print(f"  ✗ 全量拉取失败：{rs.error_msg}")
                return False
            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                print(f"  ✗ 未获取到数据")
                return False
            try:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={'date': 'time', 'open': 'start', 'high': 'max',
                                        'low': 'min', 'close': 'end'})
                df['time'] = df['time'].str.replace('-', '')
                df = df[df['time'] != '']
                df['time'] = df['time'].astype(int)
                for col in ['start', 'max', 'min', 'end']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['start', 'max', 'min', 'end'])
                if 'amount' in df.columns:
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                    df['volume'] = (df['amount'] / 1000.0).fillna(0.0)
                else:
                    df['volume'] = 0.0
                # 优先使用 turnover 字段，如果不存在则使用 turn 字段
                if 'turnover' in df.columns:
                    df['exchange'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0.0)
                elif 'turn' in df.columns:
                    df['exchange'] = pd.to_numeric(df['turn'], errors='coerce').fillna(0.0)
                else:
                    df['exchange'] = 0.0
                df = df[['time', 'start', 'max', 'min', 'end', 'volume', 'exchange']]
                df = df.iloc[::-1].reset_index(drop=True)
                df.to_csv(file_path, index=False)
                print(f"  ✓ 全量写入 {len(df)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        # ── 情况3：数据滞后（WARNING）────────────────────────────────────
        if "滞后" in msg and result.details and 'latest_date' in result.details:
            latest = result.details['latest_date']
            start_date = f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
            # 从最新日期次日开始补
            next_day = (datetime.datetime.strptime(start_date, "%Y-%m-%d")
                        + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  → 修复：增量补更 {stock_code}（从 {next_day} 起）")
            patch_df = self.fetch_baostock_data(
                stock_code, next_day,
                datetime.datetime.now().strftime("%Y-%m-%d")
            )
            if patch_df is None or len(patch_df) == 0:
                print(f"  ⚠ 无新数据可补（可能已是最新）")
                return True
            try:
                old_df = pd.read_csv(file_path)
                old_df['time'] = old_df['time'].astype(str)
                patch_df['time'] = patch_df['time'].astype(str)
                patch_df = patch_df.rename(columns={'turn': 'exchange'})
                patch_df = patch_df[['time', 'start', 'max', 'min', 'end', 'volume', 'exchange']]
                existing = set(old_df['time'])
                patch_df = patch_df[~patch_df['time'].isin(existing)]
                if len(patch_df) == 0:
                    print(f"  ⚠ 补丁数据已存在，无需写入")
                    return True
                combined = pd.concat([patch_df, old_df], ignore_index=True)
                combined = combined.sort_values('time', ascending=False).reset_index(drop=True)
                combined.to_csv(file_path, index=False)
                new_latest = str(int(combined.iloc[0]['time']))
                print(f"  ✓ 补入 {len(patch_df)} 条，最新：{new_latest}")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        print(f"  ⚠ 无对应修复策略：{msg}")
        return False

    def repair_index_data(self, file_path: Path, result: CheckResult) -> bool:
        """
        尝试修复上证指数数据错误

        Args:
            file_path: 本地文件路径
            result: 对应的 CheckResult
        Returns:
            修复是否成功
        """
        msg = result.message

        if "缺失" in msg and result.details and 'missing_dates' in result.details:
            missing_dates = result.details['missing_dates']
            print(f"  → 修复：补拉 {len(missing_dates)} 个缺失交易日")
            start_bs = f"{missing_dates[0][:4]}-{missing_dates[0][4:6]}-{missing_dates[0][6:]}"
            end_bs = f"{missing_dates[-1][:4]}-{missing_dates[-1][4:6]}-{missing_dates[-1][6:]}"
            patch_df = self.fetch_index_baostock_data(start_bs, end_bs)
            if patch_df is None or len(patch_df) == 0:
                print(f"  ✗ 无法获取补丁数据，跳过")
                return False
            try:
                old_df = pd.read_csv(file_path)
                old_df['time'] = old_df['time'].astype(str)
                patch_df['time'] = patch_df['time'].astype(str)
                existing = set(old_df['time'])
                patch_df = patch_df[~patch_df['time'].isin(existing)]
                if len(patch_df) == 0:
                    print(f"  ⚠ 补丁数据已存在，无需写入")
                    return True
                combined = pd.concat([old_df, patch_df], ignore_index=True)
                combined = combined.sort_values('time', ascending=False).reset_index(drop=True)
                combined.to_csv(file_path, index=False)
                print(f"  ✓ 补入 {len(patch_df)} 条，总计 {len(combined)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        if any(k in msg for k in ["不匹配", "OHLC", "超出范围", "无法读取", "最高价", "最低价"]):
            print(f"  → 修复：重新全量拉取上证指数")
            rs = bs.query_history_k_data_plus(
                DataConfig.MARKET_CODE,
                "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,"
                "pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date="2010-01-01",
                end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                frequency="d",
                adjustflag="3"
            )
            if rs.error_code != '0':
                print(f"  ✗ 全量拉取失败：{rs.error_msg}")
                return False
            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())
            if not data_list:
                print(f"  ✗ 未获取到数据")
                return False
            try:
                df = pd.DataFrame(data_list, columns=rs.fields)
                df = df.rename(columns={'date': 'time', 'open': 'start', 'high': 'max',
                                        'low': 'min', 'close': 'end'})
                df['time'] = df['time'].str.replace('-', '')
                df = df[df['time'] != '']
                df['time'] = df['time'].astype(int)
                for col in ['start', 'max', 'min', 'end']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['start', 'max', 'min', 'end'])
                if 'amount' in df.columns:
                    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
                    df['volume'] = (df['amount'] / 1000.0).fillna(0.0)
                else:
                    df['volume'] = 0.0
                # 优先使用 turnover 字段，如果不存在则使用 turn 字段
                if 'turnover' in df.columns:
                    df['exchange'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(0.0)
                elif 'turn' in df.columns:
                    df['exchange'] = pd.to_numeric(df['turn'], errors='coerce').fillna(0.0)
                else:
                    df['exchange'] = 0.0
                df = df[['time', 'start', 'max', 'min', 'end', 'volume', 'exchange']]
                df = df.iloc[::-1].reset_index(drop=True)
                df.to_csv(file_path, index=False)
                print(f"  ✓ 全量写入 {len(df)} 条")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        if "滞后" in msg and result.details and 'latest_date' in result.details:
            latest = result.details['latest_date']
            start_date = f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
            next_day = (datetime.datetime.strptime(start_date, "%Y-%m-%d")
                        + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"  → 修复：增量补更上证指数（从 {next_day} 起）")
            patch_df = self.fetch_index_baostock_data(
                next_day, datetime.datetime.now().strftime("%Y-%m-%d")
            )
            if patch_df is None or len(patch_df) == 0:
                print(f"  ⚠ 无新数据可补（可能已是最新）")
                return True
            try:
                old_df = pd.read_csv(file_path)
                old_df['time'] = old_df['time'].astype(str)
                patch_df['time'] = patch_df['time'].astype(str)
                existing = set(old_df['time'])
                patch_df = patch_df[~patch_df['time'].isin(existing)]
                if len(patch_df) == 0:
                    print(f"  ⚠ 补丁数据已存在，无需写入")
                    return True
                combined = pd.concat([patch_df, old_df], ignore_index=True)
                combined = combined.sort_values('time', ascending=False).reset_index(drop=True)
                combined.to_csv(file_path, index=False)
                new_latest = str(int(combined.iloc[0]['time']))
                print(f"  ✓ 补入 {len(patch_df)} 条，最新：{new_latest}")
                return True
            except Exception as e:
                print(f"  ✗ 写入失败：{e}")
                return False

        print(f"  ⚠ 无对应修复策略：{msg}")
        return False

    def run_full_check(self, stock_codes: Optional[List[str]] = None,
                      verbose: bool = False, check_ohlc: bool = True) -> List[CheckResult]:
        """
        运行完整的数据检查，发现错误时自动修复

        Args:
            stock_codes: 指定要检查的股票列表
            verbose: 是否详细输出
            check_ohlc: 是否检查 OHLC 逻辑
        Returns:
            检查结果列表
        """
        print(f"检查范围：最近 {self.check_days} 天")
        print(f"检查项目：数据完整性、OHLC 逻辑、涨跌幅合理性")
        
        if not self.login_baostock():
            print("无法登录 Baostock，退出检查")
            return []
        
        try:
            print(f"\n检查上证指数数据...")
            index_result = self.check_index_data(verbose)
            results = [index_result]
            self._update_stats(index_result.status)

            if index_result.status == CheckStatus.PASS:
                print(f"✓ 上证指数 - {index_result.message}")
            elif index_result.status == CheckStatus.WARNING:
                print(f"⚠ 上证指数 - {index_result.message}")
            else:
                print(f"✗ 上证指数 - {index_result.message}")

            if index_result.status in (CheckStatus.FAIL, CheckStatus.WARNING):
                index_file_path = self.data_dir / DataConfig.MARKET_DATA_FILE
                ok = self.repair_index_data(index_file_path, index_result)
                if ok:
                    print(f"  ✓ 上证指数数据已修复")
                else:
                    print(f"  ✗ 上证指数数据修复失败")

            if stock_codes:
                files_to_check = [self.data_dir / f"{code}.csv" for code in stock_codes]
                files_to_check = [f for f in files_to_check if f.exists()]
                print(f"\n检查指定股票：{len(stock_codes)} 只")
            else:
                files_to_check = self.get_all_csv_files()
                print(f"\n检查所有股票：{len(files_to_check)} 只")
            
            if self.enable_backup:
                self.backup_data()
            
            results = []
            repair_success = 0
            repair_fail = 0

            for i, file_path in enumerate(files_to_check, 1):
                stock_code = file_path.stem

                if verbose:
                    print(f"\n[{i}/{len(files_to_check)}] 检查 {stock_code}...")
                else:
                    print(f"[{i}/{len(files_to_check)}] 检查 {stock_code}...", end=" ")

                integrity_result = self.check_data_integrity(stock_code, file_path, verbose)
                results.append(integrity_result)

                self._update_stats(integrity_result.status)

                if not verbose:
                    print(f"{integrity_result.status.value}", end="")
                    if integrity_result.status == CheckStatus.PASS:
                        print(f" - {integrity_result.message}")
                    else:
                        print(f" - {integrity_result.message}")

                if integrity_result.status in (CheckStatus.FAIL, CheckStatus.WARNING):
                    ok = self.repair_stock_data(stock_code, file_path, integrity_result)
                    if ok:
                        repair_success += 1
                    else:
                        repair_fail += 1

                if check_ohlc and integrity_result.status == CheckStatus.PASS:
                    ohlc_result = self.check_ohlc_logic(file_path, self.check_days)
                    if ohlc_result.status != CheckStatus.PASS:
                        results.append(ohlc_result)
                        if not verbose:
                            print(f"  {ohlc_result.status.value} - {ohlc_result.message}")
                        ok = self.repair_stock_data(stock_code, file_path, ohlc_result)
                        if ok:
                            repair_success += 1
                        else:
                            repair_fail += 1

                    price_result = self.check_price_changes(file_path, self.check_days)
                    if price_result.status != CheckStatus.PASS:
                        results.append(price_result)
                        if not verbose:
                            print(f"  {price_result.status.value} - {price_result.message}")
                        ok = self.repair_stock_data(stock_code, file_path, price_result)
                        if ok:
                            repair_success += 1
                        else:
                            repair_fail += 1

                if verbose and i % 50 == 0:
                    time.sleep(0.5)

            self._print_summary(results, repair_success, repair_fail)
            return results
            
        finally:
            self.logout_baostock()
    
    def _update_stats(self, status: CheckStatus):
        """更新统计"""
        self.stats['total'] += 1
        if status == CheckStatus.PASS:
            self.stats['pass'] += 1
        elif status == CheckStatus.FAIL:
            self.stats['fail'] += 1
        elif status == CheckStatus.WARNING:
            self.stats['warning'] += 1
        elif status == CheckStatus.SKIP:
            self.stats['skip'] += 1
    
    def _print_summary(self, results: List[CheckResult],
                       repair_success: int = 0,
                       repair_fail: int = 0):
        """打印摘要"""
        print("*"*32 + " 检查摘要 " + "*"*32)
        total = self.stats['total']
        print(f"总检查数：{total}")
        pass_rate = f"{(self.stats['pass']/total*100):.1f}%" if total else "N/A"
        print(f"✓ 通过：{self.stats['pass']} ({pass_rate})")
        print(f"✗ 失败：{self.stats['fail']}")
        print(f"⚠ 警告：{self.stats['warning']}")
        print(f"○ 跳过：{self.stats['skip']}")
        
        failed_results = [r for r in results if r.status == CheckStatus.FAIL]
        warning_results = [r for r in results if r.status == CheckStatus.WARNING]
        
        if failed_results:
            print(f"\n失败股票 ({len(failed_results)}):")
            for r in failed_results[:10]:
                print(f"  - {r.stock_code}: {r.message}")
            if len(failed_results) > 10:
                print(f"  ... 还有 {len(failed_results) - 10} 只")
        
        if warning_results:
            print(f"\n警告股票 ({len(warning_results)}):")
            for r in warning_results[:10]:
                print(f"  - {r.stock_code}: {r.message}")
            if len(warning_results) > 10:
                print(f"  ... 还有 {len(warning_results) - 10} 只")

        if repair_success + repair_fail > 0:
            print(f"\n修复统计：成功 {repair_success}，失败 {repair_fail}")

def main():
    parser = argparse.ArgumentParser(description='股票数据质量检查工具')
    parser.add_argument('--data-dir', type=str, default=r'src\data',help='数据存储目录 (默认：src\\data)')
    parser.add_argument('--days', type=int, default=100,help='检查最近多少天的数据 (默认：100)')
    parser.add_argument('--stocks', type=str, nargs='+',help='指定要检查的股票代码列表')
    parser.add_argument('--verbose', action='store_true',help='详细输出模式')
    parser.add_argument('--no-ohlc', action='store_true',help='跳过 OHLC 逻辑检查')
    parser.add_argument('--no-backup', action='store_true',help='禁用备份')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        script_dir = Path(__file__).parent
        data_dir = script_dir / data_dir
    
    checker = DataChecker(str(data_dir), check_days=args.days, backup=not args.no_backup)
    
    checker.run_full_check(
        stock_codes=args.stocks,
        verbose=args.verbose,
        check_ohlc=not args.no_ohlc
    )


if __name__ == "__main__":
    main()
