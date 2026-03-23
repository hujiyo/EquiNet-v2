'''
EquiNet 模型推理与选股脚本

核心流程：
1. 模型选择：读取 out/ 下可用的模型列表，用户选择模型
2. 模型评估：加载模型后执行评估（与 train.py 对模型A的评估完全一致）
3. 选股模式：用 data/ 最新数据作为最后一天，模型打分，按分数排序输出

数据一致性保证：
- 评估数据集：只包含完整样本（available_days == 3），与 train.py 完全一致
- 最近几天展示：包含临时样本（available_days < 3），仅用于展示，不参与阈值计算
'''

import os, sys, torch, numpy as np, glob, re
from datetime import datetime

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (ModelConfig, DataConfig, DeviceConfig, LossConfig)
from model import create_model
from data import (load_and_preprocess_data, create_fixed_evaluation_dataset,FeatureNormalizer,
                  create_recent_days_dataset, normalize_and_validate_context_window, GlobalDataManager)
from training_utils import evaluate_model, calculate_test_loss, DynamicWeightedBCE


# ==================== 工具函数 ====================

def parse_model_filename(filename):
    """
    从模型文件名中解析元信息
    格式示例: modelA_top1_p1_05pct_thr0_512_auc0_6377_ep15_0227_2110.pth
    """
    info = {'filename': filename, 'prefix': '', 'return_pct': '', 'threshold': '', 'auc': '', 'epoch': '', 'time': ''}
    
    # 提取模型前缀 (modelA, modelB, modelB_dft)
    prefix_match = re.match(r'^(modelA|modelB_dft|modelB)', filename)
    if prefix_match:
        info['prefix'] = prefix_match.group(1)
    
    # 提取收益率
    ret_match = re.search(r'_([pn]\d+_\d+)pct_', filename)
    if ret_match:
        ret_str = ret_match.group(1)
        ret_str = ret_str.replace('p', '+').replace('n', '-').replace('_', '.')
        info['return_pct'] = ret_str + '%'
    
    # 提取阈值
    thr_match = re.search(r'_thr(\d+_\d+)_', filename)
    if thr_match:
        thr_str = thr_match.group(1).replace('_', '.', 1)
        info['threshold'] = thr_str
    
    # 提取AUC
    auc_match = re.search(r'_auc(\d+_\d+)_', filename)
    if auc_match:
        auc_str = auc_match.group(1).replace('_', '.', 1)
        info['auc'] = auc_str
    
    # 提取epoch
    ep_match = re.search(r'_ep(\d+)_', filename)
    if ep_match:
        info['epoch'] = ep_match.group(1)
    
    # 提取时间戳
    time_match = re.search(r'_(\d{4}_\d{4})\.pth$', filename)
    if time_match:
        info['time'] = time_match.group(1)
    
    return info


def list_available_models(output_dir=DataConfig.OUTPUT_DIR):
    """列出 out/ 目录下所有可用的 .pth 模型文件"""
    if not os.path.exists(output_dir):
        print(f"  ✗ 输出目录 {output_dir} 不存在")
        return []
    
    pth_files = sorted(glob.glob(os.path.join(output_dir, '*.pth')))
    if not pth_files:
        print(f"  ✗ {output_dir} 下没有找到 .pth 模型文件")
        return []
    
    return [os.path.basename(f) for f in pth_files]


def load_model(model_path, device):
    """
    加载模型，支持两种 .pth 格式：
    - 新格式（checkpoint 字典）：含 model_arch / train_params / eval_stats / state_dict
    - 旧格式（裸 state_dict）：直接是权重字典，按 config.py 当前参数创建模型

    返回: (model, metadata)
        metadata: 新格式时为包含元数据的字典，旧格式时为 None
    """
    raw = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and 'state_dict' in raw:
        # 新格式：从内嵌的 model_arch 重建与训练时完全一致的模型
        model_arch   = raw.get('model_arch')
        train_params = raw.get('train_params')
        eval_stats   = raw.get('eval_stats')
        state_dict   = raw['state_dict']
        metadata     = {'model_arch': model_arch, 'train_params': train_params, 'eval_stats': eval_stats}
        model = create_model(model_arch=model_arch)
    else:
        # 旧格式：裸 state_dict，使用当前 config.py 参数
        state_dict = raw
        metadata   = None
        model = create_model()

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, metadata


def generate_latest_input(stock_data, latest_date, file_name, feature_normalizer=None):
    """
    为单只股票生成最新一天的模型输入（不需要未来数据）用于预测
    
    使用 data.py 的统一归一化函数，确保与训练时的数据处理逻辑完全一致。
    取数据最后 CONTEXT_LENGTH 天作为输入窗口。
    
    Args:
        stock_data: 股票原始数据
        latest_date: 最新交易日期 (YYYYMMDD 格式的整数)
        file_name: 文件名
        feature_normalizer: 可选的特征归一化器实例
    
    返回: (input_seq, market_seq, stock_code) 或 None
    """
    context_length = DataConfig.CONTEXT_LENGTH
    data_length = len(stock_data)
    
    # 需要 context_length + 1 天数据（第一天需要前一天做参照）
    if data_length < context_length + 1:
        return None
    
    # 取最后 context_length 天作为输入窗口
    start_idx = data_length - context_length
    
    # 使用 data.py 的统一归一化和验证函数
    input_seq = normalize_and_validate_context_window(
        stock_data, 
        start_idx, 
        context_length,
        check_limit_up=True,
        required_length=context_length,  # 只检查上下文窗口（无未来数据）
        feature_normalizer=feature_normalizer
    )
    
    if input_seq is None:
        return None
    
    # 获取市场数据
    gdm = GlobalDataManager.get_instance()
    market_seq = None
    if gdm.is_market_data_loaded():
        market_seq = gdm.get_market_context(latest_date)
        if market_seq is not None and feature_normalizer is not None:
            market_seq = feature_normalizer.transform_market(market_seq)
    
    # 提取股票代码（去掉.csv后缀）
    stock_code = file_name.replace('.csv', '')
    
    return input_seq, market_seq, stock_code


def load_all_stock_data(data_dir=DataConfig.DATA_DIR):
    """
    加载所有股票原始数据（用于选股推理）
    返回: [(file_name, data_array, latest_date), ...]
    """
    import pandas as pd
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv') and f != DataConfig.MARKET_DATA_FILE])
    stock_list = []
    
    for fname in all_files:
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath)
            # 原始数据按时间倒序，翻转为正序（早→晚）
            df = df.iloc[::-1].reset_index(drop=True)
            data = df[['start', 'max', 'min', 'end', 'volume', 'exchange']].values
            latest_date = str(df['time'].iloc[-1])  # 最新交易日期
            stock_list.append((fname, data, latest_date))
        except Exception as e:
            pass  # 静默跳过异常文件
    
    return stock_list


def score_all_stocks(model, stock_list, device, feature_normalizer=None):
    """
    为所有股票打分
    
    Args:
        model: 模型实例
        stock_list: 股票数据列表
        device: 设备
        feature_normalizer: 可选的特征归一化器实例
    
    返回: [(stock_code, score, latest_date, latest_close, latest_change_pct), ...]
    """
    results = []
    skipped = 0
    
    all_inputs = []
    all_market_seqs = []
    all_codes = []
    all_dates = []
    all_closes = []
    all_changes = []
    
    for fname, data, latest_date in stock_list:
        latest_date_int = int(latest_date)
        result = generate_latest_input(data, latest_date_int, fname, feature_normalizer)
        if result is None:
            skipped += 1
            continue
        
        input_seq, market_seq, stock_code = result
        all_inputs.append(input_seq)
        all_market_seqs.append(market_seq)
        all_codes.append(stock_code)
        all_dates.append(latest_date)
        
        # 最新收盘价和涨跌幅
        latest_close = data[-1, 3]
        if len(data) >= 2 and data[-2, 3] > 0:
            change_pct = (data[-1, 3] - data[-2, 3]) / data[-2, 3] * 100
        else:
            change_pct = 0.0
        all_closes.append(latest_close)
        all_changes.append(change_pct)
    
    if len(all_inputs) == 0:
        return [], skipped
    
    # 检查市场数据是否全部获取成功
    if any(m is None for m in all_market_seqs):
        raise RuntimeError("部分股票缺少市场数据，请确保 GlobalDataManager 已加载大盘数据。")
    
    # 批量推理
    batch_size = DataConfig.EVAL_BATCH_SIZE
    all_inputs_np = np.array(all_inputs)
    all_market_np = np.array(all_market_seqs)
    all_scores = []
    
    num_batches = (len(all_inputs_np) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(all_inputs_np))
            batch = torch.tensor(all_inputs_np[start:end], dtype=torch.float32, device=device)
            batch_market = torch.tensor(all_market_np[start:end], dtype=torch.float32, device=device)
            preds = torch.sigmoid(model(batch, batch_market)).cpu().numpy().flatten()
            all_scores.extend(preds)
            del batch
    
    # 组合结果
    for i in range(len(all_codes)):
        results.append((all_codes[i], float(all_scores[i]), all_dates[i], all_closes[i], all_changes[i]))
    
    return results, skipped


# ==================== 界面函数 ====================

def print_banner():
    """打印欢迎界面"""
    print()
    print("╔" + "═"*62 + "╗")
    print("║" + " "*14 + "EquiNet · 模型推理与选股" + " "*14 + "    ║")
    print("╚" + "═"*62 + "╝")
    print()


def print_section(title):
    """打印分节标题"""
    print()
    print(f"┌─── {title} " + "─" * max(1, 55 - len(title)*2) + "┐")


def print_section_end():
    """打印分节结束"""
    print(f"└" + "─"*62 + "┘")


def select_model(models):
    """模型选择界面"""
    print_section("可用模型列表")
    print(f"│")
    
    for i, fname in enumerate(models):
        info = parse_model_filename(fname)
        
        # 格式化显示
        prefix_display = {
            'modelA': '模型A(原始)',
            'modelB': '模型B(克隆)',
            'modelB_dft': '模型B(DFT)',
        }.get(info['prefix'], info['prefix'])
        
        detail_parts = []
        if info['return_pct']:
            detail_parts.append(f"收益{info['return_pct']}")
        if info['auc']:
            detail_parts.append(f"AUC={info['auc']}")
        if info['threshold']:
            detail_parts.append(f"阈值={info['threshold']}")
        if info['epoch']:
            detail_parts.append(f"Ep{info['epoch']}")
        
        detail_str = ', '.join(detail_parts)
        
        print(f"│  [{i+1}] {prefix_display}")
        print(f"│      {detail_str}")
        print(f"│      文件: {fname}")
        if i < len(models) - 1:
            print(f"│")
    
    print(f"│")
    print_section_end()
    
    while True:
        try:
            choice = input(f"\n  请选择模型 [1-{len(models)}]（输入 q 退出）: ").strip()
            if choice.lower() == 'q':
                print("  已退出。")
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return idx
            print(f"  ✗ 请输入 1 到 {len(models)} 之间的数字")
        except ValueError:
            print(f"  ✗ 无效输入，请输入数字")


def run_evaluation(model, test_stock_info, device, feature_normalizer=None):
    """
    执行模型评估（与 train.py 中对模型A的评估完全一致）
    返回评估统计字典

    Args:
        feature_normalizer: 可选的特征归一化器实例
    """
    print_section("模型评估")
    print(f"│  正在创建评估数据集...")

    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_market_seqs = \
        create_fixed_evaluation_dataset(test_stock_info, feature_normalizer)
    
    print(f"│  评估样本数: {len(eval_inputs)}")
    print(f"│  正在评估模型...")
    
    stats = evaluate_model(
        model, eval_inputs, eval_targets, eval_cumulative_returns,
        device, model_name="选中模型",
        eval_day_indices=eval_day_indices,
        eval_daily_returns=eval_daily_returns,
        market_seqs=eval_market_seqs
    )
    
    # 创建评估损失函数（与 train.py 一致）
    if LossConfig.use_dynamic_bce():
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        
        # 测试集权重
        test_targets = np.array(eval_targets)
        test_pos_count = np.sum(test_targets >= 0.5)
        test_neg_count = np.sum(test_targets < 0.5)
        if test_pos_count > 0 and test_neg_count > 0:
            test_neg_weight = LossConfig.POS_WEIGHT * (test_pos_count / test_neg_count)
        elif test_pos_count == 0:
            test_neg_weight = float(LossConfig.POS_WEIGHT)
        else:
            test_neg_weight = 0.1
        eval_criterion.weight_0_0.fill_(test_neg_weight)
    else:
        import torch.nn as nn
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # 计算测试集损失
    test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device, market_seqs=eval_market_seqs)
    
    # 打印评估结果（与 train.py 格式一致）
    print(f"│")
    print(f"│  ┌── 评估结果 ──────────────────────────────────┐")
    print(f"│  │  测试损失:          {test_loss:.4f}")
    print(f"│  │  AUC:              {stats['auc']:.4f}")
    print(f"│  │  预测均值:          {stats['pred_mean']:.3f}")
    print(f"│  │  预测标准差:        {stats['pred_std']:.4f}")
    print(f"│  │  高置信(>0.7):      {stats['high_conf_count']} 个")
    print(f"│  │  低置信(<0.2):      {stats['low_conf_count']} 个")
    print(f"│  │  Top{DataConfig.TOP_K}%样本数:        {stats['top_count']} 个")
    print(f"│  │  Top{DataConfig.TOP_K}%平均收益:      {stats['top_return']*100:+.2f}%")
    print(f"│  │")
    print(f"│  │  ★ Top{DataConfig.TOP_K}%阈值:        {stats['top_threshold']:.10f}")
    print(f"│  │")
    
    # 实战收益率
    if stats['realistic_stats'] is not None:
        rs = stats['realistic_stats']
        daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
        mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else \
                   f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else \
                   "全局阈值,不限数量"
        print(f"│  │  【实战收益率({mode_str})】")
        print(f"│  │  每日统计: {{{daily_stats_str}}}")
        print(f"│  │  平均实战收益率: {rs['avg_realistic_return']*100:.1f}%")
        print(f"│  │")
    
    if stats.get('smart_exit_stats') is not None:
        se = stats['smart_exit_stats']
        print(f"│  │  【智能止损】")
        print(f"│  │  收益率: {se['avg_realistic_return']*100:.1f}%")
        print(f"│  │  Day1止损: {se['stop_loss_day1_count']}次, 累计止损: {se['stop_loss_cum_count']}次, 止盈: {se['take_profit_count']}次")
        print(f"│  │")
    
    print(f"│  └───────────────────────────────────────────────┘")
    print_section_end()
    
    return stats


def run_stock_selection(model, threshold, device, feature_normalizer=None):
    """
    执行选股
    
    Args:
        model: 模型实例
        threshold: 选股阈值
        device: 设备
        feature_normalizer: 可选的特征归一化器实例
    """
    print_section("选股推理")
    print(f"│  正在加载全部股票数据...")
    
    stock_list = load_all_stock_data()
    total_stocks = len(stock_list)
    print(f"│  共加载 {total_stocks} 只股票数据")
    
    # 检查数据日期一致性
    dates = set(s[2] for s in stock_list)
    if len(dates) > 1:
        date_counts = {}
        for s in stock_list:
            d = s[2]
            date_counts[d] = date_counts.get(d, 0) + 1
        main_date = max(date_counts, key=date_counts.get)
        print(f"│  ⚠ 数据日期不完全一致，主要日期: {main_date} ({date_counts[main_date]}只)")
    else:
        main_date = list(dates)[0]
    
    print(f"│  数据截至日期: {main_date}")
    print(f"│  使用阈值: {threshold:.10f}")
    print(f"│  正在对所有股票打分...")
    
    results, skipped = score_all_stocks(model, stock_list, device, feature_normalizer)
    
    print(f"│  有效股票: {len(results)} 只，跳过（涨停/数据不足）: {skipped} 只")
    
    # 按分数降序排列
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 找到阈值分界位置
    threshold_idx = -1
    for i, (code, score, date, close, change) in enumerate(results):
        if score < threshold:
            threshold_idx = i
            break
    
    if threshold_idx == -1:
        threshold_idx = len(results)  # 全部都在阈值之上
    
    # 打印结果
    print(f"│")
    print(f"│  超过阈值: {threshold_idx} 只")
    print(f"│  低于阈值: {len(results) - threshold_idx} 只")
    print_section_end()
    
    # 打印选股列表
    print()
    print("╔" + "═"*78 + "╗")
    print("║" + " "*26 + "选 股 结 果 列 表" + " "*26 + "    ║")
    print("╠" + "═"*78 + "╣")
    print(f"║  {'排名':^4}  {'代码':^8}  {'模型分数':^10}  {'收盘价':^8}  {'涨跌幅':^8}  {'日期':^10}  {'':^6}  ║")
    print("╠" + "═"*78 + "╣")
    
    # 决定显示多少条
    # 阈值线上方全部显示 + 阈值线下方显示到前30名或阈值线后10条（取较大者）
    show_below = max(10, 30 - threshold_idx)
    total_show = min(len(results), threshold_idx + show_below)
    
    # 但如果阈值线上方已经超过50条，只显示上方前50条 + 下方5条
    if threshold_idx > 50:
        # 显示前20条 + 省略 + 阈值线附近10条 + 阈值线下方5条
        display_ranges = []
        display_ranges.append((0, min(20, threshold_idx)))
        if threshold_idx > 30:
            display_ranges.append(('ellipsis', threshold_idx - 20, threshold_idx))
            display_ranges.append((max(20, threshold_idx - 10), threshold_idx))
        display_ranges.append((threshold_idx, min(len(results), threshold_idx + 5)))
    else:
        display_ranges = [(0, total_show)]
    
    printed_indices = set()
    
    for item in display_ranges:
        if isinstance(item, tuple) and item[0] == 'ellipsis':
            if not any(i in printed_indices for i in range(item[1], item[2])):
                print(f"║  {'':^4}  {'...':^8}  {'':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}  ║")
            continue
        
        start_r, end_r = item if isinstance(item, tuple) else item
        for i in range(start_r, end_r):
            if i in printed_indices:
                continue
            printed_indices.add(i)
            
            code, score, date, close, change = results[i]
            rank = i + 1
            
            # 涨跌幅颜色标记
            change_str = f"{change:+.2f}%"
            
            # 阈值标记
            marker = ""
            if i == threshold_idx - 1 and threshold_idx > 0:
                marker = "┈阈值┈"
            elif i == threshold_idx:
                marker = "  ↓  "
            elif i < threshold_idx:
                marker = "  ★  "
            
            print(f"║  {rank:>4}   {code:>8}   {score:>10.8f}   {close:>8.2f}  {change_str:>8}   {date:>10}  {marker:^6}  ║")
            
            # 在阈值分界处画线
            if i == threshold_idx - 1 and threshold_idx < len(results):
                print("╠" + "─"*78 + "╣")
                print(f"║  {'':^4}  {'':^8}  {'↑ 超过阈值 ↑':^10}  {'│':^8}  {'↓ 低于阈值 ↓':^8}  {'':^10}  {'':^6}  ║")
                print("╠" + "─"*78 + "╣")
    
    # 如果还有更多未显示的
    remaining = len(results) - len(printed_indices)
    if remaining > 0:
        print(f"║  {'':^4}  {'':^8}  {f'... 还有 {remaining} 只未显示':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}  ║")
    
    print("╚" + "═"*78 + "╝")
    
    # 打印汇总统计
    print()
    print_section("选股汇总")
    above_scores = [r[1] for r in results[:threshold_idx]]
    if len(above_scores) > 0:
        print(f"│  超过阈值的股票: {threshold_idx} 只")
        print(f"│  最高分: {above_scores[0]:.8f}")
        print(f"│  最低分: {above_scores[-1]:.8f}")
        print(f"│  平均分: {np.mean(above_scores):.8f}")
        print(f"│")
        print(f"│  推荐关注（Top{DataConfig.TOP_K}%阈值以上）:")
        for i in range(min(threshold_idx, 10)):
            code, score, date, close, change = results[i]
            print(f"│    {i+1}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%")
    else:
        print(f"│  没有股票超过阈值 ({threshold:.10f})")
        print(f"│  当前最高分: {results[0][1]:.8f}" if results else "│  无有效股票")
    
    print_section_end()
    
    return results


def print_recent_days_chart(daily_stats, last_n=10):
    """
    打印最近N天的实战收益率表格
    
    参数:
        daily_stats: 每日统计列表 [(count, return, available_days), ...]
        last_n: 显示最近多少天
    """
    if not daily_stats or len(daily_stats) == 0:
        return
    
    total_days = len(daily_stats)
    start_idx = max(0, total_days - last_n)
    recent_stats = daily_stats[start_idx:]
    
    print()
    print("╔" + "═"*52 + "╗")
    title = f"最近{last_n}天实战收益率"
    padding = (52 - 2 - len(title)) // 2
    print("║" + " "*padding + title + " "*(52 - 2 - padding - len(title)) + "║")
    print("╠" + "═"*52 + "╣")
    print("║  Day  │ Count │ Return   │ 相对日期   │ 数据      ║")
    print("╠" + "─"*52 + "╣")
    
    for i, (count, ret, available_days) in enumerate(recent_stats):
        # day_num 表示"倒数第几天"
        day_num = last_n - i
        
        # 相对日期
        if i == last_n - 1:
            relative_date = "昨天"
        elif i == last_n - 2:
            relative_date = "前天"
        elif i == last_n - 3:
            relative_date = "大前天"
        else:
            relative_date = f"T-{day_num}"
        
        if available_days == 3:
            data_status = "完整"
        elif available_days == 2:
            data_status = "临时(2天)"
        elif available_days == 1:
            data_status = "临时(1天)"
        else:
            data_status = "-"
        
        ret_str = f"{ret*100:+.1f}%"
        
        print(f"║  {day_num:>3}  │  {count:>3}  │ {ret_str:>8} │ {relative_date:<8} │ {data_status:<9} ║")
    
    print("╚" + "═"*52 + "╝")


def calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=4, threshold=None, feature_normalizer=None):
    """
    计算最近几天的实战收益率（用于展示，包含临时数据）
    
    关键设计：
    - 阈值来源：直接使用传入的阈值（由 run_evaluation 计算，基于固定评估集）
    - 选股范围：所有样本（包括临时样本），用于展示最近几天的选股情况
    - 临时样本：仅用于展示，方便用户决策，不参与任何阈值计算
    
    Args:
        model: 模型实例
        test_stock_info: 测试集股票信息列表
        device: 设备
        top_n_per_day: 每日选股数量
        threshold: 选股阈值
        feature_normalizer: 可选的特征归一化器实例
    
    返回: daily_stats [(count, return, available_days), ...]
    """
    recent_result = create_recent_days_dataset(test_stock_info, feature_normalizer)
    recent_inputs, recent_returns, recent_day_indices, recent_available_days, recent_market_seqs = recent_result
    
    if recent_inputs is None or len(recent_inputs) == 0:
        return []
    
    if recent_market_seqs is None:
        raise RuntimeError("市场数据未加载，无法计算最近天数统计。请确保 GlobalDataManager 已加载大盘数据。")
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        batch_size = DataConfig.EVAL_BATCH_SIZE
        for i in range(0, len(recent_inputs), batch_size):
            batch = torch.tensor(recent_inputs[i:i+batch_size], dtype=torch.float32, device=device)
            batch_market = torch.tensor(recent_market_seqs[i:i+batch_size], dtype=torch.float32, device=device)
            preds = torch.sigmoid(model(batch, batch_market)).cpu().numpy().flatten()
            all_preds.extend(preds)
    
    all_preds = np.array(all_preds)
    all_returns = np.array(recent_returns)
    all_available_days = np.array(recent_available_days)
    
    unique_days = np.unique(recent_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    
    use_threshold_mode = (top_n_per_day == 0 and threshold is not None)
    
    if use_threshold_mode:
        max_select = DataConfig.MAX_SELECT_PER_DAY
        
        above_threshold_mask = all_preds > threshold
        
        for day in unique_days:
            day_mask = recent_day_indices == day
            day_above_threshold = above_threshold_mask & day_mask
            day_indices = np.where(day_above_threshold)[0]
            
            count = len(day_indices)
            if count > 0:
                if max_select > 0 and count > max_select:
                    day_preds = all_preds[day_indices]
                    top_local = np.argsort(day_preds)[::-1][:max_select]
                    selected_indices = day_indices[top_local]
                    count = max_select
                else:
                    selected_indices = day_indices
                
                day_return = np.mean(all_returns[selected_indices])
                min_available = int(np.min(all_available_days[selected_indices]))
                daily_stats.append((count, day_return, min_available))
            else:
                daily_stats.append((0, 0.0, 0))
    else:
        for day in unique_days:
            day_mask = recent_day_indices == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) == 0:
                daily_stats.append((0, 0.0, 0))
                continue
            
            day_preds = all_preds[day_indices]
            day_returns = all_returns[day_indices]
            day_available = all_available_days[day_indices]
            
            sorted_local_indices = np.argsort(day_preds)[::-1]
            select_count = min(top_n_per_day, len(day_indices))
            top_local_indices = sorted_local_indices[:select_count]
            
            if select_count == 0 or len(top_local_indices) == 0:
                daily_stats.append((0, 0.0, 0))
                continue
            
            day_return = np.mean(day_returns[top_local_indices])
            min_available = int(np.min(day_available[top_local_indices]))
            
            daily_stats.append((select_count, day_return, min_available))
    
    return daily_stats


# ==================== 主函数 ====================

def main():
    print_banner()
    
    # 获取设备
    device = DeviceConfig.get_device()
    if device.type == "cuda":
        print(f"  设备: GPU ({torch.cuda.get_device_name()})")
    else:
        print(f"  设备: CPU")
    
    # 列出可用模型
    models = list_available_models()
    if not models:
        print("\n  没有可用的模型，请先训练模型。")
        return
    
    # 选择模型
    model_idx = select_model(models)
    selected_file = models[model_idx]
    model_path = os.path.join(DataConfig.OUTPUT_DIR, selected_file)
    
    print(f"\n  正在加载模型: {selected_file}")
    model, metadata = load_model(model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    if metadata is not None:
        arch = metadata.get('model_arch') or {}
        tp   = metadata.get('train_params') or {}
        es   = metadata.get('eval_stats') or {}
        print(f"  ┌── 模型内嵌元数据 ──────────────────────────────")
        print(f"  │  架构: "
              f"d_model={arch.get('d_model','?')}  "
              f"layers={arch.get('num_layers','?')}  "
              f"heads={arch.get('nhead','?')}  "
              f"ctx={arch.get('context_length','?')}")
        print(f"  │  训练: lr={tp.get('learning_rate','?')}  "
              f"bs={tp.get('batch_size','?')}  "
              f"loss={tp.get('loss_type','?')}")
        print(f"  │  评估: Top{es.get('top_k','?')}%收益={es.get('top_return',0)*100:+.2f}%  "
              f"AUC={es.get('auc',0):.4f}  "
              f"Ep={es.get('epoch','?')}")
        print(f"  └───────────────────────────────────────────────")
    else:
        print(f"  (旧格式模型，无内嵌元数据，使用当前 config.py 参数)")

    print(f"  ✓ 模型加载成功")
    
    # 加载数据并评估
    print(f"\n  正在加载数据集...")

    # ========== 特征归一化器配置 ==========
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        print(f"\n  [特征归一化] 正在加载归一化器...")
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
        print(f"  [特征归一化] ✓ 已启用")
    else:
        print(f"\n  ⚠ 错误: 归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")
        print(f"  请先运行: python data.py")
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print(f"\n  [市场Token] 正在加载大盘数据...")
    gdm = GlobalDataManager.get_instance()
    try:
        gdm.load_market_data()
        print(f"  [市场Token] ✓ 大盘数据加载成功")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"大盘数据加载失败: {e}。请确保 {DataConfig.MARKET_DATA_FILE} 文件存在。")

    stats = run_evaluation(model, test_stock_info, device, feature_normalizer)
    threshold = stats['top_threshold']
    
    print()
    while True:
        choice = input("  是否进入选股模式？(y/n): ").strip().lower()
        if choice in ('y', 'yes', ''):
            break
        elif choice in ('n', 'no', 'q'):
            print("  已退出。")
            return
        print("  请输入 y 或 n")
    
    results = run_stock_selection(model, threshold, device, feature_normalizer)
    
    recent_stats = calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=DataConfig.TOP_N_PER_DAY, threshold=threshold, feature_normalizer=feature_normalizer)
    if recent_stats:
        print_recent_days_chart(recent_stats, last_n=10)
    
    # 询问是否使用自定义阈值重新选股
    while True:
        print()
        choice = input("  输入自定义阈值重新筛选（直接回车退出）: ").strip()
        if not choice:
            break
        try:
            custom_threshold = float(choice)
            if 0 <= custom_threshold <= 1:
                print(f"\n  使用自定义阈值: {custom_threshold:.10f}")
                
                # 重新标记
                threshold_idx = 0
                for i, (code, score, date, close, change) in enumerate(results):
                    if score < custom_threshold:
                        threshold_idx = i
                        break
                else:
                    threshold_idx = len(results)
                
                print(f"  超过阈值: {threshold_idx} 只")
                if threshold_idx > 0:
                    print(f"\n  推荐关注:")
                    for i in range(min(threshold_idx, 20)):
                        code, score, date, close, change = results[i]
                        marker = "┈阈值┈" if i == threshold_idx - 1 else "  ★  "
                        print(f"    {i+1:>3}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%  {marker}")
                    if threshold_idx > 20:
                        print(f"    ... 还有 {threshold_idx - 20} 只")
            else:
                print("  ✗ 阈值应在 0 到 1 之间")
        except ValueError:
            print("  ✗ 无效输入")

if __name__ == "__main__":
    main()
