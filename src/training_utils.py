'''
训练工具模块

提供训练相关的工具类和函数，供其他训练脚本导入使用：
- WarmupScheduler: 学习率预热调度器
- GradientMonitor: 梯度监控器
- DynamicWeightedBCE: 动态加权BCE损失函数
- TaskAlignedLoss: 任务对齐损失函数
- EarlyStopping: 早停机制
- evaluate_model: 模型评估函数
- calculate_test_loss: 计算测试集损失
- generate_pseudo_labels: 伪标签生成
- save_model_with_metadata: 带元数据的模型保存
- print_dispersion_sparkline: 预测值分布可视化
- create_optimizer_from_config: 根据配置创建优化器
- create_scheduler_from_config: 根据配置创建学习率调度器
'''

import os,torch,torch.nn as nn,numpy as np
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import DataConfig,LossConfig,TrainingConfig
import torch.optim as optim

class WarmupScheduler:
    """
    学习率预热调度器
    在前几轮训练中，学习率从很小的值逐步增加到目标学习率
    这有助于模型在训练初期更稳定地收敛
    """
    def __init__(self, optimizer, warmup_epochs, target_lr, start_lr=None):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热轮数
            target_lr: 目标学习率（预热结束后的学习率）
            start_lr: 预热起始学习率，如果为None则使用target_lr的1/100
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.start_lr = start_lr if start_lr is not None else target_lr / 100
        self.current_epoch = 0
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr
    
    def step(self, epoch=None):
        """
        更新学习率
        Args:
            epoch: 当前轮数，如果为None则使用内部计数器
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.start_lr + (self.target_lr - self.start_lr) * ((self.current_epoch + 1) / self.warmup_epochs)
        else:
            lr = self.target_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率（兼容PyTorch调度器接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def is_warmup_phase(self):
        """判断是否还在预热阶段"""
        return self.current_epoch < self.warmup_epochs

def print_dispersion_sparkline(all_preds, epoch_returns_history=None):
    """
    打印预测值在0-1区间上的分布直方图（终端字符可视化）
    
    Args:
        all_preds: 所有样本的预测值数组
        epoch_returns_history: 历史epoch记录列表（用于显示趋势）
    """
    print(f'  【预测值分布直方图】')
    
    all_preds = np.array(all_preds)
    
    num_bins = 20
    counts, _ = np.histogram(all_preds, bins=num_bins, range=(0, 1))
    max_count = max(counts) if max(counts) > 0 else 1
    
    chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    hist_line = ""
    for count in counts:
        idx = int(count / max_count * (len(chars) - 1))
        idx = min(max(idx, 0), len(chars) - 1)
        hist_line += chars[idx]
    
    print(f'    0.0  {hist_line}  1.0')
    print(f'         ├────────────────────┤')
    
    std = float(np.std(all_preds))
    mean = float(np.mean(all_preds))
    min_val = float(np.min(all_preds))
    max_val = float(np.max(all_preds))
    pos_ratio = float(np.mean(all_preds >= 0.5)) * 100
    high_conf_ratio = float(np.mean(all_preds >= 0.7)) * 100
    
    print(f'    均值={mean:.3f}, 标准差={std:.4f}, 范围=[{min_val:.3f}, {max_val:.3f}]')
    print(f'    >0.5: {pos_ratio:.1f}%, >0.7: {high_conf_ratio:.1f}%')
    
    if epoch_returns_history and len(epoch_returns_history) >= 2:
        stds = [e.get('dispersion_std', 0) for e in epoch_returns_history]
        returns = [e.get('return', 0) for e in epoch_returns_history]
        
        window_size = min(10, len(stds))
        baseline_std = np.mean(stds[-window_size:-1]) if window_size > 1 else stds[-2]
        baseline_return = np.mean(returns[-window_size:-1]) if window_size > 1 else returns[-2]
        
        std_change = (stds[-1] - baseline_std) / baseline_std * 100 if baseline_std > 1e-6 else 0
        return_change = (returns[-1] - baseline_return) / abs(baseline_return) * 100 if abs(baseline_return) > 1e-6 else 0
        
        if std_change < -20:
            status = "⚠️ 分散度下降"
        elif std_change > 10:
            status = "📈 分散度上升"
        else:
            status = "➡️ 分散度稳定"
        
        print(f'    趋势: {status} ({std_change:+.1f}%) | 收益率变化: ({return_change:+.1f}%)')

class DynamicWeightedBCE(nn.Module):
    """
    动态加权BCE损失函数：按标签桶分配权重
    - 标签1.0固定权重4.0
    - 标签0.6/0.3/0.0按样本数量动态分配权重（样本少=权重高）
    """
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(DynamicWeightedBCE, self).__init__()
        self.reduction = reduction
        
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        
        self.register_buffer('weight_0_6', torch.tensor(1.0))
        self.register_buffer('weight_0_3', torch.tensor(1.0))
        self.register_buffer('weight_0_0', torch.tensor(1.0))
        
    def update_weights(self, targets):
        """
        二分类动态权重：根据正负样本比例动态调整
        targets: [batch_size] 标签 (1.0/0.0)
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.float().cpu().numpy()
        
        count_positive = np.sum(targets >= 0.5)
        count_negative = np.sum(targets < 0.5)
        
        if count_positive > 0 and count_negative > 0:
            neg_weight = float(self.pos_weight) * (count_positive / count_negative)
            self.weight_0_0.fill_(neg_weight)
        elif count_positive == 0:
            self.weight_0_0.fill_(float(self.pos_weight))
        else:
            self.weight_0_0.fill_(0.1)
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, 1] 模型输出的logits
        targets: [batch_size] 真实标签 (1.0/0.0)
        """
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)

        inputs_fp32 = inputs.float()
        targets_fp32 = targets.float()

        loss = F.binary_cross_entropy_with_logits(inputs_fp32, targets_fp32, reduction='none')
        
        pos_weight = self.pos_weight.to(dtype=loss.dtype, device=loss.device)
        neg_weight = self.weight_0_0.to(dtype=loss.dtype, device=loss.device)

        weights = torch.where(targets_fp32 >= 0.5, pos_weight, neg_weight)
        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TaskAlignedLoss(nn.Module):
    """
    任务对齐损失函数：专为Top-K选股任务设计的多目标组合损失
    
    总损失 = L_bce + λ1·L_rank + λ2·L_return + λ3·L_topk
    
    四个组件各司其职：
    1. L_bce:    基础分类能力（DynamicWeightedBCE，保持正负样本均衡）
    2. L_rank:   排序损失（高收益样本的预测分数应高于低收益样本）
    3. L_return: 收益加权损失（收益越高/亏损越大的样本，分类错误代价越高）
    4. L_topk:   头部聚焦损失（只关注模型预测分数最高的那批样本的质量）
    """
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(TaskAlignedLoss, self).__init__()
        self.reduction = reduction
        
        self.bce = DynamicWeightedBCE(pos_weight=pos_weight, reduction=reduction)
        
        self.rank_weight = LossConfig.RANK_LOSS_WEIGHT
        self.return_weight = LossConfig.RETURN_LOSS_WEIGHT
        self.topk_weight = LossConfig.TOPK_LOSS_WEIGHT
        
        self.rank_margin = LossConfig.RANK_MARGIN
        self.rank_num_pairs = LossConfig.RANK_NUM_PAIRS
        
        self.return_alpha = LossConfig.RETURN_ALPHA
        self.return_beta = LossConfig.RETURN_BETA
        self.return_clip = LossConfig.RETURN_CLIP
        
        self.topk_ratio = LossConfig.TOPK_RATIO
    
    def update_weights(self, targets):
        """动态更新BCE组件的正负样本权重"""
        self.bce.update_weights(targets)
    
    def _ranking_loss(self, logits, returns):
        """
        排序损失：确保高收益样本的预测分数高于低收益样本
        """
        batch_size = logits.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        sorted_indices = torch.argsort(returns, descending=True)
        
        half = batch_size // 2
        high_indices = sorted_indices[:half]
        low_indices = sorted_indices[half:]
        
        num_pairs = min(self.rank_num_pairs, half * len(low_indices))
        if num_pairs == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        high_sample = high_indices[torch.randint(0, len(high_indices), (num_pairs,), device=logits.device)]
        low_sample = low_indices[torch.randint(0, len(low_indices), (num_pairs,), device=logits.device)]
        
        score_diff = logits[high_sample] - logits[low_sample]
        loss = F.relu(self.rank_margin - score_diff)
        
        return loss.mean()
    
    def _return_weighted_loss(self, logits, targets, returns):
        """
        收益加权损失：用收益率大小调制BCE的梯度
        """
        clipped_returns = torch.clamp(returns, -self.return_clip, self.return_clip)
        
        bce_per_sample = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), reduction='none'
        )
        
        pos_mask = targets >= 0.5
        neg_loss_mask = (targets < 0.5) & (returns < 0)
        
        weights = torch.ones_like(returns)
        weights[pos_mask] = 1.0 + self.return_alpha * clipped_returns[pos_mask].abs()
        weights[neg_loss_mask] = 1.0 + self.return_beta * clipped_returns[neg_loss_mask].abs()
        
        weights = weights / (weights.mean() + 1e-8)
        
        loss = (bce_per_sample * weights).mean()
        return loss
    
    def _topk_focus_loss(self, logits, targets, returns):
        """
        Top-K聚焦损失：只关注模型预测分数最高的那些样本
        """
        batch_size = logits.size(0)
        k = max(1, int(batch_size * self.topk_ratio))
        
        _, topk_indices = torch.topk(logits.detach(), k)
        
        topk_logits = logits[topk_indices]
        topk_targets = targets[topk_indices]
        topk_returns = returns[topk_indices]
        
        topk_bce = F.binary_cross_entropy_with_logits(
            topk_logits.float(), topk_targets.float(), reduction='none'
        )
        
        clipped_returns = torch.clamp(topk_returns, -self.return_clip, self.return_clip)
        penalty = torch.where(
            topk_returns < 0,
            1.0 + self.return_beta * clipped_returns.abs(),
            torch.ones_like(topk_returns)
        )
        
        loss = (topk_bce * penalty).mean()
        return loss
    
    def forward(self, logits, targets, returns=None):
        """
        前向计算：组合所有子损失
        
        Args:
            logits: [batch_size] 或 [batch_size, 1] 模型原始输出
            targets: [batch_size] 真实标签 (0/1)
            returns: [batch_size] 真实累计收益率（可选，不提供则退化为纯BCE）
        """
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(-1)
        
        loss_bce = self.bce(logits, targets)
        
        if returns is None:
            return loss_bce
        
        if returns.dim() == 2:
            returns = returns.squeeze(-1)
        returns = returns.to(dtype=logits.dtype, device=logits.device)
        
        loss_rank = self._ranking_loss(logits, returns)
        
        loss_return = self._return_weighted_loss(logits, targets, returns)
        
        loss_topk = self._topk_focus_loss(logits, targets, returns)
        
        total_loss = loss_bce + \
                     self.rank_weight * loss_rank + \
                     self.return_weight * loss_return + \
                     self.topk_weight * loss_topk
        
        return total_loss


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns,
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name="", eval_day_indices=None, top_n_per_day=None, eval_daily_returns=None, market_seqs=None):
    """
    模型评估函数
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

    分批处理，减少显存占用

    返回统计字典，包含：
        auc：AUC得分
        top_return：Top1%收益率
        top_count：Top1%样本数
        top_threshold：Top1%最低置信度
        high_conf_count：高置信(>0.7)样本数
        low_conf_count：低置信(<0.2)样本数
        pred_mean：预测均值
        pred_std：预测标准差
        filtered_count：被过滤的涨停样本数（始终为0，因已在生成阶段过滤）
        realistic_stats：实战收益率统计（如果提供了eval_day_indices）
        smart_exit_stats：智能止损策略统计（如果提供了eval_daily_returns）
    """
    model.eval()

    num_samples = len(eval_inputs)
    if num_samples == 0:
        return {
            'auc': 0.5, 'top_return': 0.0, 'top_count': 0, 'top_threshold': 0.0,
            'high_conf_count': 0, 'low_conf_count': 0, 'pred_mean': 0.0, 
            'pred_std': 0.0, 'filtered_count': 0, 'realistic_stats': None, 'smart_exit_stats': None
        }

    all_preds = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_market = torch.tensor(market_seqs[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_preds = torch.sigmoid(model(batch_inputs, batch_market))
            all_preds.append(batch_preds.cpu().numpy().flatten())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(eval_targets)
    all_returns = np.array(eval_cumulative_returns)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    percent = DataConfig.TOP_K
    top_k = max(1, int(len(all_preds) * percent / 100))
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    top_return = np.mean(top_returns)
    top_threshold = all_preds[sorted_indices[top_k - 1]]

    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2

    stats = {
        'auc': auc,
        'top_return': top_return,
        'top_count': top_k,
        'top_threshold': top_threshold,
        'high_conf_count': np.sum(high_conf),
        'low_conf_count': np.sum(low_conf),
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
        'filtered_count': 0,
        'dispersion_std': float(np.std(all_preds)),
        'dispersion_range': float(np.max(all_preds) - np.min(all_preds)),
        'dispersion_iqr': float(np.percentile(all_preds, 75) - np.percentile(all_preds, 25)),
    }

    if eval_day_indices is not None:
        actual_top_n = top_n_per_day if top_n_per_day is not None else DataConfig.TOP_N_PER_DAY
        if actual_top_n == 0:
            actual_top_n = None
        stats['realistic_stats'] = calculate_realistic_return(all_preds, all_returns, eval_day_indices, percent, actual_top_n)
        
        if eval_daily_returns is not None and actual_top_n is not None:
            stats['smart_exit_stats'] = calculate_smart_exit_return(all_preds, eval_daily_returns, eval_day_indices, actual_top_n)
        else:
            stats['smart_exit_stats'] = None
    else:
        stats['realistic_stats'] = None
        stats['smart_exit_stats'] = None

    stats['all_preds'] = all_preds
    return stats


def calculate_realistic_return(all_preds, all_returns, all_day_indices, top_percent=1.0, top_n_per_day=None):
    """
    计算实战收益率（高仿实战）
    
    支持两种模式:
    1. 全局阈值模式（top_n_per_day=None）: 按全局Top%确定阈值，每天选超过阈值的股票
    2. 每日Top N模式（top_n_per_day指定）: 每天选预测分数最高的前N只股票
    """
    unique_days = np.unique(all_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    daily_returns = []
    
    if top_n_per_day is not None:
        for day in unique_days:
            day_mask = all_day_indices == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) == 0:
                daily_stats.append((0, 0.0))
                continue
            
            day_preds = all_preds[day_indices]
            day_returns = all_returns[day_indices]
            
            sorted_local_indices = np.argsort(day_preds)[::-1]
            select_count = min(top_n_per_day, len(day_indices))
            top_local_indices = sorted_local_indices[:select_count]
            
            day_return = np.mean(day_returns[top_local_indices])
            daily_returns.append(day_return)
            daily_stats.append((select_count, day_return))
        
        threshold = None
    else:
        top_k = max(1, int(len(all_preds) * top_percent / 100))
        sorted_indices = np.argsort(all_preds)[::-1]
        threshold = all_preds[sorted_indices[top_k - 1]]
        
        above_threshold_mask = all_preds > threshold
        max_select = DataConfig.MAX_SELECT_PER_DAY
        
        for day in unique_days:
            day_mask = all_day_indices == day
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
                daily_returns.append(day_return)
                daily_stats.append((count, day_return))
            else:
                daily_stats.append((0, 0.0))
    
    if len(daily_returns) > 0:
        avg_realistic_return = np.mean(daily_returns)
        cumulative_return = np.sum(daily_returns)
    else:
        avg_realistic_return = 0.0
        cumulative_return = 0.0
    
    return {
        'threshold': threshold,
        'daily_stats': daily_stats,
        'cumulative_return': cumulative_return,
        'valid_days': len(daily_returns),
        'avg_realistic_return': avg_realistic_return,
        'mode': 'top_n_per_day' if top_n_per_day else 'global_threshold'
    }


def calculate_smart_exit_return(all_preds, all_daily_returns, all_day_indices, top_n_per_day=4,
                                 stop_loss_day1=-0.05, stop_loss_cum=-0.05, take_profit=0.08,
                                 sell_at_day2_close=True):
    """
    智能止损策略收益率计算（A股T+1规则）
    """
    unique_days = np.unique(all_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    daily_returns = []
    
    total_trades = 0
    stop_loss_day1_count = 0
    stop_loss_cum_count = 0
    take_profit_count = 0
    normal_exit_count = 0
    
    for day in unique_days:
        day_mask = all_day_indices == day
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0:
            daily_stats.append((0, 0.0, 'none'))
            continue
        
        day_preds = all_preds[day_indices]
        day_daily_returns = [all_daily_returns[i] for i in day_indices]
        
        sorted_local_indices = np.argsort(day_preds)[::-1]
        select_count = min(top_n_per_day, len(day_indices))
        top_local_indices = sorted_local_indices[:select_count]
        
        day_trade_returns = []
        day_exit_types = {'stop_day1': 0, 'stop_cum': 0, 'profit': 0, 'normal': 0, 'partial': 0}
        
        for idx in top_local_indices:
            daily_ret = day_daily_returns[idx]
            if daily_ret is None or len(daily_ret) == 0:
                continue

            available = len(daily_ret)
            r1 = daily_ret[0]
            r2 = daily_ret[1] if available >= 2 else 0.0
            r3 = daily_ret[2] if available >= 3 else 0.0
            has_day2 = available >= 2
            has_day3 = available >= 3

            total_trades += 1
            
            if r1 < stop_loss_day1:
                if has_day2:
                    final_ret = r1 + (r2 if sell_at_day2_close else 0.0)
                    stop_loss_day1_count += 1
                    day_exit_types['stop_day1'] += 1
                else:
                    final_ret = r1
                    day_exit_types['partial'] += 1
            elif has_day2 and (r1 + r2) < stop_loss_cum:
                final_ret = r1 + r2
                stop_loss_cum_count += 1
                day_exit_types['stop_cum'] += 1
            elif has_day3 and (r1 + r2 + r3) >= take_profit:
                final_ret = r1 + r2 + r3
                take_profit_count += 1
                day_exit_types['profit'] += 1
            elif has_day3:
                final_ret = r1 + r2 + r3
                normal_exit_count += 1
                day_exit_types['normal'] += 1
            elif has_day2:
                final_ret = r1 + r2
                day_exit_types['partial'] += 1
            else:
                final_ret = r1
                day_exit_types['partial'] += 1
            
            day_trade_returns.append(final_ret)
        
        if len(day_trade_returns) == 0:
            daily_stats.append((0, 0.0, 'none'))
            continue

        avg_day_return = np.mean(day_trade_returns)
        daily_returns.append(avg_day_return)
        
        exit_type = max(day_exit_types, key=day_exit_types.get)
        daily_stats.append((len(day_trade_returns), avg_day_return, exit_type))
    
    if len(daily_returns) > 0:
        avg_realistic_return = np.mean(daily_returns)
        cumulative_return = np.sum(daily_returns)
    else:
        avg_realistic_return = 0.0
        cumulative_return = 0.0
    
    return {
        'daily_stats': daily_stats,
        'cumulative_return': cumulative_return,
        'valid_days': len(daily_returns),
        'avg_realistic_return': avg_realistic_return,
        'total_trades': total_trades,
        'stop_loss_day1_count': stop_loss_day1_count,
        'stop_loss_cum_count': stop_loss_cum_count,
        'take_profit_count': take_profit_count,
        'normal_exit_count': normal_exit_count,
        'stop_loss_day1_ratio': stop_loss_day1_count / total_trades if total_trades > 0 else 0,
        'stop_loss_cum_ratio': stop_loss_cum_count / total_trades if total_trades > 0 else 0,
        'take_profit_ratio': take_profit_count / total_trades if total_trades > 0 else 0,
        'strategy': f'smart_exit(stop_day1={stop_loss_day1*100:.1f}%, stop_cum={stop_loss_cum*100:.1f}%, profit={take_profit*100:.1f}%)'
    }


def generate_pseudo_labels(pred_scores, original_targets,
                           pseudo_pos_ratio=0.01,
                           pseudo_neg_ratio=0.05):
    """
    统一的伪标签生成函数（按数量取Top-K%方式）

    核心思想：
    - 按预测分数排序，取前 pseudo_pos_ratio 比例的样本 → 强制标签=1.0（伪正）
    - 按预测分数排序，取倒数 pseudo_neg_ratio 比例的样本 → 强制标签=0.0（伪负）
    - 其余样本保持原始标签不变
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.float().detach().cpu().numpy()
    if isinstance(original_targets, torch.Tensor):
        original_targets = original_targets.float().detach().cpu().numpy()

    pred_scores = np.asarray(pred_scores).flatten()
    original_targets = np.asarray(original_targets).copy()

    if len(pred_scores) == 0:
        stats = {
            'pseudo_pos_count': 0,
            'pseudo_neg_count': 0,
            'unchanged_count': 0,
            'threshold_pos': 0.0,
            'threshold_neg': 0.0,
        }
        return original_targets, stats

    k_pos = max(1, int(len(pred_scores) * pseudo_pos_ratio))
    k_pos = min(k_pos, len(pred_scores))
    threshold_pos = np.sort(pred_scores)[-k_pos]

    k_neg = max(1, int(len(pred_scores) * pseudo_neg_ratio))
    k_neg = min(k_neg, len(pred_scores))
    threshold_neg = np.sort(pred_scores)[k_neg - 1]

    pseudo_targets = original_targets.copy()

    high_mask = pred_scores >= threshold_pos
    pseudo_targets[high_mask] = 1.0

    low_mask = pred_scores <= threshold_neg
    pseudo_targets[low_mask] = 0.0

    stats = {
        'pseudo_pos_count': int(np.sum(high_mask)),
        'pseudo_neg_count': int(np.sum(low_mask)),
        'unchanged_count': int(len(pred_scores) - np.sum(high_mask) - np.sum(low_mask)),
        'threshold_pos': float(threshold_pos),
        'threshold_neg': float(threshold_neg),
    }

    return pseudo_targets, stats


def save_model_with_metadata(model_state_dict, top_return, top_threshold, auc,
                             epoch, model_prefix="model", extra_info="",
                             output_dir=DataConfig.OUTPUT_DIR):
    """
    通用的模型保存函数，带详细元数据

    保存格式为包含以下键的字典：
    - 'model_arch': 模型架构参数（用于 run.py 自动重建正确大小的模型）
    - 'train_params': 训练超参数快照
    - 'eval_stats': 评估指标
    - 'state_dict': 模型权重
    """
    from config import ModelConfig, TrainingConfig, LossConfig
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    return_str = f"{top_return*100:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
    thr_str = f"{top_threshold:.3f}".replace('.', '_')
    auc_str = f"{auc:.4f}".replace('.', '_')

    if extra_info:
        filename = f"{model_prefix}_top{DataConfig.TOP_K}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{extra_info}_{timestamp}.pth"
    else:
        filename = f"{model_prefix}_top{DataConfig.TOP_K}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{timestamp}.pth"

    checkpoint = {
        'model_arch': {
            'model_type':       ModelConfig.MODEL_TYPE,
            'input_dim':        ModelConfig.INPUT_DIM,
            'd_model':          ModelConfig.D_MODEL,
            'ffn_expand_ratio': ModelConfig.FFN_EXPAND_RATIO,
            'nhead':            ModelConfig.NHEAD,
            'num_layers':       ModelConfig.NUM_LAYERS,
            'output_dim':       ModelConfig.OUTPUT_DIM,
            'dropout_rate':     ModelConfig.DROPOUT_RATE,
            'attention_dropout':ModelConfig.ATTENTION_DROPOUT,
            'context_length':   DataConfig.CONTEXT_LENGTH,
        },
        'train_params': {
            'epochs':           TrainingConfig.EPOCHS,
            'learning_rate':    TrainingConfig.LEARNING_RATE,
            'batch_size':       TrainingConfig.BATCH_SIZE,
            'use_adamw':        TrainingConfig.USE_ADAMW,
            'use_mano':         TrainingConfig.USE_MANO,
            'weight_decay':     TrainingConfig.WEIGHT_DECAY,
            'loss_type':        LossConfig.LOSS_TYPE,
            'pos_weight':       LossConfig.POS_WEIGHT,
        },
        'eval_stats': {
            'top_return':   float(top_return),
            'top_threshold':float(top_threshold),
            'auc':          float(auc),
            'epoch':        int(epoch),
            'top_k':        DataConfig.TOP_K,
        },
        'state_dict': model_state_dict,
    }

    save_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, save_path)

    return save_path

def calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=1024, market_seqs=None):
    """
    计算测试集损失（官方标准：除以样本数）

    优化版本：
    - 权重在训练开始时已设置，此处直接使用
    - 支持大batch_size，提高GPU利用率
    """
    model.eval()
    total_loss = 0.0
    num_samples = len(eval_inputs)
    
    if num_samples == 0:
        return 0.0
    
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx],
                                       dtype=torch.float32).to(device)
            batch_targets = torch.tensor(eval_targets[start_idx:end_idx],
                                        dtype=torch.float32).to(device)
            batch_market = torch.tensor(market_seqs[start_idx:end_idx],
                                        dtype=torch.float32).to(device)

            outputs = model(batch_inputs, batch_market)
            loss = criterion(outputs.squeeze(-1), batch_targets)
            total_loss += loss.item() * (end_idx - start_idx)

    return total_loss / num_samples


class EarlyStopping:
    """
    早停机制类

    监控指标：
    - avg_loss: 平均损失（越低越好）
    - top_return: Top1%收益率（越高越好）

    任意一个指标改善即重置计数器
    """
    def __init__(self, patience=10):
        """
        Args:
            patience: 容忍无改善的轮数
        """
        self.patience = patience
        self.no_improve_count = 0

        self.best_loss = float('inf')
        self.best_return = -float('inf')
        self.best_return_auc = 0.0
        self.best_return_threshold = 0.0

    def check_improve(self, avg_loss=None, top_return=None, auc=None, threshold=None):
        """
        检查是否有改善

        Args:
            avg_loss: 平均损失
            top_return: Top1%收益率
            auc: AUC得分（仅当收益率改善时更新）
            threshold: Top阈值（仅当收益率改善时更新）

        Returns:
            improved: 是否有改善
            reason: 改善原因字符串
        """
        improved = False
        reasons = []

        if avg_loss is not None and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            improved = True
            reasons.append(f'损失改善: {avg_loss:.4f}')

        if top_return is not None and top_return > self.best_return:
            self.best_return = top_return
            improved = True
            if auc is not None:
                self.best_return_auc = auc
            if threshold is not None:
                self.best_return_threshold = threshold
            reasons.append(f'收益率改善: {top_return*100:+.2f}%')

        if improved:
            self.no_improve_count = 0
            return True, ' & '.join(reasons)
        else:
            self.no_improve_count += 1
            return False, None

    def should_stop(self):
        """是否应该停止训练"""
        return self.no_improve_count >= self.patience

    def get_progress(self):
        """获取当前进度"""
        return self.no_improve_count, self.patience

    def get_best_metrics(self):
        """获取最佳指标"""
        return {
            'best_loss': self.best_loss,
            'best_return': self.best_return,
            'best_return_auc': self.best_return_auc,
            'best_return_threshold': self.best_return_threshold
        }

class GradientMonitor:
    """
    梯度监控器：检测梯度爆炸和梯度消失
    在每个batch的backward后收集各层梯度统计信息
    """
    def __init__(self):
        self.grad_stats = {}
        self.hooks = []

    def _create_hook(self, name):
        def hook(grad):
            if grad is None:
                return grad

            grad_flat = grad.data.abs().flatten()

            grad_norm = grad_flat.norm(2).float().item()
            grad_max = grad_flat.max().float().item()
            grad_mean = grad_flat.mean().float().item()
            has_nan = torch.isnan(grad.data).any().item()
            has_inf = torch.isinf(grad.data).any().item()

            if name not in self.grad_stats:
                self.grad_stats[name] = {
                    'norm': [],
                    'max': [],
                    'mean': [],
                    'nan_count': 0,
                    'inf_count': 0,
                    'zero_count': 0
                }

            stats = self.grad_stats[name]
            stats['norm'].append(grad_norm)
            stats['max'].append(grad_max)
            stats['mean'].append(grad_mean)

            if len(stats['norm']) > 100:
                stats['norm'].pop(0)
                stats['max'].pop(0)
                stats['mean'].pop(0)

            if has_nan:
                stats['nan_count'] += 1
            if has_inf:
                stats['inf_count'] += 1
            if grad_norm < 1e-8:
                stats['zero_count'] += 1

            return grad
        return hook

    def register_hooks(self, model):
        """为模型所有参数注册梯度hook"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._create_hook(name))
                self.hooks.append(hook)
        print(f"  已为 {len(self.hooks)} 个参数注册梯度监控hook")

    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_epoch_summary(self):
        """获取当前epoch的梯度统计摘要"""
        summary = {}
        for name, stats in self.grad_stats.items():
            if stats['norm']:
                summary[name] = {
                    'avg_norm': np.mean(stats['norm']),
                    'max_norm': np.max(stats['norm']),
                    'avg_max': np.mean(stats['max']),
                    'avg_mean': np.mean(stats['mean']),
                    'nan_count': stats['nan_count'],
                    'inf_count': stats['inf_count'],
                    'zero_count': stats['zero_count'],
                    'total_batches': len(stats['norm'])
                }
        return summary

    def reset(self):
        """重置统计信息（新epoch开始时调用）"""
        self.grad_stats.clear()

    def diagnose(self):
        """
        诊断梯度问题，返回报告
        返回: (爆炸层列表, 消失层列表, 异常层列表)
        """
        exploding = []
        vanishing = []
        abnormal = []

        summary = self.get_epoch_summary()

        for name, stats in summary.items():
            if stats['avg_norm'] > 10 or stats['max_norm'] > 100:
                exploding.append((name, stats))

            elif stats['avg_norm'] < 1e-5:
                vanishing.append((name, stats))

            if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                abnormal.append((name, stats))

        return exploding, vanishing, abnormal


def create_optimizer_from_config(model, lr=None):
    """
    根据配置创建优化器（统一入口）

    Args:
        model: PyTorch模型
        lr: 学习率，如果为None则使用 TrainingConfig.LEARNING_RATE

    Returns:
        optimizer: 创建的优化器实例
    """
    from optimizers import create_optimizer

    actual_lr = lr if lr is not None else TrainingConfig.LEARNING_RATE

    if TrainingConfig.USE_MANO:
        optimizer = create_optimizer(
            model,
            optimizer_type='mano',
            lr=actual_lr,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            betas=TrainingConfig.MANO_ADAMW_BETAS,
            nesterov=TrainingConfig.MANO_NESTEROV,
            dual_dim_projection=TrainingConfig.MANO_DUAL_DIM_PROJECTION
        )
    elif TrainingConfig.USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=actual_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)

    return optimizer


def create_scheduler_from_config(optimizer, epochs, lr=None, eta_min=None, warmup_start_lr=None):
    """
    根据配置创建学习率调度器（预热 + 余弦退火）

    Args:
        optimizer: PyTorch优化器
        epochs: 总训练轮数
        lr: 目标学习率，如果为None则使用 TrainingConfig.LEARNING_RATE
        eta_min: 余弦退火最小学习率，如果为None则使用 TrainingConfig.COSINE_ETA_MIN
        warmup_start_lr: 预热起始学习率，如果为None则使用 TrainingConfig.WARMUP_START_LR

    Returns:
        tuple: (warmup_scheduler, main_scheduler, warmup_epochs)
            - warmup_scheduler: 预热调度器
            - main_scheduler: 余弦退火调度器
            - warmup_epochs: 预热轮数（用于打印状态）
    """
    actual_lr = lr if lr is not None else TrainingConfig.LEARNING_RATE
    actual_eta_min = eta_min if eta_min is not None else TrainingConfig.COSINE_ETA_MIN
    actual_warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else TrainingConfig.WARMUP_START_LR

    warmup_epochs = max(1, int(epochs * TrainingConfig.WARMUP_RATIO))

    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        target_lr=actual_lr,
        start_lr=actual_warmup_start_lr
    )

    # 确保余弦退火的 T_max 至少为1，防止 epochs 过小时崩溃
    total_main_epochs = max(1, epochs - warmup_epochs)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_main_epochs,
        eta_min=actual_eta_min
    )

    return warmup_scheduler, main_scheduler, warmup_epochs
