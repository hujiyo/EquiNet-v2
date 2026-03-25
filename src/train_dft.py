'''
DFT模型训练脚本（自引导式直接微调）

核心思想：
- 从out/目录加载已有模型权重
- 使用自引导DFT（Self-Guided Direct Fine-Tuning）机制继续微调
- 只训练这一个模型

自引导DFT权重机制（基于模型自身的预测排名分位数）：
- 将模型对batch内所有样本的预测值排序，得到每个样本的分位数 rank ∈ [0, 1]
- 预测排名最高的样本（rank→1）：模型已经很确定是好的 → 低权值（已学会，无需多学）
- 预测排名最低的样本（rank→0）：模型已经很确定是差的 → 低权值（已学会，无需多学）
- 预测排名在中间的样本（rank≈0.5）：模型还没分清楚的 → 高权值（最有学习价值）
- 权重公式：w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
  这是一个开口朝下的抛物线，在 rank=0.5 处取最大值 w_max，在 rank=0/1 处取最小值 w_min
- 权重随训练动态演化：随着模型学习进步，"不确定"的样本会变化，权重自然跟着调整
'''

import os
import sys

# 确保能正确导入其他模块（无论从哪里运行）
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, numpy as np
import argparse
import copy
import random
import csv
from datetime import datetime
from src.config import (TrainingConfig,DataConfig,DeviceConfig,print_config_summary,LossConfig)

from src.model import create_model

from src.data import (
    load_and_preprocess_data,
    create_sampler, sample_with_pools,
    create_fixed_evaluation_dataset,
    GlobalDataManager
)

from src.training_utils import (
    WarmupScheduler,
    evaluate_model,
    save_model_with_metadata,
    DynamicWeightedBCE,
    TaskAlignedLoss,
    EarlyStopping,
    calculate_test_loss,
    print_dispersion_sparkline,
    create_optimizer_from_config,
    create_scheduler_from_config
)


# ==================== 自引导DFT核心机制 ====================

def compute_dft_weights(pred, w_min=0.1, w_max=1.0):
    """
    根据预测值在batch内的排名分位数计算样本权重。
    rank=0.5(中间排名)权重最高，rank=0/1(头尾)权重最低。
    抛物线公式：w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
    """
    pred_squeezed = pred.squeeze().detach()
    n = pred_squeezed.shape[0]
    ranks = pred_squeezed.argsort().argsort().float() / (n - 1) if n > 1 else torch.full_like(pred_squeezed, 0.5)
    weights = w_min + (w_max - w_min) * 4.0 * ranks * (1.0 - ranks)
    return weights


def weighted_bce_with_dft(logits, targets, dft_weights, bce_criterion):
    """
    DFT加权BCE损失：先通过DynamicWeightedBCE获取含正负样本动态权重的per-sample损失，
    再叠加DFT自引导权重。
    
    Args:
        logits: 模型输出logits
        targets: 真实标签
        dft_weights: DFT自引导权重 (per-sample)
        bce_criterion: DynamicWeightedBCE实例 (reduction='none')
    """
    # bce_criterion 内部已经处理了正负样本动态权重
    per_sample_loss = bce_criterion(logits, targets)  # [batch_size], 已含正负样本权重
    dft_w = dft_weights.to(dtype=per_sample_loss.dtype)
    return (per_sample_loss * dft_w).mean()


def weighted_task_aligned_loss(logits, targets, returns, dft_weights, task_loss_fn, dft_bce_none):
    """
    DFT加权版TaskAlignedLoss：
    - BCE分量：通过独立的DynamicWeightedBCE(reduction='none')实例获取
      含正负样本动态权重的per-sample损失，再叠加DFT自引导权重
    - 排序/收益/TopK分量：直接复用TaskAlignedLoss的子组件，保持原有逻辑
    
    Args:
        dft_bce_none: 独立的DynamicWeightedBCE(reduction='none')实例，
                      与 task_loss_fn 内部的BCE共享相同的权重更新逻辑，
                      但返回 per-sample 损失而非标量
    """
    if logits.dim() == 2 and logits.size(1) == 1:
        logits = logits.squeeze(-1)

    # BCE分量：通过独立的 reduction='none' 实例获取per-sample损失
    bce_per_sample = dft_bce_none(logits, targets)  # [batch_size], 已含正负样本动态权重
    
    # 叠加DFT自引导权重
    dft_w = dft_weights.to(dtype=bce_per_sample.dtype)
    weighted_bce = (bce_per_sample * dft_w).mean()

    if returns is None:
        return weighted_bce

    if returns.dim() == 2:
        returns = returns.squeeze(-1)
    returns = returns.to(dtype=logits.dtype, device=logits.device)

    # 排序损失、收益加权损失、Top-K聚焦损失直接复用TaskAlignedLoss的子组件
    loss_rank = task_loss_fn._ranking_loss(logits, returns)
    loss_return = task_loss_fn._return_weighted_loss(logits, targets, returns)
    loss_topk = task_loss_fn._topk_focus_loss(logits, targets, returns)

    total_loss = weighted_bce + \
                 task_loss_fn.rank_weight * loss_rank + \
                 task_loss_fn.return_weight * loss_return + \
                 task_loss_fn.topk_weight * loss_topk

    return total_loss


# ==================== DFT训练主函数 ====================

def train_dft_model(model, train_stock_info, test_stock_info,
                    epochs=TrainingConfig.EPOCHS,
                    learning_rate=TrainingConfig.LEARNING_RATE,
                    device=None,
                    batch_size=TrainingConfig.BATCH_SIZE,
                    batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
                    dft_w_min=0.1,
                    dft_w_max=1.0,
                    seed=DataConfig.RANDOM_SEED):
    """
    DFT模型训练函数（自引导模式）

    训练策略：
    - 加载已有模型，使用自引导DFT继续微调
    - 样本权重基于模型自身的预测排名分位数：中间排名高权值，头尾低权值
    - w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
    - 支持TaskAlignedLoss：DFT权重调制基础BCE分量，排序/收益/TopK子损失照常计算
    """
    print("\n" + "="*60)
    print("DFT自引导微调训练")
    print("="*60)
    print(f"训练策略：")
    print(f"  - 加载已有模型，使用自引导DFT继续微调")
    print(f"  - 样本权重基于自身预测排名分位数")
    print(f"  - 权重范围: [{dft_w_min}, {dft_w_max}]")
    print(f"  - 中间排名(rank≈0.5)权重最高，头尾(rank→0/1)权重最低")
    print("="*60 + "\n")

    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 创建评估数据集
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_market_seqs = create_fixed_evaluation_dataset(test_stock_info)

    # 初始模型评估
    stats_init = evaluate_model(
        model, eval_inputs, eval_targets, eval_cumulative_returns,
        device, model_name="初始模型",
        eval_day_indices=eval_day_indices,
        eval_daily_returns=eval_daily_returns,
        market_seqs=eval_market_seqs
    )
    print(f"初始模型评估: AUC={stats_init['auc']:.4f}, Top{DataConfig.TOP_K}%收益={stats_init['top_return']*100:+.2f}%")
    if stats_init['realistic_stats'] is not None:
        rs = stats_init['realistic_stats']
        print(f"              【实战收益率】平均: {rs['avg_realistic_return']*100:.1f}%")

    # DFT使用原学习率的20%
    dft_lr = learning_rate * 0.2
    print(f"DFT学习率: {dft_lr:.6f} (原学习率的20%)")

    # 创建优化器
    optimizer = create_optimizer_from_config(model, lr=dft_lr)

    # 创建学习率调度器（DFT使用自定义的起始学习率和最小学习率）
    warmup_scheduler, main_scheduler, warmup_epochs = create_scheduler_from_config(
        optimizer,
        epochs=epochs,
        lr=dft_lr,
        eta_min=dft_lr * 0.01,
        warmup_start_lr=dft_lr * 0.1
    )

    # 损失函数选择
    use_task_aligned = LossConfig.use_task_aligned()

    if use_task_aligned:
        print("损失函数: DFT加权TaskAlignedLoss (DFT权重调制BCE + 排序损失 + 收益加权 + Top-K聚焦)")
        print(f"  权重: rank={LossConfig.RANK_LOSS_WEIGHT}, return={LossConfig.RETURN_LOSS_WEIGHT}, topk={LossConfig.TOPK_LOSS_WEIGHT}")
        task_loss_fn = TaskAlignedLoss(pos_weight=LossConfig.POS_WEIGHT, reduction='mean').to(device)
        # 独立的 reduction='none' BCE 实例，用于 DFT 加权时获取 per-sample 损失
        # 与 task_loss_fn.bce 共享相同的 pos_weight，但不修改 task_loss_fn 内部状态
        dft_train_bce = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='none').to(device)
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')

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
        print(f"测试集权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={test_neg_weight:.4f} (正负比例={test_pos_count}:{test_neg_count})")
    elif LossConfig.use_dynamic_bce():
        print("损失函数: DFT加权DynamicWeightedBCE (正负样本动态权重 × DFT自引导权重)")
        task_loss_fn = None
        # 训练用：reduction='none' 以获取per-sample损失，再叠加DFT权重
        dft_train_bce = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='none').to(device)
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')

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
        print(f"测试集权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={test_neg_weight:.4f} (正负比例={test_pos_count}:{test_neg_count})")
    else:
        print("损失函数: DFT加权BCE (标准BCE × DFT自引导权重)")
        task_loss_fn = None
        dft_train_bce = None  # 标准BCE路径不需要DynamicWeightedBCE
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 按收益率保存的最佳模型
    best_return = stats_init['top_return']
    best_auc = stats_init['auc']
    best_threshold = stats_init['top_threshold']
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # 按loss保存的最佳模型（条件：epoch >= 100, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%）
    best_loss = float('inf')
    best_loss_epoch = 0
    best_model_by_loss = None
    best_return_at_best_loss = 0.0
    best_auc_at_best_loss = 0.0
    best_threshold_at_best_loss = 0.0
    best_realistic_return_at_best_loss = 0.0

    # 按实战收益率保存的最佳模型（第100轮后）
    best_realistic_return = -float('inf')
    best_realistic_return_epoch = 0
    best_model_by_realistic_return = None
    best_return_at_best_realistic = 0.0
    best_auc_at_best_realistic = 0.0
    best_threshold_at_best_realistic = 0.0
    best_realistic_return_value_at_best = 0.0

    # 早停机制
    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    # 创建采样器
    sampler = create_sampler(train_stock_info)
    train_rng = random.Random(seed)

    # 记录每轮收益率
    epoch_returns = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_samples = 0

        # 学习率更新
        if warmup_scheduler.is_warmup_phase():
            current_lr = warmup_scheduler.step(epoch)
            lr_status = f"预热阶段 ({epoch + 1}/{warmup_epochs})"
        else:
            current_lr = main_scheduler.get_last_lr()[0]
            lr_status = "DFT微调"

        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status})')

        # 采样训练数据（包含cumulative_returns以支持TaskAlignedLoss）
        epoch_inputs, epoch_targets, epoch_cum_returns, epoch_market_seqs = sample_with_pools(
            sampler, train_stock_info, batch_size, batches_per_epoch, train_rng
        )

        # 打印循环统计
        looped_count, total_loops = sampler.get_loop_stats()
        print(f"  [循环统计] 已循环股票: {looped_count}/{len(train_stock_info)}, 总循环次数: {total_loops}")

        # 打印标签分布
        count_positive = np.sum(epoch_targets >= 0.9)
        count_boundary = np.sum((epoch_targets > 0.1) & (epoch_targets < 0.9))
        count_negative = np.sum(epoch_targets <= 0.1)
        total_count = len(epoch_targets)
        print(f'  标签分布: 上涨={count_positive}({count_positive/total_count:.1%}), 边界={count_boundary}({count_boundary/total_count:.1%}), 不涨={count_negative}({count_negative/total_count:.1%})')

        # 转换为tensor
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.float32).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.float32).to(device)
        epoch_returns_tensor = torch.tensor(epoch_cum_returns, dtype=torch.float32).to(device)
        epoch_market_tensor = torch.tensor(epoch_market_seqs, dtype=torch.float32).to(device) if epoch_market_seqs is not None else None

        # 计算实际可用的batch数量
        actual_batches = len(epoch_inputs_tensor) // batch_size
        if actual_batches < batches_per_epoch:
            print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

        for step in range(actual_batches):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size

            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            batch_returns = epoch_returns_tensor[start_idx:end_idx]
            batch_market = epoch_market_tensor[start_idx:end_idx] if epoch_market_tensor is not None else None

            optimizer.zero_grad()

            output = model(batch_inputs, batch_market)

            # 计算DFT自引导权重
            with torch.no_grad():
                pred_prob = torch.sigmoid(output)
                dft_weights = compute_dft_weights(pred_prob, w_min=dft_w_min, w_max=dft_w_max)

            # 根据损失函数类型选择计算方式
            if use_task_aligned:
                # 刷新正负样本动态权重（task_loss_fn内部BCE + 独立的dft_train_bce同步更新）
                task_loss_fn.update_weights(batch_targets)
                dft_train_bce.update_weights(batch_targets)
                loss = weighted_task_aligned_loss(
                    output, batch_targets, batch_returns, dft_weights, task_loss_fn, dft_train_bce
                )
            elif dft_train_bce is not None:
                # DynamicWeightedBCE路径：刷新正负样本动态权重，再叠加DFT权重
                dft_train_bce.update_weights(batch_targets)
                loss = weighted_bce_with_dft(output, batch_targets, dft_weights, dft_train_bce)
            else:
                # 标准BCE路径：裸BCE × DFT权重
                logits = output.squeeze(-1)
                bce_per_sample = F.binary_cross_entropy_with_logits(
                    logits.float(), batch_targets.float(), reduction='none'
                )
                dft_w = dft_weights.to(dtype=bce_per_sample.dtype)
                loss = (bce_per_sample * dft_w).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item() * (end_idx - start_idx)
            total_samples += (end_idx - start_idx)

            progress = (step + 1) / actual_batches * 100
            avg_loss = total_loss / total_samples
            print(f'\r  训练进度: {progress:.1f}%, Loss(DFT): {avg_loss:.4f}', end='', flush=True)

        print()

        # 清理内存
        del epoch_inputs_tensor, epoch_targets_tensor, epoch_returns_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新学习率
        if not warmup_scheduler.is_warmup_phase():
            main_scheduler.step()

        # 评估模型
        stats = evaluate_model(
            model, eval_inputs, eval_targets, eval_cumulative_returns,
            device, model_name="DFT",
            eval_day_indices=eval_day_indices,
            eval_daily_returns=eval_daily_returns,
            market_seqs=eval_market_seqs
        )

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device, market_seqs=eval_market_seqs)

        print(f'  [DFT模型] 训练损失: {avg_loss:.4f}, 测试损失: {test_loss:.4f}, AUC: {stats["auc"]:.4f}')
        print(f'            预测均值: {stats["pred_mean"]:.3f}, 高置信(>0.7): {stats["high_conf_count"]}, 低置信(<0.2): {stats["low_conf_count"]}')
        print(f'            Top{DataConfig.TOP_K}%收益: {stats["top_return"]*100:+.2f}%')

        if stats['realistic_stats'] is not None:
            rs = stats['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
            print(f'            【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'            【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')

        if stats.get('smart_exit_stats') is not None:
            se = stats['smart_exit_stats']
            print(f'            【智能止损】收益率: {se["avg_realistic_return"]*100:.1f}%, Day1止损: {se["stop_loss_day1_count"]}次, 累计止损: {se["stop_loss_cum_count"]}次, 止盈: {se["take_profit_count"]}次')

        epoch_return = {
            'turn': epoch + 1,
            'return': stats['top_return'] * 100,
            'train_loss': avg_loss,
            'test_loss': test_loss,
            'dispersion_std': stats.get('dispersion_std', 0),
            'dispersion_range': stats.get('dispersion_range', 0),
            'dispersion_iqr': stats.get('dispersion_iqr', 0),
            'pos_ratio': stats.get('pred_mean', 0),
            'high_conf_ratio': stats.get('high_conf_count', 0) / len(eval_targets) if eval_targets is not None else 0,
        }
        epoch_returns.append(epoch_return)

        # 预测值分布可视化
        print_dispersion_sparkline(stats.get('all_preds', []), epoch_returns)

        # 早停检测
        improved, improve_reason = early_stopping.check_improve(
            avg_loss=test_loss,
            top_return=stats['top_return'],
            auc=stats['auc'],
            threshold=stats['top_threshold']
        )

        if improved:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'            ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
        else:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'            ⚠ 无改善 ({no_improve_count}/{patience_limit})')

        # 按收益率更新最佳模型
        if stats['top_return'] > best_return:
            best_return = stats['top_return']
            best_auc = stats['auc']
            best_threshold = stats['top_threshold']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f'            ✓ 新最佳模型（收益率）！Top{DataConfig.TOP_K}%收益: {best_return*100:+.2f}% (第{best_epoch}轮)')

        # 按loss评估最佳模型（条件：epoch >= 100, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%）
        realistic_return = stats['realistic_stats']['avg_realistic_return'] if stats.get('realistic_stats') else 0.0
        if (epoch + 1) >= 100 and realistic_return >= 0.014 and stats['top_return'] > 0.008 and stats['auc'] > 0.65:
            if test_loss < best_loss:
                best_loss = test_loss
                best_loss_epoch = epoch + 1
                best_model_by_loss = copy.deepcopy(model.state_dict())
                best_return_at_best_loss = stats['top_return']
                best_auc_at_best_loss = stats['auc']
                best_threshold_at_best_loss = stats['top_threshold']
                best_realistic_return_at_best_loss = realistic_return
                print(f'            ✓ 新最佳模型（loss）！Loss: {best_loss:.4f}, 实战收益率: {best_realistic_return_at_best_loss*100:.1f}% (第{best_loss_epoch}轮)')

        # 按实战收益率评估最佳模型（第100轮后）
        if (epoch + 1) >= 100:
            if realistic_return > best_realistic_return:
                best_realistic_return = realistic_return
                best_realistic_return_epoch = epoch + 1
                best_model_by_realistic_return = copy.deepcopy(model.state_dict())
                best_return_at_best_realistic = stats['top_return']
                best_auc_at_best_realistic = stats['auc']
                best_threshold_at_best_realistic = stats['top_threshold']
                best_realistic_return_value_at_best = realistic_return
                print(f'            ✓ 新最佳模型（实战收益率）！实战: {best_realistic_return*100:.1f}%, Top{DataConfig.TOP_K}%: {best_return_at_best_realistic*100:+.2f}% (第{best_realistic_return_epoch}轮)')

        print("-" * 60)

        if early_stopping.should_stop():
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break

    # ==================== 训练完成，保存结果 ====================
    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"最佳模型（按收益率）: 第{best_epoch}轮, Top{DataConfig.TOP_K}%收益: {best_return*100:+.2f}%, AUC: {best_auc:.4f}")
    if best_model_by_loss is not None:
        print(f"最佳模型（按loss）: 第{best_loss_epoch}轮, Loss: {best_loss:.4f}, 实战收益率: {best_realistic_return_at_best_loss*100:.1f}%")
    if best_model_by_realistic_return is not None:
        print(f"最佳模型（按实战收益率）: 第{best_realistic_return_epoch}轮, 实战收益率: {best_realistic_return_value_at_best*100:.1f}%, Top{DataConfig.TOP_K}%: {best_return_at_best_realistic*100:+.2f}%")

    # 保存每轮收益率到CSV
    timestamp_csv = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"dft_epoch_returns_{timestamp_csv}.csv")
    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['turn', 'return', 'train_loss', 'test_loss'])
        writer.writeheader()

        for er in epoch_returns:
            row = {
                'turn': er['turn'],
                'return': f"{er['return']:.2f}",
                'train_loss': f"{er['train_loss']:.4f}",
                'test_loss': f"{er['test_loss']:.4f}"
            }
            writer.writerow(row)

    print(f"✓ 每轮收益率已保存: {os.path.basename(returns_csv_path)}")
    print(f"  共记录 {len(epoch_returns)} 轮训练数据")

    # 保存模型（按收益率的最佳模型）
    save_path = save_model_with_metadata(
        best_model_state,
        best_return,
        best_threshold,
        best_auc,
        best_epoch,
        model_prefix="modelB_dft",
        output_dir=DataConfig.OUTPUT_DIR
    )
    print(f"✓ DFT模型(收益率)已保存: {os.path.basename(save_path)}")
    print(f"  Top{DataConfig.TOP_K}%阈值: {best_threshold:.4f}")

    # 保存模型（按loss的最佳模型）
    if best_model_by_loss is not None:
        save_path_loss = save_model_with_metadata(
            best_model_by_loss,
            best_return_at_best_loss, best_threshold_at_best_loss, best_auc_at_best_loss,
            best_loss_epoch,
            model_prefix="modelB_dft_loss",
            output_dir=DataConfig.OUTPUT_DIR
        )
        print(f"✓ DFT模型(loss)已保存: {os.path.basename(save_path_loss)}")
        print(f"  实战收益率: {best_realistic_return_at_best_loss*100:.1f}%")

    # 保存模型（按实战收益率的最佳模型）
    if best_model_by_realistic_return is not None:
        save_path_realistic = save_model_with_metadata(
            best_model_by_realistic_return,
            best_return_at_best_realistic, best_threshold_at_best_realistic, best_auc_at_best_realistic,
            best_realistic_return_epoch,
            model_prefix="modelB_dft_realistic",
            output_dir=DataConfig.OUTPUT_DIR
        )
        print(f"✓ DFT模型(realistic)已保存: {os.path.basename(save_path_realistic)}")
        print(f"  实战收益率: {best_realistic_return_value_at_best*100:.1f}%")

    print("=" * 60)

    return best_return, best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFT自引导微调：从已有模型加载，使用自引导权重继续微调')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='要微调的模型权重路径')
    parser.add_argument('--epochs', '-e', type=int, default=TrainingConfig.EPOCHS,
                        help=f'训练轮数（默认: {TrainingConfig.EPOCHS}）')
    parser.add_argument('--w_min', type=float, default=0.1,
                        help='DFT最小权重（默认: 0.1）')
    parser.add_argument('--w_max', type=float, default=1.0,
                        help='DFT最大权重（默认: 1.0）')
    parser.add_argument('--seed', type=int, default=DataConfig.RANDOM_SEED,
                        help=f'随机种子（默认: {DataConfig.RANDOM_SEED}）')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在: {args.model}")
        exit(1)

    print_config_summary()

    device = DeviceConfig.print_device_info()

    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)

    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()

    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    print("="*60)

    # 加载大盘数据
    print("\n正在加载大盘数据...")
    gdm = GlobalDataManager.get_instance()
    try:
        gdm.load_market_data()
        print(f"✓ 大盘数据加载成功")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"大盘数据加载失败: {e}。请确保 {DataConfig.MARKET_DATA_FILE} 文件存在。")

    print(f"\n正在加载模型: {args.model}")
    model = create_model().to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数: {total_params:,}")

    print("\n开始DFT自引导微调训练...")
    best_return, best_auc = train_dft_model(
        model, train_stock_info, test_stock_info,
        device=device,
        epochs=args.epochs,
        dft_w_min=args.w_min,
        dft_w_max=args.w_max,
        seed=args.seed
    )

    print(f"\n最终结果:")
    print(f"  最佳Top{DataConfig.TOP_K}%收益: {best_return*100:+.2f}%")
    print(f"  最佳AUC: {best_auc:.4f}")
