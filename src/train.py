'''
克隆模型训练脚本

核心思想：
- 前25%轮：只训练模型A（原始标签）
- 第25%轮：克隆模型A为模型B（完全独立的参数）
- 第25%轮起：
  - 模型A继续用原始标签训练
  - 模型B用A的高置信预测作为伪标签训练：
    - A预测前1% → B的标签 = 1（伪正标签）
    - A预测倒数5% → B的标签 = 0（伪负标签）
    - 其它 → 保持原始标签不变

这样模型B学习的是A"确信"的模式，同时过滤掉A不确定的噪声样本
'''

import os
import sys

# 确保能正确导入其他模块（无论从哪里运行）
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch, torch.nn as nn, torch.optim as optim, numpy as np
import copy
import random
import csv
from datetime import datetime
from src.config import (TrainingConfig,DataConfig,DeviceConfig,print_config_summary,LossConfig)

from src.model import create_model

from src.data import (
    load_and_preprocess_data,
    create_sampler, sample_with_pools,
    create_fixed_evaluation_dataset,FeatureNormalizer, GlobalDataManager
)

from src.training_utils import (
    WarmupScheduler,
    evaluate_model,
    generate_pseudo_labels,
    save_model_with_metadata,
    DynamicWeightedBCE,
    TaskAlignedLoss,
    EarlyStopping,
    calculate_test_loss,
    print_dispersion_sparkline,
    create_optimizer_from_config,
    create_scheduler_from_config
)

def train_clone_model(model_a, train_stock_info, test_stock_info,
                      epochs=TrainingConfig.EPOCHS,
                      learning_rate=TrainingConfig.LEARNING_RATE,
                      device=None,
                      batch_size=TrainingConfig.BATCH_SIZE,
                      batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
                      clone_epoch=TrainingConfig.EPOCHS*0.25,
                      pseudo_pos_ratio=0.01,
                      pseudo_neg_ratio=0.05,
                      enable_model_b=True,
                      feature_normalizer=None):
    """
    克隆模型训练函数

    训练策略：
    - 前 clone_epoch 轮：只训练模型A
    - 第 clone_epoch 轮：克隆模型A为模型B
    - 之后：A继续原始训练，B用A的高置信预测作为伪标签
      - 按比例选取：A预测值前 pseudo_pos_ratio (1%) 的样本 → 伪正标签
      - 按比例选取：A预测值倒数 pseudo_neg_ratio (5%) 的样本 → 伪负标签
      - 其它样本 → 保持原始标签不变

    Args:
        feature_normalizer: 可选的特征归一化器实例
    """
    print("\n" + "="*60)
    print("克隆模型训练")
    print("="*60)
    print(f"训练策略：")
    print(f"  - 前{clone_epoch}轮：只训练模型A（原始标签）")
    print(f"  - 第{clone_epoch}轮：克隆模型A为模型B")
    print(f"  - 之后：")
    print(f"    - 模型A：继续用原始标签训练")
    print(f"    - 模型B：用A的高置信预测作为伪标签")
    print(f"      - A预测值前{pseudo_pos_ratio*100:.0f}% → 伪正标签")
    print(f"      - A预测值倒数{pseudo_neg_ratio*100:.0f}% → 伪负标签")
    print(f"      - 其它 → 保持原始标签不变")
    print("="*60 + "\n")

    # 设置随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)

    # 创建评估数据集
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_market_seqs = create_fixed_evaluation_dataset(test_stock_info, feature_normalizer)

    # 模型B初始化为None
    model_b = None
    optimizer_b = None

    # 创建优化器（模型A）
    optimizer_a = create_optimizer_from_config(model_a, lr=learning_rate)

    # 创建学习率调度器（模型A）
    warmup_scheduler_a, main_scheduler_a, warmup_epochs = create_scheduler_from_config(
        optimizer_a,
        epochs=epochs,
        lr=learning_rate
    )

    # 损失函数选择
    if LossConfig.use_task_aligned():
        print("损失函数: TaskAlignedLoss (BCE + 排序损失 + 收益加权 + Top-K聚焦)")
        print(f"  权重: rank={LossConfig.RANK_LOSS_WEIGHT}, return={LossConfig.RETURN_LOSS_WEIGHT}, topk={LossConfig.TOPK_LOSS_WEIGHT}")
        criterion = TaskAlignedLoss(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
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
        print("损失函数: DynamicWeightedBCE (正样本权重4.0，负样本动态调整)")
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
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
        print("损失函数: 简单BCE (BCEWithLogitsLoss)")
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 最佳模型A缓存（按Top1%收益率判断，用于生成伪标签）
    best_return_a = -float('inf')
    best_model_a_for_pseudo = None  # 用于生成伪标签的最佳A
    best_return_epoch_a = 0

    # 最佳模型B缓存（按收益率判断，仅用于显示）
    best_return_b = None if not enable_model_b else -float('inf')
    best_return_epoch_b = None if not enable_model_b else 0

    # 按loss保存的最佳模型（需要满足条件才参与评估）
    # 条件：epoch >= 100, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%
    best_loss_a = float('inf')
    best_loss_epoch_a = 0
    best_model_a_by_loss = None
    best_return_a_at_best_loss = 0.0
    best_auc_a_at_best_loss = 0.0
    best_threshold_a_at_best_loss = 0.0
    best_realistic_return_a_at_best_loss = 0.0

    # 按实战收益率保存的最佳模型A（第100轮后）
    best_realistic_return_a = -float('inf')
    best_realistic_return_epoch_a = 0
    best_model_a_by_realistic_return = None
    best_return_a_at_best_realistic = 0.0
    best_auc_a_at_best_realistic = 0.0
    best_threshold_a_at_best_realistic = 0.0
    best_realistic_return_value_at_best = 0.0

    best_loss_b = float('inf')
    best_loss_epoch_b = 0
    best_model_b_by_loss = None
    best_return_b_at_best_loss = 0.0
    best_auc_b_at_best_loss = 0.0
    best_threshold_b_at_best_loss = 0.0
    best_realistic_return_b_at_best_loss = 0.0

    # 早停机制（patience = EPOCHS * 0.25）
    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    # 创建采样器（根据配置选择策略）
    sampler = create_sampler(train_stock_info)
    train_rng = random.Random(DataConfig.RANDOM_SEED)

    # 记录每轮收益率
    epoch_returns = []  # 格式: [{'turn': 1, 'return_a': 1.62, 'return_b': None}, ...]

    for epoch in range(epochs):
        model_a.train()
        if model_b is not None:
            model_b.train()

        total_loss_a = 0
        total_loss_b = 0
        total_pseudo_pos = 0
        total_pseudo_neg = 0
        total_unchanged = 0

        has_model_b = (epoch + 1) >= clone_epoch

        # 学习率更新
        if warmup_scheduler_a.is_warmup_phase():
            current_lr = warmup_scheduler_a.step(epoch)
            lr_status = f"预热阶段 ({epoch + 1}/{warmup_epochs})"
        else:
            current_lr = main_scheduler_a.get_last_lr()[0]
            lr_status = "正常训练"

        status = "A+B训练" if (enable_model_b and has_model_b) else "只训练A"
        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status}) [{status}]')

        # 第clone_epoch轮时克隆模型B
        if (epoch + 1) == clone_epoch and model_b is None:
            if enable_model_b:
                print(f"\n  >>> 第{clone_epoch}轮：克隆模型A为模型B <<<")
                model_b = copy.deepcopy(model_a)
                model_b = model_b.to(device)

                # 模型B使用半学习率，更保守的更新
                optimizer_b = create_optimizer_from_config(model_b, lr=learning_rate * 0.5)
                print(f"  模型B已创建，参数数: {sum(p.numel() for p in model_b.parameters()):,}")
                print()
            else:
                print(f"\n  >>> 第{clone_epoch}轮：模型B训练已禁用，跳过克隆 <<<")
                print()

        # 使用时间顺序采样器生成训练数据（与主训练流程统一）
        epoch_inputs, epoch_targets, epoch_cum_returns, epoch_market_seqs = sample_with_pools(
            sampler, train_stock_info, batch_size, batches_per_epoch, train_rng,
            feature_normalizer
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
        epoch_market_tensor = torch.tensor(epoch_market_seqs, dtype=torch.float32).to(device)

        # 计算实际可用的batch数量（防止索引越界）
        actual_batches = len(epoch_inputs_tensor) // batch_size
        if actual_batches < batches_per_epoch:
            print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

        # 训练循环：使用实际的batch数量，而不是固定的batches_per_epoch
        for step in range(actual_batches):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size  # 不需要min，因为actual_batches已经保证了不越界

            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            batch_returns = epoch_returns_tensor[start_idx:end_idx]
            batch_market = epoch_market_tensor[start_idx:end_idx] if epoch_market_tensor is not None else None

            # ========== 训练模型A ==========
            optimizer_a.zero_grad()
            output_a = model_a(batch_inputs, batch_market)
            if hasattr(criterion, 'update_weights'):
                criterion.update_weights(batch_targets)
            if isinstance(criterion, TaskAlignedLoss):
                loss_a = criterion(output_a.squeeze(-1), batch_targets, batch_returns)
            else:
                loss_a = criterion(output_a.squeeze(-1), batch_targets)
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer_a.step()
            # 累加loss时乘以batch_size，得到该batch的总损失
            total_loss_a += loss_a.item() * (end_idx - start_idx)

            # ========== 训练模型B（如果存在且启用）==========
            if enable_model_b and model_b is not None and best_model_a_for_pseudo is not None:
                optimizer_b.zero_grad()

                # 用【最佳模型A】的预测生成伪标签
                with torch.no_grad():
                    output_a_for_pseudo = best_model_a_for_pseudo(batch_inputs, batch_market)
                    pred_a_for_pseudo = torch.sigmoid(output_a_for_pseudo).squeeze()

                # 使用统一的伪标签生成函数
                pseudo_targets_numpy, pseudo_stats = generate_pseudo_labels(
                    pred_a_for_pseudo, batch_targets,
                    pseudo_pos_ratio=pseudo_pos_ratio,
                    pseudo_neg_ratio=pseudo_neg_ratio
                )
                pseudo_targets = torch.tensor(pseudo_targets_numpy, dtype=torch.float32).to(device)

                total_pseudo_pos += pseudo_stats['pseudo_pos_count']
                total_pseudo_neg += pseudo_stats['pseudo_neg_count']
                total_unchanged += pseudo_stats['unchanged_count']

                # 训练模型B
                output_b = model_b(batch_inputs, batch_market)
                if hasattr(criterion, 'update_weights'):
                    criterion.update_weights(pseudo_targets)
                if isinstance(criterion, TaskAlignedLoss):
                    loss_b = criterion(output_b.squeeze(-1), pseudo_targets, batch_returns)
                else:
                    loss_b = criterion(output_b.squeeze(-1), pseudo_targets)
                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
                optimizer_b.step()
                # 累加loss时乘以batch_size，得到该batch的总损失
                total_loss_b += loss_b.item() * (end_idx - start_idx)

            # 进度显示
            progress = (step + 1) / actual_batches * 100
            # 使用已处理的样本数计算当前平均损失
            processed_samples = (step + 1) * batch_size
            avg_loss_a = total_loss_a / processed_samples
            if enable_model_b and model_b is not None:
                avg_loss_b = total_loss_b / processed_samples
                print(f'\r  训练进度: {progress:.1f}%, Loss_A: {avg_loss_a:.4f}, Loss_B: {avg_loss_b:.4f}', end='', flush=True)
            else:
                print(f'\r  训练进度: {progress:.1f}%, Loss_A: {avg_loss_a:.4f}', end='', flush=True)

        print()
        print()

        # 清理内存
        del epoch_inputs_tensor, epoch_targets_tensor, epoch_returns_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新学习率
        if not warmup_scheduler_a.is_warmup_phase():
            main_scheduler_a.step()

        # 评估模型A（使用统一的评估函数）
        stats_a = evaluate_model(
            model_a, eval_inputs, eval_targets, eval_cumulative_returns,
            device, model_name="A",
            eval_day_indices=eval_day_indices,
            eval_daily_returns=eval_daily_returns,
            market_seqs=eval_market_seqs
        )

        # 计算训练集平均损失（除以样本数，与测试损失保持一致）
        # total_loss_a已经是所有样本的总损失（累加时乘以了batch_size）
        total_samples_a = len(epoch_inputs)
        avg_loss_a = total_loss_a / total_samples_a if total_samples_a > 0 else 0

        # 计算测试集损失（用于早停检测）
        test_loss_a = calculate_test_loss(model_a, eval_inputs, eval_targets, eval_criterion, device, market_seqs=eval_market_seqs)

        # 打印模型A结果
        print(f'  [模型A] 训练损失: {avg_loss_a:.4f}, 测试损失: {test_loss_a:.4f}, AUC: {stats_a["auc"]:.4f}')
        print(f'          预测均值: {stats_a["pred_mean"]:.3f}, 高置信(>0.7): {stats_a["high_conf_count"]}, 低置信(<0.2): {stats_a["low_conf_count"]}')
        print(f'          Top{DataConfig.TOP_K}%收益: {stats_a["top_return"]*100:+.2f}%')
        
        if stats_a['realistic_stats'] is not None:
            rs = stats_a['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
            print(f'          【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'          【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')
        
        if stats_a.get('smart_exit_stats') is not None:
            se = stats_a['smart_exit_stats']
            print(f'          【智能止损】收益率: {se["avg_realistic_return"]*100:.1f}%, Day1止损: {se["stop_loss_day1_count"]}次, 累计止损: {se["stop_loss_cum_count"]}次, 止盈: {se["take_profit_count"]}次')
        
        epoch_return = {
            'turn': epoch + 1,
            'return': stats_a['top_return'] * 100,
            'return_a': stats_a['top_return'] * 100,
            'return_b': None,
            'train_loss': avg_loss_a,
            'train_loss_a': avg_loss_a,
            'train_loss_b': None,
            'test_loss': test_loss_a,
            'test_loss_a': test_loss_a,
            'test_loss_b': None,
            'dispersion_std': stats_a.get('dispersion_std', 0),
            'dispersion_range': stats_a.get('dispersion_range', 0),
            'dispersion_iqr': stats_a.get('dispersion_iqr', 0),
            'pos_ratio': stats_a.get('pred_mean', 0),
            'high_conf_ratio': stats_a.get('high_conf_count', 0) / len(eval_targets) if eval_targets is not None else 0,
        }
        epoch_returns.append(epoch_return)

        print_dispersion_sparkline(stats_a.get('all_preds', []), epoch_returns)

        # 早停检测（使用测试集loss）
        improved, improve_reason = early_stopping.check_improve(
            avg_loss=test_loss_a,
            top_return=stats_a['top_return'],
            auc=stats_a['auc'],
            threshold=stats_a['top_threshold']
        )

        if improved:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'          ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
        else:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'          ⚠ 无改善 ({no_improve_count}/{patience_limit})')

        # 更新用于生成伪标签的最佳模型A（按Top1%收益率判断）
        if stats_a['top_return'] > best_return_a:
            best_return_a = stats_a['top_return']
            best_return_epoch_a = epoch + 1
            if best_model_a_for_pseudo is None:
                best_model_a_for_pseudo = copy.deepcopy(model_a)
            else:
                best_model_a_for_pseudo.load_state_dict(copy.deepcopy(model_a.state_dict()))
            best_model_a_for_pseudo.eval()
            print(f'          ✓ 新最佳模型A（收益率）！Top1%收益: {best_return_a*100:+.2f}% (第{best_return_epoch_a}轮)')

        # 按loss评估最佳模型A（条件：epoch >= 100, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%）
        realistic_return_a = stats_a['realistic_stats']['avg_realistic_return'] if stats_a.get('realistic_stats') else 0.0
        if (epoch + 1) >= 100 and realistic_return_a >= 0.014 and stats_a['top_return'] > 0.008 and stats_a['auc'] > 0.65:
            if test_loss_a < best_loss_a:
                best_loss_a = test_loss_a
                best_loss_epoch_a = epoch + 1
                best_model_a_by_loss = copy.deepcopy(model_a.state_dict())
                best_return_a_at_best_loss = stats_a['top_return']
                best_auc_a_at_best_loss = stats_a['auc']
                best_threshold_a_at_best_loss = stats_a['top_threshold']
                best_realistic_return_a_at_best_loss = realistic_return_a
                print(f'          ✓ 新最佳模型A（loss）！Loss: {best_loss_a:.4f}, 实战收益率: {best_realistic_return_a_at_best_loss*100:.1f}% (第{best_loss_epoch_a}轮)')

        # 按实战收益率评估最佳模型A（第100轮后）
        if (epoch + 1) >= 100:
            if realistic_return_a > best_realistic_return_a:
                best_realistic_return_a = realistic_return_a
                best_realistic_return_epoch_a = epoch + 1
                best_model_a_by_realistic_return = copy.deepcopy(model_a.state_dict())
                best_return_a_at_best_realistic = stats_a['top_return']
                best_auc_a_at_best_realistic = stats_a['auc']
                best_threshold_a_at_best_realistic = stats_a['top_threshold']
                best_realistic_return_value_at_best = realistic_return_a
                print(f'          ✓ 新最佳模型A（实战收益率）！实战: {best_realistic_return_a*100:.1f}%, Top1%: {best_return_a_at_best_realistic*100:+.2f}% (第{best_realistic_return_epoch_a}轮)')

        # 评估模型B（如果存在且启用）
        if enable_model_b and model_b is not None:
            stats_b = evaluate_model(
                model_b, eval_inputs, eval_targets, eval_cumulative_returns,
                device, model_name="B",
                eval_day_indices=eval_day_indices,
                eval_daily_returns=eval_daily_returns,
                market_seqs=eval_market_seqs
            )

            # 计算训练集平均损失（除以样本数，与测试损失保持一致）
            # total_loss_b已经是所有样本的总损失（累加时乘以了batch_size）
            total_samples_b = len(epoch_inputs)
            avg_loss_b = total_loss_b / total_samples_b if total_samples_b > 0 else 0

            # 计算测试集损失
            test_loss_b = calculate_test_loss(model_b, eval_inputs, eval_targets, eval_criterion, device, market_seqs=eval_market_seqs)

            print(f'  [模型B] 训练损失: {avg_loss_b:.4f}, 测试损失: {test_loss_b:.4f}, AUC: {stats_b["auc"]:.4f}')
            print(f'          预测均值: {stats_b["pred_mean"]:.3f}, 高置信(>0.7): {stats_b["high_conf_count"]}, 低置信(<0.2): {stats_b["low_conf_count"]}')
            print(f'          Top{DataConfig.TOP_K}%收益: {stats_b["top_return"]*100:+.2f}%')
            
            if stats_b['realistic_stats'] is not None:
                rs = stats_b['realistic_stats']
                daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
                mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
                print(f'          【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
                print(f'          【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')
            
            if stats_b.get('smart_exit_stats') is not None:
                se = stats_b['smart_exit_stats']
                print(f'          【智能止损】收益率: {se["avg_realistic_return"]*100:.1f}%, Day1止损: {se["stop_loss_day1_count"]}次, 累计止损: {se["stop_loss_cum_count"]}次, 止盈: {se["take_profit_count"]}次')
            
            print(f'          伪标签来源: 最佳A(第{best_return_epoch_a}轮, 收益{best_return_a*100:+.2f}%)')
            print(f'          伪标签统计: 伪正={total_pseudo_pos}, 伪负={total_pseudo_neg}, 不变={total_unchanged}')

            if best_return_b is not None and stats_b['top_return'] > best_return_b:
                best_return_b = stats_b['top_return']
                best_return_epoch_b = epoch + 1
                print(f'          ✓ 新最佳模型B（收益率）！Top1%收益: {best_return_b*100:+.2f}% (第{best_return_epoch_b}轮)')

            # 按loss评估最佳模型B（条件：epoch >= 100, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%）
            realistic_return_b = stats_b['realistic_stats']['avg_realistic_return'] if stats_b.get('realistic_stats') else 0.0
            if (epoch + 1) >= 100 and realistic_return_b >= 0.014 and stats_b['top_return'] > 0.008 and stats_b['auc'] > 0.65:
                if test_loss_b < best_loss_b:
                    best_loss_b = test_loss_b
                    best_loss_epoch_b = epoch + 1
                    best_model_b_by_loss = copy.deepcopy(model_b.state_dict())
                    best_return_b_at_best_loss = stats_b['top_return']
                    best_auc_b_at_best_loss = stats_b['auc']
                    best_threshold_b_at_best_loss = stats_b['top_threshold']
                    best_realistic_return_b_at_best_loss = realistic_return_b
                    print(f'          ✓ 新最佳模型B（loss）！Loss: {best_loss_b:.4f}, 实战收益率: {best_realistic_return_b_at_best_loss*100:.1f}% (第{best_loss_epoch_b}轮)')

            epoch_return['return_b'] = stats_b['top_return'] * 100
            epoch_return['train_loss_b'] = avg_loss_b
            epoch_return['test_loss_b'] = test_loss_b

        print("-" * 60)

        # 早停检查
        if early_stopping.should_stop():
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break

    # 保存最佳模型（使用统一的保存函数）
    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"最佳模型A（按收益率用于伪标签）: 第{best_return_epoch_a}轮, Top1%收益: {best_return_a*100:+.2f}%")
    print(f"最佳模型A（按loss）: 第{best_loss_epoch_a}轮, Loss: {best_loss_a:.4f}, 实战收益率: {best_realistic_return_a_at_best_loss*100:.1f}%")
    print(f"最佳模型A（按实战收益率）: 第{best_realistic_return_epoch_a}轮, 实战收益率: {best_realistic_return_value_at_best*100:.1f}%, Top1%: {best_return_a_at_best_realistic*100:+.2f}%")
    if enable_model_b and best_model_b_by_loss is not None:
        print(f"最佳模型B（按loss）: 第{best_loss_epoch_b}轮, Loss: {best_loss_b:.4f}, 实战收益率: {best_realistic_return_b_at_best_loss*100:.1f}%")

    # 保存模型A（按loss的最佳模型）
    if best_model_a_by_loss is not None:
        save_path_a = save_model_with_metadata(
            best_model_a_by_loss,
            best_return_a_at_best_loss, best_threshold_a_at_best_loss, best_auc_a_at_best_loss,
            best_loss_epoch_a,
            model_prefix="modelA_loss",
            output_dir=DataConfig.OUTPUT_DIR
        )
        print(f"✓ 模型A(loss)已保存: {os.path.basename(save_path_a)}")
        print(f"  Top1%阈值: {best_threshold_a_at_best_loss:.4f}")
        print(f"  实战收益率: {best_realistic_return_a_at_best_loss*100:.1f}%")

    # 保存模型A（按实战收益率的最佳模型）
    if best_model_a_by_realistic_return is not None:
        save_path_a_realistic = save_model_with_metadata(
            best_model_a_by_realistic_return,
            best_return_a_at_best_realistic, best_threshold_a_at_best_realistic, best_auc_a_at_best_realistic,
            best_realistic_return_epoch_a,
            model_prefix="modelA_realistic",
            output_dir=DataConfig.OUTPUT_DIR
        )
        print(f"✓ 模型A(realistic)已保存: {os.path.basename(save_path_a_realistic)}")
        print(f"  Top1%阈值: {best_threshold_a_at_best_realistic:.4f}")
        print(f"  实战收益率: {best_realistic_return_value_at_best*100:.1f}%")

    # 保存模型B（按loss的最佳模型）
    if enable_model_b and best_model_b_by_loss is not None:
        save_path_b = save_model_with_metadata(
            best_model_b_by_loss,
            best_return_b_at_best_loss, best_threshold_b_at_best_loss, best_auc_b_at_best_loss,
            best_loss_epoch_b,
            model_prefix="modelB",
            output_dir=DataConfig.OUTPUT_DIR
        )
        print(f"✓ 模型B已保存: {os.path.basename(save_path_b)}")
        print(f"  Top1%阈值: {best_threshold_b_at_best_loss:.4f}")
        print(f"  实战收益率: {best_realistic_return_b_at_best_loss*100:.1f}%")

    print("=" * 60)

    # 保存每轮收益率到CSV（使用时间戳避免多模型训练时覆盖）
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"clone_epoch_returns_{timestamp}.csv")
    
    # 根据 enable_model_b 动态决定 CSV 字段
    if enable_model_b:
        fieldnames = ['turn', 'A', 'B', 'train_loss_A', 'test_loss_A', 'train_loss_B', 'test_loss_B']
    else:
        fieldnames = ['turn', 'A', 'train_loss_A', 'test_loss_A']
    
    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch_return in epoch_returns:
            row = {
                'turn': epoch_return['turn'],
                'A': f"{epoch_return['return_a']:.2f}" if epoch_return['return_a'] is not None else "",
                'train_loss_A': f"{epoch_return['train_loss_a']:.4f}" if epoch_return.get('train_loss_a') is not None else "",
                'test_loss_A': f"{epoch_return['test_loss_a']:.4f}" if epoch_return.get('test_loss_a') is not None else ""
            }
            
            if enable_model_b:
                row.update({
                    'B': f"{epoch_return['return_b']:.2f}" if epoch_return['return_b'] is not None else "",
                    'train_loss_B': f"{epoch_return['train_loss_b']:.4f}" if epoch_return.get('train_loss_b') is not None else "",
                    'test_loss_B': f"{epoch_return['test_loss_b']:.4f}" if epoch_return.get('test_loss_b') is not None else ""
                })
            
            writer.writerow(row)

    print(f"✓ 每轮收益率已保存: {os.path.basename(returns_csv_path)}")
    print(f"  共记录 {len(epoch_returns)} 轮训练数据")
    print("=" * 60)

    return best_return_a, best_return_b


if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 打印配置摘要
    print_config_summary()

    # 获取设备
    device = DeviceConfig.print_device_info()

    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)

    # ========== 特征归一化器配置 ==========
    print("\n" + "="*60)
    print("特征归一化器配置")
    print(f" 归一化器路径: {DataConfig.NORMALIZER_PATH}")
    print(f" 输出分布: {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION}")
    print(f" 分位数数量: {DataConfig.NORMALIZER_N_QUANTILES}")

    # 检查归一化器文件是否存在
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n⚠ 归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")
        print("请先运行以下命令创建归一化器：")
        print(f"  python data.py --output-distribution {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION} --n-quantiles {DataConfig.NORMALIZER_N_QUANTILES}")
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    print("="*60)

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print("\n正在加载大盘数据...")
    gdm = GlobalDataManager.get_instance()
    try:
        gdm.load_market_data()
        print(f"✓ 大盘数据加载成功")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"大盘数据加载失败: {e}。请确保 {DataConfig.MARKET_DATA_FILE} 文件存在。")

    print("\n" + "="*60)
    print("数据集统计")
    print(f" 训练集: {len(train_stock_info)} 只股票")
    print(f" 测试集: {len(test_stock_info)} 只股票")

    print("\n正在创建模型A (FP32精度)...")
    model_a = create_model().to(device)

    total_params = sum(p.numel() for p in model_a.parameters())
    print(f" 模型A参数数: {total_params:,}")

    print("\n开始克隆模型训练...")
    best_return_a, best_return_b = train_clone_model(
        model_a, train_stock_info, test_stock_info,
        device=device,
        clone_epoch=TrainingConfig.EPOCHS*0.25,
        pseudo_pos_ratio=0.01,
        pseudo_neg_ratio=0.05,
        enable_model_b=False,
        feature_normalizer=feature_normalizer
    )

    print(f"\n最终结果:")
    print(f"  模型A: 最佳Top1%收益={best_return_a*100:+.2f}%")
    if best_return_b is not None:
        print(f"  模型B: 最佳Top1%收益={best_return_b*100:+.2f}%")
    else:
        print(f"  模型B: 训练已禁用")
