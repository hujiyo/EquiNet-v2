"""
EquiNet 模型定义文件

包含所有模型架构相关的类：
- PositionalEncoding: 位置编码
- TwoDimensionalPositionalEncoding: 二维位置编码（用于Token化模型）
- MarketTokenEncoder: 市场Token编码器
- MultiHeadAttention: 多头注意力机制
- TransformerLayer: Transformer层
- AttentionPooling: 多注意力聚合（可学习query token + cross-attention）
- StockTransformer: 连续值模型
- create_model(): 工厂函数，创建模型
"""

import math
import torch
import torch.nn as nn
from config import ModelConfig, DataConfig

def init_weights(module):
    """
    当代主流Transformer初始化策略

    设计原则：
    1. Embedding层: Xavier初始化，小增益确保输入方差合理
    2. FFN第一层: Xavier初始化，gain=1.7补偿GELU压缩
    3. FFN第二层: Xavier初始化，gain=1.0（无激活函数）
    4. 输出层: 小增益，避免sigmoid饱和
    5. LayerNorm: weight=1, bias=0

    各层初始化范围计算（基于当前模型配置）：
    - Embedding (6→48): gain=0.5 → 范围±0.167, std≈0.096
    - 位置编码 (30→48): gain=0.7 → 范围±0.168, std≈0.097
    - FFN第一层 (48→192): gain=1.7 → 范围±0.270, std≈0.155
    - FFN第二层 (192→48): gain=1.0 → 范围±0.158, std≈0.091
    - 输出层 (24→1): gain=0.1 → 范围±0.058, std≈0.034, bias=log(p/(1-p))
    """
    ffn_hidden_dim = ModelConfig.D_MODEL * ModelConfig.FFN_EXPAND_RATIO

    if isinstance(module, nn.Linear):
        if module.out_features == 1:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.OUTPUT_LAYER_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ModelConfig.INPUT_DIM and module.out_features == ModelConfig.D_MODEL:
            # stock_token embedding: 使用默认gain
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.EMBEDDING_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.out_features == ModelConfig.D_MODEL and module.in_features != ModelConfig.D_MODEL and module.in_features != ffn_hidden_dim:
            # market_token embedding: 使用配置文件中定义的专用gain
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.MARKET_EMBEDDING_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ModelConfig.D_MODEL and module.out_features == ffn_hidden_dim:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.FFN_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ffn_hidden_dim and module.out_features == ModelConfig.D_MODEL:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learned Positional Embedding）
    类似 BERT / GPT 的做法：每个位置对应一个可训练的向量
    让模型自己学习最优的位置表示，而非使用固定的正弦公式
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()        
        # 考虑因子数量，增加位置编码长度
        self.pe = nn.Embedding(seq_len + DataConfig.FACTOR_NUM, d_model)

    def forward(self, x):
        #添加位置编码，LayerNorm在后续层中可能使用
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块（不含残差连接和归一化）
    仅负责注意力计算，残差连接由上层TransformerLayer统一管理
    """
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0
        
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)

    def forward(self, x, attn_mask=None):
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)
        
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return self.dropout(attn_output)


class TransformerLayer(nn.Module):
    """
    标准 Transformer 层（Pre-Norm架构，主流大厂风格）
    统一管理归一化和残差连接，数据流清晰易懂
    Pre-Norm相比Post-Norm有更好的训练稳定性
    """
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        
        # 注意力子层
        self.attn = MultiHeadAttention(d_model, nhead)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # 前馈网络子层
        ffn_hidden_dim = int(d_model * ModelConfig.FFN_EXPAND_RATIO)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(ffn_hidden_dim, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

    def forward(self, x):
        # 注意力子层: x = x + Dropout(Attention(LayerNorm(x)))
        x = x + self.attn(self.attn_norm(x), attn_mask=None)
        
        # 前馈网络子层: x = x + Dropout(FFN(LayerNorm(x)))
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
        
        return x


class AttentionPooling(nn.Module):
    """
    多头注意力聚合（Multi-Head Attention Pooling）

    使用一个可学习的 query token 通过多头 cross-attention 聚合序列信息。
    相比单向量点积聚合，每个注意力头可以学到不同的时间聚合模式，
    表达能力更强，且与 Transformer 架构风格一致。

    参考: Set Transformer (Lee et al., 2019), Perceiver (Jaegle et al., 2021)
    """
    def __init__(self, d_model, nhead):
        super(AttentionPooling, self).__init__()

        # 可学习的 query token: [1, 1, d_model]
        self.query = nn.Parameter(torch.empty(1, 1, d_model))

        # Pre-Norm: 分别对 query 和 key-value 进行归一化
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # 多头 cross-attention: query 关注序列所有位置
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        # 初始化 query token（使用Xavier初始化）
        nn.init.xavier_uniform_(self.query, gain=ModelConfig.EMBEDDING_INIT_GAIN)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] - Transformer 编码后的序列

        Returns:
            pooled: [batch_size, d_model] - 聚合后的表示向量
        """
        batch_size = x.size(0)

        # 将 query 扩展到 batch 维度: [1, 1, d_model] -> [batch_size, 1, d_model]
        query = self.query.expand(batch_size, -1, -1)

        # Pre-Norm
        query_normed = self.norm_q(query)
        kv_normed = self.norm_kv(x)

        # Cross-attention: query 关注序列所有位置
        # attn_output: [batch_size, 1, d_model]
        attn_output, _ = self.cross_attn(query_normed, kv_normed, kv_normed)

        # 残差连接 + 去掉 seq_len=1 的维度
        pooled = (query + self.dropout(attn_output)).squeeze(1)  # [batch_size, d_model]

        return pooled


class StockTransformer(nn.Module):
    """
    Transformer 模型（Pre-Norm 架构 + Linear-Embedding）

    核心设计：回归主流 Transformer 架构
    - 6 个输入特征 (OHLC + Volume + Exchange) -> 单一线性层映射到 d_model 维
    - 遵循 BERT/GPT/LLaMA 等主流模型的设计：embedding = nn.Linear(input_dim, d_model)
    - 简化结构，减少不必要的非线性变换，让模型更容易训练

    架构统一性：
    - Embedding 层：线性投影（无激活函数）
    - Transformer 层：标准 Attention + FFN 结构
    
    市场token：
    - 市场token由大盘涨跌序列编码而来
    - 放在序列末尾，作为全局上下文
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_len, market_context_length=None):
        super(StockTransformer, self).__init__()

        # Linear-Embedding：单一线性层，主流 Transformer 标准做法
        # 输入特征直接线性映射到 d_model 维，无中间层和激活函数
        if market_context_length is None:
            market_context_length = DataConfig.MARKET_CONTEXT_LENGTH

        self.embedding = nn.Linear(input_dim, d_model)

        # 使用标准位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 统一架构：所有层都使用 Attention + FFN
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead)
            for i in range(num_layers)
        ])

        # Pre-Norm架构：在最后添加一个LayerNorm
        # 因为Pre-Norm的最后一层没有归一化输出
        self.final_norm = nn.LayerNorm(d_model)

        # 多头注意力聚合：通过 cross-attention 聚合序列信息
        # 相比单向量点积，每个注意力头可以学到不同的时间聚合模式
        self.attention_pooling = AttentionPooling(d_model, nhead)

        # 简化输出层，减少过拟合
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 降维
            nn.GELU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(d_model // 2, output_dim)  # 最终输出
        )

        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        self.market_encoder = MarketTokenEncoder(market_context_length, d_model)
        # 应用初始化
        self.apply(init_weights)

    def forward(self, x, market_seq):
        # x: [batch_size, seq_len, 6] (OHLC + volume + exchange)

        # 1. 统一Embedding：6个特征一起映射到d_model维

        """
        Args:
            x: [batch_size, seq_len, 6] (OHLC + volume + exchange)
            market_seq: [batch_size, market_context_length] 大盘涨跌序列

        Returns:
            output: [batch_size, 1] 预测logits
        """
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        market_token = self.market_encoder(market_seq)
        x = torch.cat([x, market_token], dim=1)
        # 2. 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3. Transformer层（Pre-Norm架构）
        for layer in self.layers:
            x = layer(x)

        # 4. Pre-Norm架构需要在最后进行归一化
        #    因为每层的输出没有经过归一化
        x = self.final_norm(x)

        # 5. 多头注意力聚合
        aggregated = self.attention_pooling(x)  # [batch_size, d_model]

        output = self.output_projection(aggregated)  # [batch_size, output_dim]
        return output

class MarketTokenEncoder(nn.Module):
    """
    市场Token编码器

    将N天的大盘涨跌序列编码为单个token，类似于BERT的[CLS] token。
    
    输入: [batch_size, market_context_length] 大盘涨跌序列
    输出: [batch_size, 1, d_model] 市场token
    
    设计原理：
    - 使用线性层将大盘序列直接映射到d_model维
    - 市场token作为全局上下文信息参与Transformer处理
    - 放在序列末尾，通过attention与个股token交互
    """
    def __init__(self, market_context_length, d_model):
        super(MarketTokenEncoder, self).__init__()
        self.market_proj = nn.Linear(market_context_length, d_model)
        
    def forward(self, market_seq):
        """
        Args:
            market_seq: [batch_size, market_context_length] 大盘涨跌序列
            
        Returns:
            market_token: [batch_size, 1, d_model] 市场token
        """
        return self.market_proj(market_seq).unsqueeze(1)

# ==================== 工厂函数 ====================

def create_model(input_dim=ModelConfig.INPUT_DIM, d_model=ModelConfig.D_MODEL, 
                 nhead=ModelConfig.NHEAD, num_layers=ModelConfig.NUM_LAYERS,
                 output_dim=ModelConfig.OUTPUT_DIM, seq_len=DataConfig.CONTEXT_LENGTH,
                 market_context_length=DataConfig.MARKET_CONTEXT_LENGTH, model_arch=None):
    """
    Args:
        参数均为可选，如果不提供则使用 ModelConfig 中的默认值
        model_arch: 可选的元数据字典（来自 .pth 内的 'model_arch' 键），
                    若提供则优先使用其中的参数覆盖默认值，
                    用于 run.py 自动重建与训练时架构一致的模型

    Returns:
        model: StockTransformer 模型实例
    """
    if model_arch is not None:
        input_dim  = model_arch.get('input_dim',  input_dim)
        d_model    = model_arch.get('d_model',    d_model)
        nhead      = model_arch.get('nhead',      nhead)
        num_layers = model_arch.get('num_layers', num_layers)
        output_dim = model_arch.get('output_dim', output_dim)
        seq_len    = model_arch.get('context_length', seq_len)
        market_context_length = model_arch.get('market_context_length', market_context_length)

    model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        seq_len=seq_len,
        market_context_length=market_context_length
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"模型架构 (StockTransformer)")
    print(f"{'='*50}")
    print(f"输入维度: {input_dim}")
    print(f"序列长度: {seq_len}")
    print(f"市场Token: 启用 (窗口长度: {market_context_length})")
    print(f"Embedding维度: {d_model}")
    print(f"注意力头数: {nhead}")
    print(f"Transformer层数: {num_layers}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"{'='*50}\n")

    return model
