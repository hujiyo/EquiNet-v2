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

import os
import sys

# 确保能正确导入其他模块（无论从哪里运行）
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_file_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import math
import torch
import torch.nn as nn
from src.config import ModelConfig, DataConfig

def init_weights(module):
    """
    当代主流Transformer初始化策略

    设计原则：
    1. Embedding层: 根据层类型和维度计算gain，统一输出std=0.2
    2. FFN第一层: Xavier初始化，gain=1.7补偿GELU压缩
    3. FFN第二层: Xavier初始化，gain=1.0（无激活函数）
    4. 输出层: 小增益，避免sigmoid饱和
    5. LayerNorm: weight=1, bias=0

    Embedding初始化计算（目标std=0.2）：
    - Linear层: 输出std = σ_input × gain × sqrt(2×fan_in/(fan_in+fan_out))
    - Embedding层: 输出std = gain × sqrt(2/(vocab_size+embedding_dim))

    各层gain计算结果：
    - Stock Token (Linear 6→48): gain=0.42
    - Market Token (Linear 30→48): gain=0.23
    - Position (Embedding 31→48): gain=1.26
    - Segment (Embedding 2→48): gain=1.0
    """
    ffn_hidden_dim = ModelConfig.D_MODEL * ModelConfig.FFN_EXPAND_RATIO

    if isinstance(module, nn.Linear):
        if module.out_features == 1:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.OUTPUT_LAYER_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ModelConfig.INPUT_DIM and module.out_features == ModelConfig.D_MODEL:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.EMBEDDING_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.out_features == ModelConfig.D_MODEL and module.in_features != ModelConfig.D_MODEL and module.in_features != ffn_hidden_dim:
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
    elif isinstance(module, nn.Embedding):
        # 区分Position和Segment embedding
        # Position: vocab_size=CONTEXT_LENGTH+1=31
        # Segment: vocab_size=TOKEN_TYPE_NUM=2
        if module.weight.shape[0] == DataConfig.CONTEXT_LENGTH + 1:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.POSITION_EMBEDDING_INIT_GAIN)
        elif module.weight.shape[0] == DataConfig.TOKEN_TYPE_NUM:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.TOKEN_TYPE_EMBEDDING_INIT_GAIN)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1.0)


class PositionalAndTypeEncoding(nn.Module):
    """
    位置编码 + 类型编码（BERT风格）
    
    位置编码：所有token都加，表示在序列中的位置
    类型编码：区分不同类型的token（day token vs market token）
    
    参考BERT的实现：
    - position_embeddings: Embedding(max_position_embeddings, hidden_size)
    - token_type_embeddings: Embedding(type_vocab_size, hidden_size)
    - final_embedding = token_embedding + position_embedding + token_type_embedding
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalAndTypeEncoding, self).__init__()
        # 位置编码：31个位置（30个day token + 1个market token）
        self.position_embeddings = nn.Embedding(seq_len + 1, d_model)
        # 类型编码：2个类型（0=day token, 1=market token）
        self.token_type_embeddings = nn.Embedding(DataConfig.TOKEN_TYPE_NUM, d_model)

    def forward(self, x, token_type_ids):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            token_type_ids: [batch_size, seq_len]  # 0=day token, 1=market token
        
        Returns:
            [batch_size, seq_len, d_model] 添加位置编码和类型编码后的embedding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        position_emb = self.position_embeddings(positions).unsqueeze(0)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        return x + position_emb + token_type_emb


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

        # 使用位置编码 + 类型编码（BERT风格）
        self.pos_encoding = PositionalAndTypeEncoding(d_model, seq_len)

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
        """
        Args:
            x: [batch_size, seq_len, 6] (OHLC + volume + exchange)
            market_seq: [batch_size, market_context_length] 大盘涨跌序列

        Returns:
            output: [batch_size, 1] 预测logits
        """
        # 1. Embedding
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        market_token = self.market_encoder(market_seq)  # [batch_size, 1, d_model]
        x = torch.cat([x, market_token], dim=1)  # [batch_size, seq_len+1, d_model]
        
        # 2. 创建token_type_ids：区分day token和market token
        # token_type_ids: [batch_size, seq_len+1]
        # 0 = day token (前seq_len个)
        # 1 = market token (最后1个)
        token_type_ids = torch.zeros(x.size(0), x.size(1), dtype=torch.long, device=x.device)
        token_type_ids[:, -1] = 1  # 最后一个token是market token
        
        # 3. 位置编码 + 类型编码
        x = self.pos_encoding(x, token_type_ids)
        x = self.dropout(x)

        # 4. Transformer层（Pre-Norm架构）
        for layer in self.layers:
            x = layer(x)

        # 5. Pre-Norm架构需要在最后进行归一化
        #    因为每层的输出没有经过归一化
        x = self.final_norm(x)

        # 6. 多头注意力聚合
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
