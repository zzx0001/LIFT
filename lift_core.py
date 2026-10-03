import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SemanticsEmbedding(nn.Module):
    """
    Module A: 语义感知嵌入
    负责处理多变量类型、数值，并引入基于采样间隔的门控机制。
    """
    def __init__(self, num_types, input_dim, hidden_dim):
        super().__init__()
        # 变量类型的 Embedding (e.g., 心率, 血压)
        self.type_emb = nn.Embedding(num_types, hidden_dim)
        # 数值的映射层
        self.value_proj = nn.Linear(input_dim, hidden_dim)

        # Missingness Gating 机制网络
        # 输入是 [Type_Emb; Delta_t]，输出一个 0-1 的门控值
        self.gating_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, values, times, types, mask, delta_t=None):
        """
        values: (B, L, 1) 观测值
        times: (B, L) 绝对时间戳
        types: (B, L) 变量类型索引
        mask: (B, L) Padding mask (1 for valid, 0 for padded)
        """
        B, L = times.shape

        # 1. 计算时间间隔 Delta t (用于密度感知和门控)
        # 默认保留原逻辑：按 observation 序列相邻差分。
        # LIFT plugin 可传入外部 delta_t，避免 dense 多变量展开带来的伪高采样密度。
        if delta_t is None:
            delta_t = torch.zeros_like(times)
            delta_t[:, 1:] = times[:, 1:] - times[:, :-1]
        else:
            delta_t = delta_t.to(device=times.device, dtype=times.dtype)
        # 避免 delta_t 为 0 或负数（由于数据噪音），加一个小 epsilon
        delta_t = torch.clamp(delta_t, min=1e-4) * mask

        # 2. 获取语义特征
        type_h = self.type_emb(types)  # (B, L, H)
        value_h = self.value_proj(values)  # (B, L, H)

        # 3. 计算门控 (Missingness Gating)
        # 拼接类型信息和时间间隔信息
        gating_input = torch.cat([type_h, delta_t.unsqueeze(-1)], dim=-1)
        gate = self.gating_net(gating_input)  # (B, L, H)

        # 4. 融合：(数值特征 * 门控) + 类型特征
        # 门控决定了我们多大程度上信任当前的数值观测
        combined_h = (value_h * gate) + type_h

        # Apply mask and normalize
        combined_h = combined_h * mask.unsqueeze(-1)
        combined_h = self.layer_norm(combined_h)

        return combined_h, delta_t


class CausalContinuousGaborTokenizer(nn.Module):
    """
    Module B (Enhanced): 物理感知因果 Gabor 分词器
    包含:
    1. Learnable Gabor Atoms
    2. Strictly Causal Masking
    3. Density Compensation
    4. Frequency Trust Mask (FTM) - 抑制欠采样区域的高频幻觉
    """
    def __init__(
        self,
        input_dim,
        num_atoms,
        grid_size,
        max_time=48.0,
        dc=False,
        use_phase=False,
        trend_period_min=12.0,
        trend_period_max=48.0,
        event_period_min=None,
        event_period_max=12.0,
        trend_beta_init=24.0,
        event_beta_init=4.0,
        trust_temperature_init=5.0,
        use_ftm=True,
        use_density_comp=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_atoms = num_atoms
        self.grid_size = grid_size
        self.use_phase = use_phase
        self.use_ftm = use_ftm
        self.use_density_comp = use_density_comp

        self.dc = dc

        # 定义输出的规则时间网格点 tau_j
        # 我们将不规则的 t_n 投影到这些规则的 tau_j 上
        self.register_buffer('grid_points', torch.linspace(0, max_time, grid_size))

        # --- 可学习参数 ---
        # ==========================================
        # 1. 计算分组数量 (50/50)
        # ==========================================
        n_trend = num_atoms // 2
        n_event = num_atoms - n_trend

        # ==========================================
        # 2. 分别生成初始化频率 (Log-Uniform)
        # ==========================================
        dt = max_time / grid_size

        # --- A. 趋势组 ---
        trend_min_p, trend_max_p = float(trend_period_min), float(trend_period_max)
        omega_trend_min = 2 * torch.pi / trend_max_p
        omega_trend_max = 2 * torch.pi / trend_min_p
        log_trend = torch.rand(n_trend) * (np.log(omega_trend_max) - np.log(omega_trend_min)) + np.log(omega_trend_min)

        # --- B. 事件组 ---
        if event_period_min is None:
            event_period_min = 6.0 * dt
        event_min_p, event_max_p = float(event_period_min), float(event_period_max)
        omega_event_min = 2 * torch.pi / event_max_p
        omega_event_max = 2 * torch.pi / event_min_p
        log_event = torch.rand(n_event) * (np.log(omega_event_max) - np.log(omega_event_min)) + np.log(omega_event_min)

        # ==========================================
        # 3. 定义参数 (开放 Trend, 开放 Event)
        # ==========================================
        # Trend 组: requires_grad=True (训练)
        self.omega_trend = nn.Parameter(torch.exp(log_trend), requires_grad=True)
        self.raw_beta_trend = nn.Parameter(torch.ones(n_trend) * float(trend_beta_init), requires_grad=True)

        # Event 组: requires_grad=True (训练)
        self.omega_event = nn.Parameter(torch.exp(log_event), requires_grad=True)
        self.raw_beta_event = nn.Parameter(torch.ones(n_event) * float(event_beta_init), requires_grad=True)

        # --- FTM 参数 ---
        # 温度系数，控制 Mask 的陡峭程度 (Sigmoid 的斜率)
        # 温度越高，Mask 越接近 0/1 硬截断；温度越低，边界越平滑
        self.trust_temperature = nn.Parameter(torch.tensor(float(trust_temperature_init)))

    def get_sigma(self):
        # 拼接 beta
        raw_beta = torch.cat([self.raw_beta_trend, self.raw_beta_event])
        beta = F.softplus(raw_beta)

        # 拼接 omega
        omega = torch.cat([self.omega_trend, self.omega_event])
        curr_omega = torch.abs(omega)

        sigma = beta / (curr_omega + 1.0)
        return sigma

    def forward(self, h, times, delta_t, mask):
        """
        h: (B, L, H)
        times: (B, L)
        delta_t: (B, L)
        mask: (B, L)
        """
        B, L, H_in = h.shape
        G = self.grid_size
        K = self.num_atoms

        # 1. 准备广播: Input Time vs Output Grid
        t_n = times.view(B, L, 1, 1)
        tau_j = self.grid_points.view(1, 1, G, 1)
        dist = tau_j - t_n  # (B, L, G, 1)

        # 2. 严格因果 Mask (Causal Mask)
        causal_mask = (dist >= 0).float()
        final_mask = causal_mask * mask.view(B, L, 1, 1)

        # 3. 计算 Gabor 核心组件
        omega_full = torch.cat([self.omega_trend, self.omega_event])
        omega_k = omega_full.view(1, 1, 1, K)
        sigma_k = self.get_sigma().view(1, 1, 1, K)

        # 复数指数部分: exp(i * omega * (tau - t))
        # 利用欧拉公式 e^(ix) = cos(x) + i*sin(x)
        phase = omega_k * dist
        real_part = torch.cos(phase)
        imag_part = torch.sin(phase)

        # 高斯窗部分: exp(- (tau - t)^2 / 2*sigma^2)
        # Shape: (B, L, G, K) - 注意这里每个频率 atom 的窗宽可能不同
        gaussian_window = torch.exp(- (dist**2) / (2 * sigma_k**2))

        # 组合 Gabor 权重 (复数)
        # Shape: (B, L, G, K)
        gabor_weights_real = real_part * gaussian_window * final_mask
        gabor_weights_imag = imag_part * gaussian_window * final_mask

        # =========================================================
        # Frequency Trust Mask (FTM) 计算模块
        # =========================================================

        # 归一化权重 (防止除以0)
        weight_per_atom = gaussian_window * final_mask  # (B, L, G, K)
        sum_weights = weight_per_atom.sum(dim=1) + 1e-6

        # 加权平均 Delta T: (B, L, 1, 1) * (B, L, G, K) -> sum -> (B, G, K)
        dt_input = delta_t.view(B, L, 1, 1)

        if self.use_ftm:
            local_avg_dt = (dt_input * weight_per_atom).sum(dim=1) / sum_weights
            # B. 计算局部 Nyquist 极限频率
            nyquist_limit = torch.pi / (local_avg_dt + 1e-6)  # (B, G, K)
            # C. 生成 Mask
            current_omega = torch.abs(omega_k).squeeze(1)  # (1, 1, K) ->广播到 (B, G, K)
            freq_diff = nyquist_limit - current_omega  # (B, G, K)
            trust_mask = torch.sigmoid(self.trust_temperature * freq_diff)
        else:
            # Ablation: disable FTM. trust_mask ≡ 1 (all atoms equally trusted).
            trust_mask = torch.ones(B, self.grid_size, self.num_atoms,
                                    device=h.device, dtype=h.dtype)

        # 将 Mask 扩展维度以便应用到特征上: (B, G, K) -> (B, G, K, 1)
        trust_mask = trust_mask.unsqueeze(-1)

        # =========================================================
        # 4. 物理感知投影 (带 Density Compensation)
        # =========================================================
        if self.use_density_comp:
            density_weight = dt_input.unsqueeze(-1)  # (B, L, 1, 1, 1)
        else:
            # Ablation: disable density compensation. density_weight ≡ 1.
            density_weight = torch.ones_like(dt_input).unsqueeze(-1)
        h_weighted = h.view(B, L, 1, 1, H_in) * density_weight

        # 投影 (Einstein Summation over L)
        proj_real = torch.sum(h_weighted * gabor_weights_real.unsqueeze(-1), dim=1)
        proj_imag = torch.sum(h_weighted * gabor_weights_imag.unsqueeze(-1), dim=1)

        if self.use_phase:
            # 方案 A: 保留相位 (Real + Imag) -> 维度 2*H
            masked_real = proj_real * trust_mask
            masked_imag = proj_imag * trust_mask
            masked_spectral_feature = torch.cat([masked_real, masked_imag], dim=-1)
        else:
            # 方案 B: 仅使用模长 (Magnitude) -> 维度 1*H
            spectral_magnitude = torch.sqrt(proj_real**2 + proj_imag**2 + 1e-8)
            masked_spectral_feature = spectral_magnitude * trust_mask

        final_output = masked_spectral_feature

        if self.dc:
            # =========================================================
            # 6. [方案 A] 找回直流分量 (Trend Feature)
            # =========================================================
            max_sigma_idx = torch.argmax(self.get_sigma())

            # 取出这个最宽窗口对应的权重: (B, L, G)
            trend_window = gaussian_window[:, :, :, max_sigma_idx]

            # 计算加权平均的分母 (权重的和)
            trend_weight = trend_window * final_mask.squeeze(-1)  # (B, L, G)
            sum_trend_weight = trend_weight.sum(dim=1, keepdim=True) + 1e-6  # (B, 1, G)

            # 使用 einsum 进行加权求和: B, L, H 和 B, L, G -> B, G, H
            weighted_sum_h = torch.einsum('blh,blg->bgh', h, trend_weight)

            # 得到趋势特征 (B, G, H)
            trend_feature = weighted_sum_h / sum_trend_weight.squeeze(1).unsqueeze(-1)

            B, G, K, H = masked_spectral_feature.shape

            # A. 展平频率特征 (使用 Masked 版本!)
            freq_feat = masked_spectral_feature.view(B, G, K * H)

            # B. 拼接: [频率特征; 趋势特征]
            # Output shape: (B, G, K*H + H)
            final_output = torch.cat([freq_feat, trend_feature], dim=-1)
        else:
            # 如果没有 DC，直接展平
            B, G, K, Feat_Dim = masked_spectral_feature.shape
            final_output = masked_spectral_feature.view(B, G, K * Feat_Dim)

        return final_output
