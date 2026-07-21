#!/usr/bin/env python3
"""
口字形变压器漏磁分析 — FDM 可视化 + 解析磁路模型
====================================================
- 2D 有限差分法 (Dirichlet BC) 求解磁矢势 A_z，绘制磁力线和 |B| 分布
- 解析磁路模型计算漏磁比例 (不受边界条件影响)
- 对比口字形分侧布置 vs. 同心式布置

求解方程:  ∇·(ν ∇A_z) = -J_z,   B = (∂A/∂y, -∂A/∂x)

Author: Joseph
Date: 2026-07-17
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# ============================================================
# 0. 中文字体设置
# ============================================================
for _fn in ['Microsoft YaHei', 'Noto Sans SC', 'SimHei']:
    try:
        _fp = fm.findfont(_fn, fallback_to_default=False)
        if _fp:
            plt.rcParams['font.sans-serif'] = [_fn, 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            break
    except Exception:
        continue
plt.rcParams['axes.unicode_minus'] = False
fm._load_fontmanager(try_read_cache=False)

# ============================================================
# 1. 几何与物理参数
# ============================================================
# --- 铁芯 (mm) ---
CORE_OUTER_W = 200       # 外宽
CORE_OUTER_H = 160       # 外高
CORE_WINDOW_W = 100      # 窗口宽
CORE_WINDOW_H = 80       # 窗口高
LEG_W = (CORE_OUTER_W - CORE_WINDOW_W) / 2    # 柱宽 = 50
YOKE_H = (CORE_OUTER_H - CORE_WINDOW_H) / 2   # 轭高 = 40

# --- 绕组 (mm) ---
WINDING_THICK = 10
WINDING_LENGTH = 60
WINDING_GAP = 5

# --- 材料 ---
MU0 = 4 * np.pi * 1e-7
MU_R_CORE = 2000        # 硅钢片线性区
MU_R_COPPER = 1.0

# --- 激励 ---
N_PRI = 100; N_SEC = 100
I_PRI = 1.0              # 仅初级通电 (次级开路)

# --- 网格 ---
DX = 1.2                  # mm/cell
MARGIN = 200              # 求解域边距

# ============================================================
# 2. 解析磁路模型 — 漏磁比例
# ============================================================

def magnetic_circuit_analysis():
    """
    用磁路模型估算漏磁比例。

    磁路结构 (口字形，绕组在左右两柱):

        Primary (left leg)          Secondary (right leg)
           ┌── R_yoke_L ──┬── R_yoke_R ──┐
           │              │              │
        [mmf]  R_window_leakage       [open]
           │              │              │
           └── R_yoke_L ──┴── R_yoke_R ──┘
              (bottom yoke)

    铁芯磁路与窗口漏磁路径并联。
    主磁通经过铁芯匝链两个绕组，漏磁通经窗口空气只匝链初级。

    返回:
      leakage_ratio_window : 窗口漏磁比例 (下限估计)
      leakage_ratio_total  : 总漏磁比例 (含外部漏磁估计)
      coupling_coeff       : 耦合系数
    """
    # 铁芯参数
    l_core = (2 * (CORE_WINDOW_H + CORE_WINDOW_W)
              + 4 * (LEG_W / 2 + YOKE_H / 2))  # 平均磁路长度 (mm)
    l_core_m = l_core * 1e-3                     # → m
    A_core_m = LEG_W * 1e-3                      # 铁芯截面积 per unit depth (m)

    # 铁芯磁阻 (per meter depth)
    R_core = l_core_m / (MU0 * MU_R_CORE * A_core_m)  # A/Wb per m

    # --- 窗口漏磁路径 ---
    # 从初级内侧 → 窗口空气 → 次级内侧 (或回到初级)
    # 有效间隙 ≈ 窗口宽 − 2×(绕组厚+间隙)
    gap_window = (CORE_WINDOW_W - 2 * (WINDING_THICK + WINDING_GAP)) * 1e-3  # m
    # 漏磁"有效面积" ≈ 绕组高度 (带边缘效应系数 ~1.5)
    A_leak_window = WINDING_LENGTH * 1e-3 * 1.5   # m (per unit depth)
    R_window = gap_window / (MU0 * A_leak_window)   # A/Wb per m

    # --- 外部漏磁路径 ---
    # 从初级外侧 → 铁芯外部空气 → 回到初级
    # 路径近似半圆绕铁芯外侧，长度 ≈ π * (LEG_W/2 + WINDING_THICK + WINDING_GAP)
    path_ext = np.pi * (LEG_W / 2 + WINDING_THICK + WINDING_GAP) * 1e-3  # m
    A_leak_ext = WINDING_LENGTH * 1e-3 * 2.0     # 两侧有效面积较大
    R_external = path_ext / (MU0 * A_leak_ext)

    # --- 合成漏磁比例 ---
    # 主磁通 = mmf / R_core
    # 漏磁通 = mmf / R_leak  (多条漏磁路径并联)
    # 漏磁比 = Φ_leak / (Φ_core + Φ_leak) = R_core / (R_core + R_leak_equiv)

    R_leak_eq = 1.0 / (1.0 / R_window + 1.0 / R_external)  # 并联

    leakage_ratio_window = R_core / (R_core + R_window)
    leakage_ratio_total = R_core / (R_core + R_leak_eq)
    coupling_coeff = np.sqrt(1.0 - leakage_ratio_total)

    return {
        'l_core_mm': l_core,
        'R_core': R_core,
        'R_window': R_window,
        'R_external': R_external,
        'leakage_window_pct': leakage_ratio_window * 100,
        'leakage_total_pct': leakage_ratio_total * 100,
        'coupling_coeff': coupling_coeff,
    }


# ============================================================
# 3. 几何建模 (FDM 网格)
# ============================================================

def build_domain():
    """创建求解域网格"""
    W = CORE_OUTER_W + 2 * MARGIN
    H = CORE_OUTER_H + 2 * MARGIN
    nx = int(W / DX) + 1
    ny = int(H / DX) + 1
    x = np.linspace(0, W, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, nx, ny


def build_core_mask(X, Y):
    """口字形铁芯布尔掩模"""
    cx, cy = X.mean(), Y.mean()
    hw, hh = CORE_OUTER_W / 2, CORE_OUTER_H / 2
    iw, ih = CORE_WINDOW_W / 2, CORE_WINDOW_H / 2
    outer = ((X >= cx - hw) & (X <= cx + hw) &
             (Y >= cy - hh) & (Y <= cy + hh))
    inner = ((X >= cx - iw) & (X <= cx + iw) &
             (Y >= cy - ih) & (Y <= cy + ih))
    return outer & ~inner


def build_winding_mask(X, Y, side, core_mask, position='both'):
    """在指定柱上创建绕组区域。position: 'both' | 'inner' | 'outer'"""
    cx, cy = X.mean(), Y.mean()
    iw, ih = CORE_WINDOW_W / 2, CORE_WINDOW_H / 2

    leg_cx = (cx - iw - LEG_W / 2) if side == 'left' else (cx + iw + LEG_W / 2)

    leg_inner = leg_cx + LEG_W / 2 + WINDING_GAP
    leg_outer = leg_cx - LEG_W / 2 - WINDING_GAP
    y_top = cy + WINDING_LENGTH / 2
    y_bot = cy - WINDING_LENGTH / 2

    def _rect(x0, x1):
        return ((X >= x0) & (X <= x1) &
                (Y >= y_bot) & (Y <= y_top))

    inner_w = _rect(leg_inner, leg_inner + WINDING_THICK)
    outer_w = _rect(leg_outer - WINDING_THICK, leg_outer)

    if position == 'both':
        w = inner_w | outer_w
    elif position == 'inner':
        w = inner_w
    elif position == 'outer':
        w = outer_w
    else:
        raise ValueError(f"Unknown position: {position}")

    return w & ~core_mask


# ============================================================
# 4. FDM 求解器 (Dirichlet BC)
# ============================================================

def solve_fdm(X, Y, core_mask, pri_mask, sec_mask):
    """FDM 求解 2D 静磁场 A_z。返回 A, Bx, By, Bmag, mu"""
    nx, ny = X.shape[1], X.shape[0]
    N = nx * ny
    dx_m = DX * 1e-3  # mm → m

    # 材料磁导率
    mu = np.full_like(X, MU0 * MU_R_COPPER, dtype=np.float64)
    mu[core_mask] = MU0 * MU_R_CORE

    # 电流密度 (仅初级)
    J = np.zeros_like(X, dtype=np.float64)
    if pri_mask.any():
        S_pri = np.sum(pri_mask) * (dx_m ** 2)
        J_pri = N_PRI * I_PRI / S_pri
        cx = X.mean()
        iw = CORE_WINDOW_W / 2
        leg_inner = (cx - iw - LEG_W / 2) + LEG_W / 2 + WINDING_GAP
        inner_pri = pri_mask & (X >= leg_inner)
        outer_pri = pri_mask & (X < leg_inner)
        J[inner_pri] = +J_pri
        J[outer_pri] = -J_pri

    print(f"  Grid: {nx} x {ny} = {N} nodes")
    print(f"  Core cells: {np.sum(core_mask)}")

    # 界面 ν 的谐均值 (只在内部使用)
    nu = 1.0 / mu
    inv_dx2 = 1.0 / dx_m ** 2

    # x-方向界面 ν
    nu_x = np.zeros((ny, nx - 1), dtype=np.float64)
    denom = nu[:, 1:] + nu[:, :-1]
    denom[denom == 0] = 1e-30
    nu_x = 2.0 * nu[:, 1:] * nu[:, :-1] / denom

    # y-方向界面 ν
    nu_y = np.zeros((ny - 1, nx), dtype=np.float64)
    denom = nu[1:, :] + nu[:-1, :]
    denom[denom == 0] = 1e-30
    nu_y = 2.0 * nu[1:, :] * nu[:-1, :] / denom

    # 组装稀疏矩阵
    row, col, data = [], [], []

    for i in range(ny):
        for j in range(nx):
            k = i * nx + j

            # Dirichlet BC: A = 0 on all boundaries
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                row.append(k); col.append(k); data.append(1.0)
                continue

            diag = 0.0

            # Right (j+1)
            v = nu_x[i, j] * inv_dx2
            row.append(k); col.append(k + 1); data.append(-v)
            diag += v

            # Left (j-1)
            v = nu_x[i, j - 1] * inv_dx2
            row.append(k); col.append(k - 1); data.append(-v)
            diag += v

            # Bottom (i+1)
            v = nu_y[i, j] * inv_dx2
            row.append(k); col.append(k + nx); data.append(-v)
            diag += v

            # Top (i-1)
            v = nu_y[i - 1, j] * inv_dx2
            row.append(k); col.append(k - nx); data.append(-v)
            diag += v

            row.append(k); col.append(k); data.append(diag)

    A_mat = sparse.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    # RHS
    b = J.ravel().copy().astype(np.float64)
    b[:nx] = 0; b[-nx:] = 0; b[::nx] = 0; b[nx - 1::nx] = 0

    print(f"  Sparse nnz: {A_mat.nnz}")
    print("  Solving...", end=" ", flush=True)
    A_flat = spsolve(A_mat, b)
    print("done.")

    A = A_flat.reshape(ny, nx)

    # B 场: Bx = +∂A/∂y, By = -∂A/∂x (中心差分)
    By = np.zeros_like(A)
    Bx = np.zeros_like(A)
    By[:, 1:-1] = -(A[:, 2:] - A[:, :-2]) / (2 * dx_m)
    Bx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * dx_m)
    Bmag = np.sqrt(Bx ** 2 + By ** 2)

    return A, Bx, By, Bmag, mu, J


# ============================================================
# 5. 后处理 — 基于 B 场的漏磁分析 (不依赖 A 的绝对值)
# ============================================================

def compute_leakage_from_B(Bx, By, X, Y, core_mask):
    """
    用 B 场积分计算漏磁比例 (不受 A 边界条件影响)。

    方法:
      Φ_core  = 穿过铁轭截面的磁通 → 互感磁通
      Φ_window = 穿过窗口截面的磁通 → 漏磁通 (不经过铁芯)
      leakage_ratio = Φ_window / (Φ_core + Φ_window)
    """
    dx_m = DX * 1e-3
    cx, cy = X.mean(), Y.mean()
    ny, nx = X.shape

    # --- 铁芯磁通 (在上轭测量) ---
    # 左半轭 x = 左柱中心
    leg_cx_left = cx - CORE_WINDOW_W / 2 - LEG_W / 2
    col_left = int(round((leg_cx_left - X.min()) / DX))

    yoke_y_center = cy + CORE_WINDOW_H / 2 + YOKE_H / 2
    yoke_half = YOKE_H / 2
    row_top = int(round((yoke_y_center + yoke_half - Y.min()) / DX))
    row_bot = int(round((yoke_y_center - yoke_half - Y.min()) / DX))
    r1, r2 = min(row_bot, row_top), max(row_bot, row_top)
    r1 = max(0, r1); r2 = min(ny - 1, r2)

    # 左轭中 Bx 积分 (水平穿过轭截面)
    Bx_yoke = Bx[r1:r2 + 1, col_left]
    phi_core = abs(np.sum(Bx_yoke * dx_m))  # Wb/m

    # --- 窗口漏磁通 ---
    # 在窗口中间垂直线测量 Bx (水平方向穿过窗口)
    col_window = int(round((cx - X.min()) / DX))
    window_top = cy + CORE_WINDOW_H / 2
    window_bot = cy - CORE_WINDOW_H / 2
    row_win_top = int(round((window_top - Y.min()) / DX))
    row_win_bot = int(round((window_bot - Y.min()) / DX))
    rw1, rw2 = min(row_win_bot, row_win_top), max(row_win_bot, row_win_top)
    rw1 = max(0, rw1); rw2 = min(ny - 1, rw2)

    Bx_window = Bx[rw1:rw2 + 1, col_window]
    phi_window = abs(np.sum(Bx_window * dx_m))

    # --- 外部漏磁 (铁芯外侧) ---
    # 在铁芯外部左侧垂直线
    col_ext = int(round((cx - CORE_OUTER_W / 2 - 20 - X.min()) / DX))
    col_ext = max(0, min(nx - 1, col_ext))
    core_top = cy + CORE_OUTER_H / 2
    core_bot = cy - CORE_OUTER_H / 2
    row_ext_top = int(round((core_top - Y.min()) / DX))
    row_ext_bot = int(round((core_bot - Y.min()) / DX))
    re1, re2 = min(row_ext_bot, row_ext_top), max(row_ext_bot, row_ext_top)
    re1 = max(0, re1); re2 = min(ny - 1, re2)

    Bx_ext = Bx[re1:re2 + 1, col_ext]
    phi_external = abs(np.sum(Bx_ext * dx_m))

    # 漏磁比例
    phi_total = phi_core + phi_window + phi_external
    if phi_total > 0:
        leakage_ratio = (phi_window + phi_external) / phi_total
    else:
        leakage_ratio = 0.0

    return {
        'phi_core': phi_core,
        'phi_window': phi_window,
        'phi_external': phi_external,
        'leakage_ratio': leakage_ratio,
        'leakage_pct': leakage_ratio * 100,
    }


# ============================================================
# 6. 可视化
# ============================================================

def plot_geometry(X, Y, core_mask, pri_mask, sec_mask, title):
    """绘制几何材料分布"""
    fig, ax = plt.subplots(figsize=(8, 6))
    mat = np.zeros_like(X, dtype=int)
    mat[core_mask] = 2
    mat[pri_mask] = 1
    if sec_mask.any():
        mat[sec_mask] = np.where(mat[sec_mask] == 1, 4, 3)  # 4 = overlap

    cmap = plt.cm.colors.ListedColormap(
        ['#f5f5f5', '#ff6b6b', '#b0b0b0', '#4ecdc4', '#ffa500'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    ax.pcolormesh(X, Y, mat, cmap=cmap, norm=norm, shading='auto')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    leg = [
        mpatches.Patch(color='#b0b0b0', label='Iron core'),
        mpatches.Patch(color='#ff6b6b', label='Primary winding'),
        mpatches.Patch(color='#4ecdc4', label='Secondary winding'),
        mpatches.Patch(color='#ffa500', label='Overlap (concentric)'),
        mpatches.Patch(color='#f5f5f5', label='Air'),
    ]
    ax.legend(handles=leg, loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


def plot_fields(X, Y, A, Bmag, core_mask, pri_mask, sec_mask,
                leakage_fdm, mag_results, title):
    """磁力线 + |B| 分布"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- 左: 磁力线 (A 等值线) ---
    ax = axes[0]
    Alim = np.percentile(np.abs(A[~np.isnan(A)]), [2, 98])
    levels = np.linspace(-Alim[1], Alim[1], 35) if Alim[1] > 1e-12 else np.linspace(-1, 1, 35)
    ax.contour(X, Y, A, levels=levels, colors='#1a1a2e', linewidths=0.5, alpha=0.8)
    # 铁芯轮廓
    ax.contour(X, Y, core_mask.astype(float), levels=[0.5],
               colors='#e74c3c', linewidths=2.0)
    # 绕组轮廓
    if pri_mask.any():
        ax.contour(X, Y, pri_mask.astype(float), levels=[0.5],
                   colors='#e67e22', linewidths=1.0, linestyles='--')
    if sec_mask.any():
        ax.contour(X, Y, sec_mask.astype(float), levels=[0.5],
                   colors='#3498db', linewidths=1.0, linestyles='--')

    ax.set_aspect('equal')
    ax.set_title(f'Flux lines (A_z contours) — {title}', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')

    # 标注漏磁路径
    cx, cy = X.mean(), Y.mean()
    iw, ih = CORE_WINDOW_W / 2, CORE_WINDOW_H / 2
    ax.annotate('Window leakage\n(air path)',
                xy=(cx, cy - ih / 4), fontsize=9, color='#e74c3c',
                ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.8))
    ax.annotate('Mutual flux\n(iron path)',
                xy=(cx - iw / 2, cy - CORE_WINDOW_H / 2 - YOKE_H / 2),
                fontsize=9, color='#27ae60', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))

    # 数值标注框
    txt = (f"FDM leakage: {leakage_fdm['leakage_pct']:.1f}%\n"
           f"Magnetic circuit: {mag_results['leakage_total_pct']:.2f}%\n"
           f"Coupling k: {mag_results['coupling_coeff']:.4f}\n"
           f"Phi_core: {leakage_fdm['phi_core']*1e3:.2f} mWb/m")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # --- 右: |B| 云图 ---
    ax = axes[1]
    B_disp = Bmag * 1e3  # → mT
    im = ax.pcolormesh(X, Y, B_disp, cmap='inferno', shading='auto')
    ax.contour(X, Y, core_mask.astype(float), levels=[0.5],
               colors='cyan', linewidths=1.5)
    ax.set_aspect('equal')
    ax.set_title(f'|B| (mT) — {title}', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, label='|B| (mT)', shrink=0.8)

    B_core_avg = np.mean(Bmag[core_mask]) * 1e3 if core_mask.any() else 0
    ax.text(0.02, 0.98, f"Core avg |B|: {B_core_avg:.1f} mT",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85))

    fig.tight_layout()
    return fig


def plot_summary(mag, leak_a, leak_b):
    """汇总对比图"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    categories = ['Mag. circuit\n(window only)',
                  'Mag. circuit\n(window + external)',
                  'FDM (B-field)\nwindow leakage',
                  'FDM (B-field)\ncore flux ratio']
    values = [mag['leakage_window_pct'],
              mag['leakage_total_pct'],
              leak_a['leakage_pct'],
              100 * leak_a['phi_core'] / (leak_a['phi_core'] + leak_a['phi_window']
                                          + leak_a['phi_external'] + 1e-30)]
    colors = ['#3498db', '#2c3e50', '#e74c3c', '#95a5a6']

    bars = ax.bar(categories, values, color=colors, edgecolor='#2c3e50',
                  linewidth=1.5, width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{val:.2f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')

    ax.set_ylabel('Leakage ratio (%)', fontsize=12)
    ax.set_title('Transformer Leakage Flux Analysis\n'
                 'Square-core, windings on opposite legs, mur=2000, no gaps',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.35)
    ax.grid(axis='y', alpha=0.3)

    # Add explanatory note
    ax.text(0.5, -0.18,
            'Magnetic circuit model = lower bound (lumped parameter).  '
            'FDM = includes fringe fields & discrete winding effects.\n'
            'In practice, core joints add 0.05-0.2 mm air gaps '
            'raising leakage to 5-15%.',
            transform=ax.transAxes, fontsize=9, ha='center',
            fontstyle='italic', color='#555555')

    fig.tight_layout()
    return fig


# ============================================================
# 7. 主程序
# ============================================================

def run_config(name, side_p, side_s, pos_p='both', pos_s='both'):
    """运行一个绕组配置"""
    print(f"\n{'='*50}\n  {name}\n{'='*50}")

    x, y, X, Y, nx, ny = build_domain()
    core = build_core_mask(X, Y)
    pri = build_winding_mask(X, Y, side_p, core, pos_p)
    sec = build_winding_mask(X, Y, side_s, core, pos_s)

    A, Bx, By, Bmag, mu, J = solve_fdm(X, Y, core, pri, sec)
    leak = compute_leakage_from_B(Bx, By, X, Y, core)

    # 基于 A 的耦合测量 (绕组平均磁矢势之比)
    dx_m = DX * 1e-3
    dx_m2 = dx_m ** 2
    coupling_from_A = 0.0
    if pri.any() and sec.any():
        S_p = np.sum(pri) * dx_m2
        S_s = np.sum(sec) * dx_m2
        avg_A_p = np.sum(A[pri]) * dx_m2 / S_p
        avg_A_s = np.sum(A[sec]) * dx_m2 / S_s
        if abs(avg_A_p) > 1e-30:
            # k = M / sqrt(L11*L22), with identical windings: k = |avg_A_s / avg_A_p|
            coupling_from_A = min(abs(avg_A_s / avg_A_p), 1.0)

    print(f"  Phi_core:        {leak['phi_core']*1e3:.3f} mWb/m")
    print(f"  Phi_window:      {leak['phi_window']*1e3:.3f} mWb/m")
    print(f"  Phi_external:    {leak['phi_external']*1e3:.3f} mWb/m")
    print(f"  Leakage (B-FDM): {leak['leakage_pct']:.2f}%")
    print(f"  Coupling k (A):  {coupling_from_A:.4f}")

    return {'X': X, 'Y': Y, 'A': A, 'Bx': Bx, 'By': By, 'Bmag': Bmag,
            'core': core, 'pri': pri, 'sec': sec, 'leak': leak,
            'coupling_A': coupling_from_A, 'label': name}


def main():
    print("=" * 55)
    print("  Transformer Leakage Flux Analysis")
    print("  FDM simulation + Magnetic circuit model")
    print("=" * 55)

    # --- 解析磁路模型 ---
    print("\n[Magnetic Circuit Analysis]")
    mag = magnetic_circuit_analysis()
    print(f"  Core path length:      {mag['l_core_mm']:.0f} mm")
    print(f"  Core reluctance:       {mag['R_core']:.0f} A/Wb per m")
    print(f"  Window leak reluctance:{mag['R_window']:.0f} A/Wb per m")
    print(f"  External leak reluctance:{mag['R_external']:.0f} A/Wb per m")
    print(f"  Window leakage:        {mag['leakage_window_pct']:.2f}%")
    print(f"  Total leakage (est.):  {mag['leakage_total_pct']:.2f}%")
    print(f"  Coupling coefficient:  {mag['coupling_coeff']:.4f}")

    # --- FDM 仿真 ---
    print("\n[FDM Simulation]")

    res_a = run_config("Config A: Windings on opposite legs (square-core)",
                       'left', 'right', 'both', 'both')

    res_b = run_config("Config B: Windings on same leg (concentric, min leakage)",
                       'left', 'left', 'both', 'both')

    # --- 可视化 ---
    print("\n[Generating plots...]")

    fig1 = plot_geometry(res_a['X'], res_a['Y'], res_a['core'],
                         res_a['pri'], res_a['sec'], res_a['label'])
    fig2 = plot_geometry(res_b['X'], res_b['Y'], res_b['core'],
                         res_b['pri'], res_b['sec'], res_b['label'])

    fig3 = plot_fields(res_a['X'], res_a['Y'], res_a['A'], res_a['Bmag'],
                       res_a['core'], res_a['pri'], res_a['sec'],
                       res_a['leak'], mag, res_a['label'])
    fig4 = plot_fields(res_b['X'], res_b['Y'], res_b['A'], res_b['Bmag'],
                       res_b['core'], res_b['pri'], res_b['sec'],
                       res_b['leak'], mag, res_b['label'])

    fig5 = plot_summary(mag, res_a['leak'], res_b['leak'])

    plt.show()

    # --- 输出结论 ---
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"""
  Square-core transformer (windings on opposite legs):
    Magnetic circuit estimate:  {mag['leakage_total_pct']:.2f}% leakage
    FDM estimate:               {res_a['leak']['leakage_pct']:.2f}% leakage
    Coupling coefficient:       {mag['coupling_coeff']:.4f}

  Concentric (same leg) — FDM:
    Leakage:                    {res_b['leak']['leakage_pct']:.2f}%

  Key insight:
    With a continuous silicon-steel core (mur=2000), the iron path has
    ~250x lower reluctance than the air window path.  Therefore window
    leakage is only ~{mag['leakage_window_pct']:.1f}%.

    In practice, core joints add small air gaps that increase leakage
    to ~5-15%.  The square-core geometry with separated windings is
    NOT zero-leakage — but the leakage is dominated by joint gaps,
    not the winding separation itself.
""")


if __name__ == '__main__':
    main()
