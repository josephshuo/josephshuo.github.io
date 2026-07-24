#!/usr/bin/env python3
"""
口字形变压器漏磁随铁芯磁导率变化的仿真分析
============================================

核心问题：铁芯 μr 从低到高变化时，漏磁比例如何变化？

物理直觉：
  - μr → ∞ : 铁芯是理想磁导体，磁通全部走铁芯，漏磁≈0
  - μr → 1  : 铁芯=空气，没有导磁作用，磁通全部"漏"
  - 实际 μr≈2000: 铁芯磁阻远小于空气，漏磁由几何决定

方法：
  1. 2D FDM 求解 ∇·(ν∇A_z) = -J_z，Dirichlet BC (A=0 远场)
  2. B = (∂A/∂y, -∂A/∂x)，从 B 场积分提取各路径磁通
  3. 解析磁路模型作为对照

Author: Joseph
Date: 2026-07-21
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm

# ============================================================
# 0. 中文字体
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
# 1. 几何与物理常量
# ============================================================
# --- 铁芯尺寸 (mm) ---
#           ←── 200 ──→
#     ┌────────────────────┐  ↑
#     │  ┌──────────┐      │  │
#     │  │  窗口    │      │ 160
#     │  │ 100×80  │      │  │
#     │  └──────────┘      │  │
#     └────────────────────┘  ↓
#     柱宽=50, 轭高=40

CORE_W = 200          # 铁芯外宽
CORE_H = 160          # 铁芯外高
WINDOW_W = 100        # 窗口宽
WINDOW_H = 80         # 窗口高
LEG_W = (CORE_W - WINDOW_W) / 2    # = 50 mm
YOKE_H = (CORE_H - WINDOW_H) / 2   # = 40 mm

# --- 绕组 (mm) ---
WDG_THICK = 10        # 绕组厚度
WDG_LEN = 60          # 绕组高度
WDG_GAP = 5           # 绕组与铁芯柱间隙

# --- 物理常量 ---
MU0 = 4 * np.pi * 1e-7
N_TURNS = 100
I_EXCITE = 1.0        # 励磁电流 (仅初级, 次级开路)

# --- 网格 ---
DX = 1.2              # mm/cell
MARGIN = 200          # 求解域边距 (远场边界)

# ============================================================
# 2. 解析磁路模型
# ============================================================

def mag_circuit_model(mu_r):
    """
    口字形变压器解析磁路模型。

    磁路结构:
        MMF → R_leak (窗口空气) ∥ R_core (铁芯路径)

    返回 dict: 各磁阻、漏磁比例、耦合系数
    """
    # 铁芯平均磁路长度 (mm → m)
    l_core_mm = (2 * (WINDOW_H + WINDOW_W)
                 + 4 * (LEG_W / 2 + YOKE_H / 2))
    l_core_m = l_core_mm * 1e-3
    A_core_m = LEG_W * 1e-3           # 铁芯截面积 per unit depth (m)

    R_core = l_core_m / (MU0 * mu_r * A_core_m)   # A/Wb per meter depth

    # 窗口漏磁路径 — 磁力线从初级经窗口空气到次级
    gap_window = (WINDOW_W - 2 * (WDG_THICK + WDG_GAP)) * 1e-3  # m
    A_leak_win = WDG_LEN * 1e-3 * 1.5            # 有效面积 (含边缘效应)
    R_window_leak = gap_window / (MU0 * A_leak_win)

    # 外部漏磁 — 绕组外侧经铁芯外部空气构成回路
    path_ext = np.pi * (LEG_W / 2 + WDG_THICK + WDG_GAP) * 1e-3
    A_leak_ext = WDG_LEN * 1e-3 * 2.0
    R_ext_leak = path_ext / (MU0 * A_leak_ext)

    # 漏磁路径并联
    R_leak_eq = 1.0 / (1.0 / R_window_leak + 1.0 / R_ext_leak)

    # 漏磁比例: Φ_leak / (Φ_core + Φ_leak) = R_core / (R_core + R_leak_eq)
    leak_window = R_core / (R_core + R_window_leak)
    leak_total = R_core / (R_core + R_leak_eq)
    k_coupling = np.sqrt(max(0, 1.0 - leak_total))

    return {
        'mu_r': mu_r,
        'l_core_mm': l_core_mm,
        'R_core': R_core,
        'R_window_leak': R_window_leak,
        'R_ext_leak': R_ext_leak,
        'R_leak_eq': R_leak_eq,
        'leak_window_pct': leak_window * 100,
        'leak_total_pct': leak_total * 100,
        'coupling_coeff': k_coupling,
        'phi_core_over_mmf': 1.0 / R_core,        # 磁通/MMF (per m depth)
        'phi_leak_over_mmf': 1.0 / R_leak_eq,
    }


# ============================================================
# 3. FDM 几何建模
# ============================================================

def build_grid():
    """创建网格坐标"""
    W = CORE_W + 2 * MARGIN
    H = CORE_H + 2 * MARGIN
    nx = int(W / DX) + 1
    ny = int(H / DX) + 1
    x = np.linspace(0, W, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, nx, ny


def build_core_mask(X, Y):
    """口字形铁芯布尔掩模: 外矩形 - 内矩形(窗口)"""
    cx, cy = X.mean(), Y.mean()
    hw, hh = CORE_W / 2, CORE_H / 2
    iw, ih = WINDOW_W / 2, WINDOW_H / 2
    outer = ((X >= cx - hw) & (X <= cx + hw) &
             (Y >= cy - hh) & (Y <= cy + hh))
    inner = ((X >= cx - iw) & (X <= cx + iw) &
             (Y >= cy - ih) & (Y <= cy + ih))
    return outer & ~inner


def build_winding_mask(X, Y, side, core_mask):
    """在指定柱(left/right)上创建绕组 (内外侧都放)"""
    cx, cy = X.mean(), Y.mean()
    iw = WINDOW_W / 2

    leg_cx = (cx - iw - LEG_W / 2) if side == 'left' else (cx + iw + LEG_W / 2)

    leg_inner = leg_cx + LEG_W / 2 + WDG_GAP
    leg_outer = leg_cx - LEG_W / 2 - WDG_GAP
    y_top = cy + WDG_LEN / 2
    y_bot = cy - WDG_LEN / 2

    def _rect(x0, x1):
        return ((X >= x0) & (X <= x1) &
                (Y >= y_bot) & (Y <= y_top))

    inner = _rect(leg_inner, leg_inner + WDG_THICK)
    outer = _rect(leg_outer - WDG_THICK, leg_outer)
    wdg = inner | outer
    return wdg & ~core_mask


# ============================================================
# 4. FDM 求解器
# ============================================================

def solve_fdm(X, Y, core_mask, pri_mask, mu_r):
    """
    FDM 求解 2D 静磁场。

    方程:  ∇·(ν ∇A_z) = -J_z
    BC:    A_z = 0 on all boundaries (远场 Dirichlet)

    返回: A, Bx, By, Bmag, mu_field, J
    """
    ny, nx = X.shape
    N = nx * ny
    dx_m = DX * 1e-3

    # 材料磁导率
    mu = np.full_like(X, MU0 * 1.0, dtype=np.float64)   # 默认空气 μr=1
    mu[core_mask] = MU0 * mu_r

    # 电流密度 (仅初级, 左侧柱)
    J = np.zeros_like(X, dtype=np.float64)
    if pri_mask.any():
        S_pri = np.sum(pri_mask) * (dx_m ** 2)               # 总面积 m²
        J_mag = N_TURNS * I_EXCITE / S_pri                    # A/m²
        # 内外侧电流方向相反 (绕组环绕铁芯柱)
        cx = X.mean()
        iw_val = WINDOW_W / 2
        leg_inner_edge = (cx - iw_val - LEG_W / 2) + LEG_W / 2 + WDG_GAP
        inner_part = pri_mask & (X >= leg_inner_edge)
        outer_part = pri_mask & (X < leg_inner_edge)
        J[inner_part] = +J_mag
        J[outer_part] = -J_mag

    # 磁阻率 ν = 1/μ
    nu = 1.0 / mu
    inv_dx2 = 1.0 / dx_m ** 2

    # x-方向界面谐均值
    denom_x = nu[:, 1:] + nu[:, :-1]
    denom_x[denom_x == 0] = 1e-30
    nu_x = 2.0 * nu[:, 1:] * nu[:, :-1] / denom_x

    # y-方向界面谐均值
    denom_y = nu[1:, :] + nu[:-1, :]
    denom_y[denom_y == 0] = 1e-30
    nu_y = 2.0 * nu[1:, :] * nu[:-1, :] / denom_y

    # 组装稀疏矩阵 (COO 格式)
    row, col, data = [], [], []

    for i in range(ny):
        for j in range(nx):
            k = i * nx + j

            # Dirichlet BC: 边界 A=0
            if i == 0 or i == ny - 1 or j == 0 or j == nx - 1:
                row.append(k); col.append(k); data.append(1.0)
                continue

            diag = 0.0

            # +x neighbor
            v = nu_x[i, j] * inv_dx2
            row.append(k); col.append(k + 1); data.append(-v)
            diag += v

            # -x neighbor
            v = nu_x[i, j - 1] * inv_dx2
            row.append(k); col.append(k - 1); data.append(-v)
            diag += v

            # +y neighbor
            v = nu_y[i, j] * inv_dx2
            row.append(k); col.append(k + nx); data.append(-v)
            diag += v

            # -y neighbor
            v = nu_y[i - 1, j] * inv_dx2
            row.append(k); col.append(k - nx); data.append(-v)
            diag += v

            row.append(k); col.append(k); data.append(diag)

    A_mat = sparse.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    # RHS: J, 边界置零
    b = J.ravel().copy().astype(np.float64)
    b[:nx] = 0; b[-nx:] = 0             # top/bottom rows
    b[::nx] = 0; b[nx - 1::nx] = 0       # left/right columns

    A_flat = spsolve(A_mat, b)
    A = A_flat.reshape(ny, nx)

    # B 场 (中心差分)
    Bx = np.zeros_like(A)
    By = np.zeros_like(A)
    Bx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * dx_m)      # Bx = +∂A/∂y
    By[:, 1:-1] = -(A[:, 2:] - A[:, :-2]) / (2 * dx_m)      # By = -∂A/∂x
    Bmag = np.sqrt(Bx ** 2 + By ** 2)

    return A, Bx, By, Bmag, mu, J, A_mat


# ============================================================
# 5. 漏磁分析 — 绕组中点截面磁通比法
# ============================================================

def analyze_leakage(A, Bx, By, X, Y, core_mask):
    """
    改进的漏磁计算 —— 比较左右两柱铁芯截面内的磁通。

    核心思想:
      在绕组高度中点做水平截面，分别穿过左柱铁芯和右柱铁芯:
        Φ_pri_leg = 左柱垂直磁通 (被初级绕组包围的铁芯截面)
        Φ_sec_leg = 右柱垂直磁通 (被次级绕组包围的铁芯截面)

      理想耦合: Φ_pri_leg = Φ_sec_leg (所有磁通经铁芯串联回路到达次级)
      实际情况: Φ_pri_leg > Φ_sec_leg (部分磁通经窗口空气"抄近路")

      漏磁通:  Φ_leak = Φ_pri_leg - Φ_sec_leg
      漏磁比:  leakage = Φ_leak / Φ_pri_leg

    数值方法:
      Φ = ∫_leg By·dx = ∫ -∂A/∂x·dx = A(x_left) - A(x_right)
      直接取 A_z 在铁芯柱两侧边缘的差值，多点采样取平均。

    同时保留 Bx 窗口积分作为独立验证。
    """
    dx_m = DX * 1e-3
    cx, cy = X.mean(), Y.mean()
    ny, nx = X.shape

    # --- 几何关键坐标 ---
    # 左柱铁芯 x 范围
    x_LL = cx - WINDOW_W / 2 - LEG_W     # 左柱左边缘
    x_LR = cx - WINDOW_W / 2              # 左柱右边缘 (= 窗口左边缘)
    # 右柱铁芯 x 范围
    x_RL = cx + WINDOW_W / 2              # 右柱左边缘 (= 窗口右边缘)
    x_RR = cx + WINDOW_W / 2 + LEG_W      # 右柱右边缘

    def _col(x_val):
        """x 坐标 → 网格列索引"""
        return max(0, min(nx - 1, int(round((x_val - X.min()) / DX))))

    def _row(y_val):
        """y 坐标 → 网格行索引"""
        return max(0, min(ny - 1, int(round((y_val - Y.min()) / DX))))

    def flux_through_surface(row_idx, col_L, col_R):
        """
        穿过水平截面 [col_L, col_R] @ row_idx 的垂直磁通 (per unit depth).
        Φ = ∫_{x_L}^{x_R} By(x, y) dx = A(x_L, y) - A(x_R, y)
        """
        return A[row_idx, col_L] - A[row_idx, col_R]

    # --- 多点采样: 在绕组高度范围内取 7 个 y 位置 ---
    n_samples = 7
    y_samples = np.linspace(cy - WDG_LEN / 2 + 2, cy + WDG_LEN / 2 - 2, n_samples)

    col_LL = _col(x_LL); col_LR = _col(x_LR)
    col_RL = _col(x_RL); col_RR = _col(x_RR)

    phi_pri_samples = []
    phi_sec_samples = []
    for y in y_samples:
        r = _row(y)
        phi_pri_samples.append(flux_through_surface(r, col_LL, col_LR))
        phi_sec_samples.append(flux_through_surface(r, col_RL, col_RR))

    phi_pri = abs(np.mean(phi_pri_samples))
    phi_sec = abs(np.mean(phi_sec_samples))
    phi_leak_leg = abs(phi_pri - phi_sec)

    # 漏磁比
    leak_ratio = phi_leak_leg / phi_pri if phi_pri > 0 else 0.0

    # --- 交叉验证: 上轭 Bx 积分 (铁芯磁通的独立测量) ---
    # 在上轭中点取 Bx 垂直线积分
    col_yoke = _col(cx - WINDOW_W / 2 - LEG_W / 2)  # 左柱中心线
    yoke_y = cy + WINDOW_H / 2 + YOKE_H / 2
    r_yoke_top = _row(yoke_y + YOKE_H / 2)
    r_yoke_bot = _row(yoke_y - YOKE_H / 2)
    r_y1, r_y2 = min(r_yoke_bot, r_yoke_top), max(r_yoke_bot, r_yoke_top)
    Bx_yoke = Bx[r_y1:r_y2 + 1, col_yoke]
    phi_yoke = abs(np.sum(Bx_yoke * dx_m))

    # --- 交叉验证: 窗口 Bx 积分 (直接测量漏磁通) ---
    col_win = _col(cx)
    r_win_top = _row(cy + WINDOW_H / 2)
    r_win_bot = _row(cy - WINDOW_H / 2)
    rw1, rw2 = min(r_win_bot, r_win_top), max(r_win_bot, r_win_top)
    Bx_win = Bx[rw1:rw2 + 1, col_win]
    phi_window = abs(np.sum(Bx_win * dx_m))

    return {
        'phi_pri_leg': phi_pri,          # 左柱磁通 (初级侧)
        'phi_sec_leg': phi_sec,          # 右柱磁通 (次级侧)
        'phi_leak_leg': phi_leak_leg,    # 漏磁通 = 差值
        'leak_ratio': leak_ratio,
        'leak_pct': leak_ratio * 100,
        'phi_yoke': phi_yoke,            # 上轭磁通 (独立验证)
        'phi_window': phi_window,        # 窗口漏磁 (独立验证)
        'y_samples': y_samples,
        'phi_pri_samples': phi_pri_samples,
        'phi_sec_samples': phi_sec_samples,
    }


# ============================================================
# 6. 可视化
# ============================================================

def plot_field_snapshot(ax, X, Y, A, Bmag, core_mask, pri_mask, mu_r, leak):
    """在给定 axes 上绘制磁力线和 |B| 分布"""
    # 磁力线
    Alim = np.percentile(np.abs(A[~np.isnan(A)]), [2, 98])
    if Alim[1] > 1e-12:
        levels = np.linspace(-Alim[1], Alim[1], 30)
    else:
        levels = np.linspace(-1, 1, 30)
    ax.contour(X, Y, A, levels=levels, colors='#1a1a2e', linewidths=0.4, alpha=0.7)

    # 铁芯轮廓
    ax.contour(X, Y, core_mask.astype(float), levels=[0.5],
               colors='#e74c3c', linewidths=1.8)
    # 绕组轮廓
    if pri_mask.any():
        ax.contour(X, Y, pri_mask.astype(float), levels=[0.5],
                   colors='#e67e22', linewidths=0.8, linestyles='--')

    ax.set_aspect('equal')
    ax.set_title(f'μr = {mu_r}', fontsize=11, fontweight='bold')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')

    # 漏磁信息
    ax.text(0.02, 0.98,
            f'Leakage: {leak["leak_pct"]:.1f}%\n'
            f'Φ_pri (left leg): {leak["phi_pri_leg"]*1e3:.2f} mWb/m\n'
            f'Φ_sec (right leg): {leak["phi_sec_leg"]*1e3:.2f} mWb/m\n'
            f'Φ_leak = Φ_pri − Φ_sec: {leak["phi_leak_leg"]*1e3:.3f} mWb/m',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))


def plot_bmag_snapshot(ax, X, Y, Bmag, core_mask, mu_r):
    """在给定 axes 上绘制 |B| 云图"""
    B_disp = Bmag * 1e3   # T → mT
    im = ax.pcolormesh(X, Y, B_disp, cmap='inferno', shading='auto',
                       norm=LogNorm(vmax=B_disp.max(), vmin=max(B_disp[B_disp > 0].min(), 1e-3)))
    ax.contour(X, Y, core_mask.astype(float), levels=[0.5],
               colors='cyan', linewidths=1.2)
    ax.set_aspect('equal')
    ax.set_title(f'|B| (mT), μr = {mu_r}', fontsize=11, fontweight='bold')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, shrink=0.85)


def plot_mu_sweep_results(mur_list, leak_list, mag_list):
    """
    主结果图 — 漏磁 vs μr。

    4 面板:
      (a) 漏磁比例 vs μr (log-log)
      (b) 各路径磁通绝对值 vs μr
      (c) 磁阻 vs μr
      (d) 耦合系数 vs μr
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    mur = np.array(mur_list)

    # 提取 FDM 结果
    leak_pct = np.array([r['leak_pct'] for r in leak_list])
    phi_pri = np.array([r['phi_pri_leg'] * 1e3 for r in leak_list])   # mWb/m
    phi_sec = np.array([r['phi_sec_leg'] * 1e3 for r in leak_list])
    phi_leak = np.array([r['phi_leak_leg'] * 1e3 for r in leak_list])
    phi_yoke = np.array([r['phi_yoke'] * 1e3 for r in leak_list])
    phi_win = np.array([r['phi_window'] * 1e3 for r in leak_list])

    # 提取解析模型结果
    mag_leak = np.array([m['leak_total_pct'] for m in mag_list])
    mag_k = np.array([m['coupling_coeff'] for m in mag_list])

    # (a) 漏磁比例 vs μr
    ax = axes[0, 0]
    ax.loglog(mur, leak_pct, 'o-', color='#e74c3c', linewidth=2, markersize=8,
              label='FDM (Φ_pri − Φ_sec)/Φ_pri')
    ax.loglog(mur, mag_leak, 's--', color='#3498db', linewidth=2, markersize=8,
              label='Magnetic circuit model')
    ax.set_xlabel('Core relative permeability μr', fontsize=11)
    ax.set_ylabel('Leakage ratio (%)', fontsize=11)
    ax.set_title('(a) Leakage % vs μr\n(lower μr → more leakage)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 标注关键区域
    ax.axvspan(1, 10, alpha=0.08, color='red')
    ax.axvspan(100, 1000, alpha=0.08, color='green')
    ax.axvspan(5000, 20000, alpha=0.08, color='blue')
    ax.text(2.5, ax.get_ylim()[0] * 2, 'Low μr\n(air-like)', fontsize=8,
            ha='center', color='red', alpha=0.7)
    ax.text(300, ax.get_ylim()[0] * 2, 'Transition', fontsize=8,
            ha='center', color='green', alpha=0.7)
    ax.text(8000, ax.get_ylim()[0] * 2, 'High μr\n(ideal iron)', fontsize=8,
            ha='center', color='blue', alpha=0.7)

    # (b) 左右柱磁通 vs μr
    ax = axes[0, 1]
    ax.loglog(mur, phi_pri, 'o-', color='#27ae60', linewidth=2, markersize=8,
              label='Φ_pri_leg (left leg, primary side)')
    ax.loglog(mur, phi_sec, 's-', color='#3498db', linewidth=2, markersize=8,
              label='Φ_sec_leg (right leg, secondary side)')
    ax.loglog(mur, phi_leak, 'D-', color='#e74c3c', linewidth=1.5, markersize=6,
              label='Φ_leak = Φ_pri − Φ_sec')
    ax.loglog(mur, phi_yoke, 'v--', color='#95a5a6', linewidth=1.5, markersize=5,
              alpha=0.7, label='Φ_yoke (Bx integral, cross-check)')
    ax.loglog(mur, phi_win, '^:', color='#f39c12', linewidth=1.5, markersize=5,
              alpha=0.7, label='Φ_window (Bx window integral)')
    ax.set_xlabel('Core relative permeability μr', fontsize=11)
    ax.set_ylabel('Flux per unit depth (mWb/m)', fontsize=11)
    ax.set_title('(b) Leg fluxes vs μr\n(Φ_pri − Φ_sec = leakage flux bypassing the secondary)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 标注: Φ_pri ≈ Φ_sec at high μr (ideal coupling)
    ax.annotate('Φ_pri ≈ Φ_sec\n(near-ideal coupling)',
                xy=(mur[-1], phi_pri[-1]),
                xytext=(mur[len(mur)//2], phi_pri[-1] * 0.3),
                fontsize=9, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    # (c) 磁阻 vs μr
    ax = axes[1, 0]
    R_core_vals = np.array([m['R_core'] for m in mag_list])
    R_leak_vals = np.array([m['R_leak_eq'] for m in mag_list])

    ax.loglog(mur, R_core_vals, 'o-', color='#8e44ad', linewidth=2, markersize=8,
              label='R_core (iron path)')
    ax.axhline(y=R_leak_vals[0], color='#e74c3c', linewidth=2, linestyle='--',
               label=f'R_leak_eq ≈ {R_leak_vals[0]:.0f} (constant, air path)')
    ax.set_xlabel('Core relative permeability μr', fontsize=11)
    ax.set_ylabel('Reluctance (A/Wb per m)', fontsize=11)
    ax.set_title('(c) Magnetic reluctance vs μr\n(R_core ∝ 1/μr, R_leak = constant)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 交叉点
    cross_idx = np.argmin(np.abs(R_core_vals - R_leak_vals[0]))
    ax.annotate(f'R_core = R_leak\nat μr ≈ {mur[cross_idx]}',
                xy=(mur[cross_idx], R_core_vals[cross_idx]),
                xytext=(mur[cross_idx] * 3, R_core_vals[cross_idx] * 0.3),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.5))

    # (d) 耦合系数 vs μr
    ax = axes[1, 1]

    # FDM 耦合系数: k = Φ_sec / Φ_pri (leg flux ratio, direct measurement)
    k_fdm = np.array([1.0 - r['leak_ratio'] for r in leak_list])

    ax.semilogx(mur, k_fdm, 'o-', color='#e74c3c', linewidth=2, markersize=8,
                label='FDM: k = Φ_sec / Φ_pri (leg flux ratio)')
    ax.semilogx(mur, mag_k, 's--', color='#3498db', linewidth=2, markersize=8,
                label='Mag. circuit: k = √(1−leakage)')
    ax.set_xlabel('Core relative permeability μr', fontsize=11)
    ax.set_ylabel('Coupling coefficient k', fontsize=11)
    ax.set_title('(d) Coupling coefficient vs μr\n(k = Φ_sec/Φ_pri → 1 as μr → ∞)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 标注理想耦合线
    ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.text(mur[0], 1.01, 'Ideal coupling (k=1)', fontsize=8,
            color='gray', alpha=0.7)

    fig.suptitle('Square-Core Transformer: Leakage Flux vs Core Permeability\n'
                 'FDM simulation + Magnetic circuit model',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def plot_flux_line_grid(all_results):
    """
    展示不同 μr 下的磁力线快照 (3×3 网格)。
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    for idx, (ax, res) in enumerate(zip(axes.flat, all_results)):
        plot_field_snapshot(ax, res['X'], res['Y'], res['A'],
                           res['Bmag'], res['core_mask'], res['pri_mask'],
                           res['mu_r'], res['leak'])
    fig.suptitle('Flux line evolution with μr\n'
                 'Low μr: flux spreads into air  |  High μr: flux confined to core',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================
# 7. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("  Square-Core Transformer Leakage vs μr — FDM Simulation")
    print("=" * 60)

    # μr 扫描范围
    # 从 μr=1 (空气) 到 μr=20000 (高导磁硅钢)
    mu_r_values = [1, 2, 5, 10, 20, 50, 100, 200, 500,
                   1000, 2000, 5000, 10000, 20000]

    # 构建一次几何 (几何不变)
    print("\n[1] Building geometry...")
    x, y, X, Y, nx, ny = build_grid()
    core_mask = build_core_mask(X, Y)
    pri_mask = build_winding_mask(X, Y, 'left', core_mask)
    sec_mask = build_winding_mask(X, Y, 'right', core_mask)

    print(f"    Grid: {nx} × {ny} = {nx * ny} nodes")
    print(f"    Core: {np.sum(core_mask)} cells")
    print(f"    Primary: {np.sum(pri_mask)} cells")
    print(f"    Secondary: {np.sum(sec_mask)} cells")

    # 解析磁路模型
    print("\n[2] Magnetic circuit model...")
    mag_results = [mag_circuit_model(mu) for mu in mu_r_values]
    for m in mag_results:
        print(f"    μr={m['mu_r']:5d}:  leakage={m['leak_total_pct']:6.2f}%,  "
              f"k={m['coupling_coeff']:.4f},  "
              f"R_core={m['R_core']:.0f},  R_leak={m['R_leak_eq']:.0f} A/Wb")

    # FDM 扫描
    print("\n[3] FDM sweep (this will take a while)...")
    fdm_results = []
    all_results = []  # 用于磁力线快照的子集

    SNAPSHOT_MUR = [1, 5, 50, 200, 500, 1000, 2000, 5000, 20000]  # 9 个快照值

    for mu_r in mu_r_values:
        print(f"    Solving μr = {mu_r:5d} ...", end=" ", flush=True)

        A, Bx, By, Bmag, mu_field, J, A_mat = solve_fdm(X, Y, core_mask, pri_mask, mu_r)
        leak = analyze_leakage(A, Bx, By, X, Y, core_mask)

        fdm_results.append(leak)

        print(f"Leakage: {leak['leak_pct']:5.1f}%,  "
              f"Φ_pri={leak['phi_pri_leg']*1e3:.3f},  "
              f"Φ_sec={leak['phi_sec_leg']*1e3:.3f},  "
              f"Φ_leak={leak['phi_leak_leg']*1e3:.3f},  "
              f"Φ_yoke={leak['phi_yoke']*1e3:.3f} mWb/m")

        # 保存快照
        if mu_r in SNAPSHOT_MUR:
            all_results.append({
                'X': X, 'Y': Y, 'A': A, 'Bmag': Bmag,
                'core_mask': core_mask, 'pri_mask': pri_mask,
                'mu_r': mu_r, 'leak': leak,
            })

    # 打印汇总表
    print("\n[4] Summary Table")
    print(f"    {'μr':>6s}  {'FDM leak%':>10s}  {'Mag. leak%':>11s}  "
          f"{'k (FDM)':>8s}  {'k (mag)':>8s}  {'Φ_pri-leak':>11s}")
    print("    " + "-" * 68)
    for mu_r, leak, mag in zip(mu_r_values, fdm_results, mag_results):
        k_fdm = 1.0 - leak['leak_ratio']   # k = Φ_sec/Φ_pri directly
        print(f"    {mu_r:6d}  {leak['leak_pct']:10.2f}  {mag['leak_total_pct']:11.2f}  "
              f"{k_fdm:8.4f}  {mag['coupling_coeff']:8.4f}  "
              f"{leak['leak_pct']:5.1f}% = (Φ_pri−Φ_sec)/Φ_pri")

    # 可视化
    print("\n[5] Generating plots...")

    fig_main = plot_mu_sweep_results(mu_r_values, fdm_results, mag_results)
    fig_main.savefig('leakage_vs_mu.png', dpi=150, bbox_inches='tight')
    print("    → leakage_vs_mu.png")

    fig_grid = plot_flux_line_grid(all_results)
    fig_grid.savefig('flux_lines_grid.png', dpi=150, bbox_inches='tight')
    print("    → flux_lines_grid.png")

    # 关键物理洞察
    print("\n" + "=" * 60)
    print("  Key Physics Insights")
    print("=" * 60)
    print(f"""
  1. 改进的漏磁测量方法:
     Φ_pri = 左柱铁芯垂直磁通 (A_z 差值法, 绕组高度内多点采样)
     Φ_sec = 右柱铁芯垂直磁通 (同上)
     Φ_leak = Φ_pri − Φ_sec  (经窗口空气路径旁路的磁通)
     leakage% = 100 × Φ_leak / Φ_pri

  2. 方法优势:
     → 直接比较原副边绕组所包围铁芯截面内的磁通
     → 物理意义清晰: Φ_pri 是初级产生的总磁通, Φ_sec 是到达次级的磁通
     → 不需要猜测测量线位置
     → 多点采样平均, 数值鲁棒

  3. 高 μr 下 Φ_pri ≈ Φ_sec (耦合接近理想)
     → Φ_pri 和 Φ_sec 都 ∝ μr, 两者几乎重合
     → 漏磁通 Φ_leak 随 μr 增大趋近于几何决定的常数值
     → k = Φ_sec/Φ_pri → 1

  4. 低 μr 下 Φ_pri >> Φ_sec (严重漏磁)
     → 铁芯导磁能力弱, 大量磁通经窗口空气"抄近路"
     → Φ_sec 远小于 Φ_pri

  5. 交叉验证:
     → Φ_yoke (上轭 Bx 积分) 应与 Φ_pri, Φ_sec 一致 (磁路连续性)
     → Φ_window (窗口 Bx 积分) 应近似等于 Φ_leak (窗口漏磁通)
     → 多点采样标准差可评估测量可靠性

  6. 实际意义:
     → 铁芯接缝气隙 (0.05-0.2mm) 等效 μr_eff << 2000
     → 实测漏磁通常 5-15%
     → 本方法可直接输出耦合系数 k = Φ_sec/Φ_pri
""")

    plt.show()


if __name__ == '__main__':
    main()
