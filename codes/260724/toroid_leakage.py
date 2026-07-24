#!/usr/bin/env python3
"""
环形铁芯 (Toroid) 变压器漏磁分析 — 2D FDM
==========================================

对比两种绕组配置:
  Config A — 交替绕组 (uniform winding): 原副边均布整个圆周
    理论上漏磁≈0，因为绕组处处紧贴铁芯，磁动势均匀分布

  Config B — 分侧绕组 (opposite sides):  原边绕左半圆，副边绕右半圆
    磁通须沿铁芯走 180° 到达副边，中心孔提供漏磁捷径

方法:
  - 俯视 2D FDM (x-y 平面): 铁芯 = 圆环 (R_out, R_in)
  - 绕组在 2D 中表现为铁芯内侧 (r<R_in) 和外侧 (r>R_out) 的载流区
  - 漏磁测量: Φ(φ) = A_z(r_in, φ) - A_z(r_out, φ)
    沿圆周多点采样，比较 Φ_pri 与 Φ_sec

物理方程:  ∇·(ν ∇A_z) = -J_z,   B_φ = -∂A_z/∂r
BC: A_z = 0 on far field (Dirichlet)

环形铁芯理想解: B_φ(r) = μ₀μᵣ N I / (2π r),  r ∈ [R_in, R_out]

Author: Joseph
Date: 2026-07-23
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.patches import Wedge, Circle

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
# 1. 几何与物理参数
# ============================================================

# --- 环形铁芯 (mm) ---
R_OUT = 100.0           # 外半径
R_IN = 55.0             # 内半径
CORE_THICK = R_OUT - R_IN  # = 45 mm, 铁芯径向厚度

# --- 绕组 (mm) ---
WDG_THICK = 8.0         # 绕组径向厚度
WDG_GAP = 3.0           # 绕组与铁芯表面间隙

# --- 材料 ---
MU0 = 4 * np.pi * 1e-7
MU_R_CORE = 2000

# --- 激励 ---
N_TURNS = 100
I_EXCITE = 1.0           # 仅初级通电 (次级开路)

# --- 网格 ---
DX = 1.0                 # mm/cell
MARGIN = 80              # 求解域边距 (mm)

# --- 分侧绕组角度范围 (Config B) ---
# 左半: 原边,  右半: 副边 (开路)
# 左半定义: x < cx, 即角度 [π/2, 3π/2]
# 为避免边缘效应，两侧各缩进 10°
ANGLE_MARGIN = np.deg2rad(10)

# ============================================================
# 2. 环形铁芯解析模型
# ============================================================

def toroid_analytical(mu_r):
    """
    理想均匀绕制环形铁芯的解析解。

    返回 B_φ(r), Φ, R_core, L (per unit height in z).
    """
    # 铁芯磁路长度 (平均周长)
    r_avg = (R_OUT + R_IN) / 2 * 1e-3  # m
    l_core = 2 * np.pi * r_avg          # m

    # 铁芯截面积 per unit height (m)
    A_core = (R_OUT - R_IN) * 1e-3      # m² per m depth

    # 铁芯磁阻 per unit height
    R_core = l_core / (MU0 * mu_r * A_core)   # A/Wb per m

    # 理想环形铁芯内的 B 场
    # B_φ(r) = μ₀μᵣ N I / (2π r)
    # 磁通 per unit height: Φ' = ∫_{R_in}^{R_out} B_φ dr
    #   = μ₀μᵣ N I / (2π) × ln(R_out / R_in)
    phi_ideal = MU0 * mu_r * N_TURNS * I_EXCITE / (2 * np.pi) * np.log(R_OUT / R_IN)

    # 中心孔漏磁路径磁阻 (近似)
    # 漏磁跨过直径 ≈ 2R_in, 有效面积 ≈ 2R_in per unit height
    R_hole_leak = (2 * R_IN * 1e-3) / (MU0 * (2 * R_IN * 1e-3))  # ≈ 1/μ₀ !
    # 实际上是: 漏磁路径长度 ≈ 2R_in, 截面积 ≈ πR_in (半圆区域)
    # 更准确的估计:
    path_hole = 2 * R_IN * 1e-3       # m (diameter)
    area_hole = np.pi * R_IN * 1e-3    # m² per m (half-circumference × 1m depth)
    R_hole_leak = path_hole / (MU0 * area_hole * 0.5)  # 0.5 factor for distributed path

    return {
        'mu_r': mu_r,
        'r_avg_mm': r_avg * 1e3,
        'l_core_m': l_core,
        'A_core_m': A_core,
        'R_core': R_core,
        'phi_ideal': phi_ideal,
        'R_hole_leak': R_hole_leak,
        'leakage_est': R_core / (R_core + R_hole_leak) * 100,
    }


# ============================================================
# 3. 几何建模 (FDM 网格)
# ============================================================

def build_domain():
    """创建求解域"""
    W = 2 * R_OUT + 2 * MARGIN
    H = 2 * R_OUT + 2 * MARGIN
    nx = int(W / DX) + 1
    ny = int(H / DX) + 1
    x = np.linspace(0, W, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, nx, ny


def build_core_mask(X, Y):
    """环形铁芯掩模: R_in ≤ r ≤ R_out"""
    cx, cy = X.mean(), Y.mean()
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (R >= R_IN) & (R <= R_OUT)


def build_winding_mask(X, Y, config, side='primary'):
    """
    构建绕组区域。

    Config A (uniform): 绕组覆盖整个圆周
      - inner: r ∈ [R_in - WDG_GAP - WDG_THICK, R_in - WDG_GAP], 全 360°
      - outer: r ∈ [R_out + WDG_GAP, R_out + WDG_GAP + WDG_THICK], 全 360°

    Config B (opposite): 绕组只在指定半圆
      - primary (left):  φ ∈ [π/2+δ, 3π/2-δ]
      - secondary (right): φ ∈ [-π/2+δ, π/2-δ]
    """
    cx, cy = X.mean(), Y.mean()
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    Phi = np.arctan2(Y - cy, X - cx)  # ∈ [-π, π]

    # 径向区间
    inner_ring = (R >= R_IN - WDG_GAP - WDG_THICK) & (R <= R_IN - WDG_GAP)
    outer_ring = (R >= R_OUT + WDG_GAP) & (R <= R_OUT + WDG_GAP + WDG_THICK)

    if config == 'A':
        # 全圆周
        return (inner_ring | outer_ring)

    elif config == 'B':
        if side == 'primary':
            # 左半: φ ∈ [π/2 + δ, 3π/2 - δ]  即 x < cx 的区域
            # 等价于 cos(φ) < 0
            left_mask = np.cos(Phi) <= -np.sin(ANGLE_MARGIN)
            ang_mask = left_mask
        else:
            # 右半: φ ∈ [-π/2 + δ, π/2 - δ]  即 x > cx 的区域
            right_mask = np.cos(Phi) >= np.sin(ANGLE_MARGIN)
            ang_mask = right_mask

        return (inner_ring | outer_ring) & ang_mask

    else:
        raise ValueError(f"Unknown config: {config}")


# ============================================================
# 4. FDM 求解器 (矢量化矩阵组装)
# ============================================================

def solve_fdm(X, Y, core_mask, pri_mask, mu_r):
    """
    FDM 求解 2D 静磁场 A_z — 矢量化矩阵组装。

    5 点差分: ∇·(ν∇A) = -J
    返回 A, Bx, By, Bmag, mu_field, J
    """
    ny, nx = X.shape
    N = nx * ny
    dx_m = DX * 1e-3
    cx, cy = X.mean(), Y.mean()

    # 材料磁导率
    mu = np.full_like(X, MU0 * 1.0, dtype=np.float64)
    mu[core_mask] = MU0 * mu_r

    # 电流密度 —— 分侧归一化: 内侧总电流 = +NI, 外侧总电流 = -NI
    J = np.zeros_like(X, dtype=np.float64)
    if pri_mask.any():
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        inner_pri = pri_mask & (R < (R_IN + R_OUT) / 2)
        outer_pri = pri_mask & (R >= (R_IN + R_OUT) / 2)
        S_inner = np.sum(inner_pri) * (dx_m ** 2)
        S_outer = np.sum(outer_pri) * (dx_m ** 2)
        if S_inner > 0:
            J[inner_pri] = +N_TURNS * I_EXCITE / S_inner    # ∫J·dS = +NI
        if S_outer > 0:
            J[outer_pri] = -N_TURNS * I_EXCITE / S_outer    # ∫J·dS = -NI

    # 磁阻率 ν = 1/μ
    nu = 1.0 / mu
    inv_dx2 = 1.0 / dx_m ** 2

    # --- 矢量化界面谐均值 ---
    # nu_x[i,j]: 界面在 (i,j) 和 (i,j+1) 之间
    denom_x = nu[:, 1:] + nu[:, :-1]
    denom_x[denom_x == 0] = 1e-30
    nu_x = 2.0 * nu[:, 1:] * nu[:, :-1] / denom_x   # shape: (ny, nx-1)

    # nu_y[i,j]: 界面在 (i,j) 和 (i+1,j) 之间
    denom_y = nu[1:, :] + nu[:-1, :]
    denom_y[denom_y == 0] = 1e-30
    nu_y = 2.0 * nu[1:, :] * nu[:-1, :] / denom_y   # shape: (ny-1, nx)

    # --- 矢量化组装 ---
    # 内部节点: i ∈ [1, ny-2], j ∈ [1, nx-2]
    ii, jj = np.meshgrid(np.arange(1, ny - 1), np.arange(1, nx - 1), indexing='ij')
    k_interior = (ii * nx + jj).ravel()   # 内部节点的全局索引

    # 四个邻居的系数
    # +x: nu_x[i, j]
    coeff_px = (nu_x[ii, jj] * inv_dx2).ravel()
    # -x: nu_x[i, j-1]
    coeff_mx = (nu_x[ii, jj - 1] * inv_dx2).ravel()
    # +y: nu_y[i, j]
    coeff_py = (nu_y[ii, jj] * inv_dx2).ravel()
    # -y: nu_y[i-1, j]
    coeff_my = (nu_y[ii - 1, jj] * inv_dx2).ravel()
    # 对角 = 四邻居之和
    coeff_diag = (coeff_px + coeff_mx + coeff_py + coeff_my)

    # 构建 COO 格式：对角
    all_k = [k_interior]
    all_col = [k_interior]
    all_data = [coeff_diag]
    # +x: k → k+1
    all_k.append(k_interior); all_col.append(k_interior + 1); all_data.append(-coeff_px)
    # -x: k → k-1
    all_k.append(k_interior); all_col.append(k_interior - 1); all_data.append(-coeff_mx)
    # +y: k → k+nx
    all_k.append(k_interior); all_col.append(k_interior + nx); all_data.append(-coeff_py)
    # -y: k → k-nx
    all_k.append(k_interior); all_col.append(k_interior - nx); all_data.append(-coeff_my)

    # 边界条件: Dirichlet A=0
    boundary = np.zeros(N, dtype=bool)
    boundary[:nx] = True; boundary[-nx:] = True           # top/bottom rows
    boundary[::nx] = True; boundary[nx - 1::nx] = True     # left/right cols
    k_boundary = np.where(boundary)[0]

    all_k.append(k_boundary); all_col.append(k_boundary)
    all_data.append(np.ones(len(k_boundary), dtype=np.float64))

    # 合并并构建 CSR 矩阵
    row = np.concatenate(all_k)
    col = np.concatenate(all_col)
    data = np.concatenate(all_data)
    A_mat = sparse.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()

    # RHS
    b = J.ravel().copy().astype(np.float64)
    b[boundary] = 0

    print("solving...", end=" ", flush=True)
    A_flat = spsolve(A_mat, b)
    print("done.", flush=True)

    A = A_flat.reshape(ny, nx)

    # B 场 (中心差分)
    Bx = np.zeros_like(A)
    By = np.zeros_like(A)
    Bx[1:-1, :] = (A[2:, :] - A[:-2, :]) / (2 * dx_m)
    By[:, 1:-1] = -(A[:, 2:] - A[:, :-2]) / (2 * dx_m)
    Bmag = np.sqrt(Bx ** 2 + By ** 2)

    return A, Bx, By, Bmag, mu, J


# ============================================================
# 5. 漏磁分析 — 沿圆周多点测量 Φ(φ)
# ============================================================

def analyze_leakage_toroid(A, Bx, By, X, Y, core_mask, pri_mask):
    """
    环形铁芯漏磁分析。

    方法:
      Φ(φ) = A_z(R_in, φ) - A_z(R_out, φ)
      即穿过径向截面 [R_in, R_out] 的方位角方向磁通 (per unit height).

      对理想均匀绕制的环形铁芯: Φ(φ) = constant (与 φ 无关)
      对分侧绕组: Φ(φ_pri) > Φ(φ_sec), 差值 = 漏磁通

    沿圆周在 N_phi 个等距角度采样, 比较 Φ 的变化。
    """
    cx, cy = X.mean(), Y.mean()
    ny, nx = X.shape

    N_PHI = 36  # 每 10° 采样一次
    phi_angles = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

    fluxes = []
    for phi in phi_angles:
        # 采样点: 铁芯内缘和外缘
        x_in = cx + (R_IN + 1.0) * np.cos(phi)
        y_in = cy + (R_IN + 1.0) * np.sin(phi)
        x_out = cx + (R_OUT - 1.0) * np.cos(phi)
        y_out = cy + (R_OUT - 1.0) * np.sin(phi)

        col_in = int(round((x_in - X.min()) / DX))
        row_in = int(round((y_in - Y.min()) / DX))
        col_out = int(round((x_out - X.min()) / DX))
        row_out = int(round((y_out - Y.min()) / DX))

        # Clamp 到有效范围
        col_in = max(0, min(nx - 1, col_in))
        row_in = max(0, min(ny - 1, row_in))
        col_out = max(0, min(nx - 1, col_out))
        row_out = max(0, min(ny - 1, row_out))

        A_in = A[row_in, col_in]
        A_out = A[row_out, col_out]
        phi_val = A_in - A_out  # Φ(φ) per unit height
        fluxes.append(phi_val)

    fluxes = np.array(fluxes)

    # 分侧绕组的 Φ_pri 和 Φ_sec
    # 原边在左半 (φ ≈ π), 副边在右半 (φ ≈ 0)
    # 取平均值
    idx_left = (phi_angles > np.pi / 2) & (phi_angles < 3 * np.pi / 2)
    idx_right = ~idx_left

    phi_pri = np.abs(np.mean(fluxes[idx_left]))
    phi_sec = np.abs(np.mean(fluxes[idx_right]))
    phi_mean = np.abs(np.mean(fluxes))

    # 漏磁比: (Φ_pri - Φ_sec) / Φ_pri
    phi_leak = abs(phi_pri - phi_sec)
    leak_ratio = phi_leak / phi_pri if phi_pri > 0 else 0.0

    # 交叉验证: 取另一条直径测量
    idx_top = (phi_angles > 0) & (phi_angles < np.pi)
    idx_bot = (phi_angles > np.pi) & (phi_angles < 2 * np.pi)
    phi_top = np.abs(np.mean(fluxes[idx_top]))
    phi_bot = np.abs(np.mean(fluxes[idx_bot]))

    # 通量变化标准差 (衡量漏磁均匀性)
    phi_std = np.std(np.abs(fluxes))
    phi_uniformity = phi_std / phi_mean if phi_mean > 0 else 1.0

    return {
        'phi_angles': phi_angles,
        'fluxes': fluxes,
        'phi_pri': phi_pri,
        'phi_sec': phi_sec,
        'phi_leak': phi_leak,
        'phi_mean': phi_mean,
        'phi_top': phi_top,
        'phi_bot': phi_bot,
        'phi_std': phi_std,
        'phi_uniformity': phi_uniformity,
        'leak_ratio': leak_ratio,
        'leak_pct': leak_ratio * 100,
        'coupling_k': 1.0 - leak_ratio,
    }


# ============================================================
# 6. 可视化
# ============================================================

def plot_config_comparison(res_a, res_b, ana):
    """
    主对比图: 2 行 × 3 列。

    上行: Config A (交替绕组)
    下行: Config B (分侧绕组)
    列: 几何 | 磁力线 | |B| 云图
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row_idx, (res, label) in enumerate([(res_a, 'Config A: Uniform winding'),
                                              (res_b, 'Config B: Opposite sides')]):
        X, Y = res['X'], res['Y']
        A, Bmag = res['A'], res['Bmag']
        core, pri, sec = res['core_mask'], res['pri_mask'], res['sec_mask']
        leak = res['leak']

        cx, cy = X.mean(), Y.mean()

        # (a) 几何
        ax = axes[row_idx, 0]
        mat = np.zeros_like(X, dtype=int)
        mat[core] = 3           # 铁芯 → 灰色
        mat[pri] = 1             # 原边 → 红色
        if sec.any():
            mat[sec] = np.where(mat[sec] == 1, 4, 2)  # 副边 → 蓝色, 重叠 → 橙色
        cmap = plt.cm.colors.ListedColormap(
            ['#f5f5f5', '#ff6b6b', '#4ecdc4', '#b0b0b0', '#ffa500'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        ax.pcolormesh(X, Y, mat, cmap=cmap, shading='auto',
                      norm=BoundaryNorm(bounds, cmap.N))
        # 中心孔虚线
        ax.add_patch(Circle((cx, cy), R_IN, fill=False, color='#e74c3c',
                           linewidth=1.5, linestyle='--'))
        # 外缘虚线
        ax.add_patch(Circle((cx, cy), R_OUT, fill=False, color='#2c3e50',
                           linewidth=1, linestyle='-', alpha=0.4))
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nGeometry (red=pri, blue=sec, gray=core)', fontsize=11, fontweight='bold')

        # (b) 磁力线
        ax = axes[row_idx, 1]
        Alim = np.percentile(np.abs(A[~np.isnan(A)]), [2, 98])
        levels = np.linspace(-Alim[1], Alim[1], 30) if Alim[1] > 1e-12 else np.linspace(-1, 1, 30)
        ax.contour(X, Y, A, levels=levels, colors='#1a1a2e', linewidths=0.4, alpha=0.7)
        ax.contour(X, Y, core.astype(float), levels=[0.5],
                   colors='#e74c3c', linewidths=1.8)
        if pri.any():
            ax.contour(X, Y, pri.astype(float), levels=[0.5],
                       colors='#e67e22', linewidths=0.6, linestyles='--')
        ax.set_aspect('equal')
        ax.set_title(f'Flux lines (A_z contours)', fontsize=11, fontweight='bold')

        # 漏磁信息
        ax.text(0.02, 0.98,
                f'Leakage: {leak["leak_pct"]:.2f}%\n'
                f'k = Φ_sec/Φ_pri = {leak["coupling_k"]:.4f}\n'
                f'Φ_uniformity σ = {leak["phi_uniformity"]:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

        # (c) |B| 云图
        ax = axes[row_idx, 2]
        B_disp = Bmag * 1e3
        vmin = max(B_disp[B_disp > 0].min() if (B_disp > 0).any() else 1e-3, 1e-3)
        im = ax.pcolormesh(X, Y, B_disp, cmap='inferno', shading='auto',
                          norm=LogNorm(vmax=B_disp.max(), vmin=vmin))
        ax.contour(X, Y, core.astype(float), levels=[0.5],
                   colors='cyan', linewidths=1.2)
        ax.set_aspect('equal')
        ax.set_title(f'|B| (mT)', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle('Toroidal Transformer: Uniform vs Opposite-Side Windings\n'
                 f'μr = {MU_R_CORE}, NI = {N_TURNS * I_EXCITE} At, open-circuit test',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_flux_vs_angle(res_a, res_b):
    """Φ(φ) 沿圆周变化 — 对比两种配置"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), subplot_kw={'projection': 'polar'})

    for ax, (res, label, color) in zip(
        [ax1, ax2],
        [(res_a, 'Config A: Uniform', '#27ae60'),
         (res_b, 'Config B: Opposite sides', '#e74c3c')]
    ):
        leak = res['leak']
        phi_angles = leak['phi_angles']
        fluxes = np.abs(leak['fluxes']) * 1e3   # → mWb/m

        ax.plot(phi_angles, fluxes, 'o-', color=color, linewidth=2, markersize=6)
        ax.fill(phi_angles, fluxes, alpha=0.15, color=color)

        # 标注原边/副边位置
        ax.annotate('Primary\n(left)', xy=(np.pi, fluxes.max() * 1.05),
                   fontsize=9, ha='center', color='#e74c3c', fontweight='bold')
        ax.annotate('Secondary\n(right)', xy=(0, fluxes.max() * 1.05),
                   fontsize=9, ha='center', color='#3498db', fontweight='bold')

        ax.set_title(f'{label}\n'
                     f'Φ_mean={leak["phi_mean"]*1e3:.2f} mWb/m, '
                     f'k={leak["coupling_k"]:.4f}',
                     fontsize=12, fontweight='bold', pad=20)
        ax.set_ylabel('Φ(φ) (mWb/m)', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Azimuthal Flux Distribution Φ(φ) Around the Toroid\n'
                 'Uniform winding → flat line  |  Opposite sides → modulation',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_summary_bar(res_a, res_b, ana):
    """汇总对比柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: 漏磁比例对比
    configs = ['Config A\nUniform winding', 'Config B\nOpposite sides']
    leak_vals = [res_a['leak']['leak_pct'], res_b['leak']['leak_pct']]
    k_vals = [res_a['leak']['coupling_k'], res_b['leak']['coupling_k']]
    colors = ['#27ae60', '#e74c3c']

    bars = ax1.bar(configs, leak_vals, color=colors, edgecolor='#2c3e50',
                   linewidth=1.5, width=0.45)
    for bar, val in zip(bars, leak_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.2f}%', ha='center', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Leakage ratio (%)', fontsize=12)
    ax1.set_title('Leakage Flux Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 右: 耦合系数 + 通量均匀性
    x = np.arange(len(configs))
    width = 0.35
    bars1 = ax2.bar(x - width / 2, k_vals, width, color=['#2ecc71', '#e74c3c'],
                    edgecolor='#2c3e50', linewidth=1.5, label='Coupling k = Φ_sec/Φ_pri')
    bars2 = ax2.bar(x + width / 2,
                    [1 - res_a['leak']['phi_uniformity'],
                     1 - res_b['leak']['phi_uniformity']],
                    width, color=['#3498db', '#e67e22'],
                    edgecolor='#2c3e50', linewidth=1.5, label='Uniformity 1−σ/μ')
    for bar, val in zip(bars1, k_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, [1 - res_a['leak']['phi_uniformity'],
                                 1 - res_b['leak']['phi_uniformity']]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylabel('Coefficient', fontsize=12)
    ax2.set_title('Coupling & Uniformity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower left')
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Toroidal Transformer Leakage Analysis\n'
                 f'μr = {MU_R_CORE}, NI = {N_TURNS * I_EXCITE} At',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================
# 7. 主程序
# ============================================================

def run_config(config_name, winding_config):
    """运行单个绕组配置"""
    print(f"\n{'=' * 55}")
    print(f"  {config_name}")
    print(f"{'=' * 55}")

    x, y, X, Y, nx, ny = build_domain()
    core = build_core_mask(X, Y)

    if winding_config == 'A':
        pri = build_winding_mask(X, Y, 'A')
        sec = np.zeros_like(pri, dtype=bool)
    else:
        pri = build_winding_mask(X, Y, 'B', 'primary')
        sec = build_winding_mask(X, Y, 'B', 'secondary')

    print(f"  Grid: {nx} × {ny} = {nx * ny} nodes")
    print(f"  Core cells: {np.sum(core)}")
    print(f"  Primary cells: {np.sum(pri)}")

    print("  Solving FDM...", end=" ", flush=True)
    A, Bx, By, Bmag, mu, J = solve_fdm(X, Y, core, pri, MU_R_CORE)
    print("done.")

    leak = analyze_leakage_toroid(A, Bx, By, X, Y, core, pri)

    print(f"  Φ_pri (left):   {leak['phi_pri'] * 1e3:.3f} mWb/m")
    print(f"  Φ_sec (right):  {leak['phi_sec'] * 1e3:.3f} mWb/m")
    print(f"  Φ_leak:         {leak['phi_leak'] * 1e3:.3f} mWb/m")
    print(f"  Leakage ratio:  {leak['leak_pct']:.3f}%")
    print(f"  Coupling k:     {leak['coupling_k']:.4f}")
    print(f"  Φ uniformity σ: {leak['phi_uniformity']:.4f}")

    return {
        'X': X, 'Y': Y, 'A': A, 'Bx': Bx, 'By': By, 'Bmag': Bmag,
        'core_mask': core, 'pri_mask': pri, 'sec_mask': sec,
        'leak': leak, 'config': winding_config, 'label': config_name,
    }


def main():
    print("=" * 60)
    print("  Toroidal Transformer Leakage Flux — 2D FDM Simulation")
    print("=" * 60)
    print(f"\n  Geometry: R_out={R_OUT:.0f}, R_in={R_IN:.0f} mm, "
          f"core thickness={CORE_THICK:.0f} mm")
    print(f"  Core μr = {MU_R_CORE}")
    print(f"  Excitation: N×I = {N_TURNS} × {I_EXCITE}A = {N_TURNS * I_EXCITE} At")
    print(f"  Grid: DX={DX:.1f} mm, margin={MARGIN} mm")

    # 解析模型
    print(f"\n[Analytical Model — ideal uniform toroid]")
    ana = toroid_analytical(MU_R_CORE)
    print(f"  Ideal Φ (uniform winding): {ana['phi_ideal'] * 1e3:.3f} mWb/m")
    print(f"  R_core: {ana['R_core']:.0f} A/Wb per m")
    print(f"  R_hole_leak (est): {ana['R_hole_leak']:.0f} A/Wb per m")
    print(f"  Est. leakage (opposite sides): {ana['leakage_est']:.2f}%")

    # Config A: 交替绕组 (均匀分布)
    res_a = run_config("Config A: Uniform winding (alternating, entire toroid)", 'A')

    # Config B: 分侧绕组 (对侧布置)
    res_b = run_config("Config B: Opposite-side winding (left pri, right sec)", 'B')

    # 对比
    print(f"\n{'=' * 60}")
    print(f"  Comparison Summary")
    print(f"{'=' * 60}")
    print(f"  {'':25s}  {'Config A':>12s}  {'Config B':>12s}")
    print(f"  {'-' * 52}")
    print(f"  {'Leakage ratio':25s}  {res_a['leak']['leak_pct']:11.4f}%  {res_b['leak']['leak_pct']:11.4f}%")
    print(f"  {'Coupling k':25s}  {res_a['leak']['coupling_k']:12.4f}  {res_b['leak']['coupling_k']:12.4f}")
    print(f"  {'Φ uniformity σ':25s}  {res_a['leak']['phi_uniformity']:12.4f}  {res_b['leak']['phi_uniformity']:12.4f}")
    print(f"  {'Φ_pri (mWb/m)':25s}  {res_a['leak']['phi_pri']*1e3:12.3f}  {res_b['leak']['phi_pri']*1e3:12.3f}")
    print(f"  {'Φ_sec (mWb/m)':25s}  {res_a['leak']['phi_sec']*1e3:12.3f}  {res_b['leak']['phi_sec']*1e3:12.3f}")

    ratio = res_b['leak']['leak_pct'] / max(res_a['leak']['leak_pct'], 1e-30)
    print(f"\n  → Opposite-side winding has {ratio:.1f}× more leakage than uniform winding")

    # 可视化
    print("\n[Generating plots...]")
    fig1 = plot_config_comparison(res_a, res_b, ana)
    fig1.savefig('toroid_comparison.png', dpi=150, bbox_inches='tight')
    print("  → toroid_comparison.png")

    fig2 = plot_flux_vs_angle(res_a, res_b)
    fig2.savefig('toroid_flux_vs_angle.png', dpi=150, bbox_inches='tight')
    print("  → toroid_flux_vs_angle.png")

    fig3 = plot_summary_bar(res_a, res_b, ana)
    fig3.savefig('toroid_summary.png', dpi=150, bbox_inches='tight')
    print("  → toroid_summary.png")

    # 物理洞察
    print(f"""
{'=' * 60}
  Key Physics Insights
{'=' * 60}

  1. 交替绕组 (Config A): Φ(φ) 沿圆周几乎恒定
     → 漏磁 ≈ {res_a['leak']['leak_pct']:.3f}%, 耦合 k ≈ {res_a['leak']['coupling_k']:.4f}
     → 绕组均匀分布 → MMF 均匀 → 磁通处处一致
     → 环形铁芯 + 均匀绕组 ≈ 理想变压器 (理论漏磁→0)

  2. 分侧绕组 (Config B): Φ(φ) 有明显调制
     → 漏磁 ≈ {res_b['leak']['leak_pct']:.2f}%, 耦合 k ≈ {res_b['leak']['coupling_k']:.4f}
     → 原边(左半)处 Φ 最大, 副边(右半)处 Φ 最小
     → 差值 = 磁通经中心孔"抄近路"的漏磁通

  3. 环形铁芯 vs 口字形铁芯 (square core):
     → 环形铁芯无接缝 → μr_eff = μr (不被气隙稀释)
     → 口字形铁芯有 4 个角接缝 → μr_eff << μr
     → 但口字形配合交替绕组也可达到低漏磁

  4. 中心孔的作用:
     → 铁芯内侧暴露的中心孔是漏磁的主要路径
     → 孔越大 → 漏磁路径越长 → R_leak 越大 → 漏磁越小
     → 细长铁芯 (R_out>>R_in) 比分侧绕组漏磁更小

  5. 工程实践:
     → 环形铁芯天然无接缝 → μr_eff = μr
     → 均匀绕组难度大 (需穿绕) → 通常分段绕制
     → 分段越多越接近均匀 → 漏磁越小
     → 医疗/仪器级变压器常用环形铁芯 + 均匀绕组 (k>0.999)
""")

    plt.show()


if __name__ == '__main__':
    main()
