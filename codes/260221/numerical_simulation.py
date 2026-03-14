import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

# ==================== 可自定义参数 ====================
xf = 0.005          # 自由端 x 坐标（固定点为原点 (0,0)）
yf = 0.0         # 自由端 y 坐标（负值表示下垂）
L_total = 1.6     # 绳子总长度，必须大于直线距离 sqrt(xf^2 + yf^2)

# ==================== 模型参数 ====================
N = 200            # 杆的数量（质点数为 N+1，含固定点）
g = -9.8           # 重力加速度 (m/s^2)，向下为正
dt = 0.0005        # 时间步长 (s)
T_total = 1.0    # 总模拟时间 (s)
n_iter = 10       # 约束投影迭代次数

# 检查绳长是否足够
straight_dist = np.sqrt(xf**2 + yf**2)
if L_total <= straight_dist:
    raise ValueError(f"绳长必须大于直线距离 {straight_dist:.3f}，当前 L_total = {L_total}")

# 每段杆长
l = L_total / N

# ==================== 生成悬链线初始位形 ====================
def generate_catenary(N, L, xf, yf):
    """
    生成从 (0,0) 到 (xf, yf) 的悬链线上的离散点（等弧长分布）
    返回数组 shape (N+1, 2)，第一个点为固定点，最后一个点为自由端
    """
    # 定义残差函数，求解悬链线参数 a 和 x0
    def equations(p):
        a, x0 = p
        # 避免除零或负 a
        if a <= 0:
            return [1e6, 1e6]
        u = x0 / a
        v = (xf - x0) / a
        eq1 = a * (np.cosh(v) - np.cosh(u)) - yf
        eq2 = a * (np.sinh(v) + np.sinh(u)) - L
        return [eq1, eq2]

    # 初始猜测：a ≈ L/2, x0 ≈ xf/2
    a_guess = L / 2
    x0_guess = xf / 2
    sol, info, ier, msg = fsolve(equations, [a_guess, x0_guess], full_output=True)
    if ier != 1:
        raise RuntimeError(f"悬链线参数求解失败: {msg}")
    a, x0 = sol
    if a <= 0:
        raise ValueError("求解得到的 a <= 0，请检查参数")

    # 弧长函数 s(x) = 从固定点 (x=0) 到 x 的弧长
    def s(x):
        return a * (np.sinh((x - x0) / a) + np.sinh(x0 / a))

    # 验证端点弧长
    s_xf = s(xf)
    if abs(s_xf - L) > 1e-6:
        print(f"警告：悬链线弧长计算误差 {abs(s_xf - L):.2e}，将强制调整最后一点")

    # 生成等距弧长的点
    positions = np.zeros((N+1, 2))
    positions[0] = [0, 0]   # 固定点

    # 对于每个内部质点 k=1..N-1，求解 x 使得 s(x) = k*l
    # 最后一个质点直接用 xf, yf 保证精确
    for k in range(1, N):
        s_target = k * l
        # 二分法求 x
        x_low, x_high = 0.0, xf
        # 确保 s(x_low) <= s_target <= s(x_high)
        if s(x_low) > s_target or s(x_high) < s_target:
            # 可能因为数值误差，稍微放宽范围
            x_low, x_high = -abs(xf)*0.1, abs(xf)*1.2
        for _ in range(50):
            x_mid = (x_low + x_high) / 2
            s_mid = s(x_mid)
            if s_mid < s_target:
                x_low = x_mid
            else:
                x_high = x_mid
            if abs(x_high - x_low) < 1e-10:
                break
        xk = (x_low + x_high) / 2
        yk = a * np.cosh((xk - x0) / a) - a * np.cosh(x0 / a)
        positions[k] = [xk, yk]

    # 最后一个点直接用自由端坐标
    positions[N] = [xf, yf]
def generate_catenary(N, L, xf, yf, tol=1e-10):
    """
    生成从 (0,0) 到 (xf, yf) 的悬链线上的离散点（等弧长分布）
    改进版本：特别处理两端等高 (yf=0) 的情况，增加数值稳定性
    返回数组 shape (N+1, 2)，第一个点为固定点，最后一个点为自由端
    """
    import numpy as np
    from scipy.optimize import fsolve, brentq

    # 检查绳长是否足够
    d = np.sqrt(xf**2 + yf**2)
    if L <= d:
        raise ValueError(f"绳长 L={L} 必须大于直线距离 d={d:.3f}")

    # 特殊处理：两端等高 (yf == 0)
    if abs(yf) < 1e-12:
        # 对称悬链线：从 (0,0) 到 (xf,0)
        # 弧长公式 L = 2a * sinh(xf/(2a))
        # 定义函数 f(a) = 2a * sinh(xf/(2a)) - L
        def f(a):
            if a <= 0:
                return np.inf
            # 避免除零和大数溢出
            if a < 1e-12:
                return 2*xf - L   # 近似直线
            return 2*a * np.sinh(xf/(2*a)) - L

        # 寻找 a 的合理区间
        # 当 a 很小时，2a*sinh(xf/(2a)) ≈ xf + xf^3/(24a^2) -> 很大
        # 当 a 很大时，2a*sinh(xf/(2a)) ≈ xf + xf^3/(24a^2) ≈ xf
        # 所以 f(a) 从正无穷单调递减到 xf - L (负值，因为 L > xf)
        # 因此根存在且唯一
        a_low = 1e-6
        a_high = 1e6
        # 确保 f(a_low) > 0 且 f(a_high) < 0
        while f(a_high) > 0 and a_high < 1e12:
            a_high *= 2
        while f(a_low) < 0 and a_low > 1e-12:
            a_low /= 2

        try:
            a = brentq(f, a_low, a_high, xtol=tol)
        except ValueError:
            # 如果二分法失败，可能因为 L 非常接近 xf，此时 a 极大，直接用直线
            a = 1e12

        # 最低点 x 坐标
        x0 = xf / 2.0
        # 常数 C = -a * cosh(x0/a)
        C = -a * np.cosh(x0 / a)

        # 生成等弧长的点
        positions = np.zeros((N+1, 2))
        positions[0] = [0, 0]
        positions[N] = [xf, 0]

        # 对于内部点，使用等弧长分布
        # 弧长函数 s(x) = 从 x=0 到 x 的弧长
        # 对于对称悬链线，从 0 到 x 的弧长 = a * |sinh((x - x0)/a) + sinh(x0/a)|
        # 但注意当 x <= x0 时，表达式为 a * (sinh(x0/a) - sinh((x0 - x)/a))
        # 为了避免符号混乱，我们直接利用对称性：先计算左半部分，再镜像
        # 更简单：用参数方程，但此处采用二分法求每个 x
        def s(x):
            # 从 0 到 x 的弧长（x 在 [0, xf]）
            return a * abs(np.sinh((x - x0)/a) + np.sinh(x0/a))

        # 验证端点弧长
        s_xf = s(xf)
        if abs(s_xf - L) > 1e-6:
            # 如果误差大，可能是 a 极大，直接用直线分布
            for k in range(1, N):
                xk = xf * k / N
                yk = 0
                positions[k] = [xk, yk]
            return positions

        # 对每个 k=1..N-1，求 x 使 s(x) = k*l
        for k in range(1, N):
            target = k * l
            # x 在 [0, xf] 内单调，用二分法
            x_low, x_high = 0.0, xf
            # 确保 s(x_low) <= target <= s(x_high)
            if s(x_low) > target or s(x_high) < target:
                # 放宽范围
                x_low = -0.1 * xf
                x_high = 1.1 * xf
            for _ in range(60):
                x_mid = (x_low + x_high) / 2
                s_mid = s(x_mid)
                if s_mid < target:
                    x_low = x_mid
                else:
                    x_high = x_mid
                if abs(x_high - x_low) < tol:
                    break
            xk = (x_low + x_high) / 2
            yk = a * np.cosh((xk - x0) / a) + C
            positions[k] = [xk, yk]

        return positions

    # 一般情况（两端不等高），使用 fsolve 但提供更好的初始猜测
    # 初始猜测：假设 a ≈ L/2，x0 ≈ xf/2 附近
    # 先估算下垂量
    if yf < 0:
        # 自由端低于固定端，下垂明显
        a_guess = L / 3
        x0_guess = xf / 2
    else:
        # 自由端高于固定端，可能接近直线
        a_guess = L / 0.8   # 较大值
        x0_guess = xf / 2

    def equations(p):
        a, x0 = p
        if a <= 0:
            return [1e12, 1e12]
        u = x0 / a
        v = (xf - x0) / a
        eq1 = a * (np.cosh(v) - np.cosh(u)) - yf
        eq2 = a * (np.sinh(v) + np.sinh(u)) - L
        return [eq1, eq2]

    # 尝试多组初始猜测，增加鲁棒性
    guesses = [
        [a_guess, x0_guess],
        [L/2, xf/2],
        [L, xf/3],
        [L/4, xf*0.6]
    ]

    sol = None
    for guess in guesses:
        try:
            sol, info, ier, msg = fsolve(equations, guess, full_output=True)
            if ier == 1 and sol[0] > 0:
                a, x0 = sol
                break
        except:
            continue

    if sol is None:
        # 所有尝试失败，采用近似：将绳长按直线比例投影，然后近似为直线
        print("警告：悬链线参数求解失败，使用直线近似（将产生微小误差）")
        positions = np.zeros((N+1, 2))
        for i in range(N+1):
            t = i / N
            positions[i] = [xf * t, yf * t]
        return positions

    a, x0 = sol

    # 弧长函数
    def s(x):
        return a * (np.sinh((x - x0)/a) + np.sinh(x0/a))

    # 验证端点弧长
    s_xf = s(xf)
    if abs(s_xf - L) > 1e-4:
        # 如果误差大，可能是数值问题，改用直线近似
        positions = np.zeros((N+1, 2))
        for i in range(N+1):
            t = i / N
            positions[i] = [xf * t, yf * t]
        return positions

    positions = np.zeros((N+1, 2))
    positions[0] = [0, 0]
    positions[N] = [xf, yf]

    for k in range(1, N):
        target = k * l
        # 二分求 x
        x_low, x_high = 0.0, xf
        # 确保 s(x_low) <= target <= s(x_high)
        if s(x_low) > target:
            x_low = -0.1 * xf
        if s(x_high) < target:
            x_high = 1.1 * xf
        for _ in range(60):
            x_mid = (x_low + x_high) / 2
            s_mid = s(x_mid)
            if s_mid < target:
                x_low = x_mid
            else:
                x_high = x_mid
            if abs(x_high - x_low) < tol:
                break
        xk = (x_low + x_high) / 2
        yk = a * np.cosh((xk - x0)/a) - a * np.cosh(x0/a)
        positions[k] = [xk, yk]

    return positions

# 生成初始位置
x_init = generate_catenary(N, L_total, xf, yf)

# 初始化位置数组（质点0为固定点，1..N为自由质点）
x = x_init.copy()
v = np.zeros((N+1, 2))   # 初始速度为零

# 固定点索引
fixed = 0

# ==================== 约束投影函数 ====================
def project_constraints(x, l, fixed_idx):
    for i in range(1, N+1):
        p1 = x[i-1]
        p2 = x[i]
        d = p2 - p1
        dist = np.linalg.norm(d)
        if dist == 0:
            continue
        delta = (dist - l) * (d / dist) * 0.5
        if i-1 != fixed_idx:
            x[i-1] += delta
        if i != fixed_idx:
            x[i] -= delta
    return x

# ==================== 数据记录 ====================
time = np.arange(0, T_total + dt, dt)
n_steps = len(time)
v_end = np.zeros((n_steps, 2))
x_end = np.zeros((n_steps, 2))

record_interval = 0.02          # 动画帧间隔 (s)
frame_steps = int(record_interval / dt)
record_positions = []
record_times = []

# ==================== 主循环 ====================
for step in range(n_steps):
    t = step * dt

    # 应用重力更新速度
    for i in range(1, N+1):
        v[i] += np.array([0, g]) * dt

    # 预测位置
    x_pred = x + v * dt

    # 多次投影约束
    for _ in range(n_iter):
        x_pred = project_constraints(x_pred, l, fixed)

    # 更新速度和位置
    v = (x_pred - x) / dt
    x = x_pred.copy()
    x[fixed] = [0, 0]   # 固定点强制归零
    v[fixed] = [0, 0]

    # 记录自由端数据
    v_end[step] = v[N]
    x_end[step] = x[N]

    # 记录动画帧
    if step % frame_steps == 0:
        record_positions.append(x.copy())
        record_times.append(t)

# 补全最后一帧
if step % frame_steps != 0:
    record_positions.append(x.copy())
    record_times.append(t)

record_positions = np.array(record_positions)

# ==================== 线性拟合0-0.3秒的速度 ====================
time_mask = time <= 0.3
time_fit = time[time_mask]
v_fit = v_end[time_mask, 1]

# 线性拟合
coeffs = np.polyfit(time_fit, v_fit, 1)
slope, intercept = coeffs
v_fit_line = np.poly1d(coeffs)

# 计算相关系数 R²
v_pred = v_fit_line(time_fit)
ss_res = np.sum((v_fit - v_pred)**2)
ss_tot = np.sum((v_fit - np.mean(v_fit))**2)
r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

print(f"\n=== 0-0.3秒速度线性拟合结果 ===")
print(f"拟合直线: v_y = {slope:.6f} * t + {intercept:.6f}")
print(f"加速度（斜率）: {slope:.6f} m/s²")
print(f"初速度（截距）: {intercept:.6f} m/s")
print(f"相关系数 R²: {r_squared:.6f}")

# ==================== 绘图1：自由端速度 ====================
plt.figure(figsize=(10, 5))
#plt.plot(time, v_end[:, 0], label='v_x')
plt.plot(time, v_end[:, 1], label='v_y')
plt.plot(time_fit, v_fit_line(time_fit), 'r-', linewidth=2, label=f'Linear fit (0-0.3s): a={slope:.4f}')
#plt.plot(time, np.linalg.norm(v_end, axis=1), label='speed')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Free End Velocity vs Time')
plt.legend()
plt.grid(True)
plt.show()

# ==================== 绘图2：动画（y轴反转，自动居中） ====================
# 计算所有帧的坐标范围
all_x = record_positions[:, :, 0].flatten()
all_y = record_positions[:, :, 1].flatten()
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
margin = 0.1 * max(x_max - x_min, y_max - y_min)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(x_min - margin, x_max + margin)
# 设置 y 轴范围（正常顺序：最小值在下，最大值在上）
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Rope Motion Animation (y-axis reversed)')

line, = ax.plot([], [], 'o-', lw=2, markersize=3, color='b')
fixed_point, = ax.plot([0], [0], 'ro', markersize=8, label='Fixed point')
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    pos = record_positions[frame]
    line.set_data(pos[:, 0], pos[:, 1])
    time_text.set_text(f't = {record_times[frame]:.2f} s')
    return line, time_text

ani = FuncAnimation(fig, update, frames=len(record_positions),
                    init_func=init, blit=True, interval=record_interval*1000, repeat=True)

plt.show()

print("Simulation complete. Plots displayed.")