import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================== 参数设置 ==================
N = 100                     # 质点个数（包括两端），共N+1个质点
L0 = 1.0                   # 绳子原长 (m)
M = 0.1                    # 绳子总质量 (kg)
EA = 100.0                 # 抗拉刚度 (N)
g = 9.8                    # 重力加速度 (m/s^2)
dt = 0.0001                 # 时间步长 (s)
T_total = 0.4             # 总模拟时间 (s)
v0_free = 0.0              # 自由端初始速度大小 (m/s)，方向向上

# 固定端和自由端初始位置
fixed_pos = np.array([0.0, 0.0])          # 固定端坐标
free_pos_given = np.array([0.01, 0.0])    # 自由端给定位置

# 离散化参数
m = M / (N + 1)            # 每个质点的质量 (kg)
l0_seg = L0 / N            # 每段弹簧的原长 (m)
k_seg = EA / l0_seg        # 每段弹簧的劲度系数 (N/m)

# ================== 第一步：求平衡位形（固定自由端） ==================
# 初始化质点位置，初始猜测为直线连接固定端和自由端
x = np.linspace(fixed_pos[0], free_pos_given[0], N+1)
y = np.linspace(fixed_pos[1], free_pos_given[1], N+1)
pos = np.vstack([x, y]).T          # shape (N+1, 2)
vel = np.zeros_like(pos)           # 初始速度为零

# 固定端和自由端在松弛过程中位置固定，故设置标记
fixed_mask = np.zeros(N+1, dtype=bool)
fixed_mask[0] = True                # 质点0固定
fixed_mask[N] = True                # 质点N也固定（自由端暂时固定）

# 松弛模拟参数
damping = 10.0                      # 阻尼系数，用于快速收敛
T_relax = 5.0                       # 松弛时间
n_relax = int(T_relax / dt)

# 用于记录动能，判断是否平衡
kinetic_energy_history = []

for step in range(n_relax):
    # 计算弹簧力
    forces = np.zeros_like(pos)
    # 重力
    forces[:, 1] -= m * g
    
    # 计算每段弹簧的力
    for i in range(N):
        p1 = pos[i]
        p2 = pos[i+1]
        r_vec = p2 - p1
        r = np.linalg.norm(r_vec)
        if r > 0:
            force_mag = k_seg * (r - l0_seg)
            force_dir = r_vec / r
            f = force_mag * force_dir
            forces[i] += f          # 作用在p1上的力（方向指向p2）
            forces[i+1] -= f        # 作用在p2上的力相反
    
    # 固定端点力清零，它们不参与运动
    forces[fixed_mask] = 0
    
    # 更新速度（半隐式欧拉，加阻尼）
    acc = forces / m
    vel[~fixed_mask] += (acc[~fixed_mask] - damping * vel[~fixed_mask]) * dt
    pos[~fixed_mask] += vel[~fixed_mask] * dt
    
    # 计算动能
    kinetic_energy = 0.5 * m * np.sum(vel[~fixed_mask]**2)
    kinetic_energy_history.append(kinetic_energy)
    if kinetic_energy < 1e-8:
        print(f"松弛在第{step}步收敛")
        break

# 松弛结束后，pos即为平衡位形（自由端仍固定在给定位置）
print("平衡位形已获得")

# ================== 第二步：释放自由端并给初始速度，进行无阻尼运动模拟 ==================
# 解除自由端的固定
fixed_mask[N] = False
# 给自由端一个初始速度（竖直向上）
vel[N, 1] = v0_free

# 准备记录自由端y轴速度
time_points = np.arange(0, T_total, dt)
v_y_free = []

# 为动画记录绳子形状（每隔一定步数记录一次）
record_interval = 20        # 每20步记录一帧（可根据需要调整）
pos_history = []            # 存储不同时刻的质点位置
time_record = []            # 存储对应的时间

# 记录能量
kinetic_energy = []
potential_gravity = []
potential_elastic = []
total_energy = []

# 运动模拟（无阻尼）
for step in range(len(time_points)):
    # 记录当前自由端y速度
    v_y_free.append(vel[N, 1])
    
    # 记录形状（每隔 record_interval 步）
    if step % record_interval == 0:
        pos_history.append(pos.copy())
        time_record.append(step * dt)
    
    # 计算当前能量
    # 动能
    ke = 0.5 * m * np.sum(vel**2)      # 固定端速度为零，不影响
    # 重力势能 (以y=0为参考)
    pe_g = m * g * np.sum(pos[:, 1])
    # 弹性势能
    pe_e = 0.0
    for i in range(N):
        r_vec = pos[i+1] - pos[i]
        r = np.linalg.norm(r_vec)
        pe_e += 0.5 * k_seg * (r - l0_seg)**2
    # 总机械能
    total = ke + pe_g + pe_e
    
    kinetic_energy.append(ke)
    potential_gravity.append(pe_g)
    potential_elastic.append(pe_e)
    total_energy.append(total)
    
    # 计算弹簧力
    forces = np.zeros_like(pos)
    forces[:, 1] -= m * g
    
    for i in range(N):
        p1 = pos[i]
        p2 = pos[i+1]
        r_vec = p2 - p1
        r = np.linalg.norm(r_vec)
        if r > 0:
            force_mag = k_seg * (r - l0_seg)
            force_dir = r_vec / r
            f = force_mag * force_dir
            forces[i] += f
            forces[i+1] -= f
    
    # 固定端点力清零
    forces[fixed_mask] = 0
    
    # 更新速度（半隐式欧拉）
    acc = forces / m
    vel[~fixed_mask] += acc[~fixed_mask] * dt
    pos[~fixed_mask] += vel[~fixed_mask] * dt

# ================== 绘制自由端y轴速度随时间变化 ==================
plt.figure(figsize=(10, 5))
plt.plot(time_points, v_y_free)
plt.xlabel('时间 (s)')
plt.ylabel('自由端y轴速度 (m/s)')
plt.title('弹性绳自由端y轴速度随时间变化')
plt.grid(True)
plt.show()

# ================== 绘制能量随时间变化 ==================
plt.figure(figsize=(12, 6))
plt.plot(time_points, kinetic_energy, label='kinetic energy')
plt.plot(time_points, potential_gravity, label='potential gravity')
plt.plot(time_points, potential_elastic, label='potential elastic')
plt.plot(time_points, total_energy, label='total energy', linewidth=2, color='black')
plt.xlabel('时间 (s)')
plt.ylabel('能量 (J)')
plt.title('系统能量随时间变化')
plt.legend()
plt.grid(True)
plt.show()

# ================== 生成动画 ==================
# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, 1.5)      # 根据运动范围调整
ax.set_ylim(-1.5, 0.5)      # 根据运动范围调整
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('弹性绳运动动画')
ax.grid(True)
line, = ax.plot([], [], 'o-', lw=2, markersize=4)   # 绳子线条
fixed_point, = ax.plot([fixed_pos[0]], [fixed_pos[1]], 'ro', markersize=8, label='固定端')
free_point, = ax.plot([], [], 'bo', markersize=8, label='自由端')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

# 初始化函数
def init():
    line.set_data([], [])
    free_point.set_data([], [])
    time_text.set_text('')
    return line, free_point, time_text

# 更新函数
def update(frame):
    pos_frame = pos_history[frame]
    line.set_data(pos_frame[:, 0], pos_frame[:, 1])
    free_point.set_data([pos_frame[-1, 0]], [pos_frame[-1, 1]])
    time_text.set_text(f't = {time_record[frame]:.2f} s')
    return line, free_point, time_text

# 创建动画
ani = FuncAnimation(fig, update, frames=len(pos_history), init_func=init, 
                    blit=True, interval=50, repeat=True)

plt.show()

# 可选：保存动画为gif（需要安装imagemagick或pillow）
# ani.save('rope_motion.gif', writer='pillow', fps=20)
# 或保存为mp4（需要ffmpeg）
# ani.save('rope_motion.mp4', writer='ffmpeg', fps=20)