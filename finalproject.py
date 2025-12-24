import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from matplotlib.widgets import Slider

# ============================================================
# 1. 物理引擎：數值求解 Blasius 方程式 (無因次解)
# ============================================================
def blasius_ode(eta, F):
    return np.vstack((F[1], F[2], -0.5 * F[0] * F[2]))

def bc(F0, F_inf):
    return np.array([F0[0], F0[1], F_inf[1] - 1])

# 預算數據：eta 從 0 到 10 (足夠涵蓋 99% 的邊界層)
eta_mesh = np.linspace(0, 10, 500)
F_guess = np.zeros((3, eta_mesh.size))
F_guess[1] = 1 - np.exp(-eta_mesh)
sol = solve_bvp(blasius_ode, bc, eta_mesh, F_guess)
eta_vals = sol.x
f_prime_vals = sol.y[1]

# ============================================================
# 2. 介面初始化 (設定固定座標軸)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.3, top=0.85, wspace=0.3)

# 設定固定的 Y 軸顯示範圍 (0 到 0.08m)，確保座標軸不會跳動
Y_LIMIT = 0.08 

# 防止拉桿失效的全域變數清單
sliders = []

# ============================================================
# 3. 核心更新函數 (圖片動，座標軸不動)
# ============================================================
def update(val):
    U = s_U.val
    nu = s_nu.val
    L = 1.0
    x_pos = 0.5  # 固定觀察 x = 0.5m 處的速度剖面
    
    # --- 左圖：速度剖面 (Velocity Profile) ---
    ax1.clear()
    ax1.set_facecolor('#fdfdfd')
    
    # 物理高度 y = eta * sqrt(nu * x / U)
    y_plot = eta_vals * np.sqrt(nu * x_pos / U)
    u_ratio = f_prime_vals # u/U
    
    # 延伸數據到視窗頂部 (解決線條斷掉問題)
    # 如果 y_plot 的最後一點沒到 Y_LIMIT，就補齊它
    y_ext = np.append(y_plot, Y_LIMIT)
    u_ext = np.append(u_ratio, 1.0)
    
    # 繪製主曲線
    ax1.plot(u_ext, y_ext, 'r-', lw=2.5)
    
    # 稀疏箭頭 (固定間隔)
    skip = 30
    ax1.quiver(np.zeros_like(y_plot[::skip]), y_plot[::skip], 
               u_ratio[::skip], np.zeros_like(y_plot[::skip]), 
               angles='xy', scale_units='xy', scale=1, 
               color='gray', alpha=0.4, width=0.005)
    
    # 計算並標示目前的邊界層厚度線
    current_delta = 4.91 * np.sqrt(nu * x_pos / U)
    ax1.axhline(current_delta, color='blue', ls='--', alpha=0.5)
    
    ax1.set_title(f"Velocity Profile at x = {x_pos}m", fontsize=10)
    ax1.set_xlabel("u / U")
    ax1.set_ylabel("Height y (m)")
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, Y_LIMIT) # 強制固定 Y 軸
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右圖：邊界層成長 (Boundary Layer Growth) ---
    ax2.clear()
    ax2.set_facecolor('#fdfdfd')
    x_range = np.linspace(0.001, L, 200)
    delta_range = 4.91 * x_range / np.sqrt(U * x_range / nu)
    
    ax2.plot(x_range, delta_range, color='#1f77b4', lw=2)
    ax2.fill_between(x_range, 0, delta_range, color='#1f77b4', alpha=0.2)
    
    ax2.set_title(f"Boundary Layer thickness (Re_L={U*L/nu:.1e})", fontsize=10)
    ax2.set_xlabel("Distance x (m)")
    ax2.set_ylabel("Thickness delta (m)")
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, Y_LIMIT) # 強制固定 Y 軸
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    fig.canvas.draw_idle()

# ============================================================
# 4. 配置互動拉桿
# ============================================================
ax_U = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor='#e1e1e1')
s_U = Slider(ax_U, 'Velocity U ', 0.1, 10.0, valinit=1.0)
sliders.append(s_U)

ax_nu = plt.axes([0.2, 0.08, 0.6, 0.03], facecolor='#e1e1e1')
s_nu = Slider(ax_nu, 'Viscosity nu ', 1e-6, 1e-4, valinit=1e-5)
sliders.append(s_nu)

s_U.on_changed(update)
s_nu.on_changed(update)

# 執行初始渲染
update(None)

plt.show()
