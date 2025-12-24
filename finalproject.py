import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. Solve Blasius using solve_bvp
# ============================================================

def blasius_ode(eta, F):
    f = F[0]
    g = F[1]   # f'
    h = F[2]   # f''
    return np.vstack((g, h, -0.5 * f * h))

def bc(F0, F_inf):
    return np.array([
        F0[0],        # f(0)=0
        F0[1],        # f'(0)=0
        F_inf[1] - 1  # f'(inf)=1
    ])

# mesh for similarity variable
eta_max = 10
eta_mesh = np.linspace(0, eta_max, 400)

# initial guess
F_guess = np.zeros((3, eta_mesh.size))
F_guess[1] = 1 - np.exp(-eta_mesh)   # guess for f'

sol = solve_bvp(blasius_ode, bc, eta_mesh, F_guess)
eta = eta_mesh
fprime = sol.y[1]

# ============================================================
# 2. Boundary layer formulas
# ============================================================

def delta_blasius(x, Rex):
    # δ ≈ 4.91 x / sqrt(Re_x)
    return 4.91 * x / np.sqrt(Rex)

# parameters
U = 1.0            # free stream velocity
nu = 1e-6          # kinematic viscosity
L = 1.0            # plate length
N_frames = 200     # frames in animation

# x distribution
x_vals = np.linspace(0.001, L, N_frames)
Rex_vals = U * x_vals / nu
delta_vals = delta_blasius(x_vals, Rex_vals)

# y domain (scaled with delta)
def compute_u_over_U(y, x):
    Rex = U * x / nu
    eta_loc = y * np.sqrt(U / (nu * x))
    return np.interp(eta_loc, eta, fprime)

# ============================================================
# 3. Setup animation figure
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# left: velocity profile
y_max = delta_vals.max() * 1.2
y_plot = np.linspace(0, y_max, 300)
u_over_U_initial = compute_u_over_U(y_plot, x_vals[0])
line1, = ax1.plot(u_over_U_initial, y_plot, lw=2)
ax1.set_xlim(0, 1.05)
ax1.set_ylim(0, y_max)
ax1.set_xlabel("u / U")
ax1.set_ylabel("y (m)")
ax1.set_title("Velocity Profile: u/U vs y")

# right: boundary layer thickness growth
ax2.plot(x_vals, delta_vals, lw=2)
point, = ax2.plot([x_vals[0]], [delta_vals[0]], 'ro')
ax2.set_xlabel("x (m)")
ax2.set_ylabel("δ(x) (m)")
ax2.set_title("Boundary Layer Thickness Growth")
ax2.set_xlim(0, L)
ax2.set_ylim(0, delta_vals.max() * 1.1)

# ============================================================
# 4. Animation function
# ============================================================

def update(frame):
    x = x_vals[frame]

    # update left plot
    u_over_U = compute_u_over_U(y_plot, x)
    line1.set_ydata(y_plot)
    line1.set_xdata(u_over_U)
    ax1.set_title(f"Velocity Profile at x = {x:.3f} m")

    # update right plot
    point.set_data([x], [delta_vals[frame]])

    return line1, point

ani = FuncAnimation(fig, update, frames=N_frames, interval=50, blit=True)

plt.tight_layout()
plt.show()
