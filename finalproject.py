import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from matplotlib.widgets import Slider

def coupled_ode(eta, F, Pr):
    f, df, ddf, theta, dtheta = F
    return np.vstack((
        df, 
        ddf, 
        -0.5 * f * ddf, 
        dtheta, 
        -0.5 * Pr * f * dtheta
    ))

def bc(F0, F_inf):
    return np.array([
        F0[0], F0[1], F_inf[1] - 1, 
        F0[3], F_inf[3] - 1         
    ])

plt.close('all') 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
plt.subplots_adjust(bottom=0.35, top=0.85, wspace=0.3)

Y_LIMIT = 0.08 
L = 1.0
slider_objects = []
def update(val):
    U = s_U.val
    nu = s_nu.val
    Pr = s_Pr.val
    x_pos = 0.5
    
    eta_mesh = np.linspace(0, 10, 400)
    F_init = np.zeros((5, eta_mesh.size))
    F_init[1] = 1.0 
    F_init[3] = 1.0
    sol = solve_bvp(lambda et, F: coupled_ode(et, F, Pr), bc, eta_mesh, F_init)
    
    eta_v = sol.x
    u_ratio = sol.y[1]
    temp_ratio = sol.y[3]
    
    scale = np.sqrt(nu * x_pos / U)
    y_plot = eta_v * scale
    
    delta = 4.91 * scale
    delta_t = delta * (Pr**(-1/3)) if Pr > 0.05 else delta * 2.0
    
    ax1.clear()
    ax1.set_facecolor('#fdfdfd')
    
    y_ext = np.append(y_plot, Y_LIMIT)
    u_ext = np.append(u_ratio, 1.0)
    temp_ext = np.append(temp_ratio, 1.0)
    
    ax1.plot(u_ext, y_ext, 'r-', lw=2, label='Velocity $u/U$')
    ax1.plot(temp_ext, y_ext, 'orange', lw=2, label='Temperature $\\theta$')
    
    ax1.axhline(delta, color='blue', ls='--', alpha=0.5, label=f'Momentum $\delta$: {delta:.4f}m')
    ax1.axhline(delta_t, color='darkorange', ls='--', alpha=0.5, label=f'Thermal $\delta_t$: {delta_t:.4f}m')
    skip = 30
    ax1.quiver(np.zeros_like(y_plot[::skip]), y_plot[::skip], 
               u_ratio[::skip], np.zeros_like(y_plot[::skip]), 
               angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.3)
    
    ax1.set_title(f"Velocity & Temp Profiles (x={x_pos}m)")
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, Y_LIMIT)
    ax1.set_xlabel("Dimensionless Value")
    ax1.set_ylabel("Height $y$ (m)")
    
    ax1.legend(loc='upper left', fontsize=8, frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.5)

    ax2.clear()
    ax2.set_facecolor('#fdfdfd')
    x_range = np.linspace(0.001, L, 100)
    d_range = 4.91 * np.sqrt(nu * x_range / U)
    dt_range = d_range * (Pr**(-1/3))
    
    ax2.plot(x_range, d_range, 'b-', label='Momentum $\delta$')
    ax2.fill_between(x_range, 0, d_range, color='blue', alpha=0.1)
    ax2.plot(x_range, dt_range, 'orange', label='Thermal $\delta_t$')
    ax2.fill_between(x_range, 0, dt_range, color='orange', alpha=0.05)
    
    ax2.set_title(f"Boundary Layer Growth (Pr={Pr:.2f})")
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, Y_LIMIT)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    fig.canvas.draw_idle()

ax_U = plt.axes([0.2, 0.2, 0.6, 0.025])
s_U = Slider(ax_U, 'Velocity U', 0.1, 10.0, valinit=1.0)
slider_objects.append(s_U)

ax_nu = plt.axes([0.2, 0.13, 0.6, 0.025])
s_nu = Slider(ax_nu, 'Viscosity nu', 1e-6, 1e-4, valinit=1e-5)
slider_objects.append(s_nu)

ax_Pr = plt.axes([0.2, 0.06, 0.6, 0.025])
s_Pr = Slider(ax_Pr, 'Prandtl Pr', 0.01, 10.0, valinit=0.7)
slider_objects.append(s_Pr)

s_U.on_changed(update)
s_nu.on_changed(update)
s_Pr.on_changed(update)

update(None)
plt.show()
