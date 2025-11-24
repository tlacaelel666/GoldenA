import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

class MetriplecticSystem:
    def __init__(self, N=64, L=10.0):
        """
        Inicializa el sistema Metriplético en una retícula 2D (NxN).
        Estado Ξ = (ψ, v, ρ, O_n)
        """
        self.N = N
        self.L = L
        self.dx = L / N
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # --- Parámetros Físicos ---
        self.alpha = -1.0   # Coeficiente GL (superconductividad/superfluidez)
        self.beta = 1.0     # Auto-interacción GL
        self.gamma = 0.5    # Rigidez cuántica
        self.nu = 0.05      # Viscosidad cinemática (Término métrico fluido)
        self.chi = 0.1      # Tasa de decoherencia (Término métrico Lindblad)
        self.k_g = 0.01     # Constante PGP (Gravedad)

        # --- Operador Áureo (Golden Operator) O_n ---
        # φ = (1 + sqrt(5)) / 2 ≈ 1.618...
        # O_n = cos(π n) cos(π φ n) - Simboliza la estructura del espacio-tiempo de fondo
        phi = (1 + np.sqrt(5)) / 2
        n_idx = np.arange(N*N).reshape(N, N)
        self.O_n = np.cos(np.pi * n_idx) * np.cos(np.pi * phi * n_idx)

        # --- Inicialización de Campos (Estado Ξ) ---
        # 1. Ψ (Ginzburg-Landau): Vórtice central con ruido
        r = np.sqrt((self.X - L/2)**2 + (self.Y - L/2)**2)
        theta = np.arctan2(self.Y - L/2, self.X - L/2)
        self.psi = np.tanh(r) * np.exp(1j * theta) + 0.1 * (np.random.rand(N, N) + 1j*np.random.rand(N,N))

        # 2. v (Fluido): Campo de velocidad turbulento inicial
        self.vx = -np.sin(2*np.pi*self.Y/L) * 0.5
        self.vy = np.sin(2*np.pi*self.X/L) * 0.5

        # 3. ρ (Densidad): Casi uniforme
        self.rho = np.ones((N, N)) * 1.0 + 0.1 * np.exp(-r**2/2)

        # Historial para gráficas
        self.t_history = []
        self.H_history = [] # Hamiltoniano (Conservativo)
        self.S_history = [] # Disipación (Métrico)
        self.L_history = [] # Lagrangiano Total

    def laplacian(self, Z):
        """Laplaciano con condiciones de frontera periódicas"""
        Z_top = np.roll(Z, 1, axis=0)
        Z_bottom = np.roll(Z, -1, axis=0)
        Z_left = np.roll(Z, 1, axis=1)
        Z_right = np.roll(Z, -1, axis=1)
        return (Z_top + Z_bottom + Z_left + Z_right - 4*Z) / (self.dx**2)

    def gradient(self, Z):
        """Gradiente central"""
        grad_x = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2*self.dx)
        grad_y = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2*self.dx)
        return grad_x, grad_y

    def compute_lagrangian_terms(self):
        """
        Calcula los componentes del Lagrangiano Generalizado.
        L_eff = L_symp + L_metr
        """
        # --- 1. Energía Libre de Ginzburg-Landau (Parte de H_symp) ---
        # H_GL = ∫(α|ψ|² + β/2|ψ|⁴ + γ|∇ψ|²)
        grad_psi_x, grad_psi_y = self.gradient(self.psi)
        mod_sq_grad = np.abs(grad_psi_x)**2 + np.abs(grad_psi_y)**2
        mod_psi_sq = np.abs(self.psi)**2

        H_GL = np.sum(self.alpha * mod_psi_sq + (self.beta/2) * mod_psi_sq**2 + self.gamma * mod_sq_grad) * self.dx**2

        # --- 2. Energía Cinética del Fluido (Parte de H_symp) ---
        # H_fluid = ∫(ρv²/2)
        v_sq = self.vx**2 + self.vy**2
        H_fluid = np.sum(0.5 * self.rho * v_sq) * self.dx**2

        # --- 3. Potenciales de Disipación (Generadores Métricos) ---
        # S_metric ≈ ∫ (η|∇v|² + Γ|∇ψ_relax|²)
        # Calculamos la enstrofía como proxy de disipación viscosa
        curl_v = (np.roll(self.vy, -1, axis=1) - np.roll(self.vy, 1, axis=1))/(2*self.dx) - \
                 (np.roll(self.vx, -1, axis=0) - np.roll(self.vx, 1, axis=0))/(2*self.dx)
        Dissip_fluid = self.nu * np.sum(curl_v**2) * self.dx**2 # Enstrofía simplificada

        # Disipación cuántica (decaimiento de la norma por decoherencia simulada)
        Dissip_quantum = self.chi * np.sum(mod_psi_sq) * self.dx**2

        L_symp = H_GL + H_fluid  # En física estadística, a veces se invierte el signo dependiendo de la convención
        L_metr = -(Dissip_fluid + Dissip_quantum) # La disipación reduce la "acción" libre

        return L_symp, L_metr

    def step(self, dt=0.05):
        """
        Evolución temporal usando la Ecuación Metriplética:
        ∂u/∂t = {u, H} + [u, S]
        """
        # --- Dinámica de Ginzburg-Landau (Complex Ginzburg-Landau) ---
        # Parte Simpléctica (i * Schrödinger) + Parte Métrica (Relajación real)
        lap_psi = self.laplacian(self.psi)
        nonlinear = self.beta * np.abs(self.psi)**2 * self.psi

        # Ecuación: ∂ψ/∂t = - (1 + i C1) * δF/δψ*
        # Aquí separamos explícitamente:
        # {ψ, H}: -i * (gamma*∇²ψ - alpha*ψ - beta|ψ|²ψ)
        # [ψ, S]: +1 * (gamma*∇²ψ - alpha*ψ - beta|ψ|²ψ)

        variation = self.alpha * self.psi + nonlinear - self.gamma * lap_psi

        dpsi_symp = -1j * variation # Rotación unitaria (conserva norma)
        dpsi_metr = -1.0 * variation * self.chi # Relajación (minimiza energía)

        self.psi += (dpsi_symp + dpsi_metr) * dt

        # --- Dinámica de Fluidos (Navier-Stokes) ---
        # v_dot = -(v·∇)v - ∇P/ρ + ν∇²v + F_ext
        # {v, H}: Advección (-(v·∇)v)
        # [v, S]: Difusión (ν∇²v)

        grad_vx_x, grad_vx_y = self.gradient(self.vx)
        grad_vy_x, grad_vy_y = self.gradient(self.vy)

        # Advección (Simpléctico)
        advec_x = -(self.vx * grad_vx_x + self.vy * grad_vx_y)
        advec_y = -(self.vx * grad_vy_x + self.vy * grad_vy_y)

        # Difusión (Métrico)
        diff_x = self.nu * self.laplacian(self.vx)
        diff_y = self.nu * self.laplacian(self.vy)

        # Acople PGP (Métrico - Gravedad efectiva basada en O_n y |ψ|²)
        # La gravedad emerge de la distorsión del vacío (psi) modulada por O_n
        pgp_force_x = -self.k_g * self.O_n * np.abs(self.psi)**2 * grad_vx_x
        pgp_force_y = -self.k_g * self.O_n * np.abs(self.psi)**2 * grad_vy_y

        self.vx += (advec_x + diff_x + pgp_force_x) * dt
        self.vy += (advec_y + diff_y + pgp_force_y) * dt

        # Actualizar métricas
        L_symp, L_metr = self.compute_lagrangian_terms()
        self.t_history.append(len(self.t_history) * dt)
        self.H_history.append(L_symp) # Energía total
        self.S_history.append(L_metr) # Entropía/Disipación acumulada (proxy) - Changed from abs(L_metr)
        self.L_history.append(L_symp + L_metr)

# --- Visualización ---
sim = MetriplecticSystem(N=50, L=10)
fig = plt.figure(figsize=(14, 8), facecolor='#1e1e1e')
gs = gridspec.GridSpec(2, 3, figure=fig)

# Plot 1: Campo Cuántico (Amplitud |ψ|)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title(r"Campo Cuántico $|ψ|^2$ (GL)", color='white')
im1 = ax1.imshow(np.abs(sim.psi)**2, cmap='inferno', vmin=0, vmax=2)
ax1.axis('off')

# Plot 2: Fase y Flujo (Fase de ψ + Velocidad de Fluido)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title(r"Fase $\theta$ & Flujo $v$ (Fluido)", color='white')
im2 = ax2.imshow(np.angle(sim.psi), cmap='twilight')
# Quiver para el fluido (submuestreado)
st = 4
Q = ax2.quiver(sim.X[::st, ::st], sim.Y[::st, ::st],
               sim.vx[::st, ::st], sim.vy[::st, ::st],
               color='cyan', scale=10, alpha=0.6)
ax2.axis('off')

# Plot 3: Operador Áureo (Background Metric)
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title(r"Operador Áureo $O_n$ (PGP)", color='white')
im3 = ax3.imshow(sim.O_n, cmap='cividis')
ax3.axis('off')

# Plot 4: Dinámica Lagrangiana (La prueba matemática)
ax4 = fig.add_subplot(gs[1, :])
ax4.set_facecolor('#2d2d2d')
ax4.set_title(r"Evolución Metriplética: $L_{total} = L_{symp} + L_{metr}$", color='white')
line_H, = ax4.plot([], [], 'r-', label=r'$L_{symp}$ (Energía/Hamiltoniano)', linewidth=2)
line_S, = ax4.plot([], [], 'c--', label=r'$L_{metr}$ (Disipación/Entropía)', linewidth=2) # Changed label
line_L, = ax4.plot([], [], 'w-', label=r'$L_{eff}$ (Lagrangiano Efectivo)', linewidth=1, alpha=0.5)
ax4.legend(loc='upper right', facecolor='#333', labelcolor='white')
ax4.grid(True, color='#444', linestyle='--')
ax4.set_xlabel('Tiempo', color='white')
ax4.tick_params(colors='white')

def update(frame):
    sim.step(dt=0.05)

    im1.set_array(np.abs(sim.psi)**2)
    im2.set_array(np.angle(sim.psi))
    Q.set_UVC(sim.vx[::st, ::st], sim.vy[::st, ::st])

    # Actualizar gráfica de líneas
    times = sim.t_history
    ax4.set_xlim(0, max(10, times[-1]))

    # Escala dinámica Y
    min_val = min(min(sim.H_history), min(sim.L_history), min(sim.S_history)) # Ensure S_history is included in min calculation
    max_val = max(max(sim.H_history), max(sim.S_history), max(sim.L_history)) # Ensure S_history is included in max calculation
    margin = (max_val - min_val) * 0.1
    ax4.set_ylim(min_val - margin, max_val + margin)

    line_H.set_data(times, sim.H_history)
    line_S.set_data(times, sim.S_history)
    line_L.set_data(times, sim.L_history)

    return im1, im2, Q, line_H, line_S, line_L

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=False)
plt.tight_layout()
plt.show()