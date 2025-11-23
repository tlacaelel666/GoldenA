import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

@dataclass
class MetriplecticSystem:
    dimension: int
    hamiltonian: Callable[[np.ndarray], float]
    entropy: Callable[[np.ndarray], float]
    dH: Callable[[np.ndarray], np.ndarray]
    dS: Callable[[np.ndarray], np.ndarray]
    J: np.ndarray
    G: np.ndarray

def create_simple_metriplectic_system(
    dimension: int,
    hamiltonian: Callable[[np.ndarray], float],
    entropy: Callable[[np.ndarray], float],
    dH: Callable[[np.ndarray], np.ndarray],
    dS: Callable[[np.ndarray], np.ndarray],
    J: np.ndarray,
    G: np.ndarray
) -> MetriplecticSystem:
    return MetriplecticSystem(
        dimension=dimension,
        hamiltonian=hamiltonian,
        entropy=entropy,
        dH=dH,
        dS=dS,
        J=J,
        G=G
    )

class MetriplecticIntegrator:
    def __init__(self, system: MetriplecticSystem):
        self.system = system

    def dynamics(self, z: np.ndarray) -> np.ndarray:
        # dz/dt = {z, H} + (z, S)
        # dz/dt = J·dH + G·dS
        dH_val = self.system.dH(z)
        dS_val = self.system.dS(z)
        
        conservative = self.system.J @ dH_val
        dissipative = self.system.G @ dS_val
        
        return conservative + dissipative

    def integrate(self, z0: np.ndarray, t_span: Tuple[float, float], n_points: int) -> Dict[str, Any]:
        t_start, t_end = t_span
        t_eval = np.linspace(t_start, t_end, n_points)
        dt = (t_end - t_start) / (n_points - 1)
        
        z_history = [z0]
        entropy_history = [self.system.entropy(z0)]
        energy_history = [self.system.hamiltonian(z0)]
        
        current_z = z0.copy()
        
        for _ in range(n_points - 1):
            # RK4 integration
            k1 = self.dynamics(current_z)
            k2 = self.dynamics(current_z + 0.5 * dt * k1)
            k3 = self.dynamics(current_z + 0.5 * dt * k2)
            k4 = self.dynamics(current_z + dt * k3)
            
            current_z = current_z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            z_history.append(current_z)
            entropy_history.append(self.system.entropy(current_z))
            energy_history.append(self.system.hamiltonian(current_z))
            
        return {
            't': t_eval,
            'z': np.array(z_history),
            'entropy': np.array(entropy_history),
            'energy': np.array(energy_history)
        }
