import numpy as np
import logging

# Intentar importar Q-CTRL
try:
    # Nota: El paquete 'qctrl' parece no estar disponible en este entorno.
    # Se usa 'qctrlclient' o 'fireopal' para otras funciones.
    # Este código asume que 'qctrl' (Boulder Opal) estará disponible eventualmente.
    from qctrl import Qctrl
    QCTRL_AVAILABLE = True
except ImportError:
    QCTRL_AVAILABLE = False

def optimize_pulse_demo(duration: float = 10e-6, segment_count: int = 50):
    """
    Demo de optimización de pulso usando Q-CTRL Boulder Opal (sintaxis de grafos).
    """
    if not QCTRL_AVAILABLE:
        print("❌ El paquete 'qctrl' (Boulder Opal) no está instalado o no se encuentra.")
        print("   Este demo requiere la suite completa de Q-CTRL.")
        return

    print(f"🚀 Iniciando optimización de pulso (Duración: {duration}s, Segmentos: {segment_count})...")
    
    qctrl = Qctrl()
    
    with qctrl.create_graph() as graph:
        # Pulse parameters.
        # segment_count = 50 (argumento)
        # duration = 10e-6  # s (argumento)

        # Maximum value for |α(t)|.
        alpha_max = 2 * np.pi * 0.25e6  # rad/s

        # Real PWC signal representing α(t).
        alpha = graph.real_optimizable_pwc_signal(
            segment_count=segment_count,
            duration=duration,
            minimum=-alpha_max,
            maximum=alpha_max,
            name="$\\alpha$",
        )
        
        # Aquí iría el resto de la definición del Hamiltoniano y la función de costo
        # Por ejemplo:
        # hamiltonian = alpha * sigma_x
        # ...
        
        print("✅ Grafo de optimización definido (Snippet del usuario integrado).")
        print("   Nota: Para ejecutar la optimización real, se necesita definir el Hamiltoniano target y la función de costo.")

if __name__ == "__main__":
    optimize_pulse_demo()
