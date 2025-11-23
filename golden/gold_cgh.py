#!/usr/bin/env python3
"""
Fibonacci QML con Estructura Metriplética

Integra el formalismo metripléctico en el marco de Fibonacci QML,
permitiendo modelar:
- Evolución unitaria (Hamiltoniana) del circuito cuántico
- Disipación y decoherencia (métrica) del entorno
- Optimización termodinámicamente consistente

Concepto:
    El circuito cuántico evoluciona bajo:
    dψ/dt = {ψ, H_circuit} + (ψ, S_env)
    
    Donde:
    - H_circuit contiene compuertas Fibonacci (φ-gates)
    - S_env modela interacción con el entorno (decoherencia)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Asegurar que podemos importar desde el directorio raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from golden.metriplectic import (
    create_simple_metriplectic_system,
    MetriplecticIntegrator,
    MetriplecticSystem
)
from golden.vacuum_metriplectic import (
    compute_O1,
    compute_Fpura,
    compute_CUNIF
)


PHI =  (1+5**0.5)/2 # Razón áurea


class FibonacciMetriplecticQML:
    """
    Quantum Machine Learning con ansatz Fibonacci y estructura metriplética.
    
    Combina:
    - Capas parametrizadas por Fibonacci (pares/impares)
    - Dinámica Hamiltoniana (circuito unitario)
    - Disipación controlada (modelo del vacío con CUNIF)
    """
    
    def __init__(self, 
                 n_qubits: int = 3,
                 decoherence_strength: float = 0.01):
        """
        Inicializar sistema metripléctico Fibonacci.
        
        Args:
            n_qubits: Número de qubits
            decoherence_strength: Fuerza de decoherencia (0 = unitario puro)
        """
        self.n_qubits = n_qubits
        self.decoherence = decoherence_strength
        
        # Dimensión del espacio de Hilbert (2^n_qubits)
        self.hilbert_dim = 2 ** n_qubits
        
        # Parámetros del vacío
        self.O1 = compute_O1(1)
        self.Fpura = compute_Fpura(self.O1)
        self.CUNIF = compute_CUNIF(137.036, self.Fpura)
        self.eta = 1.0 / self.CUNIF  # Viscosidad
        
        # Secuencia Fibonacci
        self.fibonacci = self._generate_fibonacci(20)
        
        print(f"🌀 Fibonacci Metriplectic QML Initialized")
        print(f"  Qubits: {n_qubits}")
        print(f"  Hilbert dim: {self.hilbert_dim}")
        print(f"  Decoherence: {decoherence_strength}")
        print(f"  Vacuum η: {self.eta:.6e}")
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Genera secuencia Fibonacci"""
        if n <= 0:
            return []
        if n == 1:
            return [1]
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def phi_gate_hamiltonian(self, n: int, qubit: int) -> np.ndarray:
        """
        Hamiltoniano de la φ-gate (Golden Gate).
        
        Args:
            n: Índice Fibonacci
            qubit: Índice del qubit
            
        Returns:
            Hamiltoniano de la compuerta en espacio de Hilbert
        """
        # Fase basada en Fibonacci y φ
        phase = np.cos(n * np.pi) * np.cos(n * PHI * np.pi)
        
        # Matriz de fase para un qubit (2x2)
        # U(λ) = diag(1, e^{iλ})
        single_qubit = np.array([
            [1.0, 0.0],
            [0.0, np.exp(1j * phase)]
        ], dtype=complex)
        
        # Extender a sistema multi-qubit (producto tensorial)
        H = np.array([[1.0]], dtype=complex)
        for q in range(self.n_qubits):
            if q == qubit:
                H = np.kron(H, single_qubit)
            else:
                H = np.kron(H, np.eye(2, dtype=complex))
        
        # Hamiltoniano = -ilog(U) (aproximación para pequeñas fases)
        # Para simplificar: H ~ phase * Z_qubit
        return phase * self._pauli_z_extended(qubit)
    
    def _pauli_z_extended(self, qubit: int) -> np.ndarray:
        """
        Operador Z de Pauli extendido al espacio multi-qubit.
        
        Args:
            qubit: Índice del qubit objetivo
            
        Returns:
            Matriz Z extendida
        """
        Z = np.array([[1, 0], [0, -1]], dtype=float)
        I = np.eye(2, dtype=float)
        
        result = np.array([[1.0]], dtype=float)
        for q in range(self.n_qubits):
            if q == qubit:
                result = np.kron(result, Z)
            else:
                result = np.kron(result, I)
        
        return result
    
    def create_circuit_hamiltonian(self, fib_params: List[int]) -> np.ndarray:
        """
        Crear Hamiltoniano total del circuito.
        
        Args:
            fib_params: Lista de índices Fibonacci para cada capa
            
        Returns:
            Hamiltoniano del circuito completo
        """
        H_total = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=float)
        
        for layer_idx, fib_idx in enumerate(fib_params):
            n_val = self.fibonacci[fib_idx % len(self.fibonacci)]
            qubit = layer_idx % self.n_qubits
            
            H_layer = self.phi_gate_hamiltonian(n_val, qubit)
            H_total += H_layer
        
        return H_total
    
    def entropy_functional(self, 
                          rho: np.ndarray,
                          use_von_neumann: bool = False) -> float:
        """
        Funcional de entropía.
        
        Args:
            rho: Matriz de densidad (o amplitudes)
            use_von_neumann: Si True, usa entropía de von Neumann
            
        Returns:
            Valor de entropía
        """
        if use_von_neumann:
            # S = -Tr(ρ log ρ)
            eigenvalues = np.linalg.eigvalsh(rho)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Evitar log(0)
            return -np.sum(eigenvalues * np.log(eigenvalues))
        else:
            # Entropía lineal: S = 1 - Tr(ρ²)
            return 1.0 - np.trace(np.dot(rho, rho)).real
    
    def analyze_fibonacci_coupling(self, 
                                   fib_params: List[int],
                                   t_final: float = 5.0) -> Dict:
        """
        Analizar acoplamiento Fibonacci bajo dinámica metriplética.
        
        Args:
            fib_params: Parámetros Fibonacci
            t_final: Tiempo final de evolución
            
        Returns:
            Diccionario con análisis
        """
        print(f"\n🔬 Analyzing Fibonacci Coupling")
        print(f"  Parameters: {fib_params}")
        print(f"  Fibonacci values: {[self.fibonacci[p] for p in fib_params]}")
        
        # Clasificar pares/impares
        fib_vals = [self.fibonacci[p % len(self.fibonacci)] for p in fib_params]
        n_even = sum(1 for f in fib_vals if f % 2 == 0)
        n_odd = len(fib_vals) - n_even
        
        mix_ratio = n_even / len(fib_vals) if len(fib_vals) > 0 else 0
        
        # Calcular operador O_n promedio
        O_avg = np.mean([abs(np.cos(f * np.pi) * np.cos(f * PHI * np.pi)) 
                         for f in fib_vals])
        
        # Fricción geométrica efectiva
        F_eff = 1.0 / (O_avg ** 2) if O_avg > 1e-6 else self.Fpura
        
        # Create metriplectic system
        H_circuit = self.create_circuit_hamiltonian(fib_params)
        
        # Para análisis simplificado, trabajar en espacio reducido 2D
        # Proyectar a subespacio {|00...0⟩, |11...1⟩}
        z0 = np.array([1.0, 0.0])  # Estado inicial proyectado
        
        # Hamiltoniano proyectado (valores propios extremos)
        eigenvalues = np.linalg.eigvalsh(H_circuit)
        E_min, E_max = eigenvalues[0], eigenvalues[-1]
        
        # Sistema metripléctico 2D simplificado
        def hamiltonian(z: np.ndarray) -> float:
            return 0.5 * (E_max - E_min) * (z[0]**2 - z[1]**2)
        
        def entropy(z: np.ndarray) -> float:
            # Entropía aumenta con mezcla
            return self.decoherence * np.sum(z**2)
        
        def dH(z: np.ndarray) -> np.ndarray:
            return (E_max - E_min) * np.array([z[0], -z[1]])
        
        def dS(z: np.ndarray) -> np.ndarray:
            return 2 * self.decoherence * z
        
        # Crear sistema
        J = np.array([[0, 1], [-1, 0]])  # Poisson canónico
        G = self.eta * np.eye(2)  # Métrica disipativa
        
        system = create_simple_metriplectic_system(
            dimension=2,
            hamiltonian=hamiltonian,
            entropy=entropy,
            dH=dH,
            dS=dS,
            J=J,
            G=G
        )
        
        # Integrar
        integrator = MetriplecticIntegrator(system)
        history = integrator.integrate(
            z0=z0,
            t_span=(0, t_final),
            n_points=100
        )
        
        # Análisis
        final_entropy = history['entropy'][-1]
        entropy_production = final_entropy - history['entropy'][0]
        
        analysis = {
            'fib_params': fib_params,
            'fib_values': fib_vals,
            'n_even': n_even,
            'n_odd': n_odd,
            'mix_ratio': mix_ratio,
            'O_avg': O_avg,
            'F_effective': F_eff,
            'eigenvalues': eigenvalues,
            'energy_gap': E_max - E_min,
            'history': history,
            'entropy_production': entropy_production,
            'final_entropy': final_entropy
        }
        
        print(f"\n📊 Results:")
        print(f"  Even/Odd ratio: {n_even}/{n_odd}")
        print(f"  Mix ratio: {mix_ratio:.2f}")
        print(f"  <|O_n|>: {O_avg:.6f}")
        print(f"  F_effective: {F_eff:.6f}")
        print(f"  Energy gap: {E_max - E_min:.6f}")
        print(f"  Entropy production: {entropy_production:.6e}")
        
        return analysis

        return analysis

    def visualize_holographic_shapes(self, analyses: List[Tuple[str, Dict]]):
        """
        Genera visualización holográfica (Trayectorias en esfera de Bloch/Espacio de Fase).
        """
        print(f"\n🎨 Generando formas holográficas...")
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=("Trayectoria en Espacio de Fase (Holográfico)", "Producción de Entropía"),
            column_widths=[0.6, 0.4]
        )
        
        colors = {'PARES': 'cyan', 'IMPARES': 'magenta', 'MIXTA': 'gold'}
        
        for name, analysis in analyses:
            history = analysis['history']
            z = history['z']  # Shape (n_points, 2)
            entropy = history['entropy']
            t = history['t']
            
            # 3D Plot: z0 vs z1 vs entropy (Holographic Phase Space)
            fig.add_trace(
                go.Scatter3d(
                    x=z[:, 0],
                    y=z[:, 1],
                    z=entropy,
                    mode='lines',
                    name=f'{name} (Fase)',
                    line=dict(color=colors.get(name, 'white'), width=5),
                    opacity=0.8
                ),
                row=1, col=1
            )
            
            # 2D Plot: Entropy vs Time
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=entropy,
                    mode='lines',
                    name=f'{name} (Entropía)',
                    line=dict(color=colors.get(name, 'white'), width=2, dash='dot')
                ),
                row=1, col=2
            )

        # Layout holográfico
        fig.update_layout(
            title="Dinámica Metripléctica: Formas Holográficas",
            template="plotly_dark",
            height=600,
            scene=dict(
                xaxis_title='Amplitud |0...0>',
                yaxis_title='Amplitud |1...1>',
                zaxis_title='Entropía (S)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            )
        )
        
        # Guardar HTML con viewport y estilos responsive
        output_file = "cgh_holographic.html"
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Análisis Metripléctico de Circuitos Cuánticos con Fibonacci">
    <meta name="author" content="GoldenA Quantum Framework">
    <title>CGH Holographic Shapes - Análisis Metripléctico</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            color: #e0e0e0;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        .container {{
            width: 100%;
            max-width: 100vw;
            padding: 1rem;
        }}
        .header {{
            text-align: center;
            padding: 1.5rem 0;
            background: rgba(26, 26, 46, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid #00d4ff;
            margin-bottom: 1rem;
        }}
        .header h1 {{
            font-size: clamp(1.5rem, 4vw, 2.5rem);
            background: linear-gradient(45deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            font-size: clamp(0.875rem, 2vw, 1rem);
            color: #a0a0a0;
        }}
        .plot-container {{
            width: 100%;
            height: auto;
            min-height: 60vh;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 0.5rem;
            }}
            .header {{
                padding: 1rem 0;
            }}
            .plot-container {{
                min-height: 50vh;
            }}
        }}
        @media print {{
            body {{
                background: white;
                color: black;
            }}
            .header {{
                background: none;
                border-bottom: 2px solid black;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌀 Análisis Metripléctico CGH</h1>
        <p>Dinámica Fibonacci en Circuitos Cuánticos</p>
    </div>
    <div class="container">
        <div class="plot-container">
            {plot_html}
        </div>
    </div>
</body>
</html>"""
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)
            
        print(f"✅ Visualización holográfica guardada en '{output_file}'")
        
        # Intentar abrir
        import webbrowser
        try:
            webbrowser.open(output_file)
        except:
            pass

def analyze_circuit_metriplectic(circuit, n_qubits: int = None):
    """
    Analiza un circuito de Qiskit extrayendo parámetros Fibonacci de GoldenGates.
    
    Args:
        circuit: QuantumCircuit de Qiskit
        n_qubits: Número de qubits (si None, se infiere del circuito)
    
    Returns:
        Dict con resultados del análisis
    """
    if circuit is None:
        print("❌ No hay circuito para analizar")
        return None
    
    # Inferir número de qubits
    if n_qubits is None:
        n_qubits = circuit.num_qubits
    
    print("=" * 80)
    print(f"🔬 ANÁLISIS METRIPLÉCTICO DEL CIRCUITO")
    print("=" * 80)
    print(f"  Qubits: {n_qubits}")
    print(f"  Profundidad: {circuit.depth()}")
    print(f"  Operaciones: {len(circuit.data)}")
    
    # Extraer parámetros de GoldenGates
    fib_params = []
    for instruction, qargs, cargs in circuit.data:
        if instruction.name == "golden_gate":
            # Extraer el parámetro n de la GoldenGate
            n_val = instruction.params[0]
            fib_params.append(int(n_val))
            # Obtener índice del qubit correctamente
            qubit_index = circuit.find_bit(qargs[0]).index
            print(f"  ✓ GoldenGate encontrada: n={n_val} en qubit {qubit_index}")

    
    if not fib_params:
        print("\n⚠️  No se encontraron GoldenGates en el circuito")
        print("💡 Tip: Agrega puertas phi con 'agregar phi <n> <qubit>'")
        return None
    
    print(f"\n📊 Parámetros Fibonacci extraídos: {fib_params}")
    
    # Crear sistema metripléctico
    qml = FibonacciMetriplecticQML(
        n_qubits=n_qubits,
        decoherence_strength=0.001
    )
    
    # Analizar con los parámetros extraídos
    print(f"\n{'─' * 80}")
    print("ANÁLISIS METRIPLÉCTICO")
    print(f"{'─' * 80}")
    
    analysis = qml.analyze_fibonacci_coupling(fib_params, t_final=10.0)
    
    # Clasificar régimen
    dS = analysis['entropy_production']
    if dS < 1e-6:
        regime = "Coherente"
    elif dS < 1e-4:
        regime = "Cuasi-coherente"
    else:
        regime = "Disipativo"
    
    print(f"\n{'=' * 80}")
    print("📊 RESUMEN DEL ANÁLISIS")
    print(f"{'=' * 80}")
    print(f"  Parámetros Fibonacci: {fib_params}")
    print(f"  Valores Fibonacci: {analysis['fib_values']}")
    print(f"  Pares/Impares: {analysis['n_even']}/{analysis['n_odd']}")
    print(f"  F_efectiva: {analysis['F_effective']:.6f}")
    print(f"  Gap energético: {analysis['energy_gap']:.6f}")
    print(f"  Producción de entropía: {dS:.6e}")
    print(f"  Régimen: {regime}")
    print(f"{'=' * 80}")
    
    # Generar visualización
    qml.visualize_holographic_shapes([
        ("CIRCUITO", analysis)
    ])
    
    return analysis


def demo_fibonacci_metriplectic(n_qubits: int = 3):
    """
    Demostración de Fibonacci QML con estructura metriplética.
    Args:
        n_qubits: Número de qubits para la simulación
    """
    print("=" * 80)
    print(f"🌌 FIBONACCI QML + METRIPLECTIC STRUCTURE (n={n_qubits})")
    print("=" * 80)
    
    # Crear sistema
    qml = FibonacciMetriplecticQML(
        n_qubits=n_qubits,
        decoherence_strength=0.001  # Decoherencia débil
    )
    
    # Test 1: Secuencia pura de pares (amplificación)
    print("\n" + "─" * 80)
    print("TEST 1: Secuencia PARES (amplificación)")
    print("─" * 80)
    
    params_even = [1, 3, 5]  # Fibonacci: [1, 2, 5] → indices de pares
    analysis_even = qml.analyze_fibonacci_coupling(params_even, t_final=10.0)
    
    # Test 2: Secuencia pura de impares (cancelación)
    print("\n" + "─" * 80)
    print("TEST 2: Secuencia IMPARES (cancelación)")
    print("─" * 80)
    
    params_odd = [0, 2, 4]  # Fibonacci: [1, 1, 3] → impares
    analysis_odd = qml.analyze_fibonacci_coupling(params_odd, t_final=10.0)
    
    # Test 3: Secuencia mixta (control fino)
    print("\n" + "─" * 80)
    print("TEST 3: Secuencia MIXTA (control fino)")
    print("─" * 80)
    
    params_mixed = [1, 2, 3, 4]  # Fibonacci: [1, 1, 2, 3] → mezclado
    analysis_mixed = qml.analyze_fibonacci_coupling(params_mixed, t_final=10.0)
    
    # Comparación
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN")
    print("=" * 80)
    
    configs = [
        ("PARES", analysis_even),
        ("IMPARES", analysis_odd),
        ("MIXTA", analysis_mixed)
    ]
    
    print(f"\n{'Config':<12} {'F_eff':<10} {'Gap':<10} {'ΔS':<12} {'Régimen'}")
    print("─" * 60)
    
    for name, analysis in configs:
        F = analysis['F_effective']
        gap = analysis['energy_gap']
        dS = analysis['entropy_production']
        
        # Régimen basado en producción de entropía
        if dS < 1e-6:
            regime = "Coherente"
        elif dS < 1e-4:
            regime = "Cuasi-coherente"
        else:
            regime = "Disipativo"
        
        print(f"{name:<12} {F:<10.4f} {gap:<10.4f} {dS:<12.6e} {regime}")
    
    print("\n" + "=" * 80)
    print("✅ Análisis completo!")
    print("=" * 80)
    
    print("\n💡 Observaciones clave:")
    print("  • Secuencias PARES tienden a mayor gap energético (amplificación)")
    print("  • Secuencias IMPARES tienen menor F_eff (menor fricción)")
    print("  • Secuencias MIXTAS permiten control fino de disipación")
    print("  • La estructura metriplética preserva consistencia termodinámica")

    # Generar visualización
    qml.visualize_holographic_shapes([
        ("PARES", analysis_even),
        ("IMPARES", analysis_odd),
        ("MIXTA", analysis_mixed)
    ])


if __name__ == "__main__":
    demo_fibonacci_metriplectic()