#!/usr/bin/env python3
"""
Módulo de visualización de esferas de Bloch para estados cuánticos.
Permite visualizar el estado de qubits individuales en la esfera de Bloch.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
from qiskit.visualization import plot_bloch_multivector
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple


def get_statevector_from_circuit(circuit: QuantumCircuit) -> Statevector:
    """
    Obtiene el vector de estado de un circuito cuántico sin mediciones.
    
    Args:
        circuit: Circuito cuántico (sin operaciones de medición)
        
    Returns:
        Statevector del circuito
    """
    # Crear una copia del circuito sin mediciones
    qc_copy = circuit.copy()
    qc_copy.remove_final_measurements()
    
    # Calcular el statevector
    statevector = Statevector.from_instruction(qc_copy)
    return statevector


def get_single_qubit_state(statevector: Statevector, qubit_index: int, num_qubits: int) -> np.ndarray:
    """
    Extrae el estado de un qubit individual mediante traza parcial.
    
    Args:
        statevector: Vector de estado completo del sistema
        qubit_index: Índice del qubit a extraer
        num_qubits: Número total de qubits en el sistema
        
    Returns:
        Vector de Bloch [x, y, z] para el qubit
    """
    # Convertir a matriz de densidad
    rho = DensityMatrix(statevector)
    
    # Qubits a trazar (todos excepto el que queremos)
    qubits_to_trace = [i for i in range(num_qubits) if i != qubit_index]
    
    # Traza parcial para obtener el estado reducido del qubit
    if qubits_to_trace:
        rho_reduced = partial_trace(rho, qubits_to_trace)
    else:
        rho_reduced = rho
    
    # Calcular el vector de Bloch
    # Para una matriz de densidad 2x2: rho = (I + r·σ)/2
    # donde r es el vector de Bloch y σ son las matrices de Pauli
    rho_matrix = rho_reduced.data
    
    x = 2 * np.real(rho_matrix[0, 1])
    y = 2 * np.imag(rho_matrix[0, 1])
    z = np.real(rho_matrix[0, 0] - rho_matrix[1, 1])
    
    return np.array([x, y, z])


def visualize_bloch_spheres(
    circuit: QuantumCircuit,
    num_qubits: int,
    qubit_indices: Optional[List[int]] = None,
    max_qubits: int = 10
) -> Tuple[Figure, str]:
    """
    Visualiza las esferas de Bloch para los qubits especificados.
    
    Args:
        circuit: Circuito cuántico a visualizar
        num_qubits: Número total de qubits en el circuito
        qubit_indices: Lista de índices de qubits a visualizar (None = todos)
        max_qubits: Número máximo de qubits a visualizar
        
    Returns:
        Tupla (figura matplotlib, ruta del archivo guardado)
    """
    # Determinar qué qubits visualizar
    if qubit_indices is None:
        # Visualizar todos los qubits (hasta el máximo)
        qubit_indices = list(range(min(num_qubits, max_qubits)))
    else:
        # Validar índices
        qubit_indices = [q for q in qubit_indices if 0 <= q < num_qubits]
        if len(qubit_indices) > max_qubits:
            qubit_indices = qubit_indices[:max_qubits]
    
    if not qubit_indices:
        raise ValueError("No hay qubits válidos para visualizar")
    
    # Obtener el statevector del circuito
    try:
        statevector = get_statevector_from_circuit(circuit)
    except Exception as e:
        raise RuntimeError(f"Error al calcular el statevector: {e}")
    
    # Si solo hay un qubit en el sistema, usar plot_bloch_multivector directamente
    if num_qubits == 1:
        fig = plot_bloch_multivector(statevector)
        fig.suptitle(f'Esfera de Bloch - Qubit 0', fontsize=14, fontweight='bold')
    else:
        # Para múltiples qubits, crear una figura con subplots
        num_plots = len(qubit_indices)
        cols = min(3, num_plots)  # Máximo 3 columnas
        rows = (num_plots + cols - 1) // cols  # Calcular filas necesarias
        
        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        
        for idx, qubit_idx in enumerate(qubit_indices):
            # Obtener el vector de Bloch para este qubit
            bloch_vector = get_single_qubit_state(statevector, qubit_idx, num_qubits)
            
            # Crear subplot
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            # Dibujar la esfera de Bloch
            draw_bloch_sphere(ax, bloch_vector, qubit_idx)
        
        fig.suptitle(f'Esferas de Bloch - {len(qubit_indices)} Qubit(s)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
    
    # Guardar la figura
    save_dir = Path.home() / ".qiskit_cli" / "bloch_spheres"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    qubits_str = '_'.join(map(str, qubit_indices))
    filename = f"bloch_q{qubits_str}_{timestamp}.png"
    save_path = save_dir / filename
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    return fig, str(save_path)


def draw_bloch_sphere(ax, bloch_vector: np.ndarray, qubit_idx: int):
    """
    Dibuja una esfera de Bloch en un eje 3D de matplotlib.
    
    Args:
        ax: Eje 3D de matplotlib
        bloch_vector: Vector de Bloch [x, y, z]
        qubit_idx: Índice del qubit (para el título)
    """
    # Parámetros de la esfera
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Dibujar la esfera (transparente)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                   color='lightblue', alpha=0.15, linewidth=0)
    
    # Dibujar los ejes
    axis_length = 1.2
    ax.plot([-axis_length, axis_length], [0, 0], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [-axis_length, axis_length], [0, 0], 'k-', linewidth=1, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-axis_length, axis_length], 'k-', linewidth=1, alpha=0.3)
    
    # Etiquetas de los ejes
    ax.text(axis_length, 0, 0, 'X', fontsize=12, fontweight='bold')
    ax.text(0, axis_length, 0, 'Y', fontsize=12, fontweight='bold')
    ax.text(0, 0, axis_length, '|0⟩', fontsize=12, fontweight='bold')
    ax.text(0, 0, -axis_length, '|1⟩', fontsize=12, fontweight='bold')
    
    # Dibujar círculos ecuatoriales
    circle_points = 100
    theta = np.linspace(0, 2 * np.pi, circle_points)
    
    # Círculo XY (ecuador)
    ax.plot(np.cos(theta), np.sin(theta), 0, 'gray', linewidth=1, alpha=0.3)
    # Círculo XZ
    ax.plot(np.cos(theta), 0, np.sin(theta), 'gray', linewidth=1, alpha=0.3)
    # Círculo YZ
    ax.plot(0, np.cos(theta), np.sin(theta), 'gray', linewidth=1, alpha=0.3)
    
    # Dibujar el vector de estado
    x, y, z = bloch_vector
    ax.quiver(0, 0, 0, x, y, z, 
             color='red', arrow_length_ratio=0.15, linewidth=3, alpha=0.9)
    
    # Punto en la punta del vector
    ax.scatter([x], [y], [z], color='red', s=100, alpha=1.0)
    
    # Configurar límites y aspecto
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    
    # Título
    ax.set_title(f'Qubit {qubit_idx}', fontsize=12, fontweight='bold', pad=10)
    
    # Ocultar ejes numéricos
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Información del estado
    magnitude = np.linalg.norm(bloch_vector)
    info_text = f'|r| = {magnitude:.3f}\n'
    info_text += f'x = {x:.3f}\n'
    info_text += f'y = {y:.3f}\n'
    info_text += f'z = {z:.3f}'
    
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def print_bloch_info(circuit: QuantumCircuit, num_qubits: int, qubit_indices: Optional[List[int]] = None):
    """
    Imprime información sobre los vectores de Bloch de los qubits.
    
    Args:
        circuit: Circuito cuántico
        num_qubits: Número total de qubits
        qubit_indices: Lista de índices de qubits (None = todos)
    """
    if qubit_indices is None:
        qubit_indices = list(range(min(num_qubits, 10)))
    
    try:
        statevector = get_statevector_from_circuit(circuit)
        
        print("\n📊 Vectores de Bloch:")
        print("=" * 60)
        
        for qubit_idx in qubit_indices:
            if qubit_idx >= num_qubits:
                continue
                
            bloch_vector = get_single_qubit_state(statevector, qubit_idx, num_qubits)
            x, y, z = bloch_vector
            magnitude = np.linalg.norm(bloch_vector)
            
            print(f"\nQubit {qubit_idx}:")
            print(f"  Vector: ({x:+.4f}, {y:+.4f}, {z:+.4f})")
            print(f"  Magnitud: {magnitude:.4f}")
            
            # Calcular ángulos esféricos
            theta = np.arccos(np.clip(z / max(magnitude, 1e-10), -1, 1))
            phi = np.arctan2(y, x)
            
            print(f"  θ (polar): {np.degrees(theta):.2f}°")
            print(f"  φ (azimutal): {np.degrees(phi):.2f}°")
            
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error al calcular vectores de Bloch: {e}")
