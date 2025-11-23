#!/usr/bin/env python3
"""
Tests para el módulo de visualización de esferas de Bloch.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from golden.bloch_viz import (
    get_statevector_from_circuit,
    get_single_qubit_state,
    visualize_bloch_spheres,
    print_bloch_info
)


class TestBlochVisualization:
    """Tests para visualización de esferas de Bloch"""
    
    def test_get_statevector_single_qubit(self):
        """Test: obtener statevector de un circuito de 1 qubit"""
        qc = QuantumCircuit(1)
        qc.h(0)
        
        sv = get_statevector_from_circuit(qc)
        assert isinstance(sv, Statevector)
        assert sv.num_qubits == 1
    
    def test_get_statevector_multi_qubit(self):
        """Test: obtener statevector de un circuito de múltiples qubits"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.x(2)
        
        sv = get_statevector_from_circuit(qc)
        assert isinstance(sv, Statevector)
        assert sv.num_qubits == 3
    
    def test_get_single_qubit_state_ground(self):
        """Test: estado |0⟩ debe dar vector de Bloch (0, 0, 1)"""
        qc = QuantumCircuit(1)
        # Estado inicial es |0⟩
        
        sv = get_statevector_from_circuit(qc)
        bloch_vec = get_single_qubit_state(sv, 0, 1)
        
        # |0⟩ corresponde a z = +1
        assert np.isclose(bloch_vec[0], 0, atol=1e-10)  # x
        assert np.isclose(bloch_vec[1], 0, atol=1e-10)  # y
        assert np.isclose(bloch_vec[2], 1, atol=1e-10)  # z
    
    def test_get_single_qubit_state_excited(self):
        """Test: estado |1⟩ debe dar vector de Bloch (0, 0, -1)"""
        qc = QuantumCircuit(1)
        qc.x(0)  # |0⟩ -> |1⟩
        
        sv = get_statevector_from_circuit(qc)
        bloch_vec = get_single_qubit_state(sv, 0, 1)
        
        # |1⟩ corresponde a z = -1
        assert np.isclose(bloch_vec[0], 0, atol=1e-10)  # x
        assert np.isclose(bloch_vec[1], 0, atol=1e-10)  # y
        assert np.isclose(bloch_vec[2], -1, atol=1e-10)  # z
    
    def test_get_single_qubit_state_superposition(self):
        """Test: estado |+⟩ debe dar vector de Bloch (1, 0, 0)"""
        qc = QuantumCircuit(1)
        qc.h(0)  # |0⟩ -> |+⟩ = (|0⟩ + |1⟩)/√2
        
        sv = get_statevector_from_circuit(qc)
        bloch_vec = get_single_qubit_state(sv, 0, 1)
        
        # |+⟩ corresponde a x = +1
        assert np.isclose(bloch_vec[0], 1, atol=1e-10)  # x
        assert np.isclose(bloch_vec[1], 0, atol=1e-10)  # y
        assert np.isclose(bloch_vec[2], 0, atol=1e-10)  # z
    
    def test_get_single_qubit_state_y_basis(self):
        """Test: estado |+i⟩ debe dar vector de Bloch (0, 1, 0)"""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.s(0)  # |0⟩ -> |+i⟩ = (|0⟩ + i|1⟩)/√2
        
        sv = get_statevector_from_circuit(qc)
        bloch_vec = get_single_qubit_state(sv, 0, 1)
        
        # |+i⟩ corresponde a y = +1
        assert np.isclose(bloch_vec[0], 0, atol=1e-10)  # x
        assert np.isclose(bloch_vec[1], 1, atol=1e-10)  # y
        assert np.isclose(bloch_vec[2], 0, atol=1e-10)  # z
    
    def test_bloch_vector_magnitude(self):
        """Test: magnitud del vector de Bloch debe ser 1 para estados puros"""
        qc = QuantumCircuit(1)
        qc.ry(np.pi / 3, 0)  # Rotación arbitraria
        
        sv = get_statevector_from_circuit(qc)
        bloch_vec = get_single_qubit_state(sv, 0, 1)
        
        magnitude = np.linalg.norm(bloch_vec)
        assert np.isclose(magnitude, 1.0, atol=1e-10)
    
    def test_partial_trace_multi_qubit(self):
        """Test: traza parcial en sistema de múltiples qubits"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)  # Estado de Bell
        
        sv = get_statevector_from_circuit(qc)
        
        # Para un estado de Bell, cada qubit individual está maximamente mezclado
        # El vector de Bloch debe estar cerca del origen
        bloch_vec_0 = get_single_qubit_state(sv, 0, 2)
        bloch_vec_1 = get_single_qubit_state(sv, 1, 2)
        
        # Para estados entrelazados, la magnitud del vector de Bloch < 1
        mag_0 = np.linalg.norm(bloch_vec_0)
        mag_1 = np.linalg.norm(bloch_vec_1)
        
        # Estado de Bell maximamente entrelazado -> vector de Bloch en el origen
        assert mag_0 < 0.1  # Cerca de 0
        assert mag_1 < 0.1  # Cerca de 0
    
    def test_visualize_bloch_single_qubit(self):
        """Test: visualizar esfera de Bloch para un qubit"""
        qc = QuantumCircuit(1)
        qc.h(0)
        
        fig, save_path = visualize_bloch_spheres(qc, 1)
        
        assert fig is not None
        assert save_path.endswith('.png')
        assert 'bloch_q0' in save_path
    
    def test_visualize_bloch_multiple_qubits(self):
        """Test: visualizar esferas de Bloch para múltiples qubits"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)
        qc.y(2)
        
        fig, save_path = visualize_bloch_spheres(qc, 3)
        
        assert fig is not None
        assert save_path.endswith('.png')
        assert 'bloch_q0_1_2' in save_path
    
    def test_visualize_bloch_specific_qubits(self):
        """Test: visualizar esferas de Bloch para qubits específicos"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.x(2)
        qc.y(4)
        
        qubit_indices = [0, 2, 4]
        fig, save_path = visualize_bloch_spheres(qc, 5, qubit_indices)
        
        assert fig is not None
        assert save_path.endswith('.png')
        assert 'bloch_q0_2_4' in save_path
    
    def test_visualize_bloch_max_qubits(self):
        """Test: límite de 10 qubits en visualización"""
        qc = QuantumCircuit(15)
        for i in range(15):
            qc.h(i)
        
        # Debe visualizar solo los primeros 10
        fig, save_path = visualize_bloch_spheres(qc, 15)
        
        assert fig is not None
        # Verificar que solo se visualizaron 10 qubits
        assert '0_1_2_3_4_5_6_7_8_9' in save_path
    
    def test_print_bloch_info(self, capsys):
        """Test: imprimir información de vectores de Bloch"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        
        print_bloch_info(qc, 2, [0, 1])
        
        captured = capsys.readouterr()
        assert "Vectores de Bloch" in captured.out
        assert "Qubit 0" in captured.out
        assert "Qubit 1" in captured.out
        assert "Vector:" in captured.out
        assert "Magnitud:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
