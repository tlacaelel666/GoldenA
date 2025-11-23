#!/usr/bin/env python3
"""
Script de prueba simple para verificar la funcionalidad del comando bloch.
"""

import sys
sys.path.insert(0, '/home/jako/GoldenALPHA/GoldenA')

from qiskit import QuantumCircuit
from golden.bloch_viz import visualize_bloch_spheres, print_bloch_info

def test_bloch_single_qubit():
    """Test básico: un qubit en superposición"""
    print("=" * 60)
    print("TEST 1: Un qubit en superposición (Hadamard)")
    print("=" * 60)
    
    qc = QuantumCircuit(1)
    qc.h(0)
    
    print("\nCircuito:")
    print(qc.draw(output='text'))
    
    print_bloch_info(qc, 1, [0])
    
    fig, save_path = visualize_bloch_spheres(qc, 1)
    print(f"\n✅ Visualización guardada en: {save_path}")
    print()

def test_bloch_multiple_qubits():
    """Test: múltiples qubits con diferentes estados"""
    print("=" * 60)
    print("TEST 2: Tres qubits con diferentes estados")
    print("=" * 60)
    
    qc = QuantumCircuit(3)
    qc.h(0)      # Superposición en X
    qc.x(1)      # Estado |1⟩ (Z negativo)
    qc.h(2)
    qc.s(2)      # Superposición en Y
    
    print("\nCircuito:")
    print(qc.draw(output='text'))
    
    print_bloch_info(qc, 3, [0, 1, 2])
    
    fig, save_path = visualize_bloch_spheres(qc, 3)
    print(f"\n✅ Visualización guardada en: {save_path}")
    print()

def test_bloch_specific_qubits():
    """Test: visualizar solo qubits específicos"""
    print("=" * 60)
    print("TEST 3: Cinco qubits, visualizar solo 0, 2, 4")
    print("=" * 60)
    
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.x(1)
    qc.y(2)
    qc.z(3)
    qc.h(4)
    qc.t(4)
    
    print("\nCircuito:")
    print(qc.draw(output='text'))
    
    qubit_indices = [0, 2, 4]
    print_bloch_info(qc, 5, qubit_indices)
    
    fig, save_path = visualize_bloch_spheres(qc, 5, qubit_indices)
    print(f"\n✅ Visualización guardada en: {save_path}")
    print()

def test_bloch_entangled():
    """Test: estado de Bell (entrelazado)"""
    print("=" * 60)
    print("TEST 4: Estado de Bell (entrelazado)")
    print("=" * 60)
    
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    print("\nCircuito:")
    print(qc.draw(output='text'))
    
    print("\nNOTA: En estados entrelazados, los qubits individuales")
    print("están maximamente mezclados (vector de Bloch cerca del origen)")
    
    print_bloch_info(qc, 2, [0, 1])
    
    fig, save_path = visualize_bloch_spheres(qc, 2)
    print(f"\n✅ Visualización guardada en: {save_path}")
    print()

if __name__ == "__main__":
    print("\n🌐 PRUEBAS DE VISUALIZACIÓN DE ESFERAS DE BLOCH\n")
    
    try:
        test_bloch_single_qubit()
        test_bloch_multiple_qubits()
        test_bloch_specific_qubits()
        test_bloch_entangled()
        
        print("=" * 60)
        print("✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
