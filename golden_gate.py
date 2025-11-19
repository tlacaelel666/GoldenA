import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
import math

# Constantes
PHI = (1 + math.sqrt(5)) / 2
GOLDEN_PHASE = math.pi / PHI

class GoldenGate(Gate):
    """
    Puerta Cuántica Áurea.
    Aplica una fase relativa igual a pi/phi.
    """
    
    def __init__(self, label=None):
        super().__init__("golden_gate", 1, [], label=label)
        
    def _define(self):
        """
        Descomposición de la puerta en rotaciones estándar.
        Usamos U3(theta, phi, lambda)
        Para una puerta de fase P(lambda), U3(0, 0, lambda) es equivalente.
        """
        qc = QuantumCircuit(1)
        # Descomposición usando U3 (estándar universal)
        # U3(0, 0, lambda) = [[1, 0], [0, exp(i*lambda)]]
        qc.u(0, 0, GOLDEN_PHASE, 0) 
        self.definition = qc

def verify_golden_gate():
    print("=== Verificación Matemática de GoldenGate ===")
    
    # 1. Crear la puerta y obtener su operador unitario
    golden = GoldenGate()
    op_actual = Operator(golden)
    matriz_actual = op_actual.data
    
    print(f"Fase Áurea (rad): {GOLDEN_PHASE:.6f}")
    
    # 2. Calcular la matriz teórica esperada
    # P(lambda) = [[1, 0], [0, e^(i*lambda)]]
    matriz_teorica = np.array([
        [1, 0],
        [0, np.exp(1j * GOLDEN_PHASE)]
    ])
    
    print("\nMatriz Actual (desde Qiskit):")
    print(np.round(matriz_actual, 4))
    
    print("\nMatriz Teórica:")
    print(np.round(matriz_teorica, 4))
    
    # 3. Comparar
    # Usamos allclose para comparar números de punto flotante con tolerancia
    es_valido = np.allclose(matriz_actual, matriz_teorica)
    
    if es_valido:
        print("\n✅ VERIFICACIÓN EXITOSA: La descomposición coincide con la matriz teórica.")
    else:
        print("\n❌ ERROR: Las matrices no coinciden.")
        diff = np.abs(matriz_actual - matriz_teorica)
        print(f"Diferencia máxima: {np.max(diff)}")

if __name__ == "__main__":
    verify_golden_gate()
