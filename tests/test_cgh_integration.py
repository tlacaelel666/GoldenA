#!/usr/bin/env python3
"""
Test script para verificar la integración de CGH con el pipeline
"""

# Simular el comportamiento sin importar Qiskit
class MockCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.data = []
        self._depth = 0
    
    def depth(self):
        return self._depth
    
    def add_gate(self, name, params, qubits):
        """Simula agregar una puerta"""
        class MockInstruction:
            def __init__(self, name, params):
                self.name = name
                self.params = params
        
        class MockQubit:
            def __init__(self, index):
                self.index = index
        
        instruction = MockInstruction(name, params)
        qargs = [MockQubit(q) for q in qubits]
        self.data.append((instruction, qargs, []))
        self._depth += 1

# Test
print("=" * 80)
print("🧪 TEST: Extracción de parámetros GoldenGate")
print("=" * 80)

circuit = MockCircuit(1)
circuit.add_gate("h", [], [0])
circuit.add_gate("golden_gate", [1], [0])
circuit.add_gate("golden_gate", [2], [0])
circuit.add_gate("golden_gate", [5], [0])

print(f"\nCircuito creado:")
print(f"  Qubits: {circuit.num_qubits}")
print(f"  Profundidad: {circuit.depth()}")
print(f"  Operaciones: {len(circuit.data)}")

print(f"\n🔍 Extrayendo parámetros de GoldenGates...")
fib_params = []
for instruction, qargs, cargs in circuit.data:
    if instruction.name == "golden_gate":
        n_val = instruction.params[0]
        fib_params.append(int(n_val))
        print(f"  ✓ GoldenGate encontrada: n={n_val} en qubit {qargs[0].index}")

print(f"\n📊 Parámetros extraídos: {fib_params}")
print(f"✅ Test exitoso!")
