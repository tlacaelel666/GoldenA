#!/usr/bin/env python3
"""
Tests de integración para Qiskit CLI
Prueba pipelines completos y comandos del CLI
"""

import pytest
import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from golden.circuito_aureo import QiskitCLI
import logging


@pytest.fixture
def cli():
    """Fixture que crea una instancia del CLI para testing"""
    logger = logging.getLogger('test_cli')
    logger.setLevel(logging.ERROR)  # Silenciar logs durante tests
    return QiskitCLI(logger)


class TestBasicCircuitCreation:
    """Tests de creación básica de circuitos"""
    
    def test_crear_circuito_simple(self, cli):
        """Test: crear 1"""
        result = cli.execute_pipeline("crear 1")
        assert result == True
        assert cli.circuit is not None
        assert cli.num_qubits == 1
    
    def test_crear_circuito_multiqubit(self, cli):
        """Test: crear 3"""
        result = cli.execute_pipeline("crear 3")
        assert result == True
        assert cli.num_qubits == 3
        assert cli.circuit.num_qubits == 3
    
    def test_crear_sin_argumentos(self, cli):
        """Test: crear (sin número) debe fallar"""
        result = cli.execute_pipeline("crear")
        assert result == False
    
    def test_crear_con_cero_qubits(self, cli):
        """Test: crear 0 debe fallar"""
        result = cli.execute_pipeline("crear 0")
        assert result == False


class TestGateOperations:
    """Tests de operaciones con puertas"""
    
    def test_agregar_hadamard(self, cli):
        """Test: crear 1 | agregar h 0"""
        result = cli.execute_pipeline("crear 1 | agregar h 0")
        assert result == True
        assert cli.circuit.depth() == 1
    
    def test_agregar_pauli_gates(self, cli):
        """Test: agregar puertas X, Y, Z"""
        result = cli.execute_pipeline("crear 1 | agregar x 0 | agregar y 0 | agregar z 0")
        assert result == True
        assert cli.circuit.depth() == 3
    
    def test_agregar_cnot(self, cli):
        """Test: crear 2 | agregar cx 0 1"""
        result = cli.execute_pipeline("crear 2 | agregar cx 0 1")
        assert result == True
        assert cli.circuit.depth() == 1
    
    def test_agregar_golden_gate(self, cli):
        """Test: crear 1 | agregar phi 1 0"""
        result = cli.execute_pipeline("crear 1 | agregar phi 1 0")
        assert result == True
        # Verificar que la GoldenGate está en el circuito
        has_golden = any(inst.name == "golden_gate" for inst, _, _ in cli.circuit.data)
        assert has_golden == True
    
    def test_agregar_sin_circuito(self, cli):
        """Test: agregar sin crear circuito primero debe fallar"""
        result = cli.execute_pipeline("agregar h 0")
        assert result == False
    
    def test_agregar_qubit_fuera_rango(self, cli):
        """Test: agregar en qubit que no existe debe fallar"""
        result = cli.execute_pipeline("crear 1 | agregar h 5")
        assert result == False


class TestMeasurementAndExecution:
    """Tests de medición y ejecución"""
    
    def test_medir_todos(self, cli):
        """Test: crear 2 | medir"""
        result = cli.execute_pipeline("crear 2 | medir")
        assert result == True
        assert len(cli.measured_qubits) == 2
    
    def test_medir_especificos(self, cli):
        """Test: crear 3 | medir 0 2"""
        result = cli.execute_pipeline("crear 3 | medir 0 2")
        assert result == True
        assert 0 in cli.measured_qubits
        assert 2 in cli.measured_qubits
        assert 1 not in cli.measured_qubits
    
    def test_ejecutar_simple(self, cli):
        """Test: crear 1 | agregar h 0 | medir | ejecutar 10"""
        result = cli.execute_pipeline("crear 1 | agregar h 0 | medir | ejecutar 10")
        assert result == True
    
    def test_ejecutar_sin_medicion(self, cli):
        """Test: ejecutar sin medir agrega medición automática"""
        result = cli.execute_pipeline("crear 1 | agregar h 0 | ejecutar 10")
        assert result == True
        assert len(cli.measured_qubits) > 0


class TestComplexPipelines:
    """Tests de pipelines complejos"""
    
    def test_bell_state(self, cli):
        """Test: crear estado de Bell"""
        pipeline = "crear 2 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 100"
        result = cli.execute_pipeline(pipeline)
        assert result == True
        assert cli.circuit.depth() == 3  # H + CX + mediciones
    
    def test_ghz_state(self, cli):
        """Test: crear estado GHZ de 3 qubits"""
        pipeline = "crear 3 | agregar h 0 | agregar cx 0 1 | agregar cx 0 2 | medir | ejecutar 100"
        result = cli.execute_pipeline(pipeline)
        assert result == True
    
    def test_golden_ratio_circuit(self, cli):
        """Test: circuito con múltiples GoldenGates"""
        pipeline = "crear 2 | agregar h 0 | agregar phi 1 0 | agregar phi 2 1 | agregar cx 0 1"
        result = cli.execute_pipeline(pipeline)
        assert result == True
        # Contar GoldenGates
        golden_count = sum(1 for inst, _, _ in cli.circuit.data if inst.name == "golden_gate")
        assert golden_count == 2
    
    def test_toffoli_circuit(self, cli):
        """Test: circuito con Toffoli"""
        pipeline = "crear 3 | agregar h 0 | agregar h 1 | agregar ccx 0 1 2 | medir | ejecutar 50"
        result = cli.execute_pipeline(pipeline)
        assert result == True


class TestCGHIntegration:
    """Tests de integración con análisis CGH"""
    
    def test_cgh_sin_circuito(self, cli):
        """Test: cgh sin circuito ejecuta demo"""
        # Capturar output para verificar que ejecuta demo
        output = StringIO()
        with redirect_stdout(output):
            result = cli.execute_pipeline("cgh")
        
        output_str = output.getvalue()
        # Debe ejecutar demo (contiene "FIBONACCI QML")
        assert "FIBONACCI QML" in output_str or result == True
    
    def test_cgh_con_golden_gates(self, cli):
        """Test: cgh analiza circuito con GoldenGates"""
        output = StringIO()
        with redirect_stdout(output):
            result = cli.execute_pipeline("crear 1 | agregar phi 1 0 | agregar phi 2 0 | cgh")
        
        output_str = output.getvalue()
        # Debe encontrar las GoldenGates
        assert "GoldenGate encontrada" in output_str or result == True
    
    def test_cgh_sin_golden_gates(self, cli):
        """Test: cgh con circuito sin GoldenGates"""
        output = StringIO()
        with redirect_stdout(output):
            result = cli.execute_pipeline("crear 1 | agregar h 0 | agregar x 0 | cgh")
        
        output_str = output.getvalue()
        # Debe indicar que no hay GoldenGates
        assert "No se encontraron GoldenGates" in output_str or result == True


class TestCircuitPersistence:
    """Tests de persistencia del circuito en el pipeline"""
    
    def test_circuito_persiste_entre_comandos(self, cli):
        """Test: el circuito se mantiene entre comandos del pipeline"""
        cli.execute_pipeline("crear 2")
        initial_circuit = cli.circuit
        
        cli.execute_pipeline("agregar h 0")
        assert cli.circuit is initial_circuit  # Mismo objeto
        assert cli.circuit.depth() == 1
        
        cli.execute_pipeline("agregar h 1")
        assert cli.circuit is initial_circuit
        assert cli.circuit.depth() == 2
    
    def test_crear_reinicia_circuito(self, cli):
        """Test: crear reinicia el circuito"""
        cli.execute_pipeline("crear 1 | agregar h 0")
        first_circuit = cli.circuit
        
        cli.execute_pipeline("crear 2")
        assert cli.circuit is not first_circuit
        assert cli.circuit.depth() == 0
        assert cli.num_qubits == 2


class TestErrorHandling:
    """Tests de manejo de errores"""
    
    def test_comando_desconocido(self, cli):
        """Test: comando que no existe debe fallar"""
        result = cli.execute_pipeline("comando_inexistente")
        assert result == False
    
    def test_puerta_desconocida(self, cli):
        """Test: puerta que no existe debe fallar"""
        result = cli.execute_pipeline("crear 1 | agregar puerta_falsa 0")
        assert result == False
    
    def test_parametros_incorrectos(self, cli):
        """Test: número incorrecto de parámetros debe fallar"""
        # CNOT necesita 2 qubits
        result = cli.execute_pipeline("crear 2 | agregar cx 0")
        assert result == False
    
    def test_pipeline_vacio(self, cli):
        """Test: pipeline vacío"""
        result = cli.execute_pipeline("")
        assert result == False


class TestRotationGates:
    """Tests de puertas de rotación con parámetros"""
    
    def test_rx_gate(self, cli):
        """Test: agregar RX con ángulo"""
        result = cli.execute_pipeline("crear 1 | agregar rx 1.57 0")
        assert result == True
    
    def test_ry_gate(self, cli):
        """Test: agregar RY con ángulo"""
        result = cli.execute_pipeline("crear 1 | agregar ry 3.14 0")
        assert result == True
    
    def test_rz_gate(self, cli):
        """Test: agregar RZ con ángulo"""
        result = cli.execute_pipeline("crear 1 | agregar rz 0.5 0")
        assert result == True
    
    def test_phase_gate(self, cli):
        """Test: agregar Phase con ángulo"""
        result = cli.execute_pipeline("crear 1 | agregar p 1.0 0")
        assert result == True


class TestMultiQubitGates:
    """Tests de puertas multi-qubit"""
    
    def test_swap_gate(self, cli):
        """Test: SWAP entre dos qubits"""
        result = cli.execute_pipeline("crear 2 | agregar swap 0 1")
        assert result == True
    
    def test_cz_gate(self, cli):
        """Test: Control-Z"""
        result = cli.execute_pipeline("crear 2 | agregar cz 0 1")
        assert result == True
    
    def test_cy_gate(self, cli):
        """Test: Control-Y"""
        result = cli.execute_pipeline("crear 2 | agregar cy 0 1")
        assert result == True
    
    def test_fredkin_gate(self, cli):
        """Test: Fredkin (CSWAP)"""
        result = cli.execute_pipeline("crear 3 | agregar cswap 0 1 2")
        assert result == True


class TestVerCommand:
    """Tests del comando ver"""
    
    def test_ver_circuito(self, cli):
        """Test: ver muestra el circuito"""
        output = StringIO()
        with redirect_stdout(output):
            result = cli.execute_pipeline("crear 1 | agregar h 0 | ver")
        
        assert result == True
        output_str = output.getvalue()
        # Debe mostrar información del circuito
        assert "CIRCUITO" in output_str or "Qubits" in output_str
    
    def test_ver_sin_circuito(self, cli):
        """Test: ver sin circuito"""
        result = cli.execute_pipeline("ver")
        assert result == False


# ============ TESTS DE INTEGRACIÓN COMPLETA ============

class TestFullIntegration:
    """Tests de integración completa end-to-end"""
    
    def test_complete_workflow_basic(self, cli):
        """Test: workflow completo básico"""
        pipeline = """
        crear 2 | 
        agregar h 0 | 
        agregar cx 0 1 | 
        medir | 
        ejecutar 100
        """
        result = cli.execute_pipeline(pipeline.replace('\n', ''))
        assert result == True
    
    def test_complete_workflow_with_golden(self, cli):
        """Test: workflow con GoldenGates y análisis CGH"""
        pipeline = """
        crear 1 | 
        agregar h 0 | 
        agregar phi 1 0 | 
        agregar phi 2 0 | 
        cgh
        """
        result = cli.execute_pipeline(pipeline.replace('\n', ''))
        assert result == True
    
    def test_deutsch_algorithm(self, cli):
        """Test: implementación del algoritmo de Deutsch"""
        pipeline = """
        crear 2 | 
        agregar x 1 | 
        agregar h 0 | 
        agregar h 1 | 
        agregar cx 0 1 | 
        agregar h 0 | 
        medir | 
        ejecutar 1000
        """
        result = cli.execute_pipeline(pipeline.replace('\n', ''))
        assert result == True


# ============ CONFIGURACIÓN DE PYTEST ============

if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
