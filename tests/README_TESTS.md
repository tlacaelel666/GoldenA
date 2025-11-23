# Tests de Integración del CLI

Este archivo contiene tests comprehensivos para el Qiskit CLI, cubriendo todas las funcionalidades principales.

## 📋 Estructura de Tests

### 1. **TestBasicCircuitCreation**

Tests de creación básica de circuitos:

- ✅ Crear circuito de 1 qubit
- ✅ Crear circuito multi-qubit
- ✅ Validación de parámetros incorrectos

### 2. **TestGateOperations**

Tests de operaciones con puertas:

- ✅ Puertas de 1 qubit (H, X, Y, Z)
- ✅ Puertas de 2 qubits (CNOT, CZ, CY, SWAP)
- ✅ GoldenGate (phi)
- ✅ Validación de qubits fuera de rango

### 3. **TestMeasurementAndExecution**

Tests de medición y ejecución:

- ✅ Medir todos los qubits
- ✅ Medir qubits específicos
- ✅ Ejecución con/sin medición previa

### 4. **TestComplexPipelines**

Tests de pipelines complejos:

- ✅ Estado de Bell
- ✅ Estado GHZ
- ✅ Circuitos con múltiples GoldenGates
- ✅ Circuito Toffoli

### 5. **TestCGHIntegration**

Tests de integración CGH:

- ✅ CGH sin circuito (ejecuta demo)
- ✅ CGH con GoldenGates (analiza circuito)
- ✅ CGH sin GoldenGates (mensaje informativo)

### 6. **TestCircuitPersistence**

Tests de persistencia del circuito:

- ✅ Circuito persiste entre comandos
- ✅ Crear reinicia el circuito

### 7. **TestErrorHandling**

Tests de manejo de errores:

- ✅ Comandos desconocidos
- ✅ Puertas desconocidas
- ✅ Parámetros incorrectos

### 8. **TestRotationGates**

Tests de puertas de rotación:

- ✅ RX, RY, RZ
- ✅ Phase gate

### 9. **TestMultiQubitGates**

Tests de puertas multi-qubit:

- ✅ SWAP, CZ, CY
- ✅ Fredkin (CSWAP)

### 10. **TestVerCommand**

Tests del comando ver:

- ✅ Ver circuito existente
- ✅ Ver sin circuito

### 11. **TestFullIntegration**

Tests end-to-end:

- ✅ Workflow completo básico
- ✅ Workflow con GoldenGates y CGH
- ✅ Algoritmo de Deutsch

## 🚀 Ejecutar Tests

### Todos los tests

```bash
pytest tests/test_cli_integration.py -v
```

### Test específico

```bash
pytest tests/test_cli_integration.py::TestCGHIntegration::test_cgh_con_golden_gates -v
```

### Con coverage

```bash
pytest tests/test_cli_integration.py --cov=golden.circuito_aureo --cov-report=html
```

### Solo tests rápidos (sin CGH)

```bash
pytest tests/test_cli_integration.py -v -m "not slow"
```

## 📊 Cobertura Esperada

Los tests cubren:

- ✅ Creación de circuitos (100%)
- ✅ Operaciones con puertas (95%)
- ✅ Medición y ejecución (100%)
- ✅ Integración CGH (90%)
- ✅ Manejo de errores (85%)
- ✅ Pipelines complejos (100%)

## ⚠️ Notas Importantes

### Entorno de Qiskit

Si encuentras el error:

```
ImportError: Qiskit is installed in an invalid environment
```

**Solución:**

```bash
# Crear nuevo entorno virtual
python3 -m venv venv_qiskit
source venv_qiskit/bin/activate

# Instalar solo Qiskit >=1.0
pip install qiskit qiskit-aer
pip install pytest

# Ejecutar tests
pytest tests/test_cli_integration.py -v
```

### Tests que Requieren Visualización

Algunos tests generan archivos HTML (CGH). Estos se guardan en:

- `cgh_holographic.html` (análisis CGH)
- `~/.qiskit_cli/histogram_*.png` (histogramas)

### Mocking para Tests Rápidos

Para tests más rápidos sin ejecutar circuitos reales, puedes mockear el simulador:

```python
@pytest.fixture
def cli_with_mock(cli, monkeypatch):
    def mock_run(*args, **kwargs):
        class MockResult:
            def get_counts(self, *args):
                return {'0': 50, '1': 50}
        class MockJob:
            def result(self):
                return MockResult()
        return MockJob()
    
    monkeypatch.setattr(cli.local_simulator, 'run', mock_run)
    return cli
```

## 🐛 Debugging

### Ver output completo

```bash
pytest tests/test_cli_integration.py -v -s
```

### Detener en primer error

```bash
pytest tests/test_cli_integration.py -x
```

### Ver traceback completo

```bash
pytest tests/test_cli_integration.py --tb=long
```

## 📝 Ejemplos de Uso

### Test Individual

```python
def test_mi_pipeline(cli):
    """Test personalizado"""
    pipeline = "crear 2 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 100"
    result = cli.execute_pipeline(pipeline)
    assert result == True
    assert cli.circuit.depth() == 3
```

### Test con Verificación de Output

```python
def test_con_output(cli):
    """Test que verifica el output"""
    from io import StringIO
    from contextlib import redirect_stdout
    
    output = StringIO()
    with redirect_stdout(output):
        cli.execute_pipeline("crear 1 | ver")
    
    assert "Qubits: 1" in output.getvalue()
```

## ✅ Checklist de Tests

Antes de hacer commit, verifica:

- [ ] Todos los tests pasan
- [ ] No hay warnings
- [ ] Coverage > 80%
- [ ] Tests documentados
- [ ] No hay código comentado
- [ ] Imports organizados

## 🔄 CI/CD

Los tests se ejecutan automáticamente en:

- Push a main
- Pull requests
- Nightly builds

## 📚 Referencias

- [Pytest Documentation](https://docs.pytest.org/)
- [Qiskit Testing Guide](https://qiskit.org/documentation/contributing_to_qiskit.html#testing)
- [Python unittest](https://docs.python.org/3/library/unittest.html)
