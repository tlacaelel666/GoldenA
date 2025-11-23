# Tests para GoldenA

Este directorio contiene los tests automatizados para el proyecto GoldenA.

## 📋 Requisitos

Los tests requieren las siguientes dependencias (ya incluidas en `requirements.txt`):

- pytest >= 7.0.0
- numpy >= 1.23.0
- plotly >= 5.0.0

## 🚀 Ejecución de Tests

### Ejecutar todos los tests

```bash
pytest
```

### Ejecutar tests con más detalle

```bash
pytest -v
```

### Ejecutar un archivo específico

```bash
pytest tests/test_analisis_aureo.py
```

### Ejecutar una clase específica de tests

```bash
pytest tests/test_analisis_aureo.py::TestConstantesAureas
```

### Ejecutar un test específico

```bash
pytest tests/test_analisis_aureo.py::TestConstantesAureas::test_phi_value
```

### Ejecutar tests con cobertura (si tienes pytest-cov instalado)

```bash
pytest --cov=. --cov-report=html
```

## 🏷️ Markers (Etiquetas)

Los tests están organizados con los siguientes markers:

- `@pytest.mark.slow`: Tests que tardan más tiempo
- `@pytest.mark.unit`: Tests unitarios
- `@pytest.mark.integration`: Tests de integración

### Ejecutar solo tests rápidos (excluir tests lentos)

```bash
pytest -m "not slow"
```

### Ejecutar solo tests lentos

```bash
pytest -m slow
```

## 📝 Estructura de Tests

### `test_analisis_aureo.py`

Tests para el módulo `analisis_aureo.py`:

#### `TestConstantesAureas`

- ✓ Verificación del valor de PHI (proporción áurea)
- ✓ Verificación de la propiedad fundamental: PHI² = PHI + 1

#### `TestCalculosAureos`

- ✓ Paridad para valores pares (cos(n*π) = 1)
- ✓ Paridad para valores impares (cos(n*π) = -1)
- ✓ Rango de la fase cuasiperiódica [-1, 1]
- ✓ Cálculo de dimensión (producto de paridad y cuasiperiódica)
- ✓ Cálculo del valor ponderado

#### `TestProbabilidadTeorica`

- ✓ Rango de probabilidad teórica [0, 1]
- ✓ Valor de probabilidad para n=0

#### `TestEjecucionAnalisis`

- ✓ Ejecución con valores por defecto
- ✓ Ejecución con valores personalizados de n
- ✓ Manejo de entradas inválidas

#### `TestDatosExperimentales`

- ✓ Validación de longitud de datos experimentales
- ✓ Verificación de rango de probabilidades [0, 1]

#### `TestIntegracionNumerica`

- ✓ Consistencia de tamaños de arrays
- ✓ Ausencia de valores NaN

#### `TestPerformance`

- ✓ Rendimiento con valores grandes de n (n=10,000)

## 📊 Resultados

Estado actual: **17 tests pasando** ✅

## 🛠️ Agregar Nuevos Tests

Para agregar nuevos tests:

1. Crea una nueva clase de test heredando de una clase base (opcional)
2. Nombra los métodos comenzando con `test_`
3. Usa `assert` para las verificaciones
4. Añade docstrings descriptivos

Ejemplo:

```python
class TestNuevaFuncionalidad:
    """Tests para nueva funcionalidad."""
    
    def test_algo_especifico(self):
        """Verifica comportamiento específico."""
        resultado = mi_funcion()
        assert resultado == esperado, "Mensaje de error"
```

## 📚 Recursos

- [Documentación de pytest](https://docs.pytest.org/)
- [Mejores prácticas de testing](https://docs.pytest.org/en/stable/goodpractices.html)
- [Fixtures en pytest](https://docs.pytest.org/en/stable/fixture.html)

## 🐛 Debugging

Para ejecutar tests con más información de debug:

```bash
pytest -vv --tb=long
```

Para ejecutar con pdb (debugger interactivo) al fallar:

```bash
pytest --pdb
```

Para ver print statements durante la ejecución:

```bash
pytest -s
```
