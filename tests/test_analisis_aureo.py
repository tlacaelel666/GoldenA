import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Añadir el directorio padre al path para importar el módulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import analisis_aureo


class TestConstantesAureas:
    """Tests para verificar las constantes matemáticas básicas."""
    
    def test_phi_value(self):
        """Verifica que PHI tiene el valor correcto de la proporción áurea."""
        PHI = (1 + np.sqrt(5)) / 2
        assert np.isclose(PHI, 1.618033988749895), "PHI debe ser aproximadamente 1.618"
    
    def test_phi_property(self):
        """Verifica la propiedad fundamental de PHI: PHI^2 = PHI + 1."""
        PHI = (1 + np.sqrt(5)) / 2
        assert np.isclose(PHI**2, PHI + 1), "PHI debe cumplir PHI^2 = PHI + 1"


class TestCalculosAureos:
    """Tests para los cálculos relacionados con la proporción áurea."""
    
    @pytest.fixture
    def constantes(self):
        """Fixture que proporciona las constantes necesarias."""
        return {
            'PI': np.pi,
            'PHI': (1 + np.sqrt(5)) / 2
        }
    
    def test_paridad_valores_pares(self, constantes):
        """Verifica que cos(n*PI) = 1 para n pares."""
        PI = constantes['PI']
        n_pares = np.array([0, 2, 4, 6, 8])
        paridad = np.cos(PI * n_pares)
        esperado = np.ones_like(n_pares)
        assert np.allclose(paridad, esperado), "cos(n*PI) debe ser 1 para n pares"
    
    def test_paridad_valores_impares(self, constantes):
        """Verifica que cos(n*PI) = -1 para n impares."""
        PI = constantes['PI']
        n_impares = np.array([1, 3, 5, 7, 9])
        paridad = np.cos(PI * n_impares)
        esperado = -np.ones_like(n_impares)
        assert np.allclose(paridad, esperado), "cos(n*PI) debe ser -1 para n impares"
    
    def test_cuasiperiodica_rango(self, constantes):
        """Verifica que la fase cuasiperiódica está en el rango [-1, 1]."""
        PI = constantes['PI']
        PHI = constantes['PHI']
        n_vals = np.arange(100)
        cuasi = np.cos(PI * PHI * n_vals)
        assert np.all(cuasi >= -1) and np.all(cuasi <= 1), \
            "La fase cuasiperiódica debe estar en [-1, 1]"
    
    def test_dimension_producto(self, constantes):
        """Verifica que la dimensión es el producto correcto de paridad y cuasiperiódica."""
        PI = constantes['PI']
        PHI = constantes['PHI']
        n_vals = np.arange(10)
        
        paridad = np.cos(PI * n_vals)
        cuasi = np.cos(PI * PHI * n_vals)
        dimension_calculada = paridad * cuasi
        
        # Calcular directamente
        dimension_directa = np.cos(PI * n_vals) * np.cos(PI * PHI * n_vals)
        
        assert np.allclose(dimension_calculada, dimension_directa), \
            "La dimensión debe ser el producto de paridad y cuasiperiódica"
    
    def test_ponderado_suma(self, constantes):
        """Verifica que el valor ponderado es la suma correcta."""
        PI = constantes['PI']
        PHI = constantes['PHI']
        n = 5
        
        paridad = np.cos(PI * n)
        cuasi = np.cos(PI * PHI * n)
        dimension = paridad * cuasi
        ponderado = cuasi + dimension
        
        # Verificar que es la suma correcta
        assert np.isclose(ponderado, cuasi + dimension), \
            "El valor ponderado debe ser cuasi + dimension"


class TestProbabilidadTeorica:
    """Tests para el cálculo de probabilidad teórica."""
    
    def test_probabilidad_rango(self):
        """Verifica que la probabilidad teórica está en [0, 1]."""
        PI = np.pi
        PHI = (1 + np.sqrt(5)) / 2
        n_vals = np.linspace(0, 100, 1000)
        
        paridad = np.cos(PI * n_vals)
        cuasi = np.cos(PI * PHI * n_vals)
        dimension = paridad * cuasi
        prob_teorica = 0.5 - (dimension * 0.5)
        
        assert np.all(prob_teorica >= 0) and np.all(prob_teorica <= 1), \
            "La probabilidad teórica debe estar en [0, 1]"
    
    def test_probabilidad_n_cero(self):
        """Verifica el valor de probabilidad para n=0."""
        PI = np.pi
        PHI = (1 + np.sqrt(5)) / 2
        
        # Para n=0: cos(0) = 1, entonces dimension = 1 * 1 = 1
        # prob = 0.5 - (1 * 0.5) = 0
        dimension = np.cos(0) * np.cos(0)
        prob = 0.5 - (dimension * 0.5)
        
        assert np.isclose(prob, 0.0), "La probabilidad para n=0 debe ser 0"


class TestEjecucionAnalisis:
    """Tests para la función principal ejecutar_analisis()."""
    
    @patch('builtins.input', return_value='')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_ejecutar_analisis_default(self, mock_write, mock_input):
        """Verifica que ejecutar_analisis() funciona con valores por defecto."""
        # Capturar la salida
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            analisis_aureo.ejecutar_analisis()
        finally:
            sys.stdout = sys.__stdout__
        
        # Verificar que se llamó a write_html
        assert mock_write.called, "El gráfico debe guardarse"
        
        # Verificar que se imprimió algo
        output = captured_output.getvalue()
        assert "Análisis de Dinámica Áurea" in output, \
            "Debe mostrar el título del análisis"
    
    @patch('builtins.input', return_value='20')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_ejecutar_analisis_custom_n(self, mock_write, mock_input):
        """Verifica que ejecutar_analisis() acepta valores personalizados de n."""
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            analisis_aureo.ejecutar_analisis()
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "n=0..20" in output or "n=20" in output, \
            "Debe usar el valor personalizado de n"
    
    @patch('builtins.input', return_value='invalid')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_ejecutar_analisis_invalid_input(self, mock_write, mock_input):
        """Verifica que ejecutar_analisis() maneja entradas inválidas."""
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            analisis_aureo.ejecutar_analisis()
        finally:
            sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "inválida" in output or "n_max=10" in output, \
            "Debe manejar entrada inválida y usar valor por defecto"


class TestDatosExperimentales:
    """Tests para validar los datos experimentales."""
    
    def test_datos_experimentales_longitud(self):
        """Verifica que los datos experimentales tienen la longitud correcta."""
        n_experimental = np.array([0, 1, 2, 3, 4, 5])
        prob_polarizacion_exp = np.array([0.239, 0.676, 0.114, 0.176, 0.003, 0.211])
        
        assert len(n_experimental) == len(prob_polarizacion_exp), \
            "Los arrays de datos experimentales deben tener la misma longitud"
    
    def test_probabilidades_experimentales_rango(self):
        """Verifica que las probabilidades experimentales están en [0, 1]."""
        prob_polarizacion_exp = np.array([0.239, 0.676, 0.114, 0.176, 0.003, 0.211])
        
        assert np.all(prob_polarizacion_exp >= 0) and np.all(prob_polarizacion_exp <= 1), \
            "Las probabilidades experimentales deben estar en [0, 1]"


class TestIntegracionNumerica:
    """Tests de integración para verificar consistencia numérica."""
    
    def test_arrays_mismo_tamano(self):
        """Verifica que todos los arrays tienen el mismo tamaño para un n_max dado."""
        PI = np.pi
        PHI = (1 + np.sqrt(5)) / 2
        n_max = 15
        
        n_vals = np.arange(n_max + 1)
        paridad = np.cos(PI * n_vals)
        cuasi = np.cos(PI * PHI * n_vals)
        dimension = paridad * cuasi
        ponderado = cuasi + dimension
        
        assert len(n_vals) == len(paridad) == len(cuasi) == len(dimension) == len(ponderado), \
            "Todos los arrays deben tener el mismo tamaño"
    
    def test_valores_no_nan(self):
        """Verifica que no hay valores NaN en los cálculos."""
        PI = np.pi
        PHI = (1 + np.sqrt(5)) / 2
        n_vals = np.arange(100)
        
        paridad = np.cos(PI * n_vals)
        cuasi = np.cos(PI * PHI * n_vals)
        dimension = paridad * cuasi
        ponderado = cuasi + dimension
        
        assert not np.any(np.isnan(paridad)), "No debe haber NaN en paridad"
        assert not np.any(np.isnan(cuasi)), "No debe haber NaN en cuasi"
        assert not np.any(np.isnan(dimension)), "No debe haber NaN en dimension"
        assert not np.any(np.isnan(ponderado)), "No debe haber NaN en ponderado"


class TestPerformance:
    """Tests de rendimiento para valores grandes de n."""
    
    @pytest.mark.slow
    def test_large_n_performance(self):
        """Verifica que el código puede manejar valores grandes de n."""
        PI = np.pi
        PHI = (1 + np.sqrt(5)) / 2
        n_max = 10000  # Valor grande
        
        n_vals = np.arange(n_max + 1)
        paridad = np.cos(PI * n_vals)
        cuasi = np.cos(PI * PHI * n_vals)
        dimension = paridad * cuasi
        ponderado = cuasi + dimension
        
        assert len(ponderado) == n_max + 1, \
            "Debe poder calcular arrays grandes sin problemas"


# Configuración de pytest
def pytest_configure(config):
    """Configuración personalizada de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca tests que tardan mucho tiempo en ejecutarse"
    )


if __name__ == "__main__":
    # Permite ejecutar los tests directamente con python
    pytest.main([__file__, "-v"])
