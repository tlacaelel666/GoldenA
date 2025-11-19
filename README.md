<div>
  
  <img aling="center" width="902" height="296" alt="1763533051325" src="https://github.com/user-attachments/assets/02e7067b-ff9e-47a1-9178-c10c40e78c96" />

</div>

----

# 🚀 Qiskit Runtime CLI v3.2 - One-Liner Pipeline 


![Quantum Badge](https://img.shields.io/badge/quantum-system-blue)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Qis-kit CLI](https://img.shields.io/badge/Qiskit-runtime-yellow)
![SmokApp Q](https://img.shields.io/badge/GoldenA-v3.2-black)


Una interfaz de línea de comandos interactiva para ejecutar circuitos cuánticos con Qiskit, diseñada con un enfoque en la **dinámica áurea** (golden ratio) y análisis experimental de polarización cuántica.

## 📋 Características Principales

- **Pipeline One-Liner**: Encadena múltiples comandos con `|` para construir y ejecutar circuitos en una sola línea
- **Soporte IBM Quantum**: Ejecuta en simuladores locales o en hardware real de IBM
- **Puerta Cuántica Áurea**: Implementación de `GoldenGate` que aplica fases basadas en la razón áurea (φ)
- **Visualización 3D Interactiva**: Análisis de la dinámica áurea con gráficos Plotly
- **Comparación Teoría vs Experimental**: Validación de predicciones con resultados de Qiskit

---

## 🎯 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip

### Pasos

1. **Clona o descarga el repositorio**
   ```bash
   git clone https://github.com/tlacaelel666/GoldenA.git
   cd GoldenA
   ```

2. **Instala las dependencias**
   ```bash
   python3 -m venv env
   pip install -r requirements.txt
   ```

3. **Ejecuta la CLI**
   ```bash
   python main.py
   ```

---

---

## 📖 Guía de Uso

### Modo Interactivo

Ejecuta `python main.py` para entrar en el modo interactivo:

```
qiskit (🖥️  Simulador)> 
```

### Sintaxis One-Liner

Encadena comandos separados por `|`:

```bash
crear 3 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 1024
```

### Comandos Disponibles

#### Construcción del Circuito

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `crear <n>` | Crea un circuito de n qubits | `crear 2` |
| `agregar <puerta> <args>` | Añade una puerta al circuito | `agregar h 0` |
| `medir [qubits]` | Mide qubits específicos o todos | `medir 0 1` |
| `ejecutar [shots]` | Ejecuta el circuito (default: 1024) | `ejecutar 2048` |

#### Visualización

| Comando | Descripción |
|---------|-------------|
| `ver` | Muestra el circuito actual en formato texto |
| `puertas` | Lista todas las puertas disponibles |
| **`analisis`** | **Visualización 3D de la dinámica áurea** |
| `demo` | Muestra ejemplos de circuitos |

#### Gestión de Sesión IBM

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `login <token>` | Conecta con IBM Quantum | `login <TOKEN>` |
| `backends` | Lista los backends disponibles | `backends` |
| `backend <nombre>` | Selecciona un backend de hardware | `backend ibm_sherbrooke` |
| `simulator` | Cambia al simulador local | `simulator` |
| `status` | Muestra estado del backend actual | `status` |

#### Ayuda

| Comando | Descripción |
|---------|-------------|
| `ayuda` | Muestra esta referencia |
| `salir` | Cierra la aplicación |

---

## 🎨 Puertas Disponibles

### Puertas de 1 Qubit (sin parámetros)
- `h`: Hadamard (superposición)
- `x`, `y`, `z`: Puertas de Pauli
- `s`, `sdg`: Fase ±π/2
- `t`, `tdg`: Fase ±π/4
- `i`: Identidad

### Puertas de 1 Qubit (con ángulo)
- `rx <θ>`, `ry <θ>`, `rz <θ>`: Rotaciones
- `p <λ>`: Cambio de fase
- **`phi <n>`**: Puerta áurea personalizada sustituye <n> por cualquier numero entero

### Puertas de 2 Qubits
- `cx`: CNOT (Control-NOT)
- `cy`, `cz`: Control-Y, Control-Z
- `swap`: Intercambiar qubits
- `crx <θ>`, `cry <θ>`, `crz <θ>`: Control-Rotaciones
- `cp <λ>`: Control-Phase

### Puertas de 3 Qubits
- `ccx`: Toffoli (Control-Control-NOT)
- `cswap`: Fredkin (SWAP controlado)

---

## 🌟 El Comando `analisis`: Visualización 3D de Dinámica Áurea

Este es el corazón del proyecto. El comando `analisis` genera una visualización interactiva que explora la relación entre la **razón áurea (φ = 1.618...)** y los fenómenos cuánticos.

### ¿Qué Hace?

El análisis crea dos visualizaciones integradas:

#### 1. **Gráfico 3D: Dinámica Áurea**
```
Eje X: Parámetro n (valores discretos)
Eje Y: Fase Cuasiperiódica [cos(π·φ·n)]
Eje Z: Valor Ponderado [cos(π·φ·n) + dimensión]
```

**Componentes matemáticos:**
- **Paridad**: `cos(π·n)` — Alterna entre 1 (n par) y -1 (n impar)
- **Fase Cuasiperiódica**: `cos(π·φ·n)` — Distribución no periódica basada en φ
- **Dimensión**: `paridad × fase` — Producto que modula el comportamiento
- **Valor Ponderado**: `fase + dimensión` — Síntesis de ambos efectos

Los puntos se colorean con la escala **Viridis** según el valor ponderado, mostrando visualmente cómo la dinámica evoluciona.

#### 2. **Gráfico 2D: Validación Experimental**
```
Eje X: Parámetro n
Eje Y: Probabilidad P(|01⟩)
```

**Capas:**
- 🟠 **Línea naranja**: Predicción teórica basada en la fórmula áurea
- 🔵 **Puntos azules**: Resultados experimentales reales de Qiskit

La comparación muestra cómo la teoría basada en φ predice la polarización de estados cuánticos entrelazados.

### Fórmula Teórica

La probabilidad de medir el estado `|01⟩` se modela como:

$$P_n = 0.5 - 0.5 \cdot \cos(π \cdot \phi \cdot n) \cdot \cos(π \cdot n)$$

Donde:
- φ ≈ 1.618 (razón áurea)
- n es el número de la ejecución o iteración

### Cómo Usar

**En modo interactivo:**
```bash
qiskit (🖥️  Simulador)> analisis
```

**O en pipeline:**
```bash
crear 1 | agregar h 0 | medir | ejecutar 1000 
```

### Salida

1. **Tabla en consola** con valores numéricos para n=0 hasta n_max
2. **Archivo HTML interactivo** (`analisis_aureo.html`) que contiene:
   - Gráfico 3D rotable/zoomeable
   - Gráfico 2D con leyenda
   - Interpretación de cada componente

### Ejemplo de Tabla Generada

```
n | Paridad | Fase cuasiperiódica | Dimension | Valor Ponderado
0 |  1.0000 |             1.0000 |    1.0000 |           2.0000
1 | -1.0000 |            -0.3090 |    0.3090 |          -0.0000
2 |  1.0000 |            -0.8090 |   -0.8090 |          -1.6180
3 | -1.0000 |             0.3090 |   -0.3090 |          -0.0000
...
```

### Interpretación de Resultados

- **Valores cercanos a ±2**: Máxima amplificación o cancelación de la dinámica
- **Valores cercanos a 0**: Balance perfecto entre paridad y fase cuasiperiódica
- **Patrón no periódico**: La presencia de φ genera una distribución que nunca se repite exactamente

---

## 📊 Ejemplos Prácticos

### Ejemplo 1: Superposición Simple
```bash
crear 1 | agregar h 0 | medir | ejecutar 1000
```
Crea un qubit en superposición y lo mide 1000 veces. Espera ~50% |0⟩ y ~50% |1⟩.

### Ejemplo 2: Entrelazamiento Bell
```bash
crear 2 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 1000
```
Crea un par de Bell. Resultado: siempre |00⟩ o |11⟩ (nunca |01⟩ ni |10⟩).

### Ejemplo 3: Puerta Áurea
```bash
crear 1 | agregar h 0 | agregar phi 3 0 | medir | ejecutar 1000
```
Aplica la puerta áurea con n=3 a un qubit en superposición.

### Ejemplo 4: Análisis Completo
```bash
analisis
```
Ejecuta la visualización 3D de dinámica áurea con n_max=10 (interactivo).

### Ejemplo 5: Usar IBM Quantum
```bash
login sk_ibm_1234567890abcdef
backend ibm_sherbrooke
crear 2 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 100
```
Ejecuta en hardware real de IBM.

---

## 🔧 Estructura del Proyecto

```
.
├── main.py                 # Punto de entrada principal
├── circuito_aureo.py      # CLI interactiva con lógica de comandos
├── analisis_aureo.py      # Generador de visualización 3D ⭐
├── golden_gate.py         # Implementación de GoldenGate
├── requirements.txt       # Dependencias
└── README.md             # Este archivo
```

### Archivos Generados

- `analisis_aureo.html` — Gráfico 3D interactivo (se abre automáticamente)
- `~/.qiskit_cli/logs/` — Archivos de log
- `~/.qiskit_cli/histogram_*.png` — Histogramas de resultados

---

## 🎓 Conceptos Matemáticos

### Razón Áurea (φ)
$$\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618034...$$

Es un número fundamental que aparece en:
- Naturaleza: proporción de espirales de caracol, flores, galaxias
- Arte: rectangles perfectos
- **Física Cuántica**: Este proyecto explora su rol en distribuiciones no periódicas

### Fase Áurea
$$\lambda_n = \frac{\pi}{\phi} \approx 1.944 \text{ rad}$$

Utilizada en la puerta `GoldenGate` para aplicar cambios de fase específicos basados en φ.

### Cuasiperiodicidad
La función `cos(π·φ·n)` genera un patrón que **nunca se repite** exactamente porque φ es irracional. Esto es útil para sistemas dinámicos caóticos.

---

## ⚙️ Configuración Avanzada

### Variables de Entorno
```bash
QISKIT_LOG_LEVEL=DEBUG  # Aumentar verbosidad
```

### Personalizar Shots por Defecto
Edita `circuito_aureo.py` línea ~400 para cambiar shots predeterminados.

### Agregar Puertas Personalizadas
Modifica el diccionario `GATES_DB` en `circuito_aureo.py` para añadir nuevas puertas.

---

## 🐛 Resolución de Problemas

### Error: "IBM Quantum Runtime no instalado"
```bash
pip install qiskit-ibm-runtime
```

### Error al conectar con IBM
- Verifica tu token en https://quantum.ibm.com/account
- Asegúrate de tener conexión a internet

### El análisis no abre en navegador
- Verifica que `analisis_aureo.html` se creó en el directorio actual
- Abrelo manualmente en tu navegador

---

## 📚 Referencias

- [Documentación Qiskit](https://docs.quantum.ibm.com/)
- [IBM Quantum Platform](https://quantum.ibm.com/)
- [Plotly Graphing Libraries](https://plotly.com/python/)
- [Golden Ratio en Física](https://en.wikipedia.org/wiki/Golden_ratio)

---

## 📝 Licencia

Apache 2.0.

---

## 💡 Contribuciones

¿Ideas para mejorar el análisis áureo? ¡Abre un issue o haz un pull request!

---

**Última actualización**: 2025 | **Versión**: 3.2

