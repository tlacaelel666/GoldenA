# Visualización de Esferas de Bloch

## Descripción

El comando `bloch` permite visualizar el estado cuántico de qubits individuales en esferas de Bloch 3D. Esta herramienta es fundamental para entender conceptos cuánticos como superposición, fase y entrelazamiento.

## Sintaxis

```bash
bloch              # Visualiza todos los qubits (máximo 10)
bloch <q1>         # Visualiza solo el qubit q1
bloch <q1> <q2>... # Visualiza los qubits especificados
```

## Ejemplos

### Ejemplo 1: Qubit en Superposición

```bash
crear 1 | agregar h 0 | bloch
```

Visualiza un qubit en estado |+⟩ (superposición igual de |0⟩ y |1⟩).

**Resultado esperado:**

- Vector de Bloch: (1, 0, 0) - apuntando en dirección +X
- Magnitud: 1.0 (estado puro)

### Ejemplo 2: Múltiples Qubits

```bash
crear 3 | agregar h 0 | agregar x 1 | agregar h 2 | agregar s 2 | bloch
```

Visualiza tres qubits con diferentes estados:

- Qubit 0: Estado |+⟩ (superposición en X)
- Qubit 1: Estado |1⟩ (polo sur)
- Qubit 2: Estado |+i⟩ (superposición en Y)

### Ejemplo 3: Qubits Específicos

```bash
crear 5 | agregar h 0 | agregar x 2 | agregar y 4 | bloch 0 2 4
```

Visualiza solo los qubits 0, 2 y 4, ignorando los demás.

### Ejemplo 4: Estado Entrelazado (Bell)

```bash
crear 2 | agregar h 0 | agregar cx 0 1 | bloch
```

Visualiza un estado de Bell. Los vectores de Bloch estarán en el origen (0, 0, 0), indicando que los qubits están maximamente entrelazados.

## Interpretación del Vector de Bloch

### Coordenadas Cartesianas (x, y, z)

El vector de Bloch representa el estado de un qubit en el espacio 3D:

- **Eje X**: Superposición entre |0⟩ y |1⟩ con fase 0
  - x = +1: Estado |+⟩ = (|0⟩ + |1⟩)/√2
  - x = -1: Estado |-⟩ = (|0⟩ - |1⟩)/√2

- **Eje Y**: Superposición entre |0⟩ y |1⟩ con fase π/2
  - y = +1: Estado |+i⟩ = (|0⟩ + i|1⟩)/√2
  - y = -1: Estado |-i⟩ = (|0⟩ - i|1⟩)/√2

- **Eje Z**: Probabilidad de medir |0⟩ vs |1⟩
  - z = +1: Estado |0⟩ (polo norte)
  - z = -1: Estado |1⟩ (polo sur)
  - z = 0: Superposición igual

### Coordenadas Esféricas (θ, φ)

- **θ (ángulo polar)**: 0° a 180°
  - 0°: Estado |0⟩
  - 90°: Superposición igual
  - 180°: Estado |1⟩

- **φ (ángulo azimutal)**: 0° a 360°
  - 0°: Fase 0 (eje +X)
  - 90°: Fase π/2 (eje +Y)
  - 180°: Fase π (eje -X)
  - 270°: Fase 3π/2 (eje -Y)

### Magnitud del Vector

- **|r| = 1**: Estado puro (qubit en la superficie de la esfera)
- **|r| < 1**: Estado mixto (qubit dentro de la esfera)
- **|r| ≈ 0**: Estado maximamente mezclado (común en qubits entrelazados)

## Características Técnicas

### Traza Parcial

Para sistemas multi-qubit, el módulo calcula automáticamente la traza parcial para obtener el estado reducido de cada qubit individual. Esto permite visualizar qubits en sistemas entrelazados.

### Límites

- **Máximo de qubits visualizados**: 10 (para evitar sobrecarga de memoria)
- **Formato de salida**: PNG de alta resolución (150 DPI)
- **Ubicación**: `~/.qiskit_cli/bloch_spheres/`

### Información Mostrada

Para cada qubit, se muestra:

1. Vector de Bloch en coordenadas cartesianas (x, y, z)
2. Magnitud del vector |r|
3. Ángulos esféricos θ y φ
4. Visualización 3D con:
   - Esfera semitransparente
   - Ejes X, Y, Z etiquetados
   - Círculos ecuatoriales
   - Vector de estado en rojo
   - Información numérica en recuadro

## Casos de Uso

### 1. Debugging de Circuitos

Verificar que las puertas cuánticas están aplicando las transformaciones esperadas.

### 2. Educación

Entender visualmente conceptos como superposición, fase y entrelazamiento.

### 3. Análisis de Algoritmos

Visualizar el estado de qubits en diferentes etapas de un algoritmo cuántico.

### 4. Detección de Entrelazamiento

Identificar qubits entrelazados por su vector de Bloch cerca del origen.

## Integración con Otros Comandos

El comando `bloch` se puede combinar con otros comandos en pipelines:

```bash
# Ver circuito y luego visualizar estados
crear 2 | agregar h 0 | agregar cx 0 1 | ver | bloch

# Analizar y visualizar
crear 3 | agregar h 0 | agregar phi 3 0 | bloch | analisis

# Ejecutar y luego visualizar (antes de medir)
crear 2 | agregar h 0 | agregar cx 0 1 | bloch | medir | ejecutar 1000
```

## Notas Importantes

1. **Sin Mediciones**: El comando `bloch` requiere un circuito sin mediciones. Si el circuito tiene mediciones, se eliminan automáticamente para calcular el statevector.

2. **Estados Entrelazados**: En estados entrelazados, los qubits individuales aparecen como estados mixtos (vector cerca del origen). Esto es correcto y refleja la naturaleza del entrelazamiento cuántico.

3. **Rendimiento**: Para circuitos con muchos qubits (>10), solo se visualizan los primeros 10 para mantener el rendimiento.

## Referencias

- [Bloch Sphere - Wikipedia](https://en.wikipedia.org/wiki/Bloch_sphere)
- [Qiskit Visualization](https://qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html#Bloch-Sphere)
- [Quantum States and the Bloch Sphere](https://learn.qiskit.org/course/basics/single-qubit-gates#the-bloch-sphere)
