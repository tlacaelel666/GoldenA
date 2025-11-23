#!/bin/bash
# Script de demostración del comando bloch

echo "🌐 DEMOSTRACIÓN DEL COMANDO BLOCH"
echo "=================================="
echo ""

# Activar el entorno virtual
source env/bin/activate

# Ejemplo 1: Un qubit en superposición
echo "📝 Ejemplo 1: Qubit en superposición"
echo "Comando: crear 1 | agregar h 0 | bloch"
echo ""
echo "crear 1 | agregar h 0 | bloch" | python -m main
echo ""
echo "Presiona Enter para continuar..."
read

# Ejemplo 2: Múltiples qubits
echo "📝 Ejemplo 2: Tres qubits con diferentes estados"
echo "Comando: crear 3 | agregar h 0 | agregar x 1 | agregar h 2 | agregar s 2 | bloch"
echo ""
echo "crear 3 | agregar h 0 | agregar x 1 | agregar h 2 | agregar s 2 | bloch" | python -m main
echo ""
echo "Presiona Enter para continuar..."
read

# Ejemplo 3: Qubits específicos
echo "📝 Ejemplo 3: Visualizar solo qubits específicos"
echo "Comando: crear 5 | agregar h 0 | agregar x 2 | agregar y 4 | bloch 0 2 4"
echo ""
echo "crear 5 | agregar h 0 | agregar x 2 | agregar y 4 | bloch 0 2 4" | python -m main
echo ""
echo "Presiona Enter para continuar..."
read

# Ejemplo 4: Estado entrelazado
echo "📝 Ejemplo 4: Estado de Bell (entrelazado)"
echo "Comando: crear 2 | agregar h 0 | agregar cx 0 1 | bloch"
echo ""
echo "crear 2 | agregar h 0 | agregar cx 0 1 | bloch" | python -m main
echo ""

echo "✅ Demostración completada!"
echo "Las imágenes se guardaron en: ~/.qiskit_cli/bloch_spheres/"
