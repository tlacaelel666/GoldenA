#!/usr/bin/env python3
"""
Script principal de entrada para Qiskit Runtime CLI.
Importa y ejecuta la lógica desde circuito_aureo.py
"""
import sys
from circuito_aureo import main

if __name__ == "__main__":
    try:
        # Ejecutar la función principal del CLI
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Programa interrumpido por el usuario. ¡Hasta luego!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error fatal no controlado: {e}")
        sys.exit(1)
