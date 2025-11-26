"""
Diccionario de plantillas de circuitos cuánticos pre-construidos.
Cada plantilla define una secuencia de comandos para el CLI.
"""

CIRCUIT_TEMPLATES = {
    "superposicion": {
        "desc": "Un qubit en estado de superposición (|0> + |1>)/sqrt(2)",
        "pipeline": "crear 1 | agregar h 0"
    },
    "bell": {
        "desc": "Estado de Bell (Entrelazamiento máximo) (|00> + |11>)/sqrt(2)",
        "pipeline": "crear 2 | agregar h 0 | agregar cx 0 1"
    },
    "ghz": {
        "desc": "Estado GHZ de 3 qubits (Entrelazamiento multipartito)",
        "pipeline": "crear 3 | agregar h 0 | agregar cx 0 1 | agregar cx 0 2"
    },
    "teleportacion": {
        "desc": "Protocolo de Teleportación Cuántica (preparación)",
        "pipeline": "crear 3 | agregar h 1 | agregar cx 1 2 | agregar cx 0 1 | agregar h 0 | medir 0 1"
    },
    "bernstein": {
        "desc": "Algoritmo Bernstein-Vazirani (s=11)",
        "pipeline": "crear 3 | agregar x 2 | agregar h 0 | agregar h 1 | agregar h 2 | agregar cx 0 2 | agregar cx 1 2 | agregar h 0 | agregar h 1"
    },
    "qft2": {
        "desc": "Transformada Cuántica de Fourier de 2 qubits",
        "pipeline": "crear 2 | agregar h 0 | agregar cp 0.5 1 0 | agregar h 1 | agregar swap 0 1"
    }
}
