#!/usr/bin/env python3
"""
Qiskit Runtime CLI v3.1 - One-Liner Pipeline
Sintaxis: crear 3 | agregar h 0 | agregar y 1 | agregar ccx 0 1 2 | medir | ejecutar 1024
"""

import sys
import math
from typing import Optional, List, Dict
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from analisis_aureo import ejecutar_analisis
from golden.gold_cgh import demo_fibonacci_metriplectic
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

# ============ IBM QUANTUM IMPORTS ============
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler, SamplerOptions
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# ============ Q-CTRL IMPORTS ============
try:
    import fireopal
    QCTRL_AVAILABLE = True
except ImportError:
    QCTRL_AVAILABLE = False

# ============ COLORES ANSI ============
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ============ CONSTANTES CUÁNTICAS ============
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618
GOLDEN_PHASE = math.pi / PHI  # ≈ 1.944 rad

# ============ CLASES PERSONALIZADAS ============
class GoldenGate(Gate):
    """
    Puerta Cuántica Áurea.
    Aplica una fase relativa igual a pi/phi.
    """
    def __init__(self, n, label=None):
        self.n = n
        super().__init__("golden_gate", 1, [n], label=label)
        
    def _define(self):
        qc = QuantumCircuit(1)
        # lambda_n = cos(n*pi) * cos(n*phi*pi)
        lambda_n = math.cos(self.n * math.pi) * math.cos(self.n * PHI * math.pi)
        
        # U3(0, 0, lambda) es equivalente a una fase pura P(lambda)
        qc.u(0, 0, lambda_n, 0) 
        self.definition = qc

# ============ DEFINICIÓN DE PUERTAS ============
GATES_DB = {
    # Puertas de 1 qubit (sin parámetros)
    "h": {"qubits": 1, "params": 0, "name": "Hadamard", "desc": "Superposición"},
    "x": {"qubits": 1, "params": 0, "name": "Pauli-X", "desc": "NOT cuántico"},
    "y": {"qubits": 1, "params": 0, "name": "Pauli-Y", "desc": "Rotación compleja"},
    "z": {"qubits": 1, "params": 0, "name": "Pauli-Z", "desc": "Cambio de fase"},
    "s": {"qubits": 1, "params": 0, "name": "S", "desc": "Fase π/2"},
    "sdg": {"qubits": 1, "params": 0, "name": "S†", "desc": "Fase -π/2"},
    "t": {"qubits": 1, "params": 0, "name": "T", "desc": "Fase π/4"},
    "tdg": {"qubits": 1, "params": 0, "name": "T†", "desc": "Fase -π/4"},
    "i": {"qubits": 1, "params": 0, "name": "Identidad", "desc": "No hace nada"},
    "phi": {"qubits": 1, "params": 1, "name": "Phi (Golden)", "desc": "Fase cos(nπ)cos(nφπ)"},
    
    # Puertas de 2 qubits (sin parámetros)
    "cx": {"qubits": 2, "params": 0, "name": "CNOT", "desc": "Control-NOT"},
    "cy": {"qubits": 2, "params": 0, "name": "CY", "desc": "Control-Y"},
    "cz": {"qubits": 2, "params": 0, "name": "CZ", "desc": "Control-Z"},
    "swap": {"qubits": 2, "params": 0, "name": "SWAP", "desc": "Intercambiar qubits"},
    
    # Puertas de rotación (con parámetros)
    "rx": {"qubits": 1, "params": 1, "name": "RX", "desc": "Rotación-X"},
    "ry": {"qubits": 1, "params": 1, "name": "RY", "desc": "Rotación-Y"},
    "rz": {"qubits": 1, "params": 1, "name": "RZ", "desc": "Rotación-Z"},
    "p": {"qubits": 1, "params": 1, "name": "Phase", "desc": "Cambio de fase"},
    
    # Puertas controladas de rotación
    "crx": {"qubits": 2, "params": 1, "name": "CRX", "desc": "Control-RX"},
    "cry": {"qubits": 2, "params": 1, "name": "CRY", "desc": "Control-RY"},
    "crz": {"qubits": 2, "params": 1, "name": "CRZ", "desc": "Control-RZ"},
    "cp": {"qubits": 2, "params": 1, "name": "CPhase", "desc": "Control-Phase"},
    
    # Puertas de 3 qubits
    "ccx": {"qubits": 3, "params": 0, "name": "Toffoli", "desc": "Control-Control-NOT"},
    "cswap": {"qubits": 3, "params": 0, "name": "Fredkin", "desc": "SWAP controlado"},
}

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configura logging"""
    log_dir = Path.home() / ".qiskit_cli" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('qiskit_cli')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    handler = logging.FileHandler(log_dir / f"qiskit_{datetime.now().strftime('%Y%m%d')}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    
    return logger

def print_banner():
    """Imprime banner"""
    banner = f"""
{Colors.OKCYAN}{chr(9556)}{chr(9552)*78}{chr(9559)}
{chr(9553)}{'':^78}{chr(9553)}
{chr(9553)}{' ___    _         _      _   _       ____ _     ___ ':^78}{chr(9553)}
{chr(9553)}{' / _ \\  (_)  ___  | | __ (_) | |_    / ___| |   |_ _|':^78}{chr(9553)}
{chr(9553)}{'| | | | | | / __| | |/ / | | | __|  | |   | |    | | ':^78}{chr(9553)}
{chr(9553)}{'| |_| | | | \\__ \\ |   <  | | | |_   | |___| |___ | | ':^78}{chr(9553)}
{chr(9553)}{' \\__\\_\\ |_| |___/ |_|\\_\\ |_|  \\__|   \\____|_____|___|':^78}{chr(9553)}
{chr(9553)}{'':^78}{chr(9553)}
{chr(9553)}{'🚀 Qiskit Runtime CLI v3.2 - One-Liner Pipeline':^78}{chr(9553)}
{chr(9553)}{'crear 3 | agregar h 0 | agregar y 1 | agregar ccx 0 1 2 | medir | ejecutar':^78}{chr(9553)}
{chr(9553)}{'':^78}{chr(9553)}
{chr(9562)}{chr(9552)*78}{chr(9565)}{Colors.ENDC}

{Colors.BOLD}{Colors.OKGREEN}📖 Escribe 'ayuda' para comandos | 'login <token>' para IBM Quantum
🎯 'puertas' para listar puertas | 💡 'demo' para ver ejemplos
{Colors.ENDC}
    """
    print(banner)

def show_gates():
    """Muestra todas las puertas disponibles"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📚 PUERTAS DISPONIBLES{Colors.ENDC}\n")
    
    categories = {
        "1 Qubit (sin parámetros)": [k for k, v in GATES_DB.items() if v["qubits"] == 1 and v["params"] == 0],
        "1 Qubit (con ángulo)": [k for k, v in GATES_DB.items() if v["qubits"] == 1 and v["params"] == 1],
        "2 Qubits (sin parámetros)": [k for k, v in GATES_DB.items() if v["qubits"] == 2 and v["params"] == 0],
        "2 Qubits (con ángulo)": [k for k, v in GATES_DB.items() if v["qubits"] == 2 and v["params"] == 1],
        "3 Qubits": [k for k, v in GATES_DB.items() if v["qubits"] == 3],
    }
    
    for category, gates in categories.items():
        if gates:
            print(f"{Colors.OKCYAN}{category}:{Colors.ENDC}")
            for gate_name in gates:
                gate_info = GATES_DB[gate_name]
                print(f"  {Colors.OKGREEN}{gate_name:<8}{Colors.ENDC} → {gate_info['name']:<15} ({gate_info['desc']})")
            print()

def show_help():
    """Muestra ayuda de comandos"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📖 SINTAXIS ONE-LINER{Colors.ENDC}\n")
    print(f"{Colors.OKCYAN}Formato:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}comando1 args | comando2 args | comando3 args{Colors.ENDC}\n")
    
    commands = {
        "Construcción": [
            ("crear <n>", "Crear circuito de n qubits"),
            ("agregar <puerta> <args>", "Agregar puerta(s) al circuito"),
        ],
        "Ejecución": [
            ("medir [qubits]", "Medir todos o qubits específicos"),
            ("ejecutar [shots]", "Ejecutar en simulador (default: 1024)"),
        ],
        "IBM Quantum (Opcional)": [
            ("login <token>", "Conectar con token IBM Quantum"),
            ("backends", "Listar backends disponibles"),
            ("backend <nombre>", "Seleccionar backend de hardware"),
            ("simulator", "Cambiar al simulador local"),
            ("status", "Ver estado del backend actual"),
        ],
        "Q-CTRL (Opcional)": [
            ("login_qctrl <token>", "Conectar con Q-CTRL Fire Opal"),
        ],
        "Información": [
            ("ver", "Mostrar circuito actual"),
            ("analisis", "Visualización 3D de dinámica áurea"),
            ("cgh", "Análisis Fibonacci Metripléctico (gold_cgh)"),
            ("puertas", "Ver puertas disponibles"),
            ("demo", "Ejemplos de circuitos"),
            ("ayuda", "Esta ayuda"),
        ],
    }
    
    for category, cmds in commands.items():
        print(f"{Colors.OKCYAN}{category}:{Colors.ENDC}")
        for cmd, desc in cmds:
            print(f"  {Colors.OKGREEN}{cmd:<30}{Colors.ENDC} → {desc}")
        print()

class QiskitCLI:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.local_simulator = AerSimulator()
        self.service = None
        self.backend = None
        self.use_hardware = False
        self.available_backends = []
        self.qctrl = None  # Cliente Q-CTRL
        # Estado persistente del circuito
        self.circuit = None
        self.num_qubits = 0
        self.measured_qubits = set()

    def ibm_login(self, token: str) -> bool:
        """Inicia sesión con token de IBM Quantum"""
        if not IBM_AVAILABLE:
            print(f"{Colors.FAIL}❌ IBM Quantum Runtime no instalado.{Colors.ENDC}")
            print(f"{Colors.WARNING}Instala con: pip install qiskit-ibm-runtime{Colors.ENDC}")
            return False
        
        try:
            print(f"{Colors.BOLD}🔐 Iniciando sesión con IBM Quantum...{Colors.ENDC}")
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=token, overwrite=True)
            self.service = QiskitRuntimeService()
            self.available_backends = self.service.backends()
            print(f"{Colors.OKGREEN}✅ ¡Sesión iniciada exitosamente!{Colors.ENDC}")
            print(f"{Colors.OKBLUE}📱 Backends disponibles:{Colors.ENDC}")
            for i, backend in enumerate(self.available_backends[:10]):
                print(f"   {i+1}. {backend.name}")
            if len(self.available_backends) > 10:
                print(f"   ... y {len(self.available_backends)-10} más")
            self.logger.info(f"Login exitoso: {len(self.available_backends)} backends")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error en login: {e}{Colors.ENDC}")
            self.logger.error(f"Login error: {e}")
            return False

    def qctrl_login(self, token: str) -> bool:
        """Inicia sesión con Q-CTRL Fire Opal"""
        if not QCTRL_AVAILABLE:
            print(f"{Colors.FAIL}❌ Librería fire-opal no instalada.{Colors.ENDC}")
            print(f"{Colors.WARNING}Instala con: pip install fire-opal{Colors.ENDC}")
            return False
        
        try:
            print(f"{Colors.BOLD}🔥 Iniciando sesión con Q-CTRL Fire Opal...{Colors.ENDC}")
            
            # Crear credenciales usando el método específico para Q-CTRL
            credentials = fireopal.credentials.make_credentials_for_qctrl(
                token=token
            )
            
            # Guardar las credenciales
            fireopal.credentials.save(credentials)
            
            print(f"{Colors.OKGREEN}✅ ¡Sesión Fire Opal configurada exitosamente!{Colors.ENDC}")
            self.logger.info("Login Fire Opal exitoso")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error en login Fire Opal: {e}{Colors.ENDC}")
            self.logger.error(f"Login Fire Opal error: {e}")
            return False

    def select_backend(self, backend_name: str) -> bool:
        """Selecciona un backend de hardware real"""
        if not self.service:
            print(f"{Colors.FAIL}❌ Primero conecta con 'login <token>'{Colors.ENDC}")
            return False
        
        try:
            self.backend = self.service.backend(backend_name)
            self.use_hardware = True
            print(f"{Colors.OKGREEN}✅ Backend seleccionado: {backend_name}{Colors.ENDC}")
            print(f"   Qubits: {self.backend.num_qubits}")
            self.logger.info(f"Backend seleccionado: {backend_name}")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}❌ Backend no encontrado: {e}{Colors.ENDC}")
            return False

    def switch_to_simulator(self) -> bool:
        """Cambia al simulador local"""
        self.use_hardware = False
        self.backend = None
        print(f"{Colors.OKGREEN}✅ Cambiado al simulador local{Colors.ENDC}")
        return True

    def list_backends(self) -> bool:
        """Lista todos los backends disponibles"""
        if not self.service:
            print(f"{Colors.FAIL}❌ No hay sesión IBM activa{Colors.ENDC}")
            return False
        
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}📱 BACKENDS IBM QUANTUM{Colors.ENDC}\n")
        for i, backend in enumerate(self.available_backends, 1):
            status = "🟢 ACTIVO" if backend.status().operational else "🔴 INACTIVO"
            print(f"  {i}. {backend.name:<20} {status:<15} ({backend.num_qubits} qubits)")
        print()
        return True

    def parse_parameter(self, param_str: str) -> float:
        """Parsea parámetros numéricos o expresiones con π y φ"""
        safe_dict = {"pi": math.pi, "e": math.e, "sqrt": math.sqrt, "phi": PHI}
        try:
            return float(eval(param_str, {"__builtins__": {}}, safe_dict))
        except Exception as e:
            raise ValueError(f"No se puede evaluar '{param_str}': {e}")

    def jls_extract_def(self):
        return print

    def execute_pipeline(self, pipeline_str: str) -> bool:
        """Ejecuta un pipeline de comandos en una sola línea"""
        commands = [cmd.strip() for cmd in pipeline_str.split('|')]
        
        if not commands:
            return False

        if not commands:
            return False

        print(f"\n{Colors.BOLD}{Colors.OKCYAN}🔄 EJECUTANDO COMANDO(S) ({len(commands)}){Colors.ENDC}\n")

        for step, cmd_str in enumerate(commands, 1):

            
            parts = cmd_str.split()
            if not parts:
                continue

            command = parts[0].lower()
            args = parts[1:]

            # ============ CREAR ============
            if command == "crear":
                if len(args) != 1:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Uso: crear <num_qubits>{Colors.ENDC}")
                    return False
                try:
                    num_qubits_arg = int(args[0])
                    if num_qubits_arg <= 0:
                        print(f"{Colors.FAIL}❌ [Paso {step}] Qubits debe ser > 0{Colors.ENDC}")
                        return False
                    
                    # Reiniciar circuito
                    self.num_qubits = num_qubits_arg
                    qreg = QuantumRegister(self.num_qubits, 'q')
                    creg = ClassicalRegister(self.num_qubits, 'c')
                    self.circuit = QuantumCircuit(qreg, creg)
                    self.measured_qubits = set()
                    
                    self.jls_extract_def()(f"{Colors.OKGREEN}✅ [Paso {step}] Circuito de {self.num_qubits} qubits creado (Estado reiniciado){Colors.ENDC}")
                    self.logger.info(f"Circuito creado: {self.num_qubits} qubits")
                except ValueError as e:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Error en 'crear': {e}{Colors.ENDC}")
                    return False

            # ============ AGREGAR PUERTA(S) ============
            elif command == "agregar":
                if self.circuit is None:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Primero crea un circuito con 'crear'{Colors.ENDC}")
                    return False
                if len(args) < 2:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Uso: agregar <puerta> <args>{Colors.ENDC}")
                    return False

                gate_name = args[0].lower()

                if gate_name not in GATES_DB:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Puerta desconocida: '{gate_name}'{Colors.ENDC}")
                    return False

                gate_info = GATES_DB[gate_name]
                num_params = gate_info["params"]
                num_gate_qubits = gate_info["qubits"]
                total_needed = num_params + num_gate_qubits

                if len(args) - 1 != total_needed:
                    print(f"{Colors.FAIL}❌ [Paso {step}] '{gate_name}' necesita {num_params} parámetro(s) y {num_gate_qubits} qubit(s){Colors.ENDC}")
                    return False

                try:
                    # Parsear parámetros
                    params = []
                    for i in range(num_params):
                        params.append(self.parse_parameter(args[1 + i]))

                    # Parsear qubits
                    qubit_indices = []
                    for i in range(num_params, total_needed):
                        q = int(args[1 + i])
                        if q < 0 or q >= self.num_qubits:
                            print(f"{Colors.FAIL}❌ [Paso {step}] Qubit {q} fuera de rango (0-{self.num_qubits-1}){Colors.ENDC}")
                            return False
                        qubit_indices.append(q)

                    # Aplicar puerta especial PHI
                    if gate_name == "phi":
                        n_val = params[0]
                        self.circuit.append(GoldenGate(n_val), [qubit_indices[0]])
                        print(f"{Colors.OKGREEN}✅ [Paso {step}] GoldenGate(n={n_val}) aplicada en qubit {qubit_indices[0]}{Colors.ENDC}")
                    else:
                        # Aplicar puerta normal
                        gate_method = getattr(self.circuit, gate_name)
                        if num_params > 0:
                            gate_method(params[0], *qubit_indices)
                        else:
                            gate_method(*qubit_indices)
                        print(f"{Colors.OKGREEN}✅ [Paso {step}] {gate_info['name']} en qubit(s) {qubit_indices}{Colors.ENDC}")

                    self.logger.info(f"Puerta {gate_name} agregada: parámetros={params}, qubits={qubit_indices}")

                except Exception as e:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Error al agregar puerta: {type(e).__name__}: {e}{Colors.ENDC}")
                    self.logger.error(f"Error en agregar: {e}", exc_info=True)
                    return False

            # ============ MEDIR ============
            elif command == "medir":
                if self.circuit is None:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Primero crea un circuito{Colors.ENDC}")
                    return False

                try:
                    if args and args[0].lower() != "all":
                        # Medir qubits específicos
                        qubits = [int(q) for q in args]
                        for q in qubits:
                            if q < 0 or q >= self.num_qubits:
                                print(f"{Colors.FAIL}❌ [Paso {step}] Qubit {q} fuera de rango{Colors.ENDC}")
                                return False
                            self.circuit.measure(q, q)
                            self.measured_qubits.add(q)
                        print(f"{Colors.OKGREEN}✅ [Paso {step}] Medición en qubits {qubits}{Colors.ENDC}")
                    else:
                        # Medir todos
                        for i in range(self.num_qubits):
                            self.circuit.measure(i, i)
                            self.measured_qubits.add(i)
                        print(f"{Colors.OKGREEN}✅ [Paso {step}] Medición en todos los qubits{Colors.ENDC}")
                except Exception as e:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Error en medición: {e}{Colors.ENDC}")
                    return False

            # ============ EJECUTAR ============
            elif command == "ejecutar":
                if self.circuit is None:
                    print(f"{Colors.FAIL}❌ [Paso {step}] No hay circuito para ejecutar{Colors.ENDC}")
                    return False

                if len(self.measured_qubits) == 0:
                    print(f"{Colors.WARNING}⚠️  [Paso {step}] Sin mediciones. Agregando medición global.{Colors.ENDC}")
                    for i in range(self.num_qubits):
                        self.circuit.measure(i, i)
                        self.measured_qubits.add(i)

                try:
                    shots = int(args[0]) if args else 1024
                    if shots <= 0:
                        print(f"{Colors.FAIL}❌ [Paso {step}] Shots debe ser > 0{Colors.ENDC}")
                        return False

                    if self.use_hardware and self.backend:
                        print(f"\n{Colors.BOLD}🔴 [Paso {step}] Ejecutando en HARDWARE REAL ({self.backend.name})...{Colors.ENDC}")
                        print(f"{Colors.WARNING}⏳ Esto puede tomar minutos/horas dependiendo de la cola{Colors.ENDC}\n")
                        
                        try:
                            transpiled = transpile(self.circuit, self.backend, optimization_level=3)
                            
                            # Ejecutar con Sampler V2
                            with Session(backend=self.backend) as session:
                                sampler = Sampler(session=session)
                                sampler_options = SamplerOptions(default_shots=shots)
                                sampler.options.update(sampler_options)
                                
                                job = sampler.run([transpiled])
                                job_id = job.job_id
                                print(f"{Colors.OKBLUE}📋 Job ID: {job_id}{Colors.ENDC}")
                                
                                result = job.result()
                                counts = result[0].data.meas.get_counts()
                                
                        except Exception as e:
                            print(f"{Colors.FAIL}❌ Error en hardware: {e}{Colors.ENDC}")
                            self.logger.error(f"Hardware error: {e}", exc_info=True)
                            return False
                    else:
                        print(f"\n{Colors.BOLD}🖥️  [Paso {step}] Ejecutando en simulador local ({shots} shots)...{Colors.ENDC}")
                        
                        transpiled = transpile(self.circuit, self.local_simulator, optimization_level=1)
                        job = self.local_simulator.run(transpiled, shots=shots)
                        result = job.result()
                        counts = result.get_counts(transpiled)

                    self._display_results(counts, shots)
                    self.logger.info(f"Circuito ejecutado: {shots} shots en {'hardware' if self.use_hardware else 'simulador'}")

                    # Guardar histograma
                    try:
                        plot_histogram(counts)
                        save_path = Path.home() / ".qiskit_cli" / f"histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        print(f"{Colors.OKGREEN}📊 Histograma guardado: {save_path}{Colors.ENDC}")
                        plt.close()
                    except Exception as e:
                        print(f"{Colors.WARNING}⚠️  No se pudo guardar histograma: {e}{Colors.ENDC}")

                except ValueError:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Shots inválido{Colors.ENDC}")
                    return False
                except Exception as e:
                    print(f"{Colors.FAIL}❌ [Paso {step}] Error en ejecución: {e}{Colors.ENDC}")
                    self.logger.error(f"Error en ejecutar: {e}", exc_info=True)
                    return False

            # ============ VER ============
            elif command == "ver":
                if self.circuit is None:
                    print(f"{Colors.WARNING}⚠️  [Paso {step}] No hay circuito{Colors.ENDC}")
                    return False
                print(f"\n{Colors.BOLD}{Colors.OKCYAN}📊 CIRCUITO (Paso {step}){Colors.ENDC}")
                print(f"Qubits: {self.num_qubits} | Profundidad: {self.circuit.depth()} | Operaciones: {len(self.circuit.data)}")
                print(self.circuit.draw(output='text'))
                print()

            # ============ ANALISIS ============
            elif command == "analisis":
                ejecutar_analisis()

            # ============ CGH (Fibonacci Metriplectic) ============
            elif command == "cgh":
                # Usar el número de qubits actual si es válido, sino default a 3
                n_q = self.num_qubits if self.num_qubits > 0 else 3
                demo_fibonacci_metriplectic(n_q)

            # ============ OTROS COMANDOS ============
            elif command == "puertas":
                show_gates()
            elif command == "demo":
                self.show_demos()
            elif command == "ayuda":
                show_help()
            elif command == "salir":
                print(f"{Colors.OKGREEN}👋 ¡Hasta luego!{Colors.ENDC}")
                sys.exit(0)
            else:
                print(f"{Colors.FAIL}❌ [Paso {step}] Comando desconocido: '{command}'{Colors.ENDC}")
                return False

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}✨ Pipeline completado exitosamente!{Colors.ENDC}\n")
        return True

    def _display_results(self, counts: dict, shots: int):
        """Muestra resultados visuales"""
        print(f"\n{Colors.BOLD}📊 Resultados ({shots} mediciones):{Colors.ENDC}")
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        
        for outcome, count in sorted_counts.items():
            percentage = (count / shots) * 100
            bar_length = max(1, int(percentage / 2))
            bar = '█' * bar_length
            print(f"  |{outcome}⟩: {count:>4} ({percentage:5.1f}%) {Colors.OKBLUE}{bar}{Colors.ENDC}")

    def show_demos(self):
        """Muestra demostraciones"""
        demos = [
            ("Superposición", "crear 1 | agregar h 0 | medir | ejecutar 1000"),
            ("Bell (Entrelazamiento)", "crear 2 | agregar h 0 | agregar cx 0 1 | medir | ejecutar 1000"),
            ("Deutsch", "crear 2 | agregar x 1 | agregar h 0 | agregar h 1 | agregar cx 0 1 | agregar h 0 | medir | ejecutar 1000"),
            ("Golden Ratio (Phi n -> phi 3)", "crear 1 | agregar h 0 | agregar phi 3 0 | medir | ejecutar 1000"),
            ("Toffoli", "crear 3 | agregar h 0 | agregar h 1 | agregar ccx 0 1 2 | medir | ejecutar 1000"),
        ]

        print(f"\n{Colors.BOLD}{Colors.OKCYAN}🎬 DEMOSTRACIONES{Colors.ENDC}\n")
        
        for name, pipeline in demos:
            print(f"{Colors.OKCYAN}{name}:{Colors.ENDC}")
            print(f"  {Colors.OKGREEN}{pipeline}{Colors.ENDC}\n")

    def run_interactive_mode(self):
        """Inicia el modo interactivo"""
        while True:
            try:
                status_backend = f"🔴 {self.backend.name}" if self.use_hardware else "🖥️  Simulador"
                prompt = f"{Colors.BOLD}{Colors.OKGREEN}qiskit ({status_backend})> {Colors.ENDC}"
                user_input = input(prompt).strip()
                
                # Mostrar ayuda inicial
                if not hasattr(self, '_help_shown'):
                    self._help_shown = True
                
                if not user_input:
                    continue
                
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["salir", "exit", "quit"]:
                    print(f"{Colors.OKGREEN}👋 ¡Hasta luego!{Colors.ENDC}")
                    sys.exit(0)
                
                # Comandos de sesión
                elif command == "login":
                    if not args:
                        print(f"{Colors.FAIL}❌ Uso: login <token>{Colors.ENDC}")
                        print(f"{Colors.OKCYAN}Obtén tu token en: https://quantum.ibm.com/account{Colors.ENDC}")
                        continue
                    token = args.strip()
                    self.ibm_login(token)

                elif command == "login_qctrl":
                    if not args:
                        print(f"{Colors.FAIL}❌ Uso: login_qctrl <token>{Colors.ENDC}")
                        continue
                    token = args.strip()
                    self.qctrl_login(token)

                elif command == "backend":
                    if not args:
                        self.list_backends()
                    else:
                        self.select_backend(args.strip())

                elif command == "backends":
                    self.list_backends()

                elif command == "simulator":
                    self.switch_to_simulator()

                elif command == "status":
                    if self.use_hardware and self.backend:
                        try:
                            status = self.backend.status()
                            print(f"{Colors.OKCYAN}Backend: {self.backend.name}{Colors.ENDC}")
                            print(f"  Operativo: {'🟢 Sí' if status.operational else '🔴 No'}")
                            print(f"  Qubits: {self.backend.num_qubits}")
                            print(f"  Jobs pendientes: {status.pending_jobs}")
                        except Exception as e:
                            print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
                    else:
                        print(f"{Colors.OKCYAN}Usando: Simulador AER local{Colors.ENDC}")

                elif command in ["puertas", "demo", "ayuda"]:
                    if command == "puertas":
                        show_gates()
                    elif command == "demo":
                        self.show_demos()
                    else:
                        show_help()

                else:
                    # Ejecutar como pipeline
                    self.execute_pipeline(user_input)

            except KeyboardInterrupt:
                print(f"\n{Colors.OKGREEN}👋 ¡Hasta luego!{Colors.ENDC}")
                sys.exit(0)
            except Exception as e:
                print(f"{Colors.FAIL}❌ Error inesperado: {type(e).__name__}: {e}{Colors.ENDC}")
                self.logger.error(f"Error en loop: {e}", exc_info=True)


def main():
    """Función principal"""
    logger = setup_logging()
    print_banner()
    cli = QiskitCLI(logger)
    cli.run_interactive_mode()


if __name__ == "__main__":
    main()
