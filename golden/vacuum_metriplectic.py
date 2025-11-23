import numpy as np

def compute_O1(n: int) -> float:
    """Calcula el operador O1 base."""
    # Valor dummy basado en el contexto
    return 0.5

def compute_Fpura(O1: float) -> float:
    """Calcula la fricción pura."""
    return 1.0 / (O1**2) if O1 != 0 else 1.0

def compute_CUNIF(alpha_inv: float, Fpura: float) -> float:
    """Calcula la constante de acoplamiento universal."""
    # alpha_inv es aprox 137.036
    return alpha_inv * Fpura
