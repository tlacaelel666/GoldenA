import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def ejecutar_analisis():
    PI = np.pi
    PHI = (1 + np.sqrt(5)) / 2

    print("\n=== Análisis de Dinámica Áurea ===")
    
    # --- 1. Configuración y Datos para Visualización 3D ---
    try:
        n_input = input("Lista de valores discretos para 3D (n_max) [Enter para 10]: ")
        if not n_input:
            n_max = 10
        else:
            n_max = int(n_input)
    except ValueError:
        print("Entrada inválida, usando n_max=10")
        n_max = 10

    n_vals_3d = np.arange(n_max + 1)
    paridad_arr_3d = np.cos(PI * n_vals_3d)
    cuasi_arr_3d = np.cos(PI * PHI * n_vals_3d)
    dimension_arr_3d = cuasi_arr_3d * paridad_arr_3d
    ponderado_arr_3d = cuasi_arr_3d + dimension_arr_3d

    # Imprimir tabla
    print("\nn | Paridad | Fase cuasiperiódica | Dimension | Valor Ponderado")
    print("-" * 100)
    for n in range(n_max + 1):
        print(f"{n:2d} | {paridad_arr_3d[n]:7.4f} | {cuasi_arr_3d[n]:19.4f} | {dimension_arr_3d[n]:9.4f} | {ponderado_arr_3d[n]:15.4f}")

    print("\n" + "="*100)
    print("Interpretación:")
    print("- cos(n*PI): Paridad (1 si n par, -1 si n impar)")
    print("- cos(PI*PHI*n): Distribución azimutal cuasiperiódica")
    print("Valor ponderado = Fase cuasiperiódica + Dimensión")

    # --- 2. Datos para Validación Experimental (2D) ---
    n_experimental = np.array([0, 1, 2, 3, 4, 5])
    prob_polarizacion_exp = np.array([0.239, 0.676, 0.114, 0.176, 0.003, 0.211])
    
    n_vals_teorico = np.arange(n_max + 1) 
    # Usamos más puntos para la curva teórica suave extendida hasta n_max
    n_vals_smooth = np.linspace(0, n_max, 500)
    paridad_smooth = np.cos(PI * n_vals_smooth)
    cuasi_smooth = np.cos(PI * PHI * n_vals_smooth)
    dimension_smooth = paridad_smooth * cuasi_smooth
    prob_teorica_smooth = 0.5 - (dimension_smooth * 0.5)

    # --- 3. Creación de Figura Combinada ---
    print(f"\nGenerando gráfico combinado (3D + 2D) hasta n={n_max}...")
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        subplot_titles=(f"Visualización 3D: Dinámica Áurea (n=0..{n_max})", f"Validación: Teoría (n=0..{n_max}) vs Experimental (n=0..5)"),
        vertical_spacing=0.1
    )

    # Trace 3D
    fig.add_trace(
        go.Scatter3d(
            x=n_vals_3d,
            y=cuasi_arr_3d,
            z=ponderado_arr_3d,
            mode='markers',
            marker=dict(
                size=5,
                color=ponderado_arr_3d,
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Dinámica 3D'
        ),
        row=1, col=1
    )

    # Trace 2D - Teoría
    fig.add_trace(
        go.Scatter(
            x=n_vals_smooth, 
            y=prob_teorica_smooth, 
            mode='lines', 
            name='Predicción Teórica (O_n)', 
            line=dict(color='rgba(255, 165, 0, 0.6)', width=4)
        ),
        row=2, col=1
    )

    # Trace 2D - Experimental
    fig.add_trace(
        go.Scatter(
            x=n_experimental, 
            y=prob_polarizacion_exp, 
            mode='markers', 
            name='Resultados Qiskit', 
            marker=dict(size=12, color='darkblue', symbol='x-thin')
        ),
        row=2, col=1
    )

    # Layout
    fig.update_layout(
        title_text="Análisis Completo: Dinámica Áurea y Validación Experimental",
        height=900,
        showlegend=True,
        scene=dict(
            xaxis_title='n',
            yaxis_title='Fase Cuasi (cos(πφn))',
            zaxis_title='Ponderado'
        )
    )
    
    # Ejes 2D
    fig.update_xaxes(title_text="Parámetro n", row=2, col=1)
    fig.update_yaxes(title_text="Probabilidad P(|01⟩)", range=[0, 1], row=2, col=1)

    # Guardar
    output_file = "analisis_aureo.html"
    print(f"Guardando visualización en '{output_file}'...")
    fig.write_html(output_file, auto_open=True)
    print(f"✅ Gráfico guardado exitosamente.")

if __name__ == "__main__":
    ejecutar_analisis()
