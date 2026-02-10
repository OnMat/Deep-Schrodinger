import numpy as np
import matplotlib.pyplot as plt

def graph_1d(model, x_min=0.0, x_max=2.0, n_points=500):
    """
    Plota a saída da rede (psi) no intervalo [x_min, x_max].
    """
    # Gera pontos uniformes no intervalo
    x_vals = np.linspace(x_min, x_max, n_points).reshape(-1, 1)

    # Calcula psi com o modelo
    psi_vals = model(x_vals, training=False).numpy().flatten()

    # Normaliza opcionalmente (se quiser comparar com densidade de probabilidade)
    norm = np.trapezoid(np.abs(psi_vals)**2, x_vals.flatten())
    psi_vals_norm = psi_vals

    # Plota
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, psi_vals_norm, label="Ψ(x)")
    plt.plot(x_vals, np.abs(psi_vals_norm)**2, '--', label="|Ψ(x)|²")
    plt.xlabel("x (Å)")
    plt.ylabel("Amplitude")
    plt.title("Função de onda aprendida pelo modelo")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
