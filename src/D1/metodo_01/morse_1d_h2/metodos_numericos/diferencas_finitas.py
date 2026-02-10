
import math as mt
import numpy as np
import matplotlib.pyplot as plt
n = 1500
#valor de cada x seguindo o potencial
x = np.linspace(0.001, 2, n)
d = x[1] - x[0]

def morse_potential_ev(x, D_e=4.97177, a=1.85, x_e=0.89):
    """
    Potencial de Morse em eV, x em Å.
    """
    return D_e * (1 - np.exp(-a * (x - x_e)))**2 - D_e

#Criando a matrix tridiagonal que vai receber as informações da equação de Schrodinger
C = np.eye(n)
D = np.eye(n)

for i in range(0, n):

  x_i = x[i]
  C[i, i] = -2
  D[i, i] = morse_potential_ev(x_i)

  if i < n - 1:
    C[i, i + 1] = 1
    C[i + 1, i] = 1


# Constantes físicas
hbar = 1.054571817e-34
m = 8.367e-28
e = 1.602176634e-19
A_0 = 1e10



# Constante do operador cinético em eV·Å²
const_schrodinger = ((hbar**2) / (2 * m)) * ((A_0**2) / e) * (1/(d**2))

K =  (-const_schrodinger )*C + D



#Retirando os autovalores e autovetores da matrix (Valores da Energia e da função psi)
E, Psi = np.linalg.eig(K)

indice_menor = np.argmin(E)
A1 = E[indice_menor]
E[indice_menor] = np.inf  # Remove da próxima busca

# Segundo menor
indice_segundo_menor = np.argmin(E)
A2 = E[indice_segundo_menor]
E[indice_segundo_menor] = np.inf

# Terceiro menor
indice_terceiro_menor = np.argmin(E)
A3 = E[indice_terceiro_menor]
E[indice_terceiro_menor] = np.inf

# Terceiro menor
indice_quarto_menor = np.argmin(E)
A4 = E[indice_quarto_menor]
E[indice_quarto_menor] = np.inf

# Terceiro menor
indice_quinto_menor = np.argmin(E)
A5 = E[indice_quinto_menor]

# Impressão dos valores
print(A1)
print(A2)
print(A3)
print(A4)
print(A5)



#Retirando os valores dos estados de energia para a importação no gráfico, em sequencia (1° estado de energia)
psi0 = Psi[:,indice_menor]
psi1 = Psi[:,indice_segundo_menor]
psi2 = Psi[:,indice_terceiro_menor]
psi3 = Psi[:,indice_quarto_menor]



def normalizar(psi, x):
    # calcula o integral numericamente com regra do trapézio
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    return psi / norm
psi0_norm = normalizar(psi0, x)
psi1_norm = normalizar(psi1, x)
psi2_norm = normalizar(psi2, x)
psi3_norm = normalizar(psi3, x)


plt.plot(x, psi0_norm, 'k-', label='Estado fundamental')
plt.plot(x, psi1_norm, 'k--', label='1° Estado')
plt.plot(x, psi2_norm, 'k:', label='2° Estado')
plt.plot(x, psi3_norm, 'k:', label='3° Estado')

#plt.plot(x,psi3)

plt.xlabel('Posição (Å)')
plt.ylabel('$\\psi(x)$')
plt.xlim(0,2)
plt.grid(True)
plt.legend()