import tensorflow as tf
import numpy as np

dtype = tf.float64


# -------------------------
# Potencial de Morse com parâmetros reais
# -------------------------
def morse_potential_ev(x, D_e=4.97177, a=1.85, x_e=0.89):
    """
    Potencial de Morse em eV, x em Å - Ångström.
    """
    return D_e * (1 - tf.exp(-a * (x - x_e)))**2 - D_e


def raio_sampler(r_min, r_max, n_samples, dtype=dtype):
    r = tf.random.uniform([n_samples, 1], r_min, r_max, dtype=dtype)
    return r


def monte_carlo_integral_1d(f, r_min=0.001, r_max=2.0):
    integral = tf.reduce_mean(tf.math.conj(f) * f) * (r_max - r_min)
    return integral


def Laplaciano_1d(model, x):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        f = model(x)
        f_x = tape2.gradient(f, x)
    f_xx = tape2.gradient(f_x, x)
    del tape2
    laplaciano = f_xx
    return laplaciano, f


def psi_norm(f_raw):
  integral = monte_carlo_integral_1d(f_raw)
  return f_raw / tf.sqrt(integral + 1e-12)


def energy_expectation_1d(lap, f_raw, x):
  psi = psi_norm(f_raw)
  hbar = tf.constant(1.054571817e-34, dtype=dtype)
  # m = 4.6915e8 eV/C² aprox  8.367e-28 kg
  m = tf.constant(8.367e-28, dtype=dtype)
  e = tf.constant(1.602176634e-19, dtype=dtype)
  A_0 = tf.constant(1e10, dtype=dtype)

  const_schrodinger = (hbar**2) / (2 * m) * (A_0**2) / e

  Hpsi = -(const_schrodinger * lap) + morse_potential_ev(x) * psi
  num = tf.reduce_mean(tf.math.conj(psi) * Hpsi)
  den = tf.reduce_mean(tf.math.conj(psi) * psi)
  return tf.math.real(num / den)

# -------------------------
# Variância do Hamiltoniano
# -------------------------
def Var_H_1d(model, x):
  hbar = tf.constant(1.054571817e-34, dtype=dtype)
  # m = 4.6915e8 eV/C² aprox  8.367e-28 kg
  m = tf.constant(8.367e-28, dtype=dtype)
  e = tf.constant(1.602176634e-19, dtype=dtype)
  A_0 = tf.constant(1e10, dtype=dtype)

  const_schrodinger = (hbar**2) / (2 * m) * (A_0**2) / e

  lap, psi = Laplaciano_1d(model, x)
  Hpsi = -(const_schrodinger * lap) + morse_potential_ev(x) * psi
  H_exp = tf.reduce_mean(tf.math.conj(psi) * Hpsi) / tf.reduce_mean(tf.math.conj(psi) * psi)
  H2_exp = tf.reduce_mean(tf.math.conj(Hpsi) * Hpsi) / tf.reduce_mean(tf.math.conj(psi) * psi)
  VarH = H2_exp - tf.math.real(H_exp)**2
  return tf.math.abs(VarH)


def L1_1d(model, x,E_n):
  lap, f = Laplaciano_1d(model, x)
  # Constantes físicas
  hbar = tf.constant(1.054571817e-34, dtype=dtype)
  # m = 4.6915e8 eV/C² aprox  8.367e-28 kg
  m = tf.constant(8.367e-28, dtype=dtype)
  e = tf.constant(1.602176634e-19, dtype=dtype)
  A_0 = tf.constant(1e10, dtype=dtype)

  const_schrodinger = (hbar**2) / (2 * m) * (A_0**2) / e

  V_pot = morse_potential_ev(x)

  # Operador de Schrödinger aplicado
  oper = -(lap * const_schrodinger) + (V_pot - E_n) * f
  return tf.reduce_mean(tf.square(oper))


def L2_1d(model, x, epsilon):
    f = model(x)
    return tf.reduce_mean(tf.square(f - epsilon))


def L3_1d(model, x):
    f = model(x)
    integral = monte_carlo_integral_1d(f)
    return tf.square(integral - 1.0)

def L4_1d(model, x):
    lap, f = Laplaciano_1d(model, x)
    return tf.reduce_mean(tf.square(lap))

def overlap_squared_1d(model_curr, model_prev_list, x, r_min=0.001, r_max=2.0):
  f_curr = model_curr(x)
  norm_curr = tf.sqrt(monte_carlo_integral_1d(f_curr) + 1e-12)

  loss_sum = tf.constant(0.0, dtype=tf.float64)

  for orthogonal_model in model_prev_list:
    f_prev = orthogonal_model(x)
    norm_prev = tf.sqrt(monte_carlo_integral_1d(f_prev) + 1e-12)
    inner = tf.reduce_mean(tf.math.conj(f_prev) * f_curr) * (r_max - r_min)
    loss_sum += tf.math.real(inner * tf.math.conj(inner) / (norm_curr * norm_prev)**2)

  return loss_sum