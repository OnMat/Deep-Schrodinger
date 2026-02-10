import os
import numpy as np
from D1.graphics.plot import graph_1d
from D1.metodo_01.morse_1d_h2.losses import L1_1d, L2_1d, L3_1d, Laplaciano_1d, Var_H_1d, energy_expectation_1d, monte_carlo_integral_1d, overlap_squared_1d, raio_sampler
from schordinger_DGM import DGMNet1D

import tensorflow as tf

num_layers = 3
nodes_per_layer = 100
steps_per_sample = 10
dtype = tf.float64
initial_lr = 1e-03
final_lr = 1e-04
sampling_stages = 5000
tolerancia = 1e-5
clip_grad = 1.0


orthogonal_models = []
model_1D_orthogonal_1 = DGMNet1D(layer_width=nodes_per_layer, n_layers=num_layers, input_dim=1, final_trans=None)

model_1D_orthogonal_1.build(input_shape=(None, 1))

model_1D_orthogonal_1.load_weights('../src/D1/metodo_01/morse_1d_h2/modelos/psi_0.keras')
print("modelo carregado com sucesso!")
graph_1d(model_1D_orthogonal_1)

orthogonal_models.append(model_1D_orthogonal_1)



#===========================================================================================================================================================
# Treino 
#===========================================================================================================================================================


model_1D = DGMNet1D(layer_width=nodes_per_layer, n_layers=num_layers, input_dim=1, final_trans=None)

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=clip_grad)



use_mini_batches = False

if use_mini_batches:
    mc_total_int = 10000
    mc_total_b   = 5000
    mc_total_ext = 5000
    mc_iters     = 20
    mc_batch_int = mc_total_int // mc_iters
    mc_batch_b   = mc_total_b   // mc_iters
    mc_batch_ext = mc_total_ext // mc_iters
else:
    mc_batch = 10000

total_steps = sampling_stages * steps_per_sample


use_L5 = False
find_aprox = False
fix_var = 0.0
use_ema = False
ema_decay = 0.999
ema_weights = [tf.Variable(w, trainable=False) for w in model_1D.get_weights()]

def update_ema(model):
    for w, ew in zip(model.get_weights(), ema_weights):
        ew.assign(ema_decay * ew + (1 - ema_decay) * w)

def apply_ema():
    return [ew.numpy() for ew in ema_weights]

for stage in range(sampling_stages):


    x_all = raio_sampler(0.001, 2, 8000)
    x_border = raio_sampler(1.50, 2.0, 2500)
    x_border2 = raio_sampler( 0.001, 0.5,2500)
    x_interior = raio_sampler(0.49, 2.99, 17000)
    lap, f = Laplaciano_1d(model_1D,  x_all)
    E = energy_expectation_1d(lap, f, x_all)


    # Define learning rate por etapa
    global_step = stage * steps_per_sample
    cosine_lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * global_step / total_steps))
    optimizer.learning_rate.assign(cosine_lr)

    # Forward + backward direto
    with tf.GradientTape() as tape:
      L1_val = L1_1d(model_1D,  x_interior,E)
      L2_val = L2_1d(model_1D,x_border, epsilon=1e-4)
      L3_val = L3_1d(model_1D,x_all)
      L4_val = Var_H_1d(model_1D, x_all)
      L5_val = L2_1d(model_1D,x_border2, epsilon=1e-4)
      L6_val = overlap_squared_1d(model_1D,orthogonal_models,x_all)
      loss_val = L1_val + (L2_val + L3_val + L4_val + L5_val + L6_val)

    grads = tape.gradient(loss_val, model_1D.trainable_variables)
    optimizer.apply_gradients(zip(grads, model_1D.trainable_variables))

    # EMA
    if use_ema:
        update_ema(model_1D)

    # Avaliação
    if stage % 500 == 0:
      lap, f = Laplaciano_1d(model_1D,  x_all)
      E = energy_expectation_1d(lap, f,  x_all).numpy()
      norm = monte_carlo_integral_1d(f).numpy()
      save_path = "seuDiretorio/DGMNet_treinado_com_eV_1d.keras"
      model_1D.save(save_path)
      print("Modelo Salvo")

      print(f"Stage {stage} | Loss: {loss_val:.6f} | L1: {L1_val:.6f} "
            f"L2: {L2_val:.6f} |L3: {L3_val:.6f} | L4: {L4_val:.6f} | L5: {L5_val:.6f} "
            f"|L6: {L6_val:.6f}|| <E>: {E:.6f} | Norm: {norm:.6f} | LR: {cosine_lr:.2e}")
      graph_1d(model_1D)
    if stage % 100 == 0 and stage % 500 != 0:
      lap, f = Laplaciano_1d(model_1D,  x_all)
      E = energy_expectation_1d(lap, f,  x_all).numpy()
      norm = monte_carlo_integral_1d(f).numpy()
      print(f"Stage {stage} | Loss: {loss_val:.6f} | L1: {L1_val:.6f} "
            f"L2: {L2_val:.6f} |L3: {L3_val:.6f} | L4: {L4_val:.6f} | L5: {L5_val:.6f} "
            f"|L6: {L6_val:.6f}|| <E>: {E:.6f} | Norm: {norm:.6f} | LR: {cosine_lr:.2e}")

    # Interrompe se perda estiver abaixo da tolerância
    if loss_val < tolerancia:
        print(f"Treinamento interrompido: loss {loss_val:.6f} < tolerância {tolerancia}")
        model_1D.save(save_path)
        break

lap, f = Laplaciano_1d(model_1D,  x_all)
E = energy_expectation_1d(lap, f,  x_all).numpy()
norm = monte_carlo_integral_1d(f).numpy()
print(f"Stage {stage} | Loss: {loss_val:.6f} | L1: {L1_val:.6f} "
            f"L2: {L2_val:.6f} |L3: {L3_val:.6f} | L4: {L4_val:.6f} | L5: {L5_val:.6f} "
            f"|L6: {L6_val:.6f}|| <E>: {E:.6f} | Norm: {norm:.6f} | LR: {cosine_lr:.2e}")
graph_1d(model_1D)
# Salva modelo final
model_1D.save(save_path)
print("Modelo Salvo no fim")

# Salva checkpoint (modelo + otimizador)
ckpt = tf.train.Checkpoint(model=model_1D, optimizer=optimizer)
ckpt_dir = "seuDiretorio/checkpoints/"
ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
ckpt.save(file_prefix=ckpt_prefix)
print("Checkpoint salvo com modelo + otimizador, se for o caso, redefina o lr e contine o treinamento a partir do checkpoint salvo anteriormente")
