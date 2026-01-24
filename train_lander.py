import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# 1. Configurazione Ambiente (MDP)
env_name = "LunarLanderContinuous-v3"
env = gym.make(env_name, render_mode=None)

# 2. Definizione Agente (Policy)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99, #0.99
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01, # 0.01
    device="cpu",
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# 3. Training Loop
print("Inizio Training...")
model.learn(total_timesteps=2000000)
print("Training completato.")
model.save("ppo_lunar_lander")

# 4. Validazione e Grafici
eval_env = gym.make(env_name, render_mode=None)

traj_x = []
traj_y = []
traj_vy = [] # Lista separata per le velocità per episodio

print("Inizio Valutazione...")
for i in range(5): # Test su 3 episodi
    obs, _ = eval_env.reset()
    done = False
    truncated = False

    episode_x = []
    episode_y = []
    episode_vy = []

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = eval_env.step(action)
        episode_x.append(obs[0])
        episode_y.append(obs[1])
        episode_vy.append(obs[3])

    traj_x.append(episode_x)
    traj_y.append(episode_y)
    traj_vy.append(episode_vy)

env.close()
eval_env.close()

# --- PLOTTING ---

# Grafico 1: Traiettoria Spaziale
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(zip(traj_x, traj_y)):
    plt.plot(x, y, label=f'Episodio {i+1}', alpha=0.7)
    # Segna il punto di atterraggio finale
    plt.scatter(x[-1], y[-1], c='red', s=20, zorder=5)

plt.title("Traiettorie dell'Agente (Motion Planning Implicito)")
plt.xlabel("Posizione X (0 = Landing Pad)")
plt.ylabel("Posizione Y")
plt.axhline(0, color='black', linestyle='--', linewidth=1) # Il suolo
plt.axvline(-0.2, color='gray', linestyle=':', alpha=0.5) # Limiti pad
plt.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("trajectory_plot.png")
print("Grafico traiettoria salvato.")

# Grafico 2: Spazio delle Fasi (Altezza vs Velocità Verticale)
plt.figure(figsize=(10, 6))
for i, (y, vy) in enumerate(zip(traj_y, traj_vy)):
    plt.scatter(y, vy, s=3, alpha=0.5, label=f'Episodio {i+1}')

plt.title("Spazio delle Fasi: Controllo Velocità di Discesa")
plt.xlabel("Altezza (Y)")
plt.ylabel("Velocità Verticale (Vy)")
plt.axhline(0, color='red', linestyle='--', label="Velocità Zero")
plt.gca().invert_xaxis() # Opzionale: per vedere la discesa da destra a sinistra
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("phase_portrait.png")
print("Grafico fase salvato.")