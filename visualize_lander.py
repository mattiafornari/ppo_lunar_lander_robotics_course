import gymnasium as gym
from stable_baselines3 import PPO

# 1. Carichiamo l'ambiente con la grafica ATTIVA
# Nota: usa "LunarLanderContinuous-v3" come abbiamo corretto prima
env = gym.make("LunarLanderContinuous-v3", render_mode="human")

# 2. Carichiamo il cervello addestrato (assicurati che il file .zip esista!)
try:
    model = PPO.load("ppo_lunar_lander", device="cpu")
    print("Modello caricato con successo!")
except:
    print("ERRORE: Non trovo il file 'ppo_lunar_lander.zip'. Hai fatto il training?")
    exit()

# 3. Loop di visualizzazione
episodes = 5  # Quanti atterraggi vuoi vedere
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    print(f"Inizio episodio {ep + 1}...")
    
    while not (done or truncated):
        # Chiediamo al modello cosa fare
        action, _ = model.predict(obs, deterministic=True)
        
        # Eseguiamo l'azione (la grafica si aggiorna da sola)
        obs, reward, done, truncated, info = env.step(action)

env.close()
