import gymnasium as gym
from stable_baselines3 import PPO

# 1. Caricamento dell'ambiente con grafica attiva "LunarLanderContinuous-v3"
env = gym.make("LunarLanderContinuous-v3", render_mode="human")

# 2. Caricamento del modello/policy addestrato 
# (assicurarsi che il file .zip esista - i.e. eseguire prima il training)
try:
    model = PPO.load("ppo_lunar_lander", device="cpu")
    print("Modello caricato con successo!")
except:
    print("ERRORE: 'ppo_lunar_lander.zip'. Eseguire il training se non ancora effettuato!")
    exit()

# 3. Loop di visualizzazione - Inferenza
episodes = 5  # cambiare per visualiuzzare più atterraggi
for ep in range(episodes):
    obs, _ = env.reset() # imposto seed=42 o altro numero per debugging e riproducibilità
    done = False
    truncated = False
    print(f"Inizio episodio {ep + 1}...")
    
    while not (done or truncated):
        # Calcolo l'azione "migliore"  disabilitando la stocasticità/esplorazione ==> determinismo
        action, _ = model.predict(obs, deterministic=True) # 
        
        # Eseguo l'azione - la grafica si aggiorna autonomamente
        obs, reward, done, truncated, info = env.step(action)

env.close()
