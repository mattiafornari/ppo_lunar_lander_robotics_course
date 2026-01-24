# Documentazione Codice

## train_lander.py

### Panoramica

Script principale per l'addestramento dell'agente PPO sul task LunarLanderContinuous-v3.

### Flusso Esecuzione

1. Configurazione ambiente MDP
2. Inizializzazione agente PPO
3. Ciclo di addestramento
4. Validazione e generazione grafici

### Sezione 1: Configurazione Ambiente

```python
env_name = "LunarLanderContinuous-v3"
env = gym.make(env_name, render_mode=None)
```

**Spiegazione:**
- `LunarLanderContinuous-v3`: Versione con azioni continue (2D: main engine, lateral thrusters)
- `render_mode=None`: Disabilita rendering grafico durante training (maggiore velocità)

**MDP Definition:**
- Stati (8D): posizione (x,y), velocità (vx,vy), angolo, velocità angolare, contatti gambe
- Azioni (2D): potenza motore principale [0,1], controllo laterale [-1,1]
- Ricompensa: funzione di shaping basata su distanza, velocità, carburante

### Sezione 2: Definizione Agente

```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device="cpu",
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)
```

**Analisi Parametri:**

- **MlpPolicy**: Policy basata su reti neurali fully-connected. Architettura default di Stable-Baselines3 con strati [64, 64].

- **learning_rate=3e-4**: Tasso di apprendimento standard per PPO. Controlla l'entità degli aggiornamenti dei pesi.

- **n_steps=2048**: Numero di step raccolti da ogni ambiente prima di un aggiornamento della policy. Bilancia esplorazione e frequenza aggiornamenti.

- **batch_size=64**: Dimensione del minibatch per SGD. Determina quanti campioni vengono usati per calcolare ogni gradiente.

- **n_epochs=10**: Numero di passate complete sui dati raccolti. Più epoche = più ottimizzazione ma rischio overfitting.

- **gamma=0.99**: Fattore di sconto per ricompense future. Formula: R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
  Valore 0.99 significa che ricompense a 100 step hanno ancora peso circa 37%.

- **gae_lambda=0.95**: Parametro per Generalized Advantage Estimation. Controlla bias-variance tradeoff nella stima del vantaggio.

- **clip_range=0.2**: Parametro caratteristico di PPO. Limita l'aggiornamento della policy entro ±20% della vecchia distribuzione. Previene aggiornamenti distruttivi.

- **ent_coef=0.01**: Coefficiente del bonus entropia. Incoraggia esplorazione penalizzando policy troppo deterministiche. Valore maggiore rispetto al default (0.001) per favorire esplorazione iniziale.

- **device="cpu"**: Dispositivo per computazione. Per reti piccole come questa, CPU è sufficiente.

- **verbose=1**: Stampa informazioni di training ogni aggiornamento.

- **tensorboard_log**: Directory per log TensorBoard.

### Sezione 3: Training Loop

```python
model.learn(total_timesteps=2000000)
model.save("ppo_lunar_lander")
```

**Meccanismo:**
1. Raccolta di 2048 step di esperienza
2. Calcolo vantaggi con GAE
3. 10 epoche di ottimizzazione SGD
4. Ripeti fino a 2M timesteps totali

**Tempo stimato:** 60-90 minuti su CPU moderna (Intel i7/AMD Ryzen).

**Output:** File `ppo_lunar_lander.zip` contenente:
- Pesi della rete neurale
- Parametri di normalizzazione (se applicati)
- Configurazione algoritmo

### Sezione 4: Validazione

```python
eval_env = gym.make(env_name, render_mode=None)

traj_x = []
traj_y = []
traj_vy = []

for i in range(5):
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
```

**Logica:**
- 5 episodi di test con policy deterministica
- Raccolta posizioni (x,y) e velocità verticale (vy)
- `deterministic=True`: usa media della distribuzione invece di campionare

**Variabili stato osservate:**
- `obs[0]`: posizione x (0 = landing pad)
- `obs[1]`: altitudine y
- `obs[3]`: velocità verticale vy

### Generazione Grafici

#### Grafico 1: Traiettorie Spaziali

```python
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(zip(traj_x, traj_y)):
    plt.plot(x, y, label=f'Episodio {i+1}', alpha=0.7)
    plt.scatter(x[-1], y[-1], c='red', s=20, zorder=5)

plt.title("Traiettorie dell'Agente (Motion Planning Implicito)")
plt.xlabel("Posizione X (0 = Landing Pad)")
plt.ylabel("Posizione Y")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.axvline(-0.2, color='gray', linestyle=':', alpha=0.5)
plt.axvline(0.2, color='gray', linestyle=':', alpha=0.5)
```

**Interpretazione:**
- Linee: traiettoria completa dell'agente
- Punti rossi: punto di atterraggio finale
- Linee verticali grigie: confini landing pad
- Asse x=0: centro landing pad

**Cosa cercare:**
- Convergenza verso x=0
- Approccio controllato senza oscillazioni eccessive
- Atterraggio all'interno del pad (-0.2 < x < 0.2)

#### Grafico 2: Spazio delle Fasi

```python
plt.figure(figsize=(10, 6))
for i, (y, vy) in enumerate(zip(traj_y, traj_vy)):
    plt.scatter(y, vy, s=3, alpha=0.5, label=f'Episodio {i+1}')

plt.title("Spazio delle Fasi: Controllo Velocità di Discesa")
plt.xlabel("Altezza (Y)")
plt.ylabel("Velocità Verticale (Vy)")
plt.axhline(0, color='red', linestyle='--', label="Velocità Zero")
plt.gca().invert_xaxis()
```

**Interpretazione fisica:**
- Ogni punto rappresenta uno stato (altezza, velocità verticale)
- La traiettoria nello spazio delle fasi mostra la strategia di controllo
- Linea rossa: velocità zero (hover o atterraggio)

**Strategia ottimale:**
- Ad alta quota: velocità di discesa moderata (vy negativa)
- In prossimità del suolo: decelerazione progressiva
- All'atterraggio: vy ≈ 0

**Pattern non desiderati:**
- Discesa troppo rapida (vy molto negativa vicino a y=0)
- Oscillazioni ampie della velocità
- Mancata decelerazione prima dell'impatto

---

## visualize_lander.py

### Panoramica

Script per visualizzazione grafica dell'agente addestrato.

### Struttura Codice

#### Caricamento Ambiente

```python
env = gym.make("LunarLanderContinuous-v3", render_mode="human")
```

**Differenza con training:**
- `render_mode="human"`: Abilita rendering grafico real-time
- Ogni step viene visualizzato a schermo

#### Caricamento Modello

```python
try:
    model = PPO.load("ppo_lunar_lander", device="cpu")
    print("Modello caricato con successo!")
except:
    print("ERRORE: Non trovo il file 'ppo_lunar_lander.zip'.")
    exit()
```

**Gestione errori:**
- Verifica esistenza file modello
- Notifica utente se file mancante
- Termina execution in modo pulito

#### Loop Visualizzazione

```python
episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

env.close()
```

**Comportamento:**
1. Reset ambiente per nuovo episodio
2. Policy deterministica per comportamento consistente
3. Rendering automatico ad ogni step
4. Episodio termina quando done=True o truncated=True
5. Chiusura pulita dell'ambiente

**Condizioni terminazione:**
- `done=True`: Lander atterrato o crashato
- `truncated=True`: Limite timesteps raggiunto (1000 default)

### Note Tecniche

**Performance:**
- Rendering rallenta esecuzione a circa 30 FPS
- Ogni episodio dura 5-30 secondi reali
- Non adatto per valutazione quantitativa massiva

**Requisiti sistema:**
- Display grafico (X11 su Linux, nativo su Windows/macOS)
- PyGame installato (incluso in gymnasium[box2d])