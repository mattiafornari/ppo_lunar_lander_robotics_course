# PPO Lunar Lander Training

Robotics course project: application of the PPO (Proximal Policy Optimization) algorithm to solve the continuous control task in Gymnasium LunarLanderContinuous-v3.

![Demo](ppo_lunar_lander_demo.gif)

## Project Structure

```
ppo_lunar_lander_robotics_course/
├── train_lander.py           # Training script
├── visualize_lander.py       # Trained agent visualization
├── requirements.txt          # Python dependencies
├── ppo_lunar_lander_demo.gif # Video Demo

├── LICENSE.md                # MIT License
├── .gitignore
├── .gitattributes

├── README.md             
```

## Requirements

- Operating System:
  - Linux (recommended)
  - Windows with WSL (Windows Subsystem for Linux)
- Python 3.10 or higher
- pip 

## Installation

### Option 1: Local Installation

```bash
git clone [https://github.com/mattiafornari/ppo_lunar_lander_robotics_course.git](https://github.com/mattiafornari/ppo_lunar_lander_robotics_course.git)
cd ppo_lunar_lander_robotics_course

sudo apt update && sudo apt install -y python3-pip python3-venv swig
python3 -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
```

## Usage

### Training

**Local:**
```bash
python train_lander.py
```

The training takes about 20/30 minutes (variable) on a modern CPU and generates:
- File `ppo_lunar_lander.zip` containing `policy.optmizer.pth`, `policy.optmizer.pth`, `pytorch\_variables.pth` and others according to Stable-Baselines3.

The script also includes a brief testing and evaluation phase.
Specifically, it runs inference on the trained model over 5 test episodes. The agent is executed in deterministic mode (i.e. without exploration) to verify the stability of the learned trajectory and its generalization capability. Data Logging: collects telemetry (X/Y coordinates, velocity) to generate the following plots:
- Trajectories plot: `trajectory_plot.png`
- Phase space plot: `phase_portrait.png`

Note that using an Intel Core i7-14700 the training took about 20 minutes.

### Monitoring with TensorBoard [OPTIONAL]

**Local:**
```bash
tensorboard --logdir tensorboard_logs/
```
Open your browser at `http://localhost:6006` to view the training progress.

### Visualization

After training, you can visualize the agent in action:

**Local:**
```bash
python visualize_lander.py
```
## Training Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| Algorithm | PPO | Proximal Policy Optimization |
| Learning Rate | 3e-4 | Learning Rate |
| Steps per Update | 2048 | Steps collected before each policy update |
| Batch Size | 64 | Minibatch size for SGD |
| Epochs | 10 | Number of epochs |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | GAE parameter for advantage estimation |
| Clip Range | 0.2 | PPO clipping range |
| Entropy Coefficient | 0.01 | Entropy coefficient |
| Total Timesteps | 2,000,000 | Total training steps |

## Network Architecture

- Type: Multi-Layer Perceptron (MLP) Stable-Baselines3 default
- Hidden layers: [64, 64] (Stable-Baselines3 default)
- Activation function: tanhObservation space: 8 dimensions
- Action space: 2 dimensions (continuous)
- Input: 8-D state vector
- Output: A vector $\mu \in \mathbb{R}^2$ representing the means of the multivariate Gaussian distribution (from which actions are sampled) for the two continuous control commands (main engine thrust, side engine thrust). A scalar $V(s) \in \mathbb{R}$ estimating the value of the current state, i.e., the expected return (discounted sum of future rewards), used to calculate the Advantage during training.

Note that this configuration is a baseline: MLP [64,64] with Tanh. Significant improvements are possible through hyperparameter tuning (e.g., learning rate, ent_coef, n_steps, batch_size, observation normalization, etc.) and through policy redesign (e.g., ReLU/Swish, deeper or wider networks, policy/value separation, etc.).

## Metrics 

- Average reward > 200: Environment solved (see the site https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Expected reward: 250-280 after convergence
- Landing speed < 0.5 m/s
- Landing within the landing pad

## Output Analysis

### Trajectories plot

The plot shows the agent's spatial trajectories over 5 validation episodes. A well-trained agent exhibits:

- Controlled approach to the landing pad (x ≈ 0)
- Gradual altitude reduction
- Landing points concentrated near the center

### Phase space plot

The plot represents the height-vertical velocity relationship (y-vy). Expected behavior:

- Moderate descent speed at high altitudes
- Progressive deceleration near the ground and thus the pad
- Velocity close to zero upon landing

## Troubleshooting

### Error: Box2D not instlled

```bash
# Ubuntu/Debian
sudo apt-get install swig
pip install gymnasium[box2d]

# macOS
brew install swig
pip install gymnasium[box2d]
```

### Error: Model not found in visualize_lander.py

Verify that `ppo_lunar_lander.zip` exists in the current directory.
The mentioned zip folder is created by `train_lander.py` when the training finishes.

### TensorBoard shows empty plots

Wait a few minutes after starting the training. The first data is written after about 2048 steps. Please note that TensorBoard updates every 30 seconds by default.

## References

- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Stable-Baselines3: [Documentazione](https://stable-baselines3.readthedocs.io/)
- Gymnasium: [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Authors

- **Mattia Fornari** - [@mattiafornari](https://github.com/mattiafornari)
- **Luca Pugnetti** - [@luca-pugnetti](https://github.com/luca-pugnetti)

## Licenza

Distributed under the MIT License. See the `LICENSE` file for more details.

## Contact

For bug reports or questions, please open an issue on GitHub or contact via email.
