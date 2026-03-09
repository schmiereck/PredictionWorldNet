# PredictionWorldNet

## Project Overview
PredictionWorldNet is a Predictive World Model based on the principles of **Active Inference**. It serves as an alternative to classical Reinforcement Learning by having the agent actively anticipate reality and minimize its "Variational Free Energy" (prediction error) rather than passively chasing rewards.

The architecture combines:
- A local generative neural network (VAE Encoder, CNN Decoder, RSSM/GRU-based World Model) for fast, low-resolution processing and "planning-as-inference" (imagination).
- High-resolution semantic evaluations from the **Gemini Vision API** acting as a high-level reward signal and strategy generator.
- A **CLIP** Text-Encoder for translating natural language goals into the latent space.
- A simulated 3D environment using **MiniWorld**.

## Building and Running

### Prerequisites
- Python 3.11+
- Set the `GOOGLE_API_KEY` environment variable for Gemini API access.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
*(Note: Windows users may also need to install `pyopengl` if MiniWorld OpenGL errors occur).*

### Running the System
The main entry point for running the live simulation and training loop in the MiniWorld environment is:
```bash
python B19OrchestratorModeMiniworld.py
```

### Pre-training (Optional)
The models can be pre-trained offline before running the live simulation to significantly speed up convergence:
1. **VAE (Image Compression):**
   ```bash
   python B20PreTrainVAE.py --source miniworld --epochs 50
   ```
2. **CLIP Goal-Projection:**
   ```bash
   python B21PreTrainCLIP.py --vae-checkpoint checkpoints/pwn_*.pt --epochs 60
   ```
3. **Dynamics-Head (State Transitions):**
   ```bash
   python B24PreTrainDynamics.py --checkpoint checkpoints/pwn_*.pt --epochs 30
   ```

## Development Conventions & Architecture
- **Active Inference & EFE:** The system optimizes for Expected Free Energy (EFE). Actions are selected to balance epistemic behavior (reducing uncertainty/exploring) and pragmatic behavior (achieving semantic goals).
- **Neural Network Components:**
  - *Encoder:* Compresses RGB images into a 256-dim latent representation. Uses GroupNorm instead of BatchNorm to prevent train/eval drift during online learning.
  - *Decoder:* Reconstructs predicted images from the latent space.
  - *RSSM (GRU):* Recurrent State-Space Model for temporal dynamics and state prediction.
  - *Action Head:* Outputs continuous 6D actions (linear, angular, pan, tilt, arc, duration) and an uncertainty metric ($\sigma$).
- **Gemini Integration:** The system adaptively queries the Gemini API (`gemini-robotics-er` and `gemini-2.5-flash`) based on a calculated urgency metric (e.g., high prediction error). This minimizes API costs while acquiring essential semantic rewards and exploration strategies.
- **Visualization:** Uses `matplotlib` for real-time tracking.
  - `B18Dashboard.py`: A comprehensive real-time dashboard showing camera views, predictions, loss curves, rewards, and latent space PCA. (Note: High CPU load is expected when maximizing this window due to software rendering in `TkAgg`).
  - `OverheadMapView.py`: A 2D overhead map displaying the agent's trajectory, camera FOV, and semantic Gemini call locations.

### Key Files
- `B16FullIntegration.py`: The core ML system containing the Encoder, Decoder, RSSM, ActionHead, and Imagination rollout logic.
- `B19Orchestrator.py` & `B19OrchestratorModeMiniworld.py`: The central orchestrators that tie together the ML models, the environment, and the visualizers.
- `B10PredictionLoss.py` & `B11TrainingLoop.py`: Implementations of the objective functions (MSE, SSIM, KL divergence) and the online learning loop.
