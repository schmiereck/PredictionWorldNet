# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PredictionWorldNet is a **Predictive World Model** based on Active Inference principles. The agent learns to predict what it will observe next and derives actions from minimizing prediction errors (Variational Free Energy), rather than chasing rewards. It combines a fast, low-resolution neural network with high-resolution Gemini Vision evaluations for semantic feedback.

**Language:** Python 3.11, Windows environment
**Main frameworks:** PyTorch, MiniWorld (Gymnasium), Gemini API

## Running the System

### Main Entry Point
```bash
python B19OrchestratorModeMiniworld.py
```

This starts the full system with:
- MiniWorld 3D simulation
- Live training with dashboard visualization
- Overhead map view
- Adaptive Gemini API integration

### Pretraining Pipeline (Optional)
```bash
# 1. Train VAE (Encoder + Decoder)
python B20PreTrainVAE.py --source miniworld --epochs 50

# 2. Train CLIP Goal Projection
python B21PreTrainCLIP.py --vae-checkpoint checkpoints/pwn_*.pt --epochs 60

# 3. Run live training (loads checkpoint automatically)
python B19OrchestratorModeMiniworld.py
```

### Environment Setup
- Virtual environment exists at `.venv/`
- Activate with `.venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`
- **CRITICAL:** Requires `GOOGLE_API_KEY` environment variable for Gemini API

## Architecture Overview

### Module Organization (B-prefixed files)

The codebase uses a **dual-architecture pattern**:

1. **Standalone modules** (B04b-B11): Individual components with demos, designed for 16×16 images
2. **Integrated system** (B16): Re-implements all components for 128×128 images, used at runtime

**IMPORTANT:** When making changes to neural network components:
- B16FullIntegration.py contains the **runtime implementations** (128×128)
- B04b-B09 are standalone demos (16×16) that may diverge
- B20/B21 pretraining scripts use B16's implementations
- Focus changes on B16 for production code

### Core System Flow

```
B19Orchestrator
    ├─> B16FullIntegration (Main ML System)
    │   ├─> Encoder (VAE): 128×128 RGB → 64-dim latent z
    │   ├─> Temporal Transformer: z + goal + history → 128-dim context
    │   ├─> Decoder: z → 128×128 predicted image
    │   ├─> ActionHead: context → 6D action + uncertainty (σ)
    │   └─> goal_proj: CLIP 512-dim → 64-dim latent
    ├─> B17RobotInterfaces (Abstraction for MiniWorld/ROS2/Mock)
    ├─> B13GeminiApi + B14AdaptiveGemini (Semantic evaluation)
    ├─> B22StrategyGenerator + B23StrategyExecutor (Rule-based exploration)
    ├─> B18Dashboard (Real-time visualization)
    └─> OverheadMapView (2D map with trail)
```

### Key Dimensions
- `LATENT_DIM = 64`: VAE latent space
- `D_MODEL = 128`: Transformer hidden dimension
- `ACTION_DIM = 6`: Action vector dimensions
- `OBS_SHAPE = (128, 128, 3)`: RGB observation size

### Action Space (6D Continuous)
Actions are vectors in [-1, 1]⁶ representing:
1. `linear_x`: Forward/backward velocity
2. `angular_z`: Rotation left/right
3. `camera_pan`: Camera horizontal angle
4. `camera_tilt`: Camera vertical angle
5. `arc_radius`: Curve radius (0 = straight)
6. `duration`: Action execution time

## Critical Implementation Details

### MiniWorld Coordinate System Gotchas
- **Z-axis is negated** for map display: `map_y = -env.agent.z`
- **Camera pan subtracts from heading**: `agent.dir = original_dir - cam_pan`
- Agent direction is in radians, counter-clockwise positive
- Use `agent.dir % (2π)` for normalized heading

### MiniWorld Entity Color System
- `Box.color` is a **string** (e.g., "red", "yellow"), NOT an RGB tuple
- `Ball` objects **lack** both `.color` and `.mesh_name` attributes
- Color extraction for Ball requires parsing `ObjMesh.cache` keys (filename contains color)
- Custom colors registered: `COLORS["orange"]` and `COLORS["white"]`

### Custom Environment: PredictionWorld-OneRoom-v0
Registered via `_register_pw_env()` which exists in **4 files** (B19Orchestrator, B19OrchestratorModeMiniworld, B20, B21) - this is a known code duplication issue.

Environment contains 6 objects:
- 4 colored boxes (red, yellow, white, orange)
- 2 colored balls (green, blue)

### Checkpoint Format
Checkpoints (.pt files) contain:
```python
{
    "encoder": state_dict,       # B16 Encoder
    "decoder": state_dict,       # B16 Decoder
    "goal_proj": state_dict,     # Linear(512→64) or Sequential after T09
    "total_steps": int,
    "train_steps": int,
    "beta": float,               # KL-divergence weight
    "current_goal": str,
    "constants": {"LATENT_DIM": 64, "D_MODEL": 128, "ACTION_DIM": 6},
    "result": dict,              # VAE training results
    "result_clip": dict,         # CLIP training results
}
```

Training pipeline: B20 (VAE) → B21 (CLIP) → B19 (Live)

### Reward System Architecture
Multiple reward sources combined in B15RewardCombination:
- **r_intrinsic**: Prediction error (curiosity) via B12
- **r_gemini**: Semantic evaluation from Gemini Vision API (B13/B14)
- **r_goal**: Cosine similarity in latent space to goal
- **r_action**: Action smoothness penalty

**All rewards must be in [0, 1] range** - normalization is critical (see TODO.md T01)

### Adaptive Gemini Strategy
B14AdaptiveGemini controls API call frequency based on:
- **Urgency** = 0.6×u_fe + 0.2×u_novelty + 0.2×u_timeout
- Interval varies from 5 steps (high urgency) to 80 steps (low urgency)
- Achieves 8-16× cost reduction compared to fixed intervals
- **Language**: Gemini responses are in German, but CLIP goals must be in English

### Strategy Blending System
B23StrategyExecutor implements sigma-based blending:
```python
blend_factor = sigmoid((mean_sigma - 0.4) * 8.0)
final_action = blend_factor * strategy_action + (1 - blend_factor) * nn_action
```
- High uncertainty (σ > 0.4) → rule-based strategy dominates (early training)
- Low uncertainty → neural network takes over (after learning)

### Loss Function (Variational Free Energy)
```python
FE = L_recon          # Reconstruction error (MSE + SSIM)
   + β * L_KL         # KL-divergence (annealed 0 → 0.05)
   + 0.3 * L_temporal # Next-latent prediction
   + 0.2 * L_action   # Action prediction
   + 0.1 * L_goal     # Goal alignment
   + 0.05 * L_sigma   # Uncertainty calibration (NLL)
   + 0.2 * L_gemini   # Gemini-weighted reconstruction
```

Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
Scheduler: ReduceLROnPlateau (factor=0.5, patience=80)
Gradient clipping: max_norm=1.0

### Dashboard Update Signature
```python
dashboard.update(
    obs=np.ndarray,           # (128,128,3) or (60,80,3)
    pred=np.ndarray,          # (128,128,3) Predicted frame
    metrics=dict,             # Keys: fe, recon, kl, r_intrinsic, r_gemini,
                              #       r_total, goal_progress, beta, lr
    gemini_event=dict|None,   # reward, situation, recommendation, image
    goal=str,
    scene=str,
    latent_z=np.ndarray|None, # (64,) for PCA visualization
)
```

## Development Guidelines

### When Modifying Neural Network Architecture
1. **Primary target**: Edit B16FullIntegration.py (runtime system)
2. **Secondary**: Update standalone demos (B04b-B09) if they diverge significantly
3. **Test pretraining**: Verify B20/B21 still work after changes
4. **Checkpoint compatibility**: Add try/except for loading old checkpoints

### When Adding New Features
- Follow the B-prefix naming convention (B00-B23 are taken)
- MiniWorld modifications may require changes to B17RobotInterfaces
- Visualization changes go in B18Dashboard or OverheadMapView
- New reward components should integrate via B15RewardCombination

### Known Issues and TODOs
See TODO.md for prioritized improvement list. Key completed items:
- ✅ T01: Reward normalization (all rewards now [0, 1])
- ✅ T03: Transformer loss now predicts next latent (task-aligned)
- ✅ T04: Sigma loss uses NLL (calibrated uncertainty)
- ✅ T05: Cosine beta annealing (prevents KL collapse)
- ✅ T06: SSIM perceptual loss added
- ✅ T07: HSV color detection (robust to lighting)
- ✅ T09: Two-stage CLIP projection (512→128→64)

Outstanding issues:
- `_register_pw_env()` duplicated across 4 files (needs refactoring)
- B04b-B09 standalone demos diverge from B16 implementations

### Testing Changes
- Run standalone demos to verify component behavior: `python B04bVariationalEncoder.py`
- Test full pipeline: `python B19OrchestratorModeMiniworld.py`
- Monitor dashboard for training stability (KL divergence, reconstruction quality)
- Watch for KL-collapse warning (KL < 0.1 nats after step 50)

## Reference Documentation

- `README.md`: Detailed architecture diagrams and theory
- `AGENTS.md`: Technical reference for AI agents (detailed module breakdown)
- `TODO.md`: Improvement roadmap with status tracking
- `doc/LOGBOOK.md`: Development history and decisions
