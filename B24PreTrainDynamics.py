"""
B24 – Pre-Training Dynamics Head
==================================
Trainiert den dynamics_head (T10) offline auf (obs, action, next_obs)-Tripeln.
Macht das Transitions-Modell P(z_{t+1} | z_t, a_t) schneller konvergieren,
bevor das Live-Training beginnt.

Voraussetzung:
    B20 + B21 Checkpoint (Encoder vortrainiert, LATENT_DIM=256).

Ablauf:
    1. Sammle N Episoden in MiniWorld (Mischung aus zufälligen + gerichteten Aktionen)
    2. Speichere (obs, action, next_obs) Tripel im Speicher
    3. Lade Encoder aus Checkpoint (wird eingefroren – Repräsentation bleibt stabil)
    4. Trainiere dynamics_head: cat([z_cur, action]) → z_next
       Hinweis: Funktioniert weil LATENT_DIM == D_MODEL == 256 (T16),
       d.h. z_cur kann direkt als Context-Substitute übergeben werden.
    5. Speichere aktualisierten Checkpoint (nur dynamics_head überschrieben)

Nutzung:
    python B24PreTrainDynamics.py
    python B24PreTrainDynamics.py --checkpoint checkpoints/pwn_*.pt --episodes 500
    python B24PreTrainDynamics.py --episodes 200 --epochs 30 --lr 5e-4
"""

import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import importlib.util


# ─────────────────────────────────────────────
# B16 laden (Encoder, TemporalTransformer, Konstanten)
# ─────────────────────────────────────────────

def _load_module(filename: str):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(pattern: str) -> str:
    import glob as g
    matches = sorted(g.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Kein Checkpoint gefunden: {pattern}")
    return matches[-1]


_b16 = _load_module("B16FullIntegration.py")
Encoder            = _b16.Encoder
TemporalTransformer = _b16.TemporalTransformer
LATENT_DIM         = _b16.LATENT_DIM
D_MODEL            = _b16.D_MODEL
ACTION_DIM         = _b16.ACTION_DIM


def _register_prediction_world_env(gym):
    """Registriert PredictionWorld-OneRoom (falls noch nicht geschehen)."""
    env_id = "PredictionWorld-OneRoom-v0"
    if env_id in gym.envs.registry:
        return
    from miniworld.envs.oneroom import OneRoom
    from miniworld.entity import Box, Ball, COLORS, COLOR_NAMES
    if "orange" not in COLORS:
        COLORS["orange"] = np.array([1.0, 0.5, 0.0])
    if "white" not in COLORS:
        COLORS["white"] = np.array([1.0, 1.0, 1.0])
    for c in ("orange", "white"):
        if c not in COLOR_NAMES:
            COLOR_NAMES.append(c)

    class PredictionWorldRoom(OneRoom):
        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            self.box = self.place_entity(Box(color="red"))   # OneRoom.step() braucht self.box
            self.place_entity(Box(color="yellow"))
            self.place_entity(Box(color="white"))
            self.place_entity(Box(color="orange"))
            self.place_entity(Ball(color="green"))
            self.place_entity(Ball(color="blue"))
            self.place_agent()

    gym.register(id=env_id,
                 entry_point=lambda **kw: PredictionWorldRoom(**kw),
                 max_episode_steps=300)


# ─────────────────────────────────────────────
# DATEN SAMMELN
# ─────────────────────────────────────────────

# MiniWorld hat 3 diskrete Aktionen: 0=turn_left, 1=turn_right, 2=move_forward
# Wir mappen sie auf 6D-Aktionsvektoren (wie B16 sie erwartet)
_MW_ACTION_VECS = {
    0: np.array([ 0.0,  0.7,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),  # turn_left
    1: np.array([ 0.0, -0.7,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),  # turn_right
    2: np.array([ 0.8,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),  # move_forward
}


def collect_transitions(n_episodes: int = 300,
                        steps_per_episode: int = 30,
                        env_name: str = "PredictionWorld-OneRoom-v0"):
    """
    Sammelt (obs, action_6d, next_obs) Tripel aus MiniWorld.

    Aktionsmix pro Episode:
        - 60% move_forward  (häufig um Zustandsübergänge mit Tiefenwechsel zu lernen)
        - 20% turn_left
        - 20% turn_right
    """
    obs_list      = []
    action_list   = []
    next_obs_list = []

    try:
        import gymnasium as gym
        import miniworld  # noqa: F401
        from PIL import Image as PILImage

        _register_prediction_world_env(gym)
        env = gym.make(env_name, render_mode="rgb_array", view="agent")

        print(f"  Sammle Transitionen: {n_episodes} Episoden × {steps_per_episode} Steps")
        print(f"  = max {n_episodes * steps_per_episode} Tripel")

        total = 0
        for ep in range(n_episodes):
            obs_raw, _ = env.reset()

            for _ in range(steps_per_episode):
                # Aktionsmix: 60% forward, 20% left, 20% right
                r = np.random.random()
                if r < 0.60:
                    mw_action = 2
                elif r < 0.80:
                    mw_action = 0
                else:
                    mw_action = 1

                next_obs_raw, _, terminated, truncated, _ = env.step(mw_action)

                # Auf 128×128 skalieren
                obs_img = np.array(
                    PILImage.fromarray(obs_raw).resize((128, 128), PILImage.BILINEAR),
                    dtype=np.uint8
                )
                next_img = np.array(
                    PILImage.fromarray(next_obs_raw).resize((128, 128), PILImage.BILINEAR),
                    dtype=np.uint8
                )

                obs_list.append(obs_img)
                action_list.append(_MW_ACTION_VECS[mw_action].copy())
                next_obs_list.append(next_img)
                total += 1

                if terminated or truncated:
                    obs_raw, _ = env.reset()
                    break
                else:
                    obs_raw = next_obs_raw

            if (ep + 1) % 50 == 0:
                print(f"    Episode {ep+1}/{n_episodes}  |  {total} Tripel gesammelt")

        env.close()
        print(f"  Gesammelt: {total} Tripel ✓")

    except ImportError as e:
        print(f"  MiniWorld nicht verfügbar ({e}) → Fallback auf Zufalls-Tripel")
        obs_list, action_list, next_obs_list = _mock_transitions(
            n_episodes * steps_per_episode
        )

    return (np.stack(obs_list),
            np.stack(action_list),
            np.stack(next_obs_list))


def _mock_transitions(n: int):
    """Fallback: zufällige Bilder + Aktionen (nur für Smoke-Tests)."""
    obs  = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(n)]
    acts = [_MW_ACTION_VECS[np.random.randint(0, 3)].copy() for _ in range(n)]
    nxt  = [np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8) for _ in range(n)]
    return obs, acts, nxt


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class TransitionDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray, next_obs: np.ndarray):
        self.obs      = obs       # (N, 128, 128, 3) uint8
        self.actions  = actions   # (N, 6)  float32
        self.next_obs = next_obs  # (N, 128, 128, 3) uint8

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        o  = torch.from_numpy(self.obs[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        a  = torch.from_numpy(self.actions[idx])
        no = torch.from_numpy(self.next_obs[idx].astype(np.float32) / 255.0).permute(2, 0, 1)
        return o, a, no


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def pretrain_dynamics(
        encoder:     Encoder,
        transformer: TemporalTransformer,
        dataset:     TransitionDataset,
        epochs:      int   = 40,
        batch_size:  int   = 32,
        lr:          float = 3e-4,
):
    """
    Trainiert dynamics_head offline auf Transitions-Tripeln.

    Encoder wird eingefroren — nur dynamics_head wird trainiert.
    Möglich weil LATENT_DIM == D_MODEL (256): z_cur dient direkt
    als Context-Substitute für den dynamics_head-Input.

    Loss: MSE(dynamics_head(cat([z_cur, action])), z_next.detach())
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Encoder einfrieren
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # Nur dynamics_head trainieren
    dyn_params = list(transformer.dynamics_head.parameters())
    optimizer  = torch.optim.AdamW(dyn_params, lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    n_dyn = sum(p.numel() for p in dyn_params)
    print(f"\nPre-Training Dynamics Head")
    print(f"  Encoder:       eingefroren ({sum(p.numel() for p in encoder.parameters()):,} Param.)")
    print(f"  dynamics_head: {n_dyn:,} Parameter (trainierbar)")
    print(f"  Dataset:       {len(dataset)} Tripel")
    print(f"  Epochen:       {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Hinweis: z_cur (LATENT_DIM={LATENT_DIM}) = Context-Substitute")
    print(f"           (möglich weil LATENT_DIM == D_MODEL nach T16)")
    print()

    transformer.train()
    best_loss = float('inf')
    t_start   = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for obs_b, act_b, nxt_b in loader:
            # Latent-Vektoren berechnen (Encoder eingefroren)
            with torch.no_grad():
                _, _, z_cur  = encoder(obs_b)
                _, _, z_next = encoder(nxt_b)

            # Dynamics-Vorhersage: z_cur als Context-Substitute
            # dynamics_head erwartet cat([context, action]) = (B, 256+6 = 262)
            z_pred = transformer.dynamics_head(
                torch.cat([z_cur, act_b], dim=-1)
            )

            loss = F.mse_loss(z_pred, z_next.detach())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dyn_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " ★"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
                  f"MSE: {avg_loss:.6f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}  "
                  f"({elapsed:.0f}s){marker}")

    elapsed = time.time() - t_start
    print(f"\n  Fertig in {elapsed:.1f}s  |  Best MSE: {best_loss:.6f}")

    # Encoder wieder entfrieren (damit Aufrufer ihn normal weiter nutzen kann)
    for p in encoder.parameters():
        p.requires_grad_(True)

    return {"final_loss": avg_loss, "best_loss": best_loss, "epochs": epochs}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B24 – Pre-Training Dynamics Head (T17)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/pwn_checkpoint_*.pt",
        help="Checkpoint mit vortrainiertem Encoder (B20/B21)"
    )
    parser.add_argument(
        "--env", default="PredictionWorld-OneRoom-v0",
        help="MiniWorld Environment"
    )
    parser.add_argument(
        "--episodes", type=int, default=400,
        help="Anzahl Episoden zum Sammeln (default: 400)"
    )
    parser.add_argument(
        "--steps-per-episode", type=int, default=25,
        help="Steps pro Episode (default: 25)"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Trainings-Epochen (default: 40)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch-Größe (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Lernrate für dynamics_head (default: 3e-4)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("B24 – Pre-Training Dynamics Head (T17)")
    print("=" * 55)
    print(f"  LATENT_DIM: {LATENT_DIM}  |  D_MODEL: {D_MODEL}  |  ACTION_DIM: {ACTION_DIM}")
    print()

    # ── Modelle erstellen ──────────────────────────────
    encoder     = Encoder()
    transformer = TemporalTransformer()

    # ── Checkpoint laden ───────────────────────────────
    try:
        ckpt_path = resolve_checkpoint(args.checkpoint)
        print(f"Lade Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)

        # Dimensions-Guard
        ckpt_latent = ckpt.get("constants", {}).get("LATENT_DIM", LATENT_DIM)
        if ckpt_latent != LATENT_DIM:
            print(f"  ⚠ Checkpoint LATENT_DIM={ckpt_latent} ≠ {LATENT_DIM} "
                  f"→ erst B20 neu ausführen.")
            sys.exit(1)

        encoder.load_state_dict(ckpt["encoder"])
        print("  Encoder geladen ✓")

        if "transformer" in ckpt:
            try:
                transformer.load_state_dict(ckpt["transformer"], strict=False)
                print("  Transformer (inkl. dynamics_head) geladen ✓")
            except Exception as e:
                print(f"  Transformer: partiell geladen ({e})")
        print()
    except FileNotFoundError:
        print("WARNUNG: Kein Checkpoint gefunden → frische Gewichte.")
        print("  Erst B20 ausführen für sinnvolle Encoder-Repräsentationen.")
        print()

    # ── Transitionen sammeln ───────────────────────────
    obs_arr, act_arr, nxt_arr = collect_transitions(
        n_episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        env_name=args.env,
    )
    dataset = TransitionDataset(obs_arr, act_arr, nxt_arr)

    # ── Training ───────────────────────────────────────
    result = pretrain_dynamics(
        encoder=encoder,
        transformer=transformer,
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # ── Checkpoint speichern ───────────────────────────
    import datetime
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(ckpt_dir, f"pwn_checkpoint_{ts}.pt")

    # Basis-Checkpoint übernehmen, nur transformer/dynamics_head aktualisieren
    try:
        base_path = resolve_checkpoint(args.checkpoint)
        checkpoint = torch.load(base_path, weights_only=False)
    except FileNotFoundError:
        checkpoint = {}

    checkpoint["transformer"]       = transformer.state_dict()
    checkpoint["encoder"]           = encoder.state_dict()
    checkpoint["tag"]               = "pretrain_dynamics"
    checkpoint["current_goal"]      = "pretrain_dynamics"
    checkpoint["result_dynamics"]   = result
    checkpoint["constants"]         = {
        "LATENT_DIM": LATENT_DIM,
        "D_MODEL":    D_MODEL,
        "ACTION_DIM": ACTION_DIM,
    }
    torch.save(checkpoint, out_path)

    print(f"\n  Checkpoint gespeichert: {out_path}")
    print()
    print("Nächste Schritte:")
    print(f"  Live verwenden:  python B19OrchestratorModeMiniworld.py")
    print(f"  (config: checkpoint='{out_path}')")
    print()
    print("Empfohlene Reihenfolge:")
    print("  B20 (VAE) → B21 (CLIP) → B24 (Dynamics) → B19 (Live)")


if __name__ == "__main__":
    main()
