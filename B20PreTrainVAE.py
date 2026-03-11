"""
B20 – Pre-Training VAE (Autoencoder offline)
==============================================
Trainiert Encoder + Decoder rein auf Bild-Rekonstruktion,
ohne RL, ohne Gemini, ohne Transformer.

Kann wiederholt auf dasselbe Netz angewendet werden:
    1. Lauf: python B20PreTrainVAE.py --epochs 50
    2. Lauf: python B20PreTrainVAE.py --checkpoint checkpoints/pwn_checkpoint_*.pt --epochs 50

Datenquellen:
    --source miniworld   → Frames aus MiniWorld Gym (zufällige Exploration)
    --source mock        → Frames aus draw_scene() (B16 Mock-Szenen)

Ergebnis:
    checkpoints/pwn_checkpoint_<timestamp>.pt
    → Kann in B19 geladen werden:
      python B19Orchestrator.py --checkpoint checkpoints/pwn_checkpoint_*.pt
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

# ─────────────────────────────────────────────
# B16 laden (Encoder, Decoder, Konstanten)
# ─────────────────────────────────────────────

import importlib.util

def _load_module(filename: str):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_checkpoint(pattern: str) -> str:
    """Löst Glob-Patterns auf und gibt den neuesten Checkpoint zurück."""
    import glob as g
    matches = sorted(g.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Kein Checkpoint gefunden: {pattern}")
    return matches[-1]  # neuester (Timestamp im Namen → alphabetisch sortiert)

_b16 = _load_module("B16FullIntegration.py")
Encoder    = _b16.Encoder
Decoder    = _b16.Decoder
LATENT_DIM = _b16.LATENT_DIM
D_MODEL    = _b16.D_MODEL
draw_scene = _b16.draw_scene
SCENE_TYPES = _b16.SCENE_TYPES


def _register_prediction_world_env(gym):
    """Registriert PredictionWorld-OneRoom (falls noch nicht geschehen)."""
    from MiniWorldRegistry import register_prediction_world_environments
    register_prediction_world_environments()


# ─────────────────────────────────────────────
# DATASET: Frames sammeln
# ─────────────────────────────────────────────

class MiniWorldFrameDataset(Dataset):
    """Sammelt N Frames aus MiniWorld durch zufällige Exploration."""

    def __init__(self, n_frames: int = 2000,
                 env_name: str = "MiniWorld-OneRoom-v0"):
        self.frames = []
        print(f"Sammle {n_frames} Frames aus {env_name}...")

        try:
            import gymnasium as gym
            import miniworld  # noqa: F401
            from PIL import Image as PILImage

            _register_prediction_world_env(gym)

            env = gym.make(env_name, render_mode="rgb_array", view="agent")

            for i in range(n_frames):
                # Reset bei jedem Frame → neue Objektpositionen
                obs, _ = env.reset()

                # Optional: 0-5 zufällige Steps für verschiedene Distanzen/Perspektiven
                n_steps = np.random.randint(0, 6)
                for _ in range(n_steps):
                    action = env.action_space.sample()
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        obs, _ = env.reset()
                        break

                # Auf 128×128 skalieren
                img_pil = PILImage.fromarray(obs).resize(
                    (128, 128), PILImage.BILINEAR)
                self.frames.append(np.array(img_pil, dtype=np.uint8))

                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{n_frames} Frames gesammelt")

            env.close()
            print(f"  {len(self.frames)} Frames gesammelt ✓")

        except ImportError:
            print("  MiniWorld nicht installiert → Fallback auf Mock-Szenen")
            self._fill_mock(n_frames)

    def _fill_mock(self, n_frames):
        """Fallback: Mock-Szenen mit Rauschen."""
        for i in range(n_frames):
            scene = SCENE_TYPES[i % len(SCENE_TYPES)]
            noise = 0.05 + 0.1 * np.random.rand()
            self.frames.append(draw_scene(scene, noise=noise))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx].astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)  # (3, 128, 128)


class MockFrameDataset(Dataset):
    """Frames aus draw_scene() mit Variationen."""

    def __init__(self, n_frames: int = 2000):
        self.frames = []
        print(f"Generiere {n_frames} Mock-Frames...")
        for i in range(n_frames):
            scene = SCENE_TYPES[i % len(SCENE_TYPES)]
            noise = 0.03 + 0.12 * np.random.rand()
            self.frames.append(draw_scene(scene, noise=noise))
        print(f"  {len(self.frames)} Frames generiert ✓")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx].astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)


# ─────────────────────────────────────────────
# PRE-TRAINING LOOP
# ─────────────────────────────────────────────

def pretrain_vae(
        encoder: Encoder,
        decoder: Decoder,
        dataset: Dataset,
        epochs: int = 60,
        batch_size: int = 32,
        lr: float = 5e-4,
        beta_max: float = 0.005,
        beta_warmup_epochs: int = 20,
):
    """
    Trainiert Encoder + Decoder (VAE) rein auf Rekonstruktion.

    Loss = Recon (MSE) + beta * KL-Divergenz
    beta wird langsam von 0 → beta_max hochgefahren (Warmup).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    print(f"\nPre-Training VAE")
    print(f"  Encoder:  {sum(p.numel() for p in encoder.parameters()):,} Parameter")
    print(f"  Decoder:  {sum(p.numel() for p in decoder.parameters()):,} Parameter")
    print(f"  Dataset:  {len(dataset)} Frames")
    print(f"  Epochen:  {epochs}")
    print(f"  Batch:    {batch_size}")
    print(f"  LR:       {lr}")
    print(f"  Beta:     0 → {beta_max} (Warmup: {beta_warmup_epochs} Epochen)")
    print()

    encoder.train()
    decoder.train()

    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        epoch_recon = 0.0
        epoch_kl    = 0.0
        epoch_loss  = 0.0
        n_batches   = 0

        # Beta-Warmup
        beta = min(beta_max, beta_max * epoch / max(1, beta_warmup_epochs))

        for batch in loader:
            # batch shape: (B, 3, 128, 128)
            mu, log_var, z = encoder(batch)
            recon = decoder(z)

            # Reconstruction Loss
            l_recon = F.mse_loss(recon, batch)

            # KL-Divergenz
            l_kl = -0.5 * torch.mean(
                1 + log_var - mu.pow(2) - log_var.exp()
            )

            # Gesamt-Loss
            loss = l_recon + beta * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_recon += l_recon.item()
            epoch_kl    += l_kl.item()
            epoch_loss  += loss.item()
            n_batches   += 1

        scheduler.step()

        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl / n_batches
        avg_loss  = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " ★"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
                  f"Loss: {avg_loss:.5f}  "
                  f"Recon: {avg_recon:.5f}  "
                  f"KL: {avg_kl:.5f}  "
                  f"β: {beta:.4f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}  "
                  f"({elapsed:.0f}s){marker}")

    elapsed = time.time() - t_start
    print(f"\n  Fertig in {elapsed:.1f}s  |  Best Loss: {best_loss:.5f}")

    return {
        "final_loss":  avg_loss,
        "final_recon": avg_recon,
        "final_kl":    avg_kl,
        "best_loss":   best_loss,
        "epochs":      epochs,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B20 – Pre-Training VAE (Encoder + Decoder)"
    )
    parser.add_argument(
        "--source", choices=["miniworld", "mock"], default="miniworld",
        help="Datenquelle (default: miniworld)"
    )
    parser.add_argument(
        "--env", default="PredictionWorld-OneRoom-v0",
        help="MiniWorld Environment (nur bei --source miniworld)"
    )
    parser.add_argument(
        "--frames", type=int, default=4000,
        help="Anzahl Frames zum Sammeln (default: 4000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Anzahl Trainings-Epochen (default: 60)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch-Größe (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Lernrate (default: 5e-4)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Bestehenden Checkpoint laden und weitertrainieren"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("B20 – Pre-Training VAE")
    print("=" * 55)
    print()

    # ── Daten sammeln ──────────────────────────────────
    if args.source == "miniworld":
        dataset = MiniWorldFrameDataset(
            n_frames=args.frames, env_name=args.env)
    else:
        dataset = MockFrameDataset(n_frames=args.frames)

    # ── Modelle erstellen ──────────────────────────────
    encoder = Encoder()
    decoder = Decoder()

    # ── Optional: bestehenden Checkpoint laden ─────────
    if args.checkpoint:
        try:
            print(f"\nLade Checkpoint: {args.checkpoint}")
            ckpt_path = resolve_checkpoint(args.checkpoint)
            print(f"  → {ckpt_path}")
            ckpt = torch.load(ckpt_path, weights_only=False)
            encoder.load_state_dict(ckpt["encoder"])
            decoder.load_state_dict(ckpt["decoder"])
            prev_steps = ckpt.get("total_steps", 0)
            print(f"  Vorherige Steps: {prev_steps}")
            print()
        except FileNotFoundError:
            print(f"\nKein Checkpoint gefunden → starte mit frischen Gewichten.")
            print()

    # ── Training ───────────────────────────────────────
    result = pretrain_vae(
        encoder=encoder,
        decoder=decoder,
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
    ckpt_path = os.path.join(ckpt_dir, f"pwn_checkpoint_{ts}.pt")

    # Basis-Checkpoint übernehmen (enthält ggf. CLIP goal_proj etc.)
    try:
        if args.checkpoint:
            base_path = resolve_checkpoint(args.checkpoint)
            checkpoint = torch.load(base_path, weights_only=False)
        else:
            checkpoint = {}
    except FileNotFoundError:
        checkpoint = {}

    # VAE-Teile aktualisieren
    checkpoint["encoder"]      = encoder.state_dict()
    checkpoint["decoder"]      = decoder.state_dict()
    checkpoint["total_steps"]  = args.epochs * len(dataset)
    checkpoint["train_steps"]  = args.epochs * (len(dataset) // args.batch_size)
    checkpoint["beta"]         = 0.01
    checkpoint["current_goal"] = "pretrain_vae"
    checkpoint["config"]       = {"source": args.source, "frames": args.frames,
                                  "epochs": args.epochs, "lr": args.lr}
    checkpoint["constants"]    = {
        "LATENT_DIM": LATENT_DIM,
        "D_MODEL":    D_MODEL,
        "ACTION_DIM": 6,
    }
    checkpoint["tag"]          = "pretrain_vae"
    checkpoint["result"]       = result
    torch.save(checkpoint, ckpt_path)

    print(f"\n  Checkpoint gespeichert: {ckpt_path}")
    print()
    print("Nächste Schritte:")
    print(f"  Weitertrainieren:  python B20PreTrainVAE.py "
          f"--checkpoint {ckpt_path}")
    print(f"  Live verwenden:    python B19OrchestratorModeMiniworld.py")
    print(f"                     (config: checkpoint='{ckpt_path}')")
    print(f"  Oder per CLI:      python B19Orchestrator.py "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
