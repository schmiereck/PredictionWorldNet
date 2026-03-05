"""
B21 – Pre-Training CLIP / Goal-Projektion
==========================================
Trainiert die goal_proj Schicht (Text → Latent Alignment),
damit textuelle Ziele ("find red box") zu sinnvollen
latenten Vektoren führen BEVOR das RL-Training beginnt.

Voraussetzung:
    Ein vortrainierter VAE-Checkpoint (B20) wird geladen,
    damit die Encoder-Latents sinnvolle Repräsentationen sind.

Vorgehen:
    1. Frames aus MiniWorld sammeln
    2. Auto-Labels erzeugen (Farb-Heuristik: red/green/blue/wall/empty)
    3. CLIP Text-Encoder liefert Goal-Embeddings (512-dim)
    4. Encoder liefert Latent-Vektoren (64-dim) für jeden Frame
    5. goal_proj (512→64) wird trainiert:
       Contrastive Loss: passende Paare (label, frame) → nahe Vektoren

    python B21PreTrainCLIP.py
    python B21PreTrainCLIP.py --checkpoint checkpoints/pwn_checkpoint_*.pt --epochs 60
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
    return matches[-1]

_b16 = _load_module("B16FullIntegration.py")
Encoder    = _b16.Encoder
Decoder    = _b16.Decoder
LATENT_DIM = _b16.LATENT_DIM
draw_scene = _b16.draw_scene
SCENE_TYPES = _b16.SCENE_TYPES

# CLIP Text Encoder (B05)
_b05 = _load_module("B05ClipTextEncoder.py")
ClipTextEncoder = _b05.CLIPTextEncoder


# ─────────────────────────────────────────────
# AUTO-LABELING (Farb-Heuristik)
# ─────────────────────────────────────────────

LABEL_DESCRIPTIONS = {
    "red":   "a red object or red box in the scene",
    "green": "a green object or green surface",
    "blue":  "a blue object or blue wall",
    "wall":  "a plain wall or corridor with no objects",
    "empty": "an empty room with nothing interesting",
}

def classify_frame(frame: np.ndarray) -> str:
    """
    Farb-Heuristik für 16×16 MiniWorld-Frames.
    Prüft ob eine Farbe deutlich dominiert.
    """
    if frame.dtype == np.uint8:
        f = frame.astype(np.float32) / 255.0
    else:
        f = frame

    r, g, b = f[:,:,0], f[:,:,1], f[:,:,2]
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)

    # Pixel mit deutlicher Farbdominanz zählen
    red_pixels   = np.mean((r > g + 0.1) & (r > b + 0.1) & (r > 0.3))
    green_pixels = np.mean((g > r + 0.1) & (g > b + 0.1) & (g > 0.3))
    blue_pixels  = np.mean((b > r + 0.1) & (b > g + 0.1) & (b > 0.3))

    threshold = 0.05  # mind. 5% der Pixel müssen farbdominant sein

    if red_pixels > threshold and red_pixels >= green_pixels and red_pixels >= blue_pixels:
        return "red"
    if green_pixels > threshold and green_pixels >= red_pixels and green_pixels >= blue_pixels:
        return "green"
    if blue_pixels > threshold and blue_pixels >= red_pixels and blue_pixels >= green_pixels:
        return "blue"

    brightness = (mean_r + mean_g + mean_b) / 3.0
    return "wall" if brightness > 0.25 else "empty"


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class LabeledFrameDataset(Dataset):
    """Frames mit Auto-Labels."""

    def __init__(self, n_frames=2000, env_name="MiniWorld-OneRoom-v0",
                 source="miniworld"):
        self.frames = []
        self.labels = []

        if source == "miniworld":
            self._collect_miniworld(n_frames, env_name)
        else:
            self._collect_mock(n_frames)

        # Label-Verteilung ausgeben
        from collections import Counter
        counts = Counter(self.labels)
        print(f"  Label-Verteilung:")
        for label, count in sorted(counts.items()):
            print(f"    {label:8s}: {count:4d} ({100*count/len(self.labels):.1f}%)")

    def _collect_miniworld(self, n_frames, env_name):
        print(f"Sammle {n_frames} gelabelte Frames aus {env_name}...")
        print(f"  (50% gezielt auf Objekte gerichtet)")
        try:
            import gymnasium as gym
            import miniworld  # noqa: F401
            from PIL import Image as PILImage

            env = gym.make(env_name, render_mode="rgb_array", view="agent")
            obs, _ = env.reset()
            uw = env.unwrapped

            n_targeted = n_frames // 2
            n_random   = n_frames - n_targeted

            # Phase 1: Gezielte Frames auf Objekte (Agent → Objekt ausrichten)
            for i in range(n_targeted):
                obs, _ = env.reset()
                uw = env.unwrapped

                # Farbiges Objekt suchen (Box, Ball, Key, ...)
                colored = [e for e in uw.entities
                           if hasattr(e, 'color') and e.color is not None
                           and type(e).__name__ != 'Agent']
                if colored:
                    target = colored[np.random.randint(len(colored))]
                    # Agent auf Objekt ausrichten
                    dx = target.pos[0] - uw.agent.pos[0]
                    dz = target.pos[2] - uw.agent.pos[2]
                    uw.agent.dir = np.arctan2(-dz, dx)
                    # Ein paar Schritte vorwärts (näher zum Objekt)
                    for _ in range(np.random.randint(0, 8)):
                        obs, _, term, trunc, _ = env.step(2)  # vorwärts
                        if term or trunc:
                            break
                    # Leichte Variation im Blickwinkel
                    uw.agent.dir += np.random.uniform(-0.3, 0.3)
                    obs = env.unwrapped.render_obs()

                img = np.array(
                    PILImage.fromarray(obs).resize((16, 16), PILImage.BILINEAR),
                    dtype=np.uint8
                )
                self.frames.append(img)
                self.labels.append(classify_frame(img))

                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{n_frames} Frames (gezielt)")

            # Phase 2: Zufällige Exploration (Wände, leere Bereiche)
            obs, _ = env.reset()
            for i in range(n_random):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()

                img = np.array(
                    PILImage.fromarray(obs).resize((16, 16), PILImage.BILINEAR),
                    dtype=np.uint8
                )
                self.frames.append(img)
                self.labels.append(classify_frame(img))

                if (n_targeted + i + 1) % 500 == 0:
                    print(f"  {n_targeted+i+1}/{n_frames} Frames (zufällig)")

                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{n_frames} Frames")

            env.close()
        except ImportError:
            print("  MiniWorld nicht verfügbar → Fallback auf Mock")
            self._collect_mock(n_frames)

    def _collect_mock(self, n_frames):
        print(f"Generiere {n_frames} gelabelte Mock-Frames...")
        for i in range(n_frames):
            scene = SCENE_TYPES[i % len(SCENE_TYPES)]
            noise = 0.03 + 0.12 * np.random.rand()
            img = draw_scene(scene, noise=noise)
            label = classify_frame(img)
            self.frames.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx].astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3,16,16)
        label = self.labels[idx]
        description = LABEL_DESCRIPTIONS[label]
        return tensor, label, description


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def pretrain_clip(
        encoder: Encoder,
        goal_proj: nn.Linear,
        clip_encoder: ClipTextEncoder,
        dataset: LabeledFrameDataset,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 5e-4,
        temperature: float = 0.07,
):
    """
    Contrastive Training: goal_proj(clip_text) ↔ encoder(frame).

    Für jedes Batch werden N Frames mit ihren Text-Labels gepaart.
    Positive Paare = gleicher Label, Negative = verschiedene Labels.
    InfoNCE Loss minimiert den Abstand passender Paare.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True)

    # Nur goal_proj wird trainiert, Encoder ist eingefroren
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(goal_proj.parameters(), lr=lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Text-Embeddings vorberechnen
    text_cache = {}
    for desc in LABEL_DESCRIPTIONS.values():
        vec = clip_encoder.encode_text(desc)  # np.ndarray (512,)
        text_cache[desc] = torch.from_numpy(vec)

    print(f"\nPre-Training CLIP Goal-Projektion")
    print(f"  goal_proj:  {sum(p.numel() for p in goal_proj.parameters()):,} Parameter")
    print(f"  Dataset:    {len(dataset)} Frames")
    print(f"  Epochen:    {epochs}")
    print(f"  Temperatur: {temperature}")
    print()

    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        goal_proj.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch_imgs, batch_labels, batch_descs in loader:
            # batch_imgs: (B, 3, 16, 16)
            with torch.no_grad():
                mu, log_var, z = encoder(batch_imgs)
                z_norm = F.normalize(z, dim=-1)  # (B, 64)

            # Text-Embeddings aus Cache
            text_embs = torch.stack(
                [text_cache[d] for d in batch_descs], dim=0
            )  # (B, 512)

            # Projizieren
            goal_latent = goal_proj(text_embs)  # (B, 64)
            goal_norm = F.normalize(goal_latent, dim=-1)  # (B, 64)

            # InfoNCE Loss (Cosine Similarity Matrix)
            sim = torch.matmul(goal_norm, z_norm.T) / temperature  # (B, B)
            targets = torch.arange(batch_size)  # Diagonale = positive Paare

            # Symmetrischer Loss
            loss_g2z = F.cross_entropy(sim, targets)
            loss_z2g = F.cross_entropy(sim.T, targets)
            loss = (loss_g2z + loss_z2g) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(goal_proj.parameters(), 1.0)
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
                  f"Loss: {avg_loss:.5f}  "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}  "
                  f"({elapsed:.0f}s){marker}")

    elapsed = time.time() - t_start
    print(f"\n  Fertig in {elapsed:.1f}s  |  Best Loss: {best_loss:.5f}")

    # Qualitätstest: Ähnlichkeit der Klassen
    print("\n  Qualitätstest (Cosine Similarity):")
    goal_proj.eval()
    with torch.no_grad():
        for label, desc in LABEL_DESCRIPTIONS.items():
            emb = torch.from_numpy(clip_encoder.encode_text(desc))
            proj = F.normalize(goal_proj(emb.unsqueeze(0)), dim=-1)
            # Sammle ein paar passende Frames
            matches = [i for i, l in enumerate(dataset.labels) if l == label][:10]
            if matches:
                imgs = torch.stack(
                    [dataset[i][0] for i in matches], dim=0
                )
                mu, _, z = encoder(imgs)
                z_n = F.normalize(z, dim=-1)
                sim = torch.matmul(z_n, proj.squeeze()).mean().item()
                print(f"    {label:8s}: {sim:.3f}")

    return {"final_loss": avg_loss, "best_loss": best_loss, "epochs": epochs}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B21 – Pre-Training CLIP Goal-Projektion"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/pwn_checkpoint_*.pt",
        help="Checkpoint laden (Encoder + ggf. goal_proj)"
    )
    parser.add_argument(
        "--source", choices=["miniworld", "mock"], default="miniworld",
        help="Datenquelle"
    )
    parser.add_argument(
        "--env", default="MiniWorld-OneRoom-v0",
        help="MiniWorld Environment"
    )
    parser.add_argument(
        "--frames", type=int, default=2000,
        help="Anzahl Frames"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Trainings-Epochen"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("B21 – Pre-Training CLIP Goal-Projektion")
    print("=" * 55)
    print()

    # ── Modelle erstellen ──────────────────────────────
    encoder = Encoder()
    goal_proj = nn.Linear(512, LATENT_DIM, bias=False)

    # ── Checkpoint laden (Encoder + ggf. goal_proj) ────
    try:
        ckpt_path = resolve_checkpoint(args.checkpoint)
        print(f"Lade Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        print("  Encoder geladen ✓")
        if "goal_proj" in ckpt and ckpt["goal_proj"] is not None:
            goal_proj.load_state_dict(ckpt["goal_proj"])
            print("  goal_proj geladen ✓ (Nachtraining)")
        print()
    except FileNotFoundError:
        print("WARNUNG: Kein Checkpoint gefunden!")
        print("  Encoder hat zufällige Gewichte → erst B20 ausführen.")
        print()

    # ── CLIP Text Encoder ──────────────────────────────
    clip_encoder = ClipTextEncoder()

    # ── Daten sammeln ──────────────────────────────────
    dataset = LabeledFrameDataset(
        n_frames=args.frames, env_name=args.env, source=args.source
    )

    # ── Training ───────────────────────────────────────
    result = pretrain_clip(
        encoder=encoder,
        goal_proj=goal_proj,
        clip_encoder=clip_encoder,
        dataset=dataset,
        epochs=args.epochs,
    )

    # ── Checkpoint speichern (VAE-Basis + goal_proj) ─────
    import datetime
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(ckpt_dir, f"pwn_checkpoint_{ts}.pt")

    # Basis-Checkpoint übernehmen (enthält Decoder, Transformer etc.)
    try:
        base_path = resolve_checkpoint(args.checkpoint)
        checkpoint = torch.load(base_path, weights_only=False)
    except FileNotFoundError:
        checkpoint = {}

    # goal_proj hinzufügen/aktualisieren
    checkpoint["goal_proj"]    = goal_proj.state_dict()
    checkpoint["encoder"]      = encoder.state_dict()
    checkpoint["tag"]          = "pretrain_clip"
    checkpoint["current_goal"] = "pretrain_clip"
    checkpoint["result_clip"]  = result

    torch.save(checkpoint, ckpt_path)

    print(f"\n  Checkpoint gespeichert: {ckpt_path}")
    print()
    print("Nächste Schritte:")
    print(f"  Live verwenden:  python B19Orchestrator.py "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
