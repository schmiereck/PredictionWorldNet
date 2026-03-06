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
            self.box = self.place_entity(Box(color="red"))
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
# ENTITY-BASIERTES LABELING (primär)
# ─────────────────────────────────────────────

_KNOWN_COLORS = ("red", "green", "blue", "yellow", "orange", "white")


def _entity_color_name(ent) -> str | None:
    """
    Gibt den Farbnamen einer MiniWorld-Entity zurück.
    Box: ent.color ist ein String → direkt lesen.
    Ball: kein .color-Attribut → Farbe aus ObjMesh-Cache-Key extrahieren
          (Dateiname: ball_{color}.obj).
    """
    # Box / Key / etc.: .color ist ein String
    if hasattr(ent, 'color') and isinstance(ent.color, str):
        return ent.color if ent.color in _KNOWN_COLORS else None

    # Ball: Mesh-Dateiname enthält die Farbe als "ball_{color}.obj"
    if hasattr(ent, 'mesh') and ent.mesh is not None:
        try:
            import os, re
            from miniworld.objmesh import ObjMesh
            for k, v in ObjMesh.cache.items():
                if v is ent.mesh:
                    base = os.path.basename(k).lower()   # z.B. "ball_green.obj"
                    m = re.match(r'ball_([a-z]+)\.obj$', base)
                    if m and m.group(1) in _KNOWN_COLORS:
                        return m.group(1)
                    break  # Mesh gefunden – kein weiterer Treffer möglich
        except Exception:
            pass
    return None


def _entity_label(ent) -> str | None:
    """Gibt das CLIP-Label für eine Entity zurück (Farbe)."""
    color = _entity_color_name(ent)
    return color if color and color in _KNOWN_COLORS else None


# FOV = 60° (identisch zu OverheadMapView)
_FOV_DEG = 60.0


def _visual_radius(ent) -> float:
    """
    Gibt den visuell relevanten Radius einer Entity zurück.
    Box: ent.radius ist die 3D-Diagonale (zu groß für FOV-Berechnung).
         Korrektur auf visuelle Halbbreite = height/2 (Würfelseite).
    Ball: ent.radius ist der echte Kugelradius → direkt verwenden.
    """
    if 'box' in type(ent).__name__.lower():
        return ent.height / 2.0   # Würfel: Halbseite = height/2 ≈ 0.4
    return ent.radius              # Kugel: Kugelradius ≈ 0.43


def _visible_entities_in_fov(uw, fov_deg: float = _FOV_DEG,
                              max_coverage_box: float = 0.85,
                              max_coverage_ball: float = 0.65) -> list:
    """
    Gibt alle farbigen Nicht-Agent-Entities zurück, die sich aktuell im
    Kamera-FOV des Agenten befinden, sortiert nach Distanz (nächste zuerst).

    Separate Schwellwerte für Box und Ball, weil ihre Radien sich unterscheiden:
      Box  (r_visual≈0.40): max_coverage_box=0.85  → d_min ≈ 0.84 m
      Ball (r_visual≈0.43): max_coverage_ball=0.65 → d_min ≈ 1.21 m

    Herleitung:  d_min = r_visual / tan(max_coverage · fov/2)

    Koordinaten: agent.dir wächst CCW; Vorwärtsvektor = (cos(dir), 0, -sin(dir)).
    FOV-Check: Winkel zwischen Blickrichtung und Richtung zum Objekt ≤ half_fov.
    """
    agent_pos = uw.agent.pos
    agent_dir = uw.agent.dir
    half_fov  = np.radians(fov_deg / 2.0)

    visible = []
    for ent in uw.entities:
        if type(ent).__name__ == 'Agent':
            continue
        if not hasattr(ent, 'pos'):
            continue
        label = _entity_label(ent)
        if label is None:
            continue

        dx = ent.pos[0] - agent_pos[0]
        dz = ent.pos[2] - agent_pos[2]
        dist = np.sqrt(dx * dx + dz * dz)

        if dist < 1e-6:
            continue

        # Per-Typ Schwellwert
        is_ball   = 'ball' in type(ent).__name__.lower()
        coverage  = max_coverage_ball if is_ball else max_coverage_box
        max_ang_h = np.radians(fov_deg * coverage / 2.0)

        if np.arctan2(_visual_radius(ent), dist) > max_ang_h:
            continue  # zu nah – füllt Frame zu stark

        angle_to = np.arctan2(-dz, dx)
        diff = (angle_to - agent_dir + np.pi) % (2.0 * np.pi) - np.pi
        if abs(diff) <= half_fov:
            visible.append((ent, label, dist))

    return sorted(visible, key=lambda x: x[2])


# ─────────────────────────────────────────────
# AUTO-LABELING (Farb-Heuristik – Fallback)
# ─────────────────────────────────────────────

LABEL_DESCRIPTIONS = {
    "red":    "a red box in the scene",
    "green":  "a green ball in the scene",
    "blue":   "a blue ball in the scene",
    "yellow": "a yellow box in the scene",
    "orange": "an orange box in the scene",
    "white":  "a white box in the scene",
    "wall":   "a plain gray wall with no objects",
    "empty":  "an empty room with a checkered floor visible",
}

def classify_frame(frame: np.ndarray) -> str:
    """
    Farb-Heuristik für 128×128 MiniWorld-Frames.
    Erkennt: red, green, blue, yellow, orange, white.
    """
    if frame.dtype == np.uint8:
        f = frame.astype(np.float32) / 255.0
    else:
        f = frame

    r, g, b = f[:,:,0], f[:,:,1], f[:,:,2]
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)

    # Farbspezifische Pixel-Masken
    red_px    = np.mean((r > g + 0.15) & (r > b + 0.15) & (r > 0.4))
    green_px  = np.mean((g > r + 0.15) & (g > b + 0.15) & (g > 0.4))
    blue_px   = np.mean((b > r + 0.15) & (b > g + 0.15) & (b > 0.4))
    yellow_px = np.mean((r > 0.5) & (g > 0.5) & (b < 0.3) & (r > b + 0.2))
    orange_px = np.mean((r > 0.5) & (g > 0.2) & (g < 0.6) & (b < 0.2)
                         & (r > g + 0.1))
    white_px  = np.mean((r > 0.7) & (g > 0.7) & (b > 0.7)
                         & (np.abs(r - g) < 0.15) & (np.abs(r - b) < 0.15))

    threshold = 0.03
    scores = {
        "red": red_px, "green": green_px, "blue": blue_px,
        "yellow": yellow_px, "orange": orange_px, "white": white_px,
    }
    best = max(scores, key=scores.get)
    if scores[best] > threshold:
        return best

    brightness = (mean_r + mean_g + mean_b) / 3.0

    # Floor-Erkennung: kariertes Schachbrettmuster im unteren Bilddrittel.
    # Schachbrett → hohe lokale Varianz (abwechselnd hell/dunkel).
    # Wand → gleichförmig grau → niedrige Varianz.
    lower_third = f[85:, :, :]          # unteres Drittel (Zeilen 85–127)
    floor_std   = np.std(lower_third)   # Schachbrett ≈ 0.3+, Wand ≈ 0.05–0.1

    if floor_std > 0.15:
        return "empty"   # karierter Boden sichtbar
    return "wall"        # gleichförmige Wand oder Decke


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class LabeledFrameDataset(Dataset):
    """Frames mit Auto-Labels."""

    def __init__(self, n_frames=2000, env_name="PredictionWorld-OneRoom-v0",
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
        print(f"  Phase 1 (50%): gezielt auf Objekt – Entity-Label (ground truth)")
        print(f"  Phase 2 (50%): zufällig     – FOV-sichtbarstes Objekt oder wall/empty")
        try:
            import gymnasium as gym
            import miniworld  # noqa: F401
            from PIL import Image as PILImage

            _register_prediction_world_env(gym)

            env = gym.make(env_name, render_mode="rgb_array", view="agent")
            obs, _ = env.reset()
            uw = env.unwrapped

            n_targeted = n_frames // 2
            n_random   = n_frames - n_targeted

            # ── Phase 1: Gezielt auf einzelnes Objekt ausrichten ──────────────
            # Label kommt direkt vom Entity (nicht vom Pixel-Heuristik).
            for i in range(n_targeted):
                obs, _ = env.reset()
                uw = env.unwrapped

                # Alle farbigen Entities sammeln
                colored = [(e, _entity_label(e)) for e in uw.entities
                           if type(e).__name__ != 'Agent'
                           and _entity_label(e) is not None]

                label = None
                target_label = None
                if colored:
                    target, target_label = colored[np.random.randint(len(colored))]
                    # Agent exakt auf Ziel ausrichten
                    dx = target.pos[0] - uw.agent.pos[0]
                    dz = target.pos[2] - uw.agent.pos[2]
                    uw.agent.dir = np.arctan2(-dz, dx)
                    # Ein paar Schritte vorwärts (näher zum Objekt)
                    for _ in range(np.random.randint(0, 8)):
                        obs, _, term, trunc, _ = env.step(2)
                        if term or trunc:
                            break
                    # Kleine Blickwinkel-Variation (Objekt bleibt im FOV)
                    uw.agent.dir += np.random.uniform(-0.25, 0.25)
                    obs = uw.render_obs()

                    # Label = nächstes Objekt im FOV (nicht das Ziel-Entity!).
                    # Nach Vorwärtsgehen kann ein anderes Objekt näher sein.
                    visible = _visible_entities_in_fov(uw)
                    if visible:
                        label = visible[0][1]   # nächstes sichtbares Objekt
                    else:
                        label = target_label    # Fallback: Ziel außerhalb FOV

                img = np.array(
                    PILImage.fromarray(obs).resize((128, 128), PILImage.BILINEAR),
                    dtype=np.uint8
                )
                self.frames.append(img)
                # Fallback: kein Objekt gefunden → Pixel-Heuristik (wall/empty)
                self.labels.append(label if label else classify_frame(img))

                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{n_frames} Frames (gezielt)")

            # ── Phase 2: Zufällige Exploration ────────────────────────────────
            # Label = nähestes Objekt im FOV → sonst wall/empty per Pixel-Heuristik.
            obs, _ = env.reset()
            uw = env.unwrapped
            for i in range(n_random):
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    uw = env.unwrapped

                img = np.array(
                    PILImage.fromarray(obs).resize((128, 128), PILImage.BILINEAR),
                    dtype=np.uint8
                )

                # FOV-basiertes Label: nähestes sichtbares Objekt
                visible = _visible_entities_in_fov(uw)
                if visible:
                    label = visible[0][1]  # Label des nächsten Objekts im FOV
                else:
                    label = classify_frame(img)  # wall / empty Fallback

                self.frames.append(img)
                self.labels.append(label)

                if (n_targeted + i + 1) % 500 == 0:
                    print(f"  {n_targeted+i+1}/{n_frames} Frames (zufällig)")

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
        tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3,128,128)
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
            # batch_imgs: (B, 3, 128, 128)
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
        "--env", default="PredictionWorld-OneRoom-v0",
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
