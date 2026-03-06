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
    5. goal_proj (512→128→ReLU→64) wird trainiert:
       Contrastive Loss: passende Paare (label, frame) → nahe Vektoren

    python B21PreTrainCLIP.py
    python B21PreTrainCLIP.py --checkpoint checkpoints/pwn_checkpoint_*.pt --epochs 60
"""

import os
import sys
import time
import math
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


def _register_empty_env(gym):
    """Registriert PredictionWorld-Empty (kein Objekt im Raum)."""
    env_id = "PredictionWorld-Empty-v0"
    if env_id in gym.envs.registry:
        return
    from miniworld.envs.oneroom import OneRoom

    class PredictionWorldEmpty(OneRoom):
        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            self.box = None   # OneRoom.step() erwartet self.box
            self.place_agent()

        def step(self, action):
            # Kein Zielobjekt → nur Basis-Step ohne near(self.box)-Check
            from miniworld.miniworld import MiniWorldEnv
            return MiniWorldEnv.step(self, action)

    gym.register(id=env_id,
                 entry_point=lambda **kw: PredictionWorldEmpty(**kw),
                 max_episode_steps=300)


def _register_single_env(gym):
    """Registriert PredictionWorld-Single (ein zufälliges Objekt pro Reset)."""
    env_id = "PredictionWorld-Single-v0"
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

    _specs = [
        ("box",  "red"),   ("box",  "yellow"),
        ("box",  "white"), ("box",  "orange"),
        ("ball", "green"), ("ball", "blue"),
    ]

    class PredictionWorldSingle(OneRoom):
        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            etype, color = _specs[np.random.randint(len(_specs))]
            ent = Box(color=color) if etype == "box" else Ball(color=color)
            self.box = self.place_entity(ent)   # OneRoom.step() erwartet self.box
            self.place_agent()

    gym.register(id=env_id,
                 entry_point=lambda **kw: PredictionWorldSingle(**kw),
                 max_episode_steps=300)


# Gruppen-Definitionen für die 5-Strategie-Sammlung
# (env_type, aimed, kurzbeschreibung)
_GROUP_DEFS = [
    ("empty",  False, "leer / zufällig"),
    ("single", False, "1 Obj / zufällig"),
    ("single", True,  "1 Obj / gezielt"),
    ("full",   False, "alle Obj / zufällig"),
    ("full",   True,  "alle Obj / gezielt"),
]


def _collect_one_group(env, n: int, aimed: bool) -> tuple:
    """
    Sammelt n Frames für eine Gruppe.
    aimed=False → Agent behält zufällige Reset-Richtung.
    aimed=True  → Agent wird auf ein zufälliges Objekt ausgerichtet.
    Gibt (frames, labels, sources) zurück.
    sources: "entity" | "fov" | "pixel"
    """
    from PIL import Image as PILImage

    frames, labels, sources = [], [], []

    for _ in range(n):
        obs, _ = env.reset()
        uw = env.unwrapped
        target_label = None

        if aimed:
            colored = [(e, _entity_label(e)) for e in uw.entities
                       if type(e).__name__ != 'Agent'
                       and _entity_label(e) is not None]
            if colored:
                target, target_label = colored[np.random.randint(len(colored))]
                dx = target.pos[0] - uw.agent.pos[0]
                dz = target.pos[2] - uw.agent.pos[2]
                uw.agent.dir = np.arctan2(-dz, dx)
                for _s in range(np.random.randint(0, 8)):
                    obs, _, term, trunc, _ = env.step(2)
                    if term or trunc:
                        break
                uw.agent.dir += np.random.uniform(-0.25, 0.25)
                obs = uw.render_obs()

        img = np.array(
            PILImage.fromarray(obs).resize((128, 128), PILImage.BILINEAR),
            dtype=np.uint8
        )

        visible = _visible_entities_in_fov(uw)
        if visible:
            label  = visible[0][1]
            source = "entity" if aimed else "fov"
        elif aimed and target_label:
            label  = target_label
            source = "entity"
        else:
            label  = classify_frame(img)
            source = "pixel"

        frames.append(img)
        labels.append(label)
        sources.append(source)

    return frames, labels, sources
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


def _visible_entities_in_fov(uw, fov_deg: float = 80.0,
                              max_coverage_box: float = 0.85,
                              max_coverage_ball: float = 0.85) -> list:
    """
    Gibt alle farbigen Nicht-Agent-Entities zurück, die sich aktuell im
    Kamera-FOV des Agenten befinden, sortiert nach Distanz (nächste zuerst).

    Horizontal: fov_deg=80° (+20° Puffer über echte Kamera-FOV 60°), damit
    halb-sichtbare Objekte am Bildrand zuverlässig erkannt werden.

    Vertikal: Exakter physischer Check – Oberkante der Entity muss innerhalb
    der unteren Kante der vertikalen FOV liegen (30° unter Horizontal).
    Kamera-Augenhöhe = agent.pos[1] + agent.height.
    Dieser Check ersetzt den alten max_coverage_ball-Hack:
      Ball auf dem Boden (top ≈ 0.86 m, Kamera ≈ 1.6 m):
        dy_to_top ≈ 0.74 m → unsichtbar wenn dist < 1.28 m

    max_coverage: nur als Sicherheitsnetz für extreme Nähe (beide Typen 0.85).
      r_visual / tan(half_coverage) = Mindestabstand
      Box  (r_v≈0.40, 0.85): d_min ≈ 0.84 m
      Ball (r_v≈0.43, 0.85): d_min ≈ 0.90 m

    Koordinaten: agent.dir wächst CCW; Vorwärtsvektor = (cos(dir), 0, -sin(dir)).
    """
    agent_pos   = uw.agent.pos
    agent_dir   = uw.agent.dir
    agent_eye_y = agent_pos[1] + uw.agent.height   # Kamera-Augenhöhe
    half_fov_h  = np.radians(fov_deg / 2.0)         # horizontal (68°/2)
    half_fov_v  = np.radians(_FOV_DEG / 2.0)        # vertikal   (60°/2, exakt)

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

        r_v = _visual_radius(ent)

        # 1. Horizontaler Angular-Coverage-Check (zu nah = füllt Frame zu stark)
        is_ball  = 'ball' in type(ent).__name__.lower()
        coverage = max_coverage_ball if is_ball else max_coverage_box
        if np.arctan2(r_v, dist) > np.radians(_FOV_DEG * coverage / 2.0):
            continue

        # 2. Vertikaler FOV-Check: Oberkante der Entity muss im Bild sichtbar sein.
        #    Wenn die Oberkante unterhalb der unteren Kante der vertikalen FOV liegt
        #    (Kamera schaut horizontal, Ball liegt auf dem Boden), ist das Objekt
        #    physisch nicht sichtbar – egal wie gut der horizontale Winkel passt.
        entity_top_y = ent.pos[1] + r_v
        dy_to_top    = agent_eye_y - entity_top_y
        if dy_to_top > 0 and np.arctan2(dy_to_top, dist) > half_fov_v:
            continue  # Oberkante unterhalb unterer vertikaler FOV-Kante

        # 3. Horizontaler FOV-Check (mit Puffer)
        angle_to = np.arctan2(-dz, dx)
        diff = (angle_to - agent_dir + np.pi) % (2.0 * np.pi) - np.pi
        if abs(diff) <= half_fov_h:
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

    # Floor-Erkennung: kariertes Schachbrettmuster im unteren Bildbereich.
    # Schachbrett → hohe lokale Varianz (abwechselnd hell/dunkel).
    # Wand → gleichförmig grau → niedrige Varianz.
    # Region: Zeilen 60-127 (untere Hälfte) – Boden ist oft auch in der
    # Bildmitte sichtbar, nicht nur ganz unten.
    # Threshold: 0.08 – MiniWorld-Schachbrett (gedämpft grau) ≈ 0.10–0.25;
    # Wand ≈ 0.02–0.06.
    lower_half = f[60:, :, :]
    floor_std  = np.std(lower_half)

    if floor_std > 0.08:
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
        """
        5 gleich große Gruppen (je ~n_frames/5):
          G1: Leerer Raum,  zufällige Richtung
          G2: Ein Objekt,   zufällige Richtung
          G3: Ein Objekt,   auf Objekt ausgerichtet
          G4: Alle Objekte, zufällige Richtung
          G5: Alle Objekte, auf Objekt ausgerichtet
        """
        try:
            import gymnasium as gym
            import miniworld  # noqa: F401

            _register_prediction_world_env(gym)
            _register_empty_env(gym)
            _register_single_env(gym)

            envs = {
                "empty":  gym.make("PredictionWorld-Empty-v0",
                                   render_mode="rgb_array", view="agent"),
                "single": gym.make("PredictionWorld-Single-v0",
                                   render_mode="rgb_array", view="agent"),
                "full":   gym.make(env_name,
                                   render_mode="rgb_array", view="agent"),
            }

            n = n_frames // 5
            counts = [n, n, n, n, n_frames - 4 * n]

            print(f"Sammle {n_frames} Frames ({len(_GROUP_DEFS)} Gruppen)...")
            total = 0
            for (env_type, aimed, desc), n_group in zip(_GROUP_DEFS, counts):
                print(f"  {desc}: {n_group} Frames")
                imgs, lbls, _ = _collect_one_group(envs[env_type], n_group, aimed)
                self.frames.extend(imgs)
                self.labels.extend(lbls)
                total += n_group
                if total % 500 == 0:
                    print(f"  {total}/{n_frames} Frames gesammelt")

            for env in envs.values():
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
        goal_proj: nn.Module,
        clip_encoder: ClipTextEncoder,
        dataset: LabeledFrameDataset,
        epochs: int = 100,
        batch_size: int = 64,
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

    # Encoder: Conv-Layer einfrieren, FC-Layer (fc_mu, fc_log_var) mittrainieren
    for name, p in encoder.named_parameters():
        if name.startswith("fc_"):
            p.requires_grad = True
        else:
            p.requires_grad = False
    encoder.train()

    # Learnable Temperature (log-Skala für Stabilität)
    log_temp = torch.nn.Parameter(torch.tensor(math.log(temperature)))

    # Separate LR: goal_proj normal, Encoder-FC 10x kleiner
    encoder_fc_params = [p for n, p in encoder.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": goal_proj.parameters(), "lr": lr},
        {"params": encoder_fc_params, "lr": lr * 0.1},
        {"params": [log_temp], "lr": lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Text-Embeddings vorberechnen
    text_cache = {}
    for desc in LABEL_DESCRIPTIONS.values():
        vec = clip_encoder.encode_text(desc)  # np.ndarray (512,)
        text_cache[desc] = torch.from_numpy(vec)

    # Alle Label-Text-Embeddings als Tensor (für Klassifikation)
    all_descs = list(LABEL_DESCRIPTIONS.values())
    all_text_embs = torch.stack([text_cache[d] for d in all_descs], dim=0)  # (K, 512)
    label_to_id = {d: i for i, d in enumerate(all_descs)}

    n_enc_fc = sum(p.numel() for p in encoder_fc_params)
    print(f"\nPre-Training CLIP Goal-Projektion")
    print(f"  goal_proj:  {sum(p.numel() for p in goal_proj.parameters()):,} Parameter")
    print(f"  encoder FC: {n_enc_fc:,} Parameter (mittrainiert, LR×0.1)")
    print(f"  Dataset:    {len(dataset)} Frames")
    print(f"  Epochen:    {epochs}")
    print(f"  Temperatur: {temperature}")
    print(f"  Klassen:    {len(all_descs)}")
    print()

    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(epochs):
        goal_proj.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n_batches  = 0

        for batch_imgs, batch_labels, batch_descs in loader:
            # batch_imgs: (B, 3, 128, 128)
            mu, log_var, z = encoder(batch_imgs)
            z_norm = F.normalize(z, dim=-1)  # (B, 64)

            # Alle K Label-Projektionen berechnen
            all_proj = goal_proj(all_text_embs)             # (K, 64)
            all_proj_norm = F.normalize(all_proj, dim=-1)   # (K, 64)

            # Learnable Temperature (geclampt auf [0.05, 1.0])
            temp = torch.clamp(log_temp.exp(), 0.05, 1.0)

            # Klassifikations-Logits: jedes z gegen alle K Labels
            logits = torch.matmul(z_norm, all_proj_norm.T) / temp  # (B, K)

            # Targets: korrektes Label-ID pro Sample
            targets = torch.tensor(
                [label_to_id[d] for d in batch_descs], dtype=torch.long
            )

            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            all_params = list(goal_proj.parameters()) + encoder_fc_params + [log_temp]
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (logits.argmax(dim=1) == targets).sum().item()
            epoch_total += targets.shape[0]
            n_batches  += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / epoch_total * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " ★"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            elapsed = time.time() - t_start
            cur_temp = torch.clamp(log_temp.exp(), 0.01, 1.0).item()
            print(f"  Epoch {epoch+1:3d}/{epochs}  |  "
                  f"Loss: {avg_loss:.5f}  "
                  f"Acc: {accuracy:5.1f}%  "
                  f"τ: {cur_temp:.3f}  "
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
        "--epochs", type=int, default=100,
        help="Trainings-Epochen"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("B21 – Pre-Training CLIP Goal-Projektion")
    print("=" * 55)
    print()

    # ── Modelle erstellen ──────────────────────────────
    encoder = Encoder()
    goal_proj = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, LATENT_DIM),
    )

    # ── Checkpoint laden (Encoder + ggf. goal_proj) ────
    try:
        ckpt_path = resolve_checkpoint(args.checkpoint)
        print(f"Lade Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        print("  Encoder geladen ✓")
        if "goal_proj" in ckpt and ckpt["goal_proj"] is not None:
            try:
                goal_proj.load_state_dict(ckpt["goal_proj"])
                print("  goal_proj geladen ✓ (Nachtraining)")
            except RuntimeError:
                print("  goal_proj Architektur geändert → frisch initialisiert")
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

    # Rohe CLIP Text-Embeddings aller Labels speichern (512-dim, float32).
    # Werden zur Laufzeit (B19) mit goal_proj projiziert → Erkennung ohne CLIP.
    # Roh gespeichert (nicht projiziert), damit Verbesserungen von goal_proj
    # während des RL-Trainings sofort in der Erkennung sichtbar sind.
    print("  Speichere CLIP Label-Embeddings (für Laufzeit-Erkennung)...")
    label_clip_embs = {}
    with torch.no_grad():
        for lbl, desc in LABEL_DESCRIPTIONS.items():
            emb_np = clip_encoder.encode_text(desc)          # np.ndarray (512,)
            label_clip_embs[lbl] = torch.from_numpy(emb_np).float()
    checkpoint["label_clip_embeddings"] = label_clip_embs
    print(f"    {len(label_clip_embs)} Labels: {list(label_clip_embs.keys())}")

    torch.save(checkpoint, ckpt_path)

    print(f"\n  Checkpoint gespeichert: {ckpt_path}")
    print()
    print("Nächste Schritte:")
    print(f"  Live verwenden:  python B19Orchestrator.py "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
