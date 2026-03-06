"""
B21PreTrainCLIPVisual – Visuelle Überprüfung des CLIP-Labelings
===============================================================
Erzeugt 8 MiniWorld-Frames auf dieselbe Weise wie B21PreTrainCLIP.py
und zeigt sie mit dem erkannten Label als Überschrift an.
"Next"-Button lädt 8 neue Samples.

Aufruf:
    python B21PreTrainCLIPVisual.py
    python B21PreTrainCLIPVisual.py --source mock
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

import importlib.util


def _load_module(filename: str):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_b21 = _load_module("B21PreTrainCLIP.py")
classify_frame              = _b21.classify_frame
LABEL_DESCRIPTIONS          = _b21.LABEL_DESCRIPTIONS
_register_env               = _b21._register_prediction_world_env
_entity_label               = _b21._entity_label
_visible_entities_in_fov    = _b21._visible_entities_in_fov

_b16 = _load_module("B16FullIntegration.py")
draw_scene  = _b16.draw_scene
SCENE_TYPES = _b16.SCENE_TYPES


# ─────────────────────────────────────────────
# Frame-Sammler (identisch zu B21 – gezielt)
# ─────────────────────────────────────────────

def _collect_miniworld_samples(n: int, env_name: str):
    """
    Erzeugt n gezielte Frames aus MiniWorld.
    Label-Herkunft: 'entity' (ground truth) oder 'pixel' (Fallback).
    Gibt (frames, labels, sources) zurück.
    """
    import gymnasium as gym
    import miniworld  # noqa: F401
    from PIL import Image as PILImage

    _register_env(gym)
    env = gym.make(env_name, render_mode="rgb_array", view="agent")

    frames  = []
    labels  = []
    sources = []  # "entity" | "pixel"

    for _ in range(n):
        obs, _ = env.reset()
        uw = env.unwrapped

        colored = [(e, _entity_label(e)) for e in uw.entities
                   if type(e).__name__ != 'Agent'
                   and _entity_label(e) is not None]

        label  = None
        source = "pixel"
        if colored:
            target, target_label = colored[np.random.randint(len(colored))]
            dx = target.pos[0] - uw.agent.pos[0]
            dz = target.pos[2] - uw.agent.pos[2]
            uw.agent.dir = np.arctan2(-dz, dx)
            for _ in range(np.random.randint(0, 8)):
                obs, _, term, trunc, _ = env.step(2)
                if term or trunc:
                    break
            uw.agent.dir += np.random.uniform(-0.25, 0.25)
            obs = uw.render_obs()

            # Label = nächstes Objekt im FOV, nicht das ursprüngliche Ziel
            visible = _visible_entities_in_fov(uw)
            if visible:
                label  = visible[0][1]
                source = "entity"
            else:
                label  = target_label   # Fallback: Ziel außerhalb FOV
                source = "entity"

        img = np.array(
            PILImage.fromarray(obs).resize((128, 128), PILImage.BILINEAR),
            dtype=np.uint8
        )
        if label is None:
            # FOV-Check als zweite Chance
            visible = _visible_entities_in_fov(uw)
            if visible:
                label  = visible[0][1]
                source = "fov"
            else:
                label  = classify_frame(img)
                source = "pixel"

        frames.append(img)
        labels.append(label)
        sources.append(source)

    env.close()
    return frames, labels, sources


def _collect_mock_samples(n: int):
    """Erzeugt n Mock-Frames (ohne MiniWorld)."""
    frames  = []
    labels  = []
    sources = []
    for i in range(n):
        scene = SCENE_TYPES[np.random.randint(len(SCENE_TYPES))]
        noise = 0.03 + 0.12 * np.random.rand()
        img = draw_scene(scene, noise=noise)
        frames.append(img)
        labels.append(classify_frame(img))
        sources.append("pixel")
    return frames, labels, sources


# ─────────────────────────────────────────────
# Visualisierung
# ─────────────────────────────────────────────

COLS = 4
ROWS = 2
N_SAMPLES = COLS * ROWS  # 8


class LabelVisualizer:
    def __init__(self, source: str, env_name: str):
        self.source   = source
        self.env_name = env_name

        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.fig.suptitle(
            "B21 – CLIP Label-Überprüfung  (MiniWorld Frames)",
            color='white', fontsize=13, fontweight='bold'
        )

        # Grid: 2 Bild-Reihen + 1 Button-Reihe
        gs = gridspec.GridSpec(
            ROWS + 1, COLS,
            figure=self.fig,
            hspace=0.45, wspace=0.08,
            top=0.90, bottom=0.08,
            left=0.04, right=0.96
        )

        self.axes = []
        for r in range(ROWS):
            for c in range(COLS):
                ax = self.fig.add_subplot(gs[r, c])
                ax.set_facecolor('#2a2a3e')
                ax.set_xticks([])
                ax.set_yticks([])
                self.axes.append(ax)

        # "Next"-Button (zentriert unter den Bildern)
        btn_ax = self.fig.add_subplot(gs[ROWS, 1:3])
        btn_ax.set_facecolor('#1e1e2e')
        self.btn = Button(
            btn_ax, 'Next  ▶',
            color='#3a3a5e', hovercolor='#5a5a8e'
        )
        self.btn.label.set_color('white')
        self.btn.label.set_fontsize(11)
        self.btn.on_clicked(self._on_next)

        self._load_and_show()
        plt.show()

    # ── intern ────────────────────────────────

    def _load_samples(self):
        print(f"Sammle {N_SAMPLES} Samples ({self.source})...")
        if self.source == "miniworld":
            try:
                return _collect_miniworld_samples(N_SAMPLES, self.env_name)
            except Exception as e:
                print(f"  MiniWorld-Fehler: {e} → Fallback auf Mock")
        return _collect_mock_samples(N_SAMPLES)

    def _load_and_show(self):
        frames, labels, sources = self._load_samples()
        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('#2a2a3e')

            img    = frames[i]
            label  = labels[i]
            source = sources[i]
            desc   = LABEL_DESCRIPTIONS.get(label, label)

            # Source-Badge im Titel: ✔ entity/fov = zuverlässig, ⚠ pixel = Heuristik
            badge = {"entity": "✔ entity", "fov": "✔ fov", "pixel": "⚠ pixel"}.get(source, source)

            ax.imshow(img)
            ax.set_title(
                f"{label}  [{badge}]\n{desc}",
                color='white', fontsize=7.5,
                pad=4, wrap=True
            )
            # Rahmenfarbe je nach Label
            edge_color = {
                "red": "#ff4444", "green": "#44ff44", "blue": "#4488ff",
                "yellow": "#ffee44", "orange": "#ff8833", "white": "#cccccc",
                "wall": "#888888", "empty": "#444466",
            }.get(label, "#888888")
            # Rahmendicke: dünn bei Pixel-Fallback (unsicher)
            lw = 2 if source != "pixel" else 1
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(lw)

        self.fig.canvas.draw_idle()
        print("  ✓ Angezeigt.")

    def _on_next(self, _event):
        self._load_and_show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="B21 – Visuelle Label-Überprüfung"
    )
    parser.add_argument(
        "--source", choices=["miniworld", "mock"], default="miniworld",
        help="Datenquelle (default: miniworld)"
    )
    parser.add_argument(
        "--env", default="PredictionWorld-OneRoom-v0",
        help="MiniWorld Environment"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("B21 – CLIP Label-Überprüfung")
    print("=" * 55)
    print(f"  Quelle : {args.source}")
    print(f"  Umgebung: {args.env}")
    print(f"  Zeige {N_SAMPLES} Samples  (COLS={COLS}, ROWS={ROWS})")
    print()

    LabelVisualizer(source=args.source, env_name=args.env)


if __name__ == "__main__":
    main()
