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
classify_frame           = _b21.classify_frame
LABEL_DESCRIPTIONS       = _b21.LABEL_DESCRIPTIONS
_register_env            = _b21._register_prediction_world_env
_register_empty_env      = _b21._register_empty_env
_register_single_env     = _b21._register_single_env
_collect_one_group       = _b21._collect_one_group
_GROUP_DEFS              = _b21._GROUP_DEFS

_b16 = _load_module("B16FullIntegration.py")
draw_scene  = _b16.draw_scene
SCENE_TYPES = _b16.SCENE_TYPES


# ─────────────────────────────────────────────
# Frame-Sammler (5-Gruppen-Strategie)
# ─────────────────────────────────────────────

N_PER_GROUP = 2                        # Samples pro Gruppe
COLS        = len(_GROUP_DEFS)         # 5 Spalten = 5 Gruppen
ROWS        = N_PER_GROUP              # 2 Zeilen
N_SAMPLES   = COLS * ROWS              # 10 gesamt

# Gruppenfarben für den Spaltenrahmen
_GROUP_COLORS = ["#5566aa", "#aa6655", "#55aa66", "#aa9944", "#8855bb"]


def _make_miniworld_envs(env_name: str) -> dict:
    """Erstellt die drei MiniWorld-Environments (einmalig, werden wiederverwendet)."""
    import gymnasium as gym
    import miniworld  # noqa: F401

    _register_env(gym)
    _register_empty_env(gym)
    _register_single_env(gym)

    return {
        "empty":  gym.make("PredictionWorld-Empty-v0",
                           render_mode="rgb_array", view="agent"),
        "single": gym.make("PredictionWorld-Single-v0",
                           render_mode="rgb_array", view="agent"),
        "full":   gym.make(env_name,
                           render_mode="rgb_array", view="agent"),
    }


def _collect_miniworld_samples(envs: dict):
    """
    Erzeugt N_PER_GROUP Frames pro Gruppe mit bereits geöffneten Environments.
    Gibt (frames, labels, sources, group_indices) zurück.
    """
    frames, labels, sources, group_indices = [], [], [], []

    for gi, (env_type, aimed, _desc) in enumerate(_GROUP_DEFS):
        imgs, lbls, srcs = _collect_one_group(envs[env_type], N_PER_GROUP, aimed)
        frames.extend(imgs)
        labels.extend(lbls)
        sources.extend(srcs)
        group_indices.extend([gi] * N_PER_GROUP)

    return frames, labels, sources, group_indices


def _collect_mock_samples():
    """Mock-Fallback: gleichmäßig über Gruppen verteilt."""
    frames, labels, sources, group_indices = [], [], [], []
    for gi in range(len(_GROUP_DEFS)):
        for _ in range(N_PER_GROUP):
            scene = SCENE_TYPES[np.random.randint(len(SCENE_TYPES))]
            img = draw_scene(scene, noise=0.03 + 0.12 * np.random.rand())
            frames.append(img)
            labels.append(classify_frame(img))
            sources.append("pixel")
            group_indices.append(gi)
    return frames, labels, sources, group_indices


# ─────────────────────────────────────────────
# Visualisierung
# ─────────────────────────────────────────────

class LabelVisualizer:
    def __init__(self, source: str, env_name: str):
        self.source   = source
        self.env_name = env_name
        self._envs    = None  # wird lazy erstellt und bis zum Schließen gehalten

        self.fig = plt.figure(figsize=(16, 7))
        self.fig.patch.set_facecolor('#1e1e2e')
        self.fig.suptitle(
            "B21 – CLIP Label-Überprüfung  (5 Gruppen × 2 Samples)",
            color='white', fontsize=13, fontweight='bold'
        )

        # Grid: Gruppen-Header + ROWS Bild-Reihen + Button-Reihe
        gs = gridspec.GridSpec(
            ROWS + 2, COLS,
            figure=self.fig,
            hspace=0.55, wspace=0.06,
            top=0.90, bottom=0.07,
            left=0.02, right=0.98
        )

        # Gruppen-Header (Zeile 0)
        self.header_axes = []
        for c, (_env_type, aimed, desc) in enumerate(_GROUP_DEFS):
            ax = self.fig.add_subplot(gs[0, c])
            ax.set_facecolor(_GROUP_COLORS[c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"G{c+1}\n{desc}",
                    ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold',
                    transform=ax.transAxes)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GROUP_COLORS[c])
            self.header_axes.append(ax)

        # Bild-Achsen (Zeilen 1 bis ROWS)
        self.axes = []
        for r in range(ROWS):
            for c in range(COLS):
                ax = self.fig.add_subplot(gs[r + 1, c])
                ax.set_facecolor('#2a2a3e')
                ax.set_xticks([])
                ax.set_yticks([])
                self.axes.append(ax)

        # "Next"-Button
        btn_ax = self.fig.add_subplot(gs[ROWS + 1, 1:4])
        btn_ax.set_facecolor('#1e1e2e')
        self.btn = Button(btn_ax, 'Next  ▶',
                          color='#3a3a5e', hovercolor='#5a5a8e')
        self.btn.label.set_color('white')
        self.btn.label.set_fontsize(11)
        self.btn.on_clicked(self._on_next)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        self._load_and_show()
        plt.show()

    def _load_samples(self):
        print(f"Sammle {N_SAMPLES} Samples ({self.source})...")
        if self.source == "miniworld":
            try:
                if self._envs is None:
                    print("  Initialisiere MiniWorld-Environments (einmalig)...")
                    self._envs = _make_miniworld_envs(self.env_name)
                return _collect_miniworld_samples(self._envs)
            except Exception as e:
                print(f"  MiniWorld-Fehler: {e} → Fallback auf Mock")
        return _collect_mock_samples()

    def _on_close(self, _event):
        if self._envs is not None:
            print("Schließe MiniWorld-Environments...")
            for env in self._envs.values():
                try:
                    env.close()
                except Exception:
                    pass
            self._envs = None

    def _load_and_show(self):
        frames, labels, sources, group_indices = self._load_samples()

        # Samples sind nach Gruppe geordnet: G0,G0, G1,G1, ... → in 5×2 Grid
        # Spalte = Gruppe, Zeile = Sample-Index innerhalb Gruppe
        for r in range(ROWS):
            for c in range(COLS):
                idx = c * N_PER_GROUP + r   # Sample c*2+r
                ax  = self.axes[r * COLS + c]

                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor('#2a2a3e')

                img    = frames[idx]
                label  = labels[idx]
                source = sources[idx]
                desc   = LABEL_DESCRIPTIONS.get(label, label)
                badge  = {"entity": "✔", "fov": "~", "pixel": "⚠"}.get(source, source)

                ax.imshow(img)
                ax.set_title(
                    f"{label} {badge}\n{desc}",
                    color='white', fontsize=7,
                    pad=3, wrap=True
                )

                edge_color = {
                    "red": "#ff4444", "green": "#44ff44", "blue": "#4488ff",
                    "yellow": "#ffee44", "orange": "#ff8833", "white": "#cccccc",
                    "wall": "#888888", "empty": "#446688",
                }.get(label, "#888888")
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
    print(f"  Zeige {N_SAMPLES} Samples  ({COLS} Gruppen × {ROWS} Samples)")
    print()

    LabelVisualizer(source=args.source, env_name=args.env)


if __name__ == "__main__":
    main()
