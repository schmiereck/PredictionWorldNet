"""
B03 – Nicht-linearer Zeitstempel-Buffer Demo (v2)
==================================================
Speichert (Frame, Aktion)-Paare bei nicht-linearen Zeitabständen.
Zeitskala: t=1, 2, 4, 8, 16 Schritte zurück (logarithmisch).

v2-Änderung: Jeder Slot speichert jetzt zusätzlich die Aktion,
die zum jeweiligen Frame geführt hat:

    Slot t-8: (Frame_t-8, Action_t-8)
               ↑ was der Agent sah    ↑ was er dann tat

Warum nur eine Aktion pro Slot (nicht alle Zwischenaktionen)?
    - Kompakt: festes Format für den Transformer
    - Ausreichend: Das World Model lernt implizit die Dynamik
    - Konsistent mit DreamerV3 / RSSM Architektur

get_temporal_frames() gibt zurück:
    "frames"  : (n_slots, H, W, C)
    "actions" : (n_slots, action_dim)   ← neu
    "times"   : [1, 2, 4, 8, 16]
    "valid"   : [True, True, ...]

Wird als Input für den Temporal Transformer (B07) verwendet.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque


# ─────────────────────────────────────────────
# NICHT-LINEARER ZEITSTEMPEL-BUFFER
# ─────────────────────────────────────────────

class TemporalBuffer:
    """
    Hält einen vollständigen Ringpuffer aller letzten (Frame, Aktion)-Paare
    und gibt auf Anfrage Snapshots bei nicht-linearen Zeitabständen zurück.

    Zeitskala (Schritte in die Vergangenheit):
        [1, 2, 4, 8, 16]  →  logarithmische Abstände

    Aufbau:
        _ring  : Ringpuffer der letzten `max_lag` (Frame, Aktion)-Paare
        get()  : Gibt die Paare bei den definierten Zeitabständen zurück

    Vorteil gegenüber festem Abstand:
        - Kurze Vergangenheit: feingranular  (für schnelle Reaktionen)
        - Lange Vergangenheit: grob          (für langfristige Planung)
    """

    TIME_STEPS = [1, 2, 4, 8, 16]

    def __init__(self, obs_shape: tuple, action_dim: int = 3,
                 time_steps: list = None):
        self.obs_shape  = obs_shape
        self.action_dim = action_dim
        self.time_steps = time_steps or self.TIME_STEPS
        self.max_lag    = max(self.time_steps)
        self.n_slots    = len(self.time_steps)

        # Null-Werte für Initialisierung
        null_frame  = np.zeros(obs_shape, dtype=np.uint8)
        null_action = np.zeros(action_dim, dtype=np.float32)

        # Ringpuffer speichert (frame, action)-Tupel
        self._ring = deque(
            [(null_frame.copy(), null_action.copy())] * (self.max_lag + 1),
            maxlen=self.max_lag + 1
        )
        self.step_count = 0

    def add(self, obs: np.ndarray, action: np.ndarray):
        """
        Neues (Frame, Aktion)-Paar in den Ringpuffer schreiben.

        Args:
            obs    : (H, W, C) uint8  – aktuelle Beobachtung
            action : (action_dim,) float32  – Aktion die zu diesem Frame geführt hat
                     Format: [linear_x, angular_z, duration] normalisiert auf [-1, 1]
        """
        self._ring.append((obs.copy(), np.array(action, dtype=np.float32)))
        self.step_count += 1

    def get_temporal_frames(self) -> dict:
        """
        Gibt (Frame, Aktion)-Paare bei den definierten Zeitabständen zurück.

        Returns:
            dict mit:
              "frames"  : (n_slots, H, W, C)       – historische Frames
              "actions" : (n_slots, action_dim)     – zugehörige Aktionen  ← neu
              "times"   : Liste der Zeitabstände
              "valid"   : Bool-Array
        """
        ring_list = list(self._ring)
        n         = len(ring_list)

        frames  = []
        actions = []
        valid   = []

        for t in self.time_steps:
            idx = n - 1 - t
            if idx >= 0:
                frame, action = ring_list[idx]
                frames.append(frame)
                actions.append(action)
                valid.append(self.step_count >= t)
            else:
                frames.append(np.zeros(self.obs_shape, dtype=np.uint8))
                actions.append(np.zeros(self.action_dim, dtype=np.float32))
                valid.append(False)

        return {
            "frames":  np.stack(frames),    # (n_slots, H, W, C)
            "actions": np.stack(actions),   # (n_slots, action_dim)
            "times":   self.time_steps,
            "valid":   valid,
        }

    def get_current(self) -> tuple:
        """Gibt (Frame, Aktion) des aktuellsten Slots zurück."""
        return list(self._ring)[-1]

    def is_ready(self) -> bool:
        return self.step_count >= self.max_lag

    @property
    def fill_ratio(self) -> float:
        return min(self.step_count / self.max_lag, 1.0)

    def stats(self) -> dict:
        return {
            "step_count": self.step_count,
            "max_lag":    self.max_lag,
            "n_slots":    self.n_slots,
            "action_dim": self.action_dim,
            "time_steps": self.time_steps,
            "is_ready":   self.is_ready(),
            "fill_%":     f"{self.fill_ratio * 100:.1f}%",
        }


# ─────────────────────────────────────────────
# MOCK ENV – Farbige Frames mit Zeitstempel
# ─────────────────────────────────────────────

class MockEnv:
    """
    Erzeugt Frames mit sichtbarem Zeitfortschritt und zufällige ROS2-Aktionen.
    """
    OBS_SHAPE  = (16, 16, 3)
    ACTION_DIM = 3   # [linear_x, angular_z, duration] normalisiert [-1, 1]

    # Benannte Aktionen für die Visualisierung
    ACTION_NAMES = ["vorwaerts", "links", "rechts", "kurve_l", "kurve_r", "stopp"]
    ACTION_VECTORS = [
        [ 0.6,  0.0, -0.6],   # vorwaerts
        [ 0.0,  0.8, -0.6],   # links
        [ 0.0, -0.8, -0.6],   # rechts
        [ 0.4,  0.4, -0.3],   # kurve_links
        [ 0.4, -0.4, -0.3],   # kurve_rechts
        [ 0.0,  0.0, -1.0],   # stopp
    ]

    def __init__(self):
        self.step_count  = 0
        self.last_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self.last_action_name = "stopp"

    def step(self):
        self.step_count += 1
        # Zufällige Aktion aus den benannten Aktionen
        idx = np.random.randint(len(self.ACTION_NAMES))
        self.last_action = np.array(self.ACTION_VECTORS[idx], dtype=np.float32)
        self.last_action_name = self.ACTION_NAMES[idx]
        return self._make_frame(self.step_count), self.last_action

    def _make_frame(self, t: int) -> np.ndarray:
        frame = np.zeros(self.OBS_SHAPE, dtype=np.uint8)
        r = int((np.sin(t * 0.15) * 0.5 + 0.5) * 200)
        g = int((np.sin(t * 0.07 + 1) * 0.5 + 0.5) * 200)
        b = int((np.cos(t * 0.10) * 0.5 + 0.5) * 200)
        frame[:, :] = [r, g, b]
        bar_width = (t % 16) + 1
        frame[14:16, :bar_width] = [255, 255, 255]
        return frame


# ─────────────────────────────────────────────
# DEMO – VISUALISIERUNG
# ─────────────────────────────────────────────

def run_demo():
    N_STEPS    = 120
    TIME_STEPS = [1, 2, 4, 8, 16]
    OBS_SHAPE  = MockEnv.OBS_SHAPE
    ACTION_DIM = MockEnv.ACTION_DIM

    env  = MockEnv()
    tbuf = TemporalBuffer(obs_shape=OBS_SHAPE, action_dim=ACTION_DIM,
                          time_steps=TIME_STEPS)

    # ── Matplotlib Setup ──────────────────────────────────
    n_slots = len(TIME_STEPS)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('B03 v2 – Temporal Buffer mit (Frame, Aktion)-Paaren',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, n_slots + 1, figure=fig, hspace=0.6, wspace=0.35)

    ax_current = fig.add_subplot(gs[0, 0])
    ax_fill    = fig.add_subplot(gs[0, 1:])
    ax_frames  = [fig.add_subplot(gs[1, i]) for i in range(n_slots)]
    ax_frames.append(fig.add_subplot(gs[1, n_slots]))
    ax_actions = fig.add_subplot(gs[2, :n_slots])   # Aktions-Heatmap  ← neu
    ax_stats   = fig.add_subplot(gs[2, n_slots])
    ax_stats.axis('off')

    fill_history  = []
    valid_history = [[] for _ in TIME_STEPS]

    # Aktions-Dimensionsnamen für Achsenbeschriftung
    action_dim_names = ["lin_x", "ang_z", "dur"]

    print(f"Starte Demo: {N_STEPS} Schritte")
    print(f"Zeitabstaende: {TIME_STEPS} Schritte")
    print(f"Action-Dim: {ACTION_DIM} [linear_x, angular_z, duration]")
    print(f"Buffer bereit ab Schritt: {max(TIME_STEPS)}\n")

    for step in range(N_STEPS):
        obs, action = env.step()
        tbuf.add(obs, action)

        temporal = tbuf.get_temporal_frames()

        fill_history.append(tbuf.fill_ratio * 100)
        for i, v in enumerate(temporal["valid"]):
            valid_history[i].append(float(v))

        if step % 5 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(fill_history)))

            # ── Aktuelles Bild ─────────────────────────
            ax_current.clear()
            ax_current.imshow(obs, interpolation='nearest')
            ax_current.set_title(
                f'Aktuell\nStep {step + 1}\n{env.last_action_name}',
                fontsize=7
            )
            ax_current.axis('off')

            # ── Füllstand ──────────────────────────────
            ax_fill.clear()
            ax_fill.plot(steps_x, fill_history, color='steelblue', linewidth=1.5)
            ax_fill.axhline(100, color='green', linestyle='--', linewidth=1, label='Bereit')
            ax_fill.set_title('Buffer-Füllstand (%)')
            ax_fill.set_ylim(0, 110)
            ax_fill.set_xlabel('Schritt')
            ax_fill.legend(fontsize=7)

            # ── Historische Frames ──────────────────────
            for i, (t, frame, act, is_valid) in enumerate(zip(
                    temporal["times"],
                    temporal["frames"],
                    temporal["actions"],
                    temporal["valid"]
            )):
                ax_frames[i].clear()
                ax_frames[i].imshow(frame, interpolation='nearest')
                status = "OK" if is_valid else "--"
                # Aktion als kompakten Text anzeigen
                act_str = f"lx={act[0]:+.1f}\naz={act[1]:+.1f}"
                ax_frames[i].set_title(f't-{t} {status}\n{act_str}', fontsize=6)
                ax_frames[i].axis('off')
                for spine in ax_frames[i].spines.values():
                    spine.set_edgecolor('green' if is_valid else 'red')
                    spine.set_linewidth(2)

            ax_frames[n_slots].axis('off')

            # ── Aktions-Heatmap  ← neu ─────────────────
            ax_actions.clear()
            act_matrix = temporal["actions"]   # (n_slots, action_dim)
            im = ax_actions.imshow(
                act_matrix.T, cmap='coolwarm', aspect='auto',
                vmin=-1, vmax=1, interpolation='nearest'
            )
            ax_actions.set_xticks(range(n_slots))
            ax_actions.set_xticklabels(
                [f't-{t}' for t in temporal["times"]], fontsize=8
            )
            ax_actions.set_yticks(range(ACTION_DIM))
            ax_actions.set_yticklabels(action_dim_names, fontsize=8)
            ax_actions.set_title(
                'Aktions-Heatmap pro Slot  [blau=negativ, rot=positiv]',
                fontsize=9
            )
            # Werte in Zellen
            for r in range(ACTION_DIM):
                for c in range(n_slots):
                    ax_actions.text(
                        c, r, f"{act_matrix[c, r]:+.2f}",
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(act_matrix[c, r]) > 0.5 else 'black'
                    )

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            stats = tbuf.stats()
            valid_flags = temporal["valid"]
            lines = [
                "── Temporal Buffer v2 ──",
                f"Schritt:    {stats['step_count']}",
                f"Fuellstand: {stats['fill_%']}",
                f"Bereit:     {'JA' if stats['is_ready'] else 'NEIN'}",
                f"Action-Dim: {stats['action_dim']}",
                "",
                "── Slots ───────────────",
            ]
            for t, v, act in zip(TIME_STEPS, valid_flags, temporal["actions"]):
                status = "OK" if v else "--"
                lines.append(
                    f"  t-{t:>2}: {status} "
                    f"lx={act[0]:+.2f} az={act[1]:+.2f}"
                )

            ax_stats.text(
                0.02, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.05)

    print("\nDemo abgeschlossen!")
    print("Finale Statistiken:")
    for k, v in tbuf.stats().items():
        print(f"  {k:12s}: {v}")

    # ── Finales Bild: Snapshot ──────────────────────────
    temporal = tbuf.get_temporal_frames()
    current_frame, current_action = tbuf.get_current()

    fig2, axes2 = plt.subplots(2, n_slots + 1, figsize=(14, 5))
    fig2.suptitle(
        f'Finaler Snapshot – Step {N_STEPS}: Frames + Aktionen',
        fontsize=12
    )

    # Zeile 0: Frames
    axes2[0, 0].imshow(current_frame, interpolation='nearest')
    axes2[0, 0].set_title('Aktuell\n(t=0)', fontsize=8)
    axes2[0, 0].axis('off')

    for i, (t, frame, act) in enumerate(zip(
            temporal["times"], temporal["frames"], temporal["actions"]
    )):
        axes2[0, i+1].imshow(frame, interpolation='nearest')
        axes2[0, i+1].set_title(f't-{t}', fontsize=8)
        axes2[0, i+1].axis('off')

        # Zeile 1: Aktions-Balkendiagramm
        colors = ['tomato' if v < 0 else 'steelblue' for v in act]
        axes2[1, i+1].bar(action_dim_names, act, color=colors)
        axes2[1, i+1].set_ylim(-1.1, 1.1)
        axes2[1, i+1].axhline(0, color='black', linewidth=0.5)
        axes2[1, i+1].set_title(f'Aktion\nt-{t}', fontsize=7)
        axes2[1, i+1].tick_params(labelsize=6)

    # Aktueller Slot Aktion
    colors = ['tomato' if v < 0 else 'steelblue' for v in current_action]
    axes2[1, 0].bar(action_dim_names, current_action, color=colors)
    axes2[1, 0].set_ylim(-1.1, 1.1)
    axes2[1, 0].axhline(0, color='black', linewidth=0.5)
    axes2[1, 0].set_title('Aktion\n(aktuell)', fontsize=7)
    axes2[1, 0].tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
