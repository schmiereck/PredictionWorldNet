"""
B03 – Nicht-linearer Zeitstempel-Buffer Demo
=============================================
Speichert Frames bei nicht-linearen Zeitabständen in der Vergangenheit.
Zeitskala: t=1, 2, 4, 8, 16 Schritte zurück (logarithmisch).

Idee: Kurze Vergangenheit = hohe Auflösung (jeder Schritt)
      Lange Vergangenheit = niedrige Auflösung (nur Snapshots)

Wird später als Input für den Temporal Transformer (B07) verwendet.
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
    Hält einen vollständigen Ringpuffer aller letzten Frames
    und gibt auf Anfrage Snapshots bei nicht-linearen Zeitabständen zurück.

    Zeitskala (Schritte in die Vergangenheit):
        [1, 2, 4, 8, 16]  →  logarithmische Abstände

    Aufbau:
        _ring  : Ringpuffer der letzten `max_lag` Frames (vollständig)
        get()  : Gibt die Frames bei den definierten Zeitabständen zurück

    Vorteil gegenüber festem Abstand:
        - Kurze Vergangenheit: feingranular  (für schnelle Reaktionen)
        - Lange Vergangenheit: grob          (für langfristige Planung)
    """

    # Zeitabstände in Schritten (konfigurierbar)
    TIME_STEPS = [1, 2, 4, 8, 16]

    def __init__(self, obs_shape: tuple, time_steps: list = None):
        self.obs_shape  = obs_shape
        self.time_steps = time_steps or self.TIME_STEPS
        self.max_lag    = max(self.time_steps)  # Größter benötigter Rückblick
        self.n_slots    = len(self.time_steps)

        # Ringpuffer: speichert die letzten `max_lag + 1` Frames
        # Initialisiert mit schwarzen Bildern (Null-Frames)
        self._ring      = deque(
            [np.zeros(obs_shape, dtype=np.uint8)] * (self.max_lag + 1),
            maxlen=self.max_lag + 1
        )
        self.step_count = 0

    def add(self, obs: np.ndarray):
        """Neuen Frame in den Ringpuffer schreiben."""
        self._ring.append(obs.copy())
        self.step_count += 1

    def get_temporal_frames(self) -> dict:
        """
        Gibt Frames bei den definierten Zeitabständen zurück.

        Returns:
            dict mit:
              "frames"      : Array (n_slots, H, W, C) – die historischen Frames
              "time_steps"  : Liste der Zeitabstände
              "ages"        : Tatsächliches Alter in Schritten (= min(t, step_count))
              "valid"       : Bool-Array – False wenn noch kein echter Frame vorhanden
        """
        ring_list = list(self._ring)   # Index 0 = ältester, -1 = aktuellster
        n         = len(ring_list)

        frames = []
        ages   = []
        valid  = []

        for t in self.time_steps:
            idx = n - 1 - t            # Position im Ringpuffer
            if idx >= 0:
                frames.append(ring_list[idx])
                ages.append(t)
                valid.append(self.step_count >= t)
            else:
                # Noch nicht genug Frames → Null-Frame
                frames.append(np.zeros(self.obs_shape, dtype=np.uint8))
                ages.append(t)
                valid.append(False)

        return {
            "frames":     np.stack(frames),          # (n_slots, H, W, C)
            "time_steps": self.time_steps,
            "ages":       ages,
            "valid":      valid,
        }

    def get_current(self) -> np.ndarray:
        """Gibt den aktuellsten Frame zurück."""
        return list(self._ring)[-1]

    def is_ready(self) -> bool:
        """True wenn alle Zeitslots mit echten Frames befüllt sind."""
        return self.step_count >= self.max_lag

    @property
    def fill_ratio(self) -> float:
        return min(self.step_count / self.max_lag, 1.0)

    def stats(self) -> dict:
        return {
            "step_count":   self.step_count,
            "max_lag":      self.max_lag,
            "n_slots":      self.n_slots,
            "time_steps":   self.time_steps,
            "is_ready":     self.is_ready(),
            "fill_%":       f"{self.fill_ratio * 100:.1f}%",
        }


# ─────────────────────────────────────────────
# MOCK ENV – Farbige Frames mit Zeitstempel
# ─────────────────────────────────────────────

class MockEnv:
    """
    Erzeugt Frames mit sichtbarem Zeitfortschritt:
    - Hintergrundfarbe wechselt graduell (simuliert Bewegung)
    - Weißer Zähler-Balken am unteren Rand (visueller Fortschritt)
    """
    OBS_SHAPE = (16, 16, 3)

    def __init__(self):
        self.step_count = 0

    def step(self) -> np.ndarray:
        self.step_count += 1
        return self._make_frame(self.step_count)

    def _make_frame(self, t: int) -> np.ndarray:
        frame = np.zeros(self.OBS_SHAPE, dtype=np.uint8)

        # Hintergrund: Farbverlauf über Zeit (R-Kanal steigt, B-Kanal fällt)
        r = int((np.sin(t * 0.15) * 0.5 + 0.5) * 200)
        g = int((np.sin(t * 0.07 + 1) * 0.5 + 0.5) * 200)
        b = int((np.cos(t * 0.10) * 0.5 + 0.5) * 200)
        frame[:, :] = [r, g, b]

        # Weißer Fortschritts-Balken unten (zeigt Zeitschritt visuell)
        bar_width = (t % 16) + 1
        frame[14:16, :bar_width] = [255, 255, 255]

        return frame


# ─────────────────────────────────────────────
# DEMO – VISUALISIERUNG
# ─────────────────────────────────────────────

def run_demo():
    N_STEPS     = 120
    TIME_STEPS  = [1, 2, 4, 8, 16]
    OBS_SHAPE   = MockEnv.OBS_SHAPE

    env     = MockEnv()
    tbuf    = TemporalBuffer(obs_shape=OBS_SHAPE, time_steps=TIME_STEPS)

    # ── Matplotlib Setup ──────────────────────────────────
    n_slots = len(TIME_STEPS)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle('B03 – Nicht-linearer Zeitstempel-Buffer', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, n_slots + 1, figure=fig, hspace=0.55, wspace=0.35)

    # Zeile 0: aktuelles Bild + Füllstand
    ax_current  = fig.add_subplot(gs[0, 0])
    ax_fill     = fig.add_subplot(gs[0, 1:])

    # Zeile 1: die 5 historischen Frames
    ax_frames   = [fig.add_subplot(gs[1, i]) for i in range(n_slots)]
    ax_frames.append(fig.add_subplot(gs[1, n_slots]))  # Platzhalter

    # Zeile 2: Zeitachsen-Diagramm + Statistiken
    ax_timeline = fig.add_subplot(gs[2, :n_slots])
    ax_stats    = fig.add_subplot(gs[2, n_slots])
    ax_stats.axis('off')

    # Tracking
    fill_history    = []
    valid_history   = [[] for _ in TIME_STEPS]

    print(f"Starte Demo: {N_STEPS} Schritte")
    print(f"Zeitabstände: {TIME_STEPS} Schritte")
    print(f"Buffer bereit ab Schritt: {max(TIME_STEPS)}\n")

    for step in range(N_STEPS):
        obs = env.step()
        tbuf.add(obs)

        temporal = tbuf.get_temporal_frames()

        # Tracking
        fill_history.append(tbuf.fill_ratio * 100)
        for i, v in enumerate(temporal["valid"]):
            valid_history[i].append(float(v))

        if step % 5 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(fill_history)))

            # ── Aktuelles Bild ─────────────────────────
            ax_current.clear()
            ax_current.imshow(obs, interpolation='nearest')
            ax_current.set_title(f'Aktuell\nStep {step + 1}', fontsize=8)
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
            for i, (t, frame, is_valid) in enumerate(zip(
                    temporal["time_steps"],
                    temporal["frames"],
                    temporal["valid"]
            )):
                ax_frames[i].clear()
                ax_frames[i].imshow(frame, interpolation='nearest')
                status = "OK" if is_valid else "--"
                ax_frames[i].set_title(f't-{t} {status}', fontsize=8)
                ax_frames[i].axis('off')

                # Roter Rahmen wenn noch kein gültiger Frame
                for spine in ax_frames[i].spines.values():
                    spine.set_edgecolor('green' if is_valid else 'red')
                    spine.set_linewidth(2)

            ax_frames[n_slots].axis('off')

            # ── Zeitachsen-Diagramm ─────────────────────
            ax_timeline.clear()
            colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_slots))

            for i, (t, color) in enumerate(zip(TIME_STEPS, colors)):
                if len(valid_history[i]) > 0:
                    ax_timeline.plot(
                        steps_x, valid_history[i],
                        color=color, linewidth=1.5,
                        label=f't-{t}', alpha=0.8
                    )

            # Markierungen wo jeder Slot "aufgeht"
            for t in TIME_STEPS:
                ax_timeline.axvline(t - 1, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            ax_timeline.set_title('Slot-Verfügbarkeit über Zeit (0=leer, 1=gültig)')
            ax_timeline.set_xlabel('Schritt')
            ax_timeline.set_ylim(-0.1, 1.2)
            ax_timeline.legend(fontsize=7, loc='lower right', ncol=n_slots)

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            stats = tbuf.stats()
            valid_flags = temporal["valid"]
            lines = [
                "── Temporal Buffer ──",
                f"Schritt:    {stats['step_count']}",
                f"Füllstand:  {stats['fill_%']}",
                f"Bereit:     {'JA' if stats['is_ready'] else 'NEIN'}",
                "",
                "── Slots ────────────",
            ]
            for t, v in zip(TIME_STEPS, valid_flags):
                lines.append(f"  t-{t:>2}:  {'OK  gueltig' if v else '--  warte  '}")

            ax_stats.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.05)

    print("\nDemo abgeschlossen!")
    print("Finale Statistiken:")
    for k, v in tbuf.stats().items():
        print(f"  {k:15s}: {v}")

    # ── Finales Bild: Zeitachsen-Vergleich ─────────────
    fig2, axes2 = plt.subplots(1, n_slots + 1, figsize=(14, 3))
    fig2.suptitle(f'Finaler Snapshot – Step {N_STEPS}: Aktuell vs. historische Frames', fontsize=12)

    current = tbuf.get_current()
    axes2[0].imshow(current, interpolation='nearest')
    axes2[0].set_title('Aktuell (t=0)', fontsize=9)
    axes2[0].axis('off')

    temporal = tbuf.get_temporal_frames()
    for i, (t, frame) in enumerate(zip(temporal["time_steps"], temporal["frames"])):
        axes2[i + 1].imshow(frame, interpolation='nearest')
        axes2[i + 1].set_title(f't-{t} Schritte\nvor Step {N_STEPS}', fontsize=9)
        axes2[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
