"""
B02 – Replay Buffer Demo
========================
Speichert (obs, action, reward, next_obs, done) als Ringpuffer.
Demonstriert: Befüllen, Sampling, Statistiken, Visualisierung.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import random
import time


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Ringpuffer für Transitions: (obs, action, reward, next_obs, done)

    - Maximale Kapazität: max_size Einträge
    - Wenn voll: älteste Einträge werden überschrieben
    - Sample: zufällige Mini-Batches für das Training
    """

    def __init__(self, max_size: int, obs_shape: tuple):
        self.max_size   = max_size
        self.obs_shape  = obs_shape
        self.ptr        = 0       # Zeiger auf nächste Schreibposition
        self.size       = 0       # Aktuelle Anzahl gespeicherter Transitions

        # Pre-allokierte numpy Arrays (effizienter als Liste von Dicts)
        self.obs        = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.next_obs   = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.actions    = np.zeros((max_size,),             dtype=np.int32)
        self.rewards    = np.zeros((max_size,),             dtype=np.float32)
        self.dones      = np.zeros((max_size,),             dtype=np.bool_)

        # Statistik-Tracking
        self.reward_history = deque(maxlen=1000)
        self.write_count    = 0

    def add(self, obs, action, reward, next_obs, done):
        """Eine Transition in den Buffer schreiben."""
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done

        # Ringpuffer: Zeiger weiterbewegen
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.reward_history.append(reward)
        self.write_count += 1

    def sample(self, batch_size: int) -> dict:
        """Zufälligen Mini-Batch aus dem Buffer ziehen."""
        assert self.size >= batch_size, (
            f"Nicht genug Daten: {self.size} < {batch_size}"
        )
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs":      self.obs[indices].astype(np.float32) / 255.0,
            "next_obs": self.next_obs[indices].astype(np.float32) / 255.0,
            "actions":  self.actions[indices],
            "rewards":  self.rewards[indices],
            "dones":    self.dones[indices],
            "indices":  indices,
        }

    def is_ready(self, min_size: int) -> bool:
        """Prüft ob genug Daten für ein Training vorhanden sind."""
        return self.size >= min_size

    @property
    def fill_ratio(self) -> float:
        return self.size / self.max_size

    def stats(self) -> dict:
        if self.size == 0:
            return {}
        return {
            "size":         self.size,
            "max_size":     self.max_size,
            "fill_%":       f"{self.fill_ratio * 100:.1f}%",
            "write_count":  self.write_count,
            "reward_mean":  np.mean(self.rewards[:self.size]),
            "reward_std":   np.std(self.rewards[:self.size]),
            "done_ratio":   np.mean(self.dones[:self.size]),
        }


# ─────────────────────────────────────────────
# MINI-ENVIRONMENT MOCK (ohne MiniWorld-Start)
# ─────────────────────────────────────────────

class MockEnv:
    """
    Leichtgewichtige Simulation für den Buffer-Demo.
    Erzeugt zufällige 16x16 RGB-Bilder als Beobachtungen.
    Kann später durch die echte MiniWorld-Env ersetzt werden.
    """
    OBS_SHAPE   = (16, 16, 3)
    N_ACTIONS   = 4  # vor, zurück, links, rechts

    def reset(self):
        self.step_count = 0
        return self._obs(), {}

    def step(self, action):
        self.step_count += 1
        obs     = self._obs()
        reward  = random.gauss(0.0, 0.5)          # Zufälliger Reward
        done    = self.step_count >= random.randint(10, 30)  # Episode-Ende
        return obs, reward, done, False, {}

    def _obs(self):
        return np.random.randint(0, 256, self.OBS_SHAPE, dtype=np.uint8)


# ─────────────────────────────────────────────
# DEMO – VISUALISIERUNG
# ─────────────────────────────────────────────

def run_demo():
    BUFFER_SIZE = 500
    MIN_TRAIN   = 64
    BATCH_SIZE  = 32
    N_STEPS     = 300

    env    = MockEnv()
    buffer = ReplayBuffer(max_size=BUFFER_SIZE, obs_shape=MockEnv.OBS_SHAPE)

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('B02 – Replay Buffer Demo', fontsize=15, fontweight='bold')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_fill     = fig.add_subplot(gs[0, 0])   # Füllstand
    ax_rewards  = fig.add_subplot(gs[0, 1])   # Reward-Verlauf
    ax_done     = fig.add_subplot(gs[0, 2])   # Done-Ratio
    ax_obs      = fig.add_subplot(gs[1, 0])   # Aktuelles Bild
    ax_sample   = fig.add_subplot(gs[1, 1])   # Gesampletes Bild
    ax_stats    = fig.add_subplot(gs[1, 2])   # Text-Statistiken
    ax_stats.axis('off')

    fill_history    = []
    reward_history  = []
    done_history    = []
    sample_log      = []   # (step, batch_size)

    obs, _ = env.reset()

    print(f"Starte Demo: {N_STEPS} Schritte, Buffer-Größe: {BUFFER_SIZE}")
    print(f"Training startet ab: {MIN_TRAIN} Transitions\n")

    for step in range(N_STEPS):
        action              = random.randint(0, MockEnv.N_ACTIONS - 1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done                = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Statistiken sammeln
        fill_history.append(buffer.fill_ratio * 100)
        reward_history.append(reward)
        done_history.append(float(done))

        # Sampling sobald genug Daten
        sampled_batch = None
        if buffer.is_ready(MIN_TRAIN) and step % 10 == 0:
            sampled_batch = buffer.sample(BATCH_SIZE)
            sample_log.append(step)

        # Live-Update alle 20 Schritte
        if step % 20 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(fill_history)))

            # Füllstand
            ax_fill.clear()
            ax_fill.plot(steps_x, fill_history, color='steelblue', linewidth=1.5)
            ax_fill.axhline(100, color='red', linestyle='--', linewidth=1, label='Max')
            ax_fill.axhline(
                (MIN_TRAIN / BUFFER_SIZE) * 100,
                color='orange', linestyle='--', linewidth=1, label='Min Train'
            )
            ax_fill.set_title('Buffer-Füllstand (%)')
            ax_fill.set_ylim(0, 110)
            ax_fill.set_xlabel('Schritt')
            ax_fill.legend(fontsize=7)

            # Reward
            ax_rewards.clear()
            ax_rewards.plot(steps_x, reward_history, color='green',
                            alpha=0.5, linewidth=1, label='Reward')
            # Gleitender Durchschnitt
            if len(reward_history) >= 20:
                ma = np.convolve(reward_history, np.ones(20)/20, mode='valid')
                ax_rewards.plot(range(19, len(reward_history)), ma,
                                color='darkgreen', linewidth=2, label='MA-20')
            for s in sample_log:
                if s <= step:
                    ax_rewards.axvline(s, color='purple', alpha=0.15, linewidth=1)
            ax_rewards.set_title('Rewards & Sampling (lila)')
            ax_rewards.set_xlabel('Schritt')
            ax_rewards.legend(fontsize=7)

            # Done-Ratio
            ax_done.clear()
            window = 30
            if len(done_history) >= window:
                done_ma = np.convolve(done_history, np.ones(window)/window, mode='valid')
                ax_done.plot(range(window - 1, len(done_history)), done_ma,
                             color='tomato', linewidth=1.5)
            ax_done.set_title(f'Episode-Ende-Rate (MA-{window})')
            ax_done.set_ylim(0, 1)
            ax_done.set_xlabel('Schritt')

            # Aktuelles Bild
            ax_obs.clear()
            ax_obs.imshow(obs, interpolation='nearest')
            ax_obs.set_title(f'Aktuelles Obs (Step {step})')
            ax_obs.axis('off')

            # Gesampletes Bild
            ax_sample.clear()
            if sampled_batch is not None:
                sample_img = (sampled_batch["obs"][0] * 255).astype(np.uint8)
                ax_sample.imshow(sample_img, interpolation='nearest')
                ax_sample.set_title(
                    f'Sample aus Buffer\n'
                    f'Action={sampled_batch["actions"][0]} '
                    f'R={sampled_batch["rewards"][0]:.2f}'
                )
            else:
                ax_sample.text(0.5, 0.5, f'Warte auf\n{MIN_TRAIN} Transitions...',
                               ha='center', va='center', transform=ax_sample.transAxes,
                               fontsize=9, color='gray')
                ax_sample.set_title('Sample aus Buffer')
            ax_sample.axis('off')

            # Statistiken
            ax_stats.clear()
            ax_stats.axis('off')
            stats = buffer.stats()
            if stats:
                lines = [
                    "── Buffer Statistiken ──",
                    f"Einträge:     {stats['size']} / {stats['max_size']}",
                    f"Füllstand:    {stats['fill_%']}",
                    f"Geschrieben:  {stats['write_count']}",
                    f"Reward Ø:     {stats['reward_mean']:.3f}",
                    f"Reward σ:     {stats['reward_std']:.3f}",
                    f"Done-Ratio:   {stats['done_ratio']:.3f}",
                    "",
                    f"Batch-Size:   {BATCH_SIZE}",
                    f"Samplings:    {len(sample_log)}",
                    "",
                    "Training: " + ("✓ AKTIV" if buffer.is_ready(MIN_TRAIN) else "⏳ Warten..."),
                    ]
                ax_stats.text(0.05, 0.95, "\n".join(lines),
                              transform=ax_stats.transAxes,
                              fontsize=9, verticalalignment='top',
                              fontfamily='monospace',
                              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            plt.pause(0.01)

    print("\nDemo abgeschlossen!")
    print("Buffer-Statistiken:")
    for k, v in buffer.stats().items():
        print(f"  {k:15s}: {v}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_demo()
