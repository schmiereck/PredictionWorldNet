"""
B04 – CNN-Encoder Demo
======================
Kodiert ein 16x16 RGB-Bild in einen kompakten Latent-Vektor.

Architektur:
    Input  : (B, 3, 16, 16)   – Batch x RGB x H x W
    Conv1  : (B, 32, 8,  8)   – 32 Filter, Stride 2
    Conv2  : (B, 64, 4,  4)   – 64 Filter, Stride 2
    Conv3  : (B, 128, 2, 2)   – 128 Filter, Stride 2
    Flatten: (B, 512)
    FC     : (B, latent_dim)  – z.B. 64 oder 128

Wird später als Eingang für den Temporal Transformer (B07) verwendet.
Jeder der 6 Frames (aktuell + 5 historisch) wird einzeln durch den Encoder geschickt.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# CNN ENCODER
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Konvertiert ein 16x16 RGB-Bild in einen Latent-Vektor.

    Input:  (B, 3, 16, 16)   – normalisiert auf [0, 1]
    Output: (B, latent_dim)  – L2-normalisierter Latent-Vektor

    Aufbau:
        3x Conv2d mit BatchNorm + ReLU (je Stride=2 → halbiert Auflösung)
        Flatten → FC → LayerNorm

    Warum L2-Normalisierung am Ausgang?
        Stabilisiert das Training des nachgelagerten Transformers,
        da alle Latent-Vektoren auf der Einheitskugel liegen.
    """

    def __init__(self, latent_dim: int = 64, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional Backbone
        self.conv = nn.Sequential(
            # Block 1: (3, 16, 16) → (32, 8, 8)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (32, 8, 8) → (64, 4, 4)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, 4, 4) → (128, 2, 2)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Berechne Flatten-Größe automatisch
        self._flat_size = self._get_flat_size(in_channels)

        # Fully Connected → Latent
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def _get_flat_size(self, in_channels: int) -> int:
        """Berechnet die Größe nach dem Flatten automatisch."""
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 16, 16)
            out   = self.conv(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 16, 16) – normalisiert auf [0, 1]
        Returns:
            z: (B, latent_dim) – L2-normalisierter Latent-Vektor
        """
        features = self.conv(x)                    # (B, 128, 2, 2)
        flat     = features.view(features.size(0), -1)  # (B, 512)
        z        = self.fc(flat)                   # (B, latent_dim)
        z        = F.normalize(z, dim=-1)          # L2-Normalisierung
        return z

    def encode_numpy(self, obs: np.ndarray) -> np.ndarray:
        """
        Hilfsfunktion: numpy uint8 (H,W,3) → numpy float (latent_dim,)
        Für einfache Verwendung außerhalb des Trainingsprozesses.
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs).float() / 255.0   # [0,1]
            x = x.permute(2, 0, 1).unsqueeze(0)         # (1, 3, H, W)
            z = self.forward(x)
            return z.squeeze(0).numpy()

    def summary(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "latent_dim":   self.latent_dim,
            "flat_size":    self._flat_size,
            "params_total": total,
            "params_train": trainable,
        }


# ─────────────────────────────────────────────
# MOCK ENV
# ─────────────────────────────────────────────

class MockEnv:
    OBS_SHAPE = (16, 16, 3)

    def __init__(self):
        self.t = 0

    def step(self) -> np.ndarray:
        self.t += 1
        frame = np.zeros(self.OBS_SHAPE, dtype=np.uint8)
        r = int((np.sin(self.t * 0.15) * 0.5 + 0.5) * 200)
        g = int((np.sin(self.t * 0.07 + 1) * 0.5 + 0.5) * 200)
        b = int((np.cos(self.t * 0.10) * 0.5 + 0.5) * 200)
        frame[:, :] = [r, g, b]
        bar_width = (self.t % 16) + 1
        frame[14:16, :bar_width] = [255, 255, 255]
        return frame


# ─────────────────────────────────────────────
# DEMO – VISUALISIERUNG
# ─────────────────────────────────────────────

def visualize_feature_maps(encoder: CNNEncoder, obs: np.ndarray, ax_list):
    """Zeigt die Aktivierungen der drei Conv-Layer als Heatmaps."""
    encoder.eval()
    with torch.no_grad():
        x = torch.from_numpy(obs).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)

        # Layer-weise durchlaufen
        # Block 1
        out1 = encoder.conv[:3](x)   # (1, 32, 8, 8)
        # Block 2
        out2 = encoder.conv[3:6](out1)  # (1, 64, 4, 4)
        # Block 3
        out3 = encoder.conv[6:](out2)   # (1, 128, 2, 2)

    titles = ['Conv1 (32x8x8)', 'Conv2 (64x4x4)', 'Conv3 (128x2x2)']
    outputs = [out1, out2, out3]

    for ax, title, out in zip(ax_list, titles, outputs):
        ax.clear()
        # Mittlere Aktivierung über alle Filter (zeigt "was wurde wo aktiviert")
        heatmap = out.squeeze(0).mean(0).numpy()
        im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.axis('off')


def run_demo():
    LATENT_DIM  = 64
    N_STEPS     = 80

    env     = MockEnv()
    encoder = CNNEncoder(latent_dim=LATENT_DIM)

    # Info ausgeben
    info = encoder.summary()
    print("CNN Encoder initialisiert:")
    print(f"  Latent Dim  : {info['latent_dim']}")
    print(f"  Flatten Size: {info['flat_size']}")
    print(f"  Parameter   : {info['params_total']:,}")
    print()

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle('B04 – CNN Encoder Demo', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    ax_obs      = fig.add_subplot(gs[0, 0])      # Eingabebild
    ax_latent   = fig.add_subplot(gs[0, 1:])     # Latent-Vektor als Balkendiagramm
    ax_feat     = [fig.add_subplot(gs[1, i]) for i in range(3)]  # Feature Maps
    ax_feat_placeholder = fig.add_subplot(gs[1, 3])
    ax_feat_placeholder.axis('off')
    ax_cosine   = fig.add_subplot(gs[2, :2])     # Cosinus-Ähnlichkeit über Zeit
    ax_norm     = fig.add_subplot(gs[2, 2])      # L2-Norm (sollte ~1 sein)
    ax_stats    = fig.add_subplot(gs[2, 3])      # Statistiken
    ax_stats.axis('off')

    # Tracking
    latent_history  = []   # Liste von Latent-Vektoren
    cosine_history  = []   # Ähnlichkeit zwischen aufeinanderfolgenden Frames
    norm_history    = []   # L2-Norm der Vektoren

    prev_z = None

    print(f"Starte Demo: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        obs = env.step()
        z   = encoder.encode_numpy(obs)          # (latent_dim,)

        latent_history.append(z.copy())
        norm_history.append(float(np.linalg.norm(z)))

        # Cosinus-Ähnlichkeit zum vorherigen Frame
        if prev_z is not None:
            cos_sim = float(np.dot(z, prev_z))   # Bereits L2-normalisiert
            cosine_history.append(cos_sim)
        prev_z = z.copy()

        if step % 5 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(norm_history)))

            # ── Eingabebild ────────────────────────────
            ax_obs.clear()
            ax_obs.imshow(obs, interpolation='nearest')
            ax_obs.set_title(f'Eingabe\nStep {step + 1}', fontsize=8)
            ax_obs.axis('off')

            # ── Latent-Vektor ──────────────────────────
            ax_latent.clear()
            colors = ['steelblue' if v >= 0 else 'tomato' for v in z]
            ax_latent.bar(range(LATENT_DIM), z, color=colors, width=1.0)
            ax_latent.axhline(0, color='black', linewidth=0.5)
            ax_latent.set_title(f'Latent-Vektor z (dim={LATENT_DIM})  |  L2-Norm: {np.linalg.norm(z):.4f}', fontsize=9)
            ax_latent.set_xlabel('Dimension')
            ax_latent.set_ylabel('Wert')
            ax_latent.set_ylim(-0.4, 0.4)

            # ── Feature Maps ──────────────────────────
            visualize_feature_maps(encoder, obs, ax_feat)

            # ── Cosinus-Ähnlichkeit ────────────────────
            ax_cosine.clear()
            if len(cosine_history) > 1:
                ax_cosine.plot(range(1, len(cosine_history) + 1),
                               cosine_history, color='purple', linewidth=1.5)
                # Gleitender Durchschnitt
                if len(cosine_history) >= 10:
                    ma = np.convolve(cosine_history, np.ones(10)/10, mode='valid')
                    ax_cosine.plot(range(10, len(cosine_history) + 1), ma,
                                   color='darkviolet', linewidth=2, label='MA-10')
                    ax_cosine.legend(fontsize=7)
            ax_cosine.set_title('Cosinus-Aehnlichkeit aufeinanderfolgender Frames')
            ax_cosine.set_xlabel('Schritt')
            ax_cosine.set_ylim(-1.1, 1.1)
            ax_cosine.axhline(0, color='gray', linestyle='--', linewidth=0.8)

            # ── L2-Norm ────────────────────────────────
            ax_norm.clear()
            ax_norm.plot(steps_x, norm_history, color='green', linewidth=1.5)
            ax_norm.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Soll=1.0')
            ax_norm.set_title('L2-Norm der Latent-Vektoren')
            ax_norm.set_xlabel('Schritt')
            ax_norm.set_ylim(0, 1.5)
            ax_norm.legend(fontsize=7)

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            mean_cos = np.mean(cosine_history) if cosine_history else 0.0
            lines = [
                "── CNN Encoder ──────",
                f"Input:      16x16x3",
                f"Latent Dim: {LATENT_DIM}",
                f"Flat Size:  {info['flat_size']}",
                f"Parameter:  {info['params_total']:,}",
                "",
                "── Laufzeit ─────────",
                f"Schritt:    {step + 1}",
                f"L2-Norm:    {norm_history[-1]:.4f}",
                f"Cos-Sim Ø:  {mean_cos:.4f}",
                "",
                "── Architektur ──────",
                "Conv1: 3->32  s=2",
                "Conv2: 32->64 s=2",
                "Conv3: 64->128 s=2",
                f"FC:    512->{LATENT_DIM}",
                "LayerNorm + L2-Norm",
            ]
            ax_stats.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.05)

    # ── Finales Bild: Latent-Raum Ähnlichkeitsmatrix ───
    print("\nErstelle Aehnlichkeitsmatrix der letzten 30 Frames...")
    last_n  = 30
    vectors = np.stack(latent_history[-last_n:])    # (30, 64)
    sim_mat = vectors @ vectors.T                    # (30, 30) – Cosinus-Ähnlichkeit

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('B04 – Finale Analyse: Latent Space', fontsize=12)

    im = axes2[0].imshow(sim_mat, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
    axes2[0].set_title(f'Cosinus-Aehnlichkeitsmatrix\n(letzte {last_n} Frames)', fontsize=10)
    axes2[0].set_xlabel('Frame Index')
    axes2[0].set_ylabel('Frame Index')
    fig2.colorbar(im, ax=axes2[0])

    # Latent-Vektoren als Heatmap über Zeit
    axes2[1].imshow(vectors.T, cmap='coolwarm', aspect='auto',
                    vmin=-0.3, vmax=0.3, interpolation='nearest')
    axes2[1].set_title(f'Latent-Vektoren über Zeit\n(letzte {last_n} Frames)', fontsize=10)
    axes2[1].set_xlabel('Frame Index')
    axes2[1].set_ylabel('Latent-Dimension')

    plt.tight_layout()
    plt.show()

    print("\nDemo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
