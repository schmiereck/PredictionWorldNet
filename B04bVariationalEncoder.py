"""
B04b – Variational CNN Encoder (VAE Encoder) Demo
==================================================
Erweiterung von B04: Statt eines festen Latent-Vektors z
lernt der Encoder eine VERTEILUNG über den Latent Space.

Unterschied zu B04:
    B04:  z = encoder(obs)              → Vektor (deterministisch)
    B04b: mu, log_var = encoder(obs)    → Verteilung (stochastisch)
          z = mu + eps * std            → Sampling via Reparametrization Trick

Warum das wichtig ist (Active Inference):
    - mu       = "Was der Agent glaubt zu sehen"
    - std      = "Wie sicher ist der Agent"
    - KL-Loss  = Free Energy Komplexitäts-Term
                 → zwingt den Posterior nahe am Prior P(z) = N(0,1) zu bleiben
    - Recon-Loss = Free Energy Ungenauigkeits-Term
                 → zwingt z die Beobachtung gut zu erklären

Free Energy = Reconstruction Loss + Beta * KL-Divergenz
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
# VARIATIONAL CNN ENCODER
# ─────────────────────────────────────────────

class VariationalCNNEncoder(nn.Module):
    """
    Variational Encoder: Bild → Verteilung im Latent Space.

    Input:  (B, 3, 16, 16)
    Output: mu (B, latent_dim), log_var (B, latent_dim), z (B, latent_dim)

    Reparametrization Trick:
        z = mu + eps * exp(0.5 * log_var)   mit eps ~ N(0, I)
        → ermöglicht Backpropagation durch den Sampling-Schritt

    Zwei Köpfe am Ende des Encoders:
        fc_mu      → Mittelwert der Verteilung
        fc_log_var → Log-Varianz der Verteilung (log statt var für numerische Stabilität)
    """

    def __init__(self, latent_dim: int = 64, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim

        # Geteilter Convolutional Backbone (identisch mit B04)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self._flat_size = self._get_flat_size(in_channels)

        # Zwei separate Köpfe statt einem FC
        self.fc_mu      = nn.Linear(self._flat_size, latent_dim)
        self.fc_log_var = nn.Linear(self._flat_size, latent_dim)

    def _get_flat_size(self, in_channels: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 16, 16)
            return int(self.conv(dummy).numel())

    def forward(self, x: torch.Tensor):
        """
        Returns:
            mu      : (B, latent_dim) – Mittelwert
            log_var : (B, latent_dim) – Log-Varianz
            z       : (B, latent_dim) – Gesampelter Latent-Vektor
        """
        features = self.conv(x)
        flat     = features.reshape(features.size(0), -1)

        mu      = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)

        # Clamp log_var für numerische Stabilität
        log_var = torch.clamp(log_var, min=-10, max=10)

        z = self._reparametrize(mu, log_var)
        return mu, log_var, z

    def _reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparametrization Trick:
            z = mu + eps * std   mit eps ~ N(0, I)
        Nur während Training stochastisch, während eval() deterministisch (z = mu).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Im Eval-Modus: deterministisch

    def encode_numpy(self, obs: np.ndarray):
        """numpy (H,W,3) → (mu, std, z) als numpy Arrays."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0)
            mu, log_var, z = self.forward(x)
            std = torch.exp(0.5 * log_var)
            return mu.squeeze(0).numpy(), std.squeeze(0).numpy(), z.squeeze(0).numpy()

    def summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "latent_dim":   self.latent_dim,
            "flat_size":    self._flat_size,
            "params_total": total,
            # B04 hatte 1 FC-Kopf, B04b hat 2 → leicht mehr Parameter
            "extra_params": 2 * self._flat_size * self.latent_dim,
        }


# ─────────────────────────────────────────────
# EINFACHER CNN DECODER (für Rekonstruktion)
# ─────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    Dekodiert einen Latent-Vektor zurück zu einem 16x16 RGB-Bild.
    Wird für den Reconstruction Loss benötigt.

    Input:  (B, latent_dim)
    Output: (B, 3, 16, 16) – normalisiert auf [0, 1]
    """

    def __init__(self, latent_dim: int = 64, out_channels: int = 3):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128 * 2 * 2)

        self.deconv = nn.Sequential(
            # (128, 2, 2) → (64, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 4, 4) → (32, 8, 8)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (32, 8, 8) → (3, 16, 16)
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.reshape(x.size(0), 128, 2, 2)
        return self.deconv(x)


# ─────────────────────────────────────────────
# FREE ENERGY LOSS
# ─────────────────────────────────────────────

def free_energy_loss(recon_x, x, mu, log_var, beta: float = 1.0):
    """
    Variational Free Energy:
        F = Reconstruction Loss + Beta * KL-Divergenz

    Reconstruction Loss:
        Wie gut rekonstruiert der Decoder das Original?
        = MSE(recon_x, x)

    KL-Divergenz:
        Wie weit ist der Posterior q(z|x) vom Prior p(z) = N(0,I) entfernt?
        = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    Beta > 1 → stärker regularisiert (mehr Druck Richtung Prior)
    Beta < 1 → Rekonstruktion wichtiger als Prior
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL-Divergenz analytisch (für Normalverteilungen)
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


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
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    LATENT_DIM  = 64
    BETA        = 1.0       # Free Energy Beta
    LR          = 1e-3
    N_STEPS     = 150
    TRAIN_EVERY = 1         # Jeden Schritt trainieren

    env     = MockEnv()
    encoder = VariationalCNNEncoder(latent_dim=LATENT_DIM)
    decoder = CNNDecoder(latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=LR
    )

    info = encoder.summary()
    print("Variational CNN Encoder initialisiert:")
    print(f"  Latent Dim  : {info['latent_dim']}")
    print(f"  Flat Size   : {info['flat_size']}")
    print(f"  Parameter   : {info['params_total']:,}")
    print(f"  Beta        : {BETA}")
    print()

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('B04b – Variational CNN Encoder (Active Inference)', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.4)

    ax_obs      = fig.add_subplot(gs[0, 0])   # Original
    ax_recon    = fig.add_subplot(gs[0, 1])   # Rekonstruktion
    ax_mu       = fig.add_subplot(gs[0, 2:4]) # mu-Vektor
    ax_std      = fig.add_subplot(gs[0, 4])   # std als Heatmap

    ax_loss     = fig.add_subplot(gs[1, :2])  # Loss-Kurven
    ax_kl       = fig.add_subplot(gs[1, 2:4]) # KL-Divergenz
    ax_stats    = fig.add_subplot(gs[1, 4])   # Statistiken
    ax_stats.axis('off')

    ax_latent_dist = fig.add_subplot(gs[2, :3])  # Latent-Verteilung (mu ± std)
    ax_compare     = fig.add_subplot(gs[2, 3:])  # B04 vs B04b Vergleich
    ax_compare.axis('off')

    # Tracking
    loss_history   = []
    recon_history  = []
    kl_history     = []
    std_mean_history = []

    print(f"Starte Training: {N_STEPS} Schritte, Beta={BETA}\n")

    for step in range(N_STEPS):
        obs = env.step()

        # ── Training ──────────────────────────────────────
        encoder.train()
        decoder.train()

        x = torch.from_numpy(obs).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)   # (1, 3, 16, 16)

        mu, log_var, z = encoder(x)
        recon_x        = decoder(z)

        total_loss, recon_loss, kl_loss = free_energy_loss(
            recon_x, x, mu, log_var, beta=BETA
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Tracking (.detach() verhindert UserWarning beim float()-Cast)
        loss_history.append(float(total_loss.detach()))
        recon_history.append(float(recon_loss.detach()))
        kl_history.append(float(kl_loss.detach()))

        # Eval für Visualisierung
        mu_np, std_np, z_np = encoder.encode_numpy(obs)
        std_mean_history.append(float(std_np.mean()))

        if step % 5 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(loss_history)))

            # Rekonstruktion holen
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                x_vis       = torch.from_numpy(obs).float() / 255.0
                x_vis       = x_vis.permute(2, 0, 1).unsqueeze(0)
                mu_t, lv_t, z_t = encoder(x_vis)
                recon_vis   = decoder(z_t).squeeze(0).permute(1, 2, 0).numpy()

            # ── Original ──────────────────────────────
            ax_obs.clear()
            ax_obs.imshow(obs, interpolation='nearest')
            ax_obs.set_title(f'Original\nStep {step + 1}', fontsize=8)
            ax_obs.axis('off')

            # ── Rekonstruktion ─────────────────────────
            ax_recon.clear()
            ax_recon.imshow(np.clip(recon_vis, 0, 1), interpolation='nearest')
            ax_recon.set_title(f'Rekonstruktion\n(Free Energy = {total_loss:.4f})', fontsize=8)
            ax_recon.axis('off')

            # ── mu-Vektor ─────────────────────────────
            ax_mu.clear()
            colors = ['steelblue' if v >= 0 else 'tomato' for v in mu_np]
            ax_mu.bar(range(LATENT_DIM), mu_np, color=colors, width=1.0, label='mu')
            ax_mu.errorbar(range(LATENT_DIM), mu_np, yerr=std_np,
                           fmt='none', color='gray', alpha=0.4, linewidth=0.8)
            ax_mu.axhline(0, color='black', linewidth=0.5)
            ax_mu.set_title(f'Latent mu +/- std  (Unsicherheit grau)', fontsize=8)
            ax_mu.set_ylim(-3, 3)
            ax_mu.set_xlabel('Dimension')

            # ── std Heatmap ────────────────────────────
            ax_std.clear()
            std_grid = std_np.reshape(8, 8)
            im = ax_std.imshow(std_grid, cmap='hot', vmin=0, vmax=2, interpolation='nearest')
            ax_std.set_title(f'Std-Heatmap\nMean={std_np.mean():.3f}', fontsize=8)
            ax_std.axis('off')

            # ── Loss-Kurven ────────────────────────────
            ax_loss.clear()
            ax_loss.plot(steps_x, loss_history,  color='black',  linewidth=1.5, label='Free Energy')
            ax_loss.plot(steps_x, recon_history, color='steelblue', linewidth=1.2, label='Recon Loss', alpha=0.8)
            ax_loss.set_title('Free Energy = Recon Loss + Beta * KL')
            ax_loss.set_xlabel('Schritt')
            ax_loss.legend(fontsize=7)

            # ── KL-Divergenz ──────────────────────────
            ax_kl.clear()
            ax_kl.plot(steps_x, kl_history, color='darkorange', linewidth=1.5, label='KL Loss')
            ax_kl.plot(steps_x, std_mean_history, color='green', linewidth=1.2,
                       label='Std Mean', alpha=0.8)
            ax_kl.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax_kl.set_title('KL-Divergenz & mittlere Unsicherheit (std)')
            ax_kl.set_xlabel('Schritt')
            ax_kl.legend(fontsize=7)

            # ── Statistiken ───────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            lines = [
                "── VAE Encoder ──────",
                f"Latent Dim: {LATENT_DIM}",
                f"Beta:       {BETA}",
                f"LR:         {LR}",
                "",
                "── Laufzeit ─────────",
                f"Schritt:    {step + 1}",
                f"Free Energy:{loss_history[-1]:.4f}",
                f"Recon Loss: {recon_history[-1]:.4f}",
                f"KL Loss:    {kl_history[-1]:.4f}",
                f"Std Mean:   {std_mean_history[-1]:.4f}",
                "",
                "── Active Inference ─",
                "mu  = Erwartung",
                "std = Unsicherheit",
                "KL  = Komplexitaet",
                "Rec = Ungenauigkeit",
                "FE  = Komplx+Ungena",
            ]
            ax_stats.text(
                0.05, 0.95, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            # ── Latent-Verteilung ─────────────────────
            # Zeigt mu ± 2*std für die ersten 20 Dimensionen
            ax_latent_dist.clear()
            dims    = list(range(min(20, LATENT_DIM)))
            mus     = mu_np[:20]
            stds    = std_np[:20]
            ax_latent_dist.fill_between(dims,
                                        mus - 2*stds, mus + 2*stds,
                                        alpha=0.3, color='steelblue', label='mu +/- 2*std'
                                        )
            ax_latent_dist.plot(dims, mus, color='steelblue', linewidth=2, label='mu')
            ax_latent_dist.axhline(0,  color='gray',  linestyle='--', linewidth=0.8, label='Prior N(0,1) mu')
            ax_latent_dist.axhspan(-1, 1, alpha=0.05, color='green', label='Prior N(0,1) 1-sigma')
            ax_latent_dist.set_title('Latent-Verteilung: mu +/- 2*std  (erste 20 Dim) vs. Prior N(0,1)', fontsize=9)
            ax_latent_dist.set_xlabel('Latent-Dimension')
            ax_latent_dist.set_ylim(-4, 4)
            ax_latent_dist.legend(fontsize=7, loc='upper right', ncol=2)

            # ── B04 vs B04b Vergleich ─────────────────
            ax_compare.clear()
            ax_compare.axis('off')
            compare_lines = [
                "── B04 vs B04b ─────────────────────────",
                "",
                "B04   CNN Encoder (deterministisch):",
                "  z = encoder(obs)      → Vektor",
                "  Loss = MSE(pred, obs) → Reconstruction",
                "  Norm = L2-Norm auf z",
                "",
                "B04b  Variational Encoder (stochastisch):",
                "  mu, std = encoder(obs) → Verteilung",
                "  z = mu + eps * std     → Sampling",
                "  Loss = Recon + KL      → Free Energy",
                "",
                "Active Inference Bedeutung:",
                "  mu  → Was der Agent erwartet zu sehen",
                "  std → Wie sicher ist der Agent",
                "  KL  → Abstand vom Prior (Komplexitaet)",
                "  Rec → Vorhersagefehler (Ungenauigkeit)",
                "  FE  → Gesamtunsicherheit (minimieren!)",
            ]
            ax_compare.text(
                0.02, 0.98, "\n".join(compare_lines),
                transform=ax_compare.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
            )

            plt.pause(0.05)

    print("\nTraining abgeschlossen!")
    print(f"  Free Energy final : {loss_history[-1]:.4f}")
    print(f"  Recon Loss final  : {recon_history[-1]:.4f}")
    print(f"  KL Loss final     : {kl_history[-1]:.4f}")
    print(f"  Std Mean final    : {std_mean_history[-1]:.4f}")
    print()
    print("Beobachtungen:")
    if std_mean_history[-1] < std_mean_history[0]:
        print("  Std sinkt → Agent wird sicherer (Posterior naeher am Prior)")
    if recon_history[-1] < recon_history[0]:
        print("  Recon sinkt → Rekonstruktion verbessert sich")
    if kl_history[-1] < 1.0:
        print("  KL klein → Posterior bleibt nah am Prior N(0,1)")

    plt.show()


if __name__ == "__main__":
    run_demo()
