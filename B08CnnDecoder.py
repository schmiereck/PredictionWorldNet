"""
B08 – CNN Decoder Demo
=======================
Dekodiert den Kontext-Vektor aus B07 zurück zu einem vorhergesagten Frame.

Architektur:
    Input:  context (B, d_model=128)  ← Temporal Transformer B07
    FC:     (B, 128*2*2)
    Reshape:(B, 128, 2, 2)
    Deconv1:(B, 64, 4, 4)   + Skip-Connection von Encoder (optional)
    Deconv2:(B, 32, 8, 8)
    Deconv3:(B, 3, 16, 16)  → Sigmoid → predicted frame [0,1]

Zusammen mit B04b (Encoder) bildet B08 einen vollständigen VAE:
    Encoder: Frame → (mu, log_var, z)
    Decoder: z / context → predicted Frame

Active Inference Bedeutung:
    Decoder = Generatives Modell P(obs | z)
    "Gegeben meinen internen Zustand z – was erwarte ich zu sehen?"
    Rekonstruktions-Loss = Ungenauigkeits-Term der Free Energy

Demo:
    Zeigt Original vs. Rekonstruktion für synthetische Mini-Szenen
    (Raum mit roter Box, blauem Ball, grüner Tür – MiniWorld-ähnlich)
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SCHÖNE SYNTHETHISCHE SZENEN
# ─────────────────────────────────────────────

def draw_scene(scene_type: str, t: int = 0) -> np.ndarray:
    """
    Erzeugt ein 16x16 RGB-Bild einer Mini-Szene.

    scene_type:
        "red_box"    – Raum mit roter Box (Zielobjekt)
        "blue_ball"  – Raum mit blauem Ball
        "green_door" – Raum mit grüner Tür
        "corridor"   – Langer Korridor
        "corner"     – Ecke mit zwei Wänden
    """
    img = np.zeros((16, 16, 3), dtype=np.uint8)

    # ── Boden (Grau-Gradient) ──
    for y in range(10, 16):
        shade = int(60 + (y - 10) * 15)
        img[y, :] = [shade, shade, shade]

    # ── Decke ──
    img[0:2, :] = [40, 40, 60]

    # ── Wände (links/rechts) mit Perspektive ──
    for y in range(2, 10):
        shade = int(80 + y * 8)
        img[y, 0]  = [shade - 20, shade - 20, shade]
        img[y, 15] = [shade - 20, shade - 20, shade]
    img[2:10, 1]  = [70, 70, 90]
    img[2:10, 14] = [70, 70, 90]

    # ── Hinterwand ──
    for y in range(2, 8):
        img[y, 2:14] = [100, 100, 120]

    # ── Objekte je nach Scene-Type ──
    if scene_type == "red_box":
        # Rote Box in der Mitte/vorne
        box_x = 5 + int(2 * np.sin(t * 0.08))   # leichte Bewegung
        img[8:12, box_x:box_x+4] = [200, 40, 40]   # Vorderseite
        img[6:9,  box_x+1:box_x+5] = [160, 30, 30]  # Oberseite (dunkler)
        img[6:12, box_x+4]  = [120, 20, 20]          # Schattenseite

    elif scene_type == "blue_ball":
        # Blauer Ball
        cx = 8
        cy = 10
        for y in range(16):
            for x in range(16):
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                if d < 3.2:
                    # Licht-Schattierung
                    bright = int(255 * max(0, 1 - d/3.2))
                    highlight = int(80 * max(0, 1 - ((x-cx+1)**2+(y-cy-1)**2)/4))
                    img[y, x] = [0, bright//3, min(255, bright + highlight)]

    elif scene_type == "green_door":
        # Grüne Tür in der Hinterwand
        img[3:8, 6:10] = [30, 140, 50]     # Türrahmen
        img[3:8, 7:9]  = [20, 180, 60]     # Türfüllung
        img[5, 9]      = [200, 180, 0]      # Türknauf (gelb)
        # Türrahmen-Umrandung
        img[3, 6:10]   = [20, 100, 30]
        img[8, 6:10]   = [20, 100, 30]
        img[3:8, 6]    = [20, 100, 30]
        img[3:8, 10]   = [20, 100, 30]

    elif scene_type == "corridor":
        # Korridor mit Fluchtpunkt
        img[2:10, 2:14] = [90, 90, 110]   # Hinterwand
        # Seitenwände konvergieren
        for y in range(2, 10):
            span = max(1, int((y - 2) * 0.8))
            img[y, 2:2+span]    = [70, 70, 90]
            img[y, 14-span:14]  = [70, 70, 90]
        # Lichtpunkt am Ende
        img[4:6, 7:9] = [220, 220, 180]

    elif scene_type == "corner":
        # Ecke: linke Wand trifft Hinterwand
        img[2:14, 2:8]  = [95, 90, 115]   # Linke Wand
        img[2:14, 8:14] = [110, 105, 130]  # Hinterwand (heller)
        img[2:14, 8]    = [50, 45, 65]     # Eckkante

    # ── Boden-Linie (Schatten) ──
    img[10, 2:14] = [50, 50, 50]

    # ── Leichte zeitliche Variation (simuliert Bewegung) ──
    # Fester Seed pro Szene – kein Noise während Training, sonst kann der Encoder nichts lernen
    # noise = np.random.RandomState(t).randint(-8, 8, img.shape)
    # img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    return img


SCENE_TYPES = ["red_box", "blue_ball", "green_door", "corridor", "corner"]
SCENE_NAMES = {
    "red_box":    "Rote Box",
    "blue_ball":  "Blauer Ball",
    "green_door": "Gruene Tuer",
    "corridor":   "Korridor",
    "corner":     "Ecke",
}


# ─────────────────────────────────────────────
# CNN ENCODER (aus B04b, vereinfacht)
# ─────────────────────────────────────────────

class CNNEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.fc_mu      = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        f       = self.conv(x).reshape(x.size(0), -1)
        mu      = self.fc_mu(f)
        log_var = torch.clamp(self.fc_log_var(f), -10, 10)
        std     = torch.exp(0.5 * log_var)
        z       = mu + (torch.randn_like(std) if self.training else 0) * std
        return mu, log_var, z


# ─────────────────────────────────────────────
# CNN DECODER
# ─────────────────────────────────────────────

class CNNDecoder(nn.Module):
    """
    Dekodiert Latent-Vektor oder Kontext-Vektor → 16x16 RGB Frame.

    Input:  (B, input_dim)   – z aus Encoder ODER context aus Transformer
    Output: (B, 3, 16, 16)  – Vorhergesagtes Bild, normalisiert [0,1]

    Architektur:
        FC → Reshape → 3× TransposedConv (je ×2 Upsampling)
        Jede Schicht: ConvTranspose2d + BatchNorm + ReLU
        Letzte Schicht: Sigmoid statt ReLU (Output in [0,1])

    Warum kein UNet (Skip-Connections)?
        Für die Demo reicht ein einfacher Decoder.
        Skip-Connections werden in B11 (Training Loop) ergänzt
        wenn wir Encoder+Decoder gemeinsam trainieren.
    """

    def __init__(self, input_dim: int = 64, base_channels: int = 64):
        super().__init__()
        self.input_dim = input_dim

        # FC: input_dim → 128 × 2 × 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128 * 2 * 2),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            # (128, 2, 2) → (64, 4, 4)
            nn.ConvTranspose2d(128, base_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # (64, 4, 4) → (32, 8, 8)
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),

            # (32, 8, 8) → (3, 16, 16)
            nn.ConvTranspose2d(base_channels // 2, 3,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, input_dim)
        Returns:
            (B, 3, 16, 16) in [0, 1]
        """
        x = self.fc(z)
        x = x.view(x.size(0), 128, 2, 2)
        return self.deconv(x)

    def decode_numpy(self, z: np.ndarray) -> np.ndarray:
        """numpy (input_dim,) → numpy (16, 16, 3) uint8"""
        self.eval()
        with torch.no_grad():
            zt = torch.from_numpy(z).float().unsqueeze(0)
            out = self.forward(zt)
            return (out.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    def summary(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output":    "(3, 16, 16)",
            "params":    sum(p.numel() for p in self.parameters()),
        }


# ─────────────────────────────────────────────
# FREE ENERGY LOSS
# ─────────────────────────────────────────────

def free_energy_loss(recon, target, mu, log_var, beta: float = 0.5):
    from B10PredictionLoss import combined_recon_loss
    recon_loss = combined_recon_loss(recon, target, ssim_weight=0.3)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    LATENT_DIM   = 64
    N_STEPS      = 800    # Mehr Schritte für bessere Rekonstruktion
    LR           = 1e-3   # Höhere LR für schnelleres Lernen
    BETA         = 0.01   # Sehr kleines Beta: Rekonstruktion hat Vorrang
    # Bei Posterior Collapse (KL→0) hilft Beta↓

    encoder = CNNEncoder(latent_dim=LATENT_DIM)
    decoder = CNNDecoder(input_dim=LATENT_DIM)
    params  = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3)

    enc_info = {"params": sum(p.numel() for p in encoder.parameters())}
    dec_info = decoder.summary()

    print("CNN Encoder + Decoder initialisiert:")
    print(f"  Encoder params: {enc_info['params']:,}")
    print(f"  Decoder params: {dec_info['params']:,}")
    print(f"  Gesamt:         {enc_info['params'] + dec_info['params']:,}")
    print(f"  BETA:           {BETA}")
    print()

    # ── Matplotlib Setup ──────────────────────────────────
    n_scenes = len(SCENE_TYPES)

    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B08 – CNN Decoder: Original vs. Rekonstruktion',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(4, n_scenes + 1, figure=fig,
                           hspace=0.55, wspace=0.3)

    # Zeile 0: Original-Bilder
    ax_orig   = [fig.add_subplot(gs[0, i]) for i in range(n_scenes)]
    ax_orig_label = fig.add_subplot(gs[0, n_scenes])
    ax_orig_label.axis('off')

    # Zeile 1: Rekonstruktionen
    ax_recon  = [fig.add_subplot(gs[1, i]) for i in range(n_scenes)]
    ax_recon_label = fig.add_subplot(gs[1, n_scenes])
    ax_recon_label.axis('off')

    # Zeile 2: Differenz-Bilder
    ax_diff   = [fig.add_subplot(gs[2, i]) for i in range(n_scenes)]
    ax_diff_label = fig.add_subplot(gs[2, n_scenes])
    ax_diff_label.axis('off')

    # Zeile 3: Loss-Kurven + Statistiken
    ax_loss  = fig.add_subplot(gs[3, :2])
    ax_kl    = fig.add_subplot(gs[3, 2:4])
    ax_stats = fig.add_subplot(gs[3, 4])
    ax_stats.axis('off')

    # Zeilenbeschriftungen
    for ax, label in [(ax_orig_label, "Original"),
                      (ax_recon_label, "Rekonstruktion"),
                      (ax_diff_label, "Differenz")]:
        ax.text(0.5, 0.5, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='center', ha='center',
                rotation=0,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    loss_history  = []
    recon_history = []
    kl_history    = []

    print(f"Starte Training: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        encoder.train()
        decoder.train()

        # ── Batch aus allen Szenen ─────────────────────
        batch_imgs = []
        for i, scene in enumerate(SCENE_TYPES):
            img = draw_scene(scene, t=step)
            batch_imgs.append(img)

        # numpy (N, H, W, 3) → torch (N, 3, H, W) float [0,1]
        x = torch.from_numpy(np.stack(batch_imgs)).float() / 255.0
        x = x.permute(0, 3, 1, 2)

        mu, log_var, z = encoder(x)
        recon          = decoder(z)

        total, recon_l, kl_l = free_energy_loss(recon, x, mu, log_var, BETA)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        loss_history.append(float(total.detach()))
        recon_history.append(float(recon_l.detach()))
        kl_history.append(float(kl_l.detach()))

        if step % 20 == 0 or step == N_STEPS - 1:
            encoder.eval()
            decoder.eval()

            steps_x = list(range(len(loss_history)))

            with torch.no_grad():
                mu_e, lv_e, z_e = encoder(x)
                recon_e          = decoder(z_e)

            recon_np = recon_e.permute(0, 2, 3, 1).numpy()   # (N, 16, 16, 3)
            orig_np  = x.permute(0, 2, 3, 1).numpy()

            for i, scene in enumerate(SCENE_TYPES):
                orig_img  = (orig_np[i]  * 255).astype(np.uint8)
                recon_img = np.clip(recon_np[i] * 255, 0, 255).astype(np.uint8)
                diff_img  = np.abs(orig_np[i] - recon_np[i])

                # Original
                ax_orig[i].clear()
                ax_orig[i].imshow(orig_img, interpolation='nearest')
                ax_orig[i].set_title(SCENE_NAMES[scene], fontsize=8)
                ax_orig[i].axis('off')

                # Rekonstruktion
                ax_recon[i].clear()
                ax_recon[i].imshow(recon_img, interpolation='nearest')
                mse_i = float(np.mean((orig_np[i] - recon_np[i])**2))
                ax_recon[i].set_title(f'MSE={mse_i:.3f}', fontsize=7)
                ax_recon[i].axis('off')

                # Differenz (verstärkt)
                ax_diff[i].clear()
                diff_display = np.clip(diff_img * 4, 0, 1)
                ax_diff[i].imshow(diff_display, interpolation='nearest',
                                  cmap='hot')
                ax_diff[i].set_title('×4', fontsize=7)
                ax_diff[i].axis('off')

            # Loss-Kurven
            ax_loss.clear()
            ax_loss.plot(steps_x, loss_history,  color='black',
                         linewidth=1.5, label='Free Energy')
            ax_loss.plot(steps_x, recon_history, color='steelblue',
                         linewidth=1.2, label='Recon Loss', alpha=0.8)
            if len(loss_history) >= 20:
                ma = np.convolve(loss_history, np.ones(20)/20, mode='valid')
                ax_loss.plot(range(19, len(loss_history)), ma,
                             color='darkblue', linewidth=2,
                             label='MA-20', alpha=0.6)
            ax_loss.set_title('Free Energy = Recon + Beta*KL', fontsize=9)
            ax_loss.set_xlabel('Schritt')
            ax_loss.legend(fontsize=7)

            ax_kl.clear()
            ax_kl.plot(steps_x, kl_history, color='darkorange',
                       linewidth=1.5, label='KL Loss')
            ax_kl.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax_kl.set_title('KL-Divergenz (Active Inference: Komplexitaet)', fontsize=9)
            ax_kl.set_xlabel('Schritt')
            ax_kl.legend(fontsize=7)

            # Statistiken
            ax_stats.clear()
            ax_stats.axis('off')
            mean_mse = np.mean([
                np.mean((orig_np[i] - recon_np[i])**2)
                for i in range(n_scenes)
            ])
            lines = [
                "── VAE (B04b + B08) ─",
                f"Encoder:  {enc_info['params']:,}",
                f"Decoder:  {dec_info['params']:,}",
                f"Beta:     {BETA}",
                "",
                "── Laufzeit ─────────",
                f"Schritt:  {step + 1}/{N_STEPS}",
                f"FE:       {loss_history[-1]:.4f}",
                f"Recon:    {recon_history[-1]:.4f}",
                f"KL:       {kl_history[-1]:.4f}",
                f"MSE Ø:    {mean_mse:.4f}",
                "",
                "── Szenen ───────────",
                *[f"  {SCENE_NAMES[s]}" for s in SCENE_TYPES],
                "",
                "── Active Inference ─",
                "Decoder = P(obs|z)",
                "Recon = Ungenauigkeit",
                "KL    = Komplexitaet",
                "FE    = Recon + KL",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.05)

    print("\nTraining abgeschlossen!")
    print(f"  Free Energy final: {loss_history[-1]:.4f}")
    print(f"  Recon Loss final:  {recon_history[-1]:.4f}")
    print(f"  KL Loss final:     {kl_history[-1]:.4f}")

    # ── Finales Bild: Latent Space Interpolation ──────────
    print("\nErstelle Latent-Space Interpolation...")
    encoder.eval()
    decoder.eval()

    fig2, axes2 = plt.subplots(3, 9, figsize=(16, 6))
    fig2.suptitle('B08 – Latent Space Interpolation zwischen Szenen', fontsize=12)

    scene_pairs = [
        ("red_box",    "blue_ball",  "Rote Box → Blauer Ball"),
        ("green_door", "corridor",   "Tuer → Korridor"),
        ("red_box",    "green_door", "Rote Box → Tuer"),
    ]

    with torch.no_grad():
        for row, (s1, s2, title) in enumerate(scene_pairs):
            img1 = draw_scene(s1, t=50)
            img2 = draw_scene(s2, t=50)

            x1 = torch.from_numpy(img1).float() / 255.0
            x2 = torch.from_numpy(img2).float() / 255.0
            x1 = x1.permute(2, 0, 1).unsqueeze(0)
            x2 = x2.permute(2, 0, 1).unsqueeze(0)

            mu1, _, _ = encoder(x1)
            mu2, _, _ = encoder(x2)

            axes2[row, 0].set_ylabel(title, fontsize=7, rotation=0,
                                     labelpad=60, va='center')

            for j, alpha in enumerate(np.linspace(0, 1, 9)):
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                img_out  = decoder(z_interp)
                img_np   = np.clip(
                    img_out.squeeze(0).permute(1, 2, 0).numpy() * 255,
                    0, 255
                ).astype(np.uint8)

                axes2[row, j].imshow(img_np, interpolation='nearest')
                axes2[row, j].axis('off')
                if row == 0:
                    axes2[row, j].set_title(f'α={alpha:.2f}', fontsize=7)

    plt.tight_layout()
    plt.show()
    print("Demo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
