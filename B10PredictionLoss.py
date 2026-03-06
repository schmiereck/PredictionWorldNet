"""
B10 – Prediction Loss Demo
============================
Kombiniert alle Verlustterme des Systems zu einer einzigen
differenzierbaren Free Energy Funktion.

Verlustterme:
    1. Reconstruction Loss  : MSE(predicted_frame, actual_frame)
    2. KL Divergenz         : KL(posterior || prior N(0,I))
    3. Temporal Loss        : MSE(predicted_t+k, actual_t+k) für k=1,2,4,8,16
    4. Action Loss          : MSE(predicted_action, actual_action)
    5. Goal Loss            : Cosinus-Distanz(context, goal_embedding)
    6. Intrinsic Reward     : Prediction Error → Curiosity/Exploration

Gesamte Free Energy:
    FE = w_recon  * Reconstruction Loss
       + w_kl     * KL Divergenz
       + w_temp   * Temporal Loss        (Vorhersage zukünftiger Frames)
       + w_action * Action Loss
       + w_goal   * Goal Loss

Intrinsic Reward (kein Loss, aber wichtiges Signal):
    r_intrinsic = alpha * Temporal Loss  (hoher Fehler = neugierig = explorieren)

Active Inference Bedeutung:
    Free Energy = Complexity + Inaccuracy
    Complexity  = KL  (wie weit bin ich vom Prior?)
    Inaccuracy  = Reconstruction + Temporal Loss  (wie falsch sind meine Vorhersagen?)

    Gemini gibt extrinsischen Reward:
    r_total = r_intrinsic + r_gemini

    Agent minimiert FE gleichzeitig durch:
        Perception  : Encoder lernt bessere Latents (KL sinkt)
        Prediction  : Decoder lernt bessere Vorhersagen (Recon sinkt)
        Action      : Action Head wählt Aktionen die FE minimieren
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
# SSIM (Structural Similarity Index)
# ─────────────────────────────────────────────

def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 7, C1: float = 0.01**2, C2: float = 0.03**2
) -> torch.Tensor:
    """
    Berechnet SSIM zwischen pred und target.
    Args: pred, target: (B, C, H, W) in [0, 1]
    Returns: Skalar SSIM ∈ [-1, 1], typisch [0, 1]
    """
    channels = pred.size(1)
    # Uniformes Fenster (einfacher als Gaussian, kaum Qualitätsunterschied)
    kernel = torch.ones(channels, 1, window_size, window_size,
                        device=pred.device) / (window_size ** 2)
    pad = window_size // 2

    mu_x  = F.conv2d(pred,   kernel, padding=pad, groups=channels)
    mu_y  = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_xx = F.conv2d(pred * pred,     kernel, padding=pad, groups=channels) - mu_xx
    sig_yy = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_yy
    sig_xy = F.conv2d(pred * target,   kernel, padding=pad, groups=channels) - mu_xy

    num   = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sig_xx + sig_yy + C2)
    return (num / (denom + 1e-8)).mean()


def combined_recon_loss(pred: torch.Tensor, target: torch.Tensor,
                        ssim_weight: float = 0.3) -> torch.Tensor:
    """MSE + SSIM: L = (1-w)*MSE + w*(1-SSIM)"""
    mse = F.mse_loss(pred, target)
    ssim_val = ssim(pred, target)
    return (1.0 - ssim_weight) * mse + ssim_weight * (1.0 - ssim_val)


# ─────────────────────────────────────────────
# SZENEN (aus B08)
# ─────────────────────────────────────────────

def draw_scene(scene_type: str) -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(10, 16):
        shade = int(60 + (y - 10) * 15)
        img[y, :] = [shade, shade, shade]
    img[0:2,  :] = [40, 40, 60]
    img[2:10, 1]  = [70, 70, 90]
    img[2:10, 14] = [70, 70, 90]
    for y in range(2, 8):
        img[y, 2:14] = [100, 100, 120]

    if scene_type == "red_box":
        img[8:12, 5:9]  = [200, 40, 40]
        img[6:9,  6:10] = [160, 30, 30]
        img[6:12, 9]    = [120, 20, 20]
    elif scene_type == "blue_ball":
        cx, cy = 8, 10
        for y in range(16):
            for x in range(16):
                d = np.sqrt((x-cx)**2 + (y-cy)**2)
                if d < 3.2:
                    bright = int(255 * max(0, 1 - d/3.2))
                    hl = int(80 * max(0, 1-((x-cx+1)**2+(y-cy-1)**2)/4))
                    img[y, x] = [0, bright//3, min(255, bright+hl)]
    elif scene_type == "green_door":
        img[3:8, 6:10] = [30, 140, 50]
        img[3:8, 7:9]  = [20, 180, 60]
        img[5, 9]      = [200, 180, 0]
        img[3, 6:10] = img[8, 6:10] = [20, 100, 30]
        img[3:8, 6]  = img[3:8, 10] = [20, 100, 30]
    elif scene_type == "corridor":
        img[2:10, 2:14] = [90, 90, 110]
        for y in range(2, 10):
            span = max(1, int((y-2)*0.8))
            img[y, 2:2+span]   = [70, 70, 90]
            img[y, 14-span:14] = [70, 70, 90]
        img[4:6, 7:9] = [220, 220, 180]
    elif scene_type == "corner":
        img[2:14, 2:8]  = [95, 90, 115]
        img[2:14, 8:14] = [110, 105, 130]
        img[2:14, 8]    = [50, 45, 65]
    img[10, 2:14] = [50, 50, 50]
    return img

SCENE_TYPES = ["red_box", "blue_ball", "green_door", "corridor", "corner"]


# ─────────────────────────────────────────────
# MINI VAE (Encoder + Decoder aus B04b/B08)
# ─────────────────────────────────────────────

class MiniEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32,  3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128,3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
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


class MiniDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3,  3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z).reshape(z.size(0), 128, 2, 2)
        return self.deconv(x)


# ─────────────────────────────────────────────
# PREDICTION LOSS
# ─────────────────────────────────────────────

class PredictionLoss(nn.Module):
    """
    Berechnet alle Verlustterme und gibt ein detailliertes Loss-Dict zurück.

    Gewichte (konfigurierbar):
        w_recon  : Rekonstruktions-Gewicht      (Inaccuracy)
        w_kl     : KL-Divergenz-Gewicht         (Complexity)
        w_temp   : Temporaler Vorhersage-Gewicht
        w_action : Aktions-Vorhersage-Gewicht
        w_goal   : Ziel-Annäherungs-Gewicht
        beta     : KL-Annealing-Faktor (wächst von 0 → beta_max)
    """

    def __init__(
            self,
            w_recon:    float = 1.0,
            w_kl:       float = 1.0,
            w_temp:     float = 0.5,
            w_action:   float = 0.3,
            w_goal:     float = 0.2,
            beta_max:   float = 0.1,
            latent_dim: int   = 64,
            clip_dim:   int   = 512,
    ):
        super().__init__()
        self.w_recon  = w_recon
        self.w_kl     = w_kl
        self.w_temp   = w_temp
        self.w_action = w_action
        self.w_goal   = w_goal
        self.beta_max = beta_max
        self.beta     = 0.0
        # Projektion: CLIP-Dim → Latent-Dim für Cosinus-Vergleich
        self.goal_proj = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )

    def anneal_beta(self, step: int, warmup_steps: int = 200):
        """
        KL-Annealing: Beta wächst per Cosine-Schedule von 0 → beta_max.
        Startet langsam, beschleunigt in der Mitte, bremst am Ende.
        Verhindert Posterior Collapse besser als lineares Annealing.
        """
        if step >= warmup_steps:
            self.beta = self.beta_max
        else:
            import math
            t = step / warmup_steps
            self.beta = self.beta_max * 0.5 * (1.0 - math.cos(math.pi * t))

    def reconstruction_loss(self, recon: torch.Tensor,
                            target: torch.Tensor) -> torch.Tensor:
        """MSE + SSIM zwischen rekonstruiertem und echtem Frame."""
        return combined_recon_loss(recon, target, ssim_weight=0.3)

    def kl_loss(self, mu: torch.Tensor,
                log_var: torch.Tensor) -> torch.Tensor:
        """KL-Divergenz: KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu² - exp(log_var))"""
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    def temporal_loss(self, predicted_frames: torch.Tensor,
                      actual_frames: torch.Tensor,
                      time_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Vorhersage-Fehler für zukünftige Frames (nicht-lineare Zeitskala).
        predicted_frames: (B, n_future, 3, H, W)
        actual_frames:    (B, n_future, 3, H, W)
        time_weights:     (n_future,) – weiter weg = weniger gewichtet
        """
        if time_weights is None:
            time_weights = torch.ones(predicted_frames.size(1))
        time_weights = time_weights / time_weights.sum()

        loss = 0.0
        for t in range(predicted_frames.size(1)):
            loss += time_weights[t] * F.mse_loss(
                predicted_frames[:, t], actual_frames[:, t]
            )
        return loss

    def action_loss(self, pred_action: torch.Tensor,
                    true_action: torch.Tensor) -> torch.Tensor:
        """MSE zwischen vorhergesagter und tatsächlicher Aktion."""
        return F.mse_loss(pred_action, true_action)

    def goal_loss(self, context: torch.Tensor,
                  goal_embedding: torch.Tensor) -> torch.Tensor:
        """
        Cosinus-Distanz zwischen Kontext-Vektor und Ziel-Embedding.
        Goal-Embedding wird per linearer Projektion auf Latent-Dim gebracht.
        Minimieren = Agent bewegt sich semantisch Richtung Ziel.
        """
        ctx_norm  = F.normalize(context, dim=-1)
        goal_proj = F.normalize(self.goal_proj(goal_embedding), dim=-1)
        cos_sim   = (ctx_norm * goal_proj).sum(dim=-1)
        return torch.clamp(1.0 - cos_sim, 0.0, 1.0).mean()

    def intrinsic_reward(self, predicted_frame: torch.Tensor,
                         actual_frame: torch.Tensor) -> torch.Tensor:
        """
        Intrinsischer Reward = Vorhersage-Fehler.
        Hoher Fehler → Agent war neugierig / unerwartet → explorieren!
        Kein Gradient (nur für Monitoring).
        """
        with torch.no_grad():
            return F.mse_loss(predicted_frame, actual_frame)

    def forward(self, batch: dict, step: int = 0) -> dict:
        """
        Berechnet alle Verlustterme.

        batch keys:
            recon         : (B, 3, H, W)              – Rekonstruktion
            target        : (B, 3, H, W)              – Original
            mu            : (B, latent_dim)
            log_var       : (B, latent_dim)
            pred_frames   : (B, n_future, 3, H, W)    – optional
            actual_frames : (B, n_future, 3, H, W)    – optional
            pred_action   : (B, action_dim)            – optional
            true_action   : (B, action_dim)            – optional
            context       : (B, d_model)               – optional
            goal_emb      : (B, clip_dim)              – optional
        """
        self.anneal_beta(step)
        losses = {}

        # ── Pflicht-Terme ──────────────────────────────────
        losses["recon"] = self.reconstruction_loss(batch["recon"], batch["target"])
        losses["kl"]    = self.kl_loss(batch["mu"], batch["log_var"])
        losses["free_energy"] = (
                self.w_recon * losses["recon"] +
                self.w_kl * self.beta * losses["kl"]
        )

        # ── Optionale Terme ────────────────────────────────
        if "pred_frames" in batch and "actual_frames" in batch:
            n_future     = batch["pred_frames"].size(1)
            time_steps   = [1, 2, 4, 8, 16][:n_future]
            # Weiter weg = log-Gewichtung (weniger wichtig)
            raw_w = torch.tensor([1.0 / np.log1p(t) for t in time_steps])
            losses["temporal"] = self.temporal_loss(
                batch["pred_frames"], batch["actual_frames"], raw_w
            )
            losses["free_energy"] = losses["free_energy"] + \
                                    self.w_temp * losses["temporal"]

            # Intrinsic Reward (nur nächster Frame)
            losses["intrinsic_reward"] = self.intrinsic_reward(
                batch["pred_frames"][:, 0], batch["actual_frames"][:, 0]
            )

        if "pred_action" in batch and "true_action" in batch:
            losses["action"] = self.action_loss(
                batch["pred_action"], batch["true_action"]
            )
            losses["free_energy"] = losses["free_energy"] + \
                                    self.w_action * losses["action"]

        if "context" in batch and "goal_emb" in batch:
            losses["goal"] = self.goal_loss(
                batch["context"], batch["goal_emb"]
            )
            losses["free_energy"] = losses["free_energy"] + \
                                    self.w_goal * losses["goal"]

        losses["beta"] = torch.tensor(self.beta)
        return losses


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    LATENT_DIM = 64
    N_STEPS    = 600
    LR         = 1e-3
    BETA_MAX   = 0.05

    encoder   = MiniEncoder(latent_dim=LATENT_DIM)
    decoder   = MiniDecoder(latent_dim=LATENT_DIM)
    criterion = PredictionLoss(beta_max=BETA_MAX, latent_dim=LATENT_DIM, clip_dim=512)
    params    = list(encoder.parameters()) + list(decoder.parameters()) + \
                list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    total_params = sum(p.numel() for p in params)
    print("B10 – Prediction Loss Demo")
    print(f"  Encoder+Decoder: {total_params:,} Parameter")
    print(f"  Beta Max:        {BETA_MAX}")
    print(f"  Verlustterme:    FE = w_recon*Recon + w_kl*Beta*KL")
    print(f"                        + w_temp*Temporal + w_action*Action")
    print()

    # Szenen vorbereiten
    scene_imgs = [draw_scene(s) for s in SCENE_TYPES]
    x_all = torch.from_numpy(np.stack(scene_imgs)).float() / 255.0
    x_all = x_all.permute(0, 3, 1, 2)   # (5, 3, 16, 16)

    # Mock: zukünftige Frames (leicht verrauscht = simulierte Zukunft)
    def make_future_frames(x, n_future=3):
        noise_levels = [0.05, 0.10, 0.15]
        frames = []
        for noise in noise_levels[:n_future]:
            frames.append(torch.clamp(x + noise * torch.randn_like(x), 0, 1))
        return torch.stack(frames, dim=1)   # (B, n_future, 3, H, W)

    # Mock: Aktionen und Goal-Embedding
    true_actions = torch.tanh(torch.randn(len(SCENE_TYPES), 3))
    goal_emb     = F.normalize(torch.randn(len(SCENE_TYPES), 512), dim=-1)

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B10 – Prediction Loss: Free Energy Zerlegung',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    # Zeile 0: Szenen Original vs Rekonstruktion
    ax_orig  = [fig.add_subplot(gs[0, i]) for i in range(5)]
    # Zeile 1: Loss-Kurven
    ax_fe    = fig.add_subplot(gs[1, :2])   # Free Energy gesamt
    ax_terms = fig.add_subplot(gs[1, 2:4])  # Einzelne Terme
    ax_beta  = fig.add_subplot(gs[1, 4])    # Beta-Annealing
    # Zeile 2: Reward + Gewichte + Statistiken
    ax_rew   = fig.add_subplot(gs[2, :2])   # Intrinsic Reward
    ax_w     = fig.add_subplot(gs[2, 2:4])  # Loss-Gewichte Balken
    ax_stats = fig.add_subplot(gs[2, 4])
    ax_stats.axis('off')

    # Tracking
    history = {k: [] for k in ["fe", "recon", "kl", "temporal",
                               "action", "goal", "reward", "beta", "lr"]}

    print(f"Starte Training: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        encoder.train()
        decoder.train()

        mu, log_var, z = encoder(x_all)
        recon          = decoder(z)

        # Mock-Kontext (wird in B11 durch echten Transformer ersetzt)
        context = F.normalize(z, dim=-1) * np.sqrt(128)

        # Batch zusammenbauen
        batch = {
            "recon":        recon,
            "target":       x_all,
            "mu":           mu,
            "log_var":      log_var,
            "pred_frames":  make_future_frames(recon, n_future=3),
            "actual_frames":make_future_frames(x_all, n_future=3),
            "pred_action":  torch.tanh(torch.randn_like(true_actions)),
            "true_action":  true_actions,
            "context":      context,
            "goal_emb":     goal_emb,
        }

        losses = criterion(batch, step=step)

        optimizer.zero_grad()
        losses["free_energy"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step(losses["recon"].detach())

        # Tracking
        for k in ["fe", "recon", "kl", "temporal", "action", "goal", "reward", "beta"]:
            key = {"fe": "free_energy", "reward": "intrinsic_reward"}.get(k, k)
            if key in losses:
                history[k].append(float(losses[key].detach()))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if step % 20 == 0 or step == N_STEPS - 1:
            encoder.eval()
            decoder.eval()
            steps_x = list(range(len(history["fe"])))

            with torch.no_grad():
                mu_e, lv_e, z_e = encoder(x_all)
                recon_e          = decoder(z_e)
            recon_np = recon_e.permute(0, 2, 3, 1).numpy()
            orig_np  = x_all.permute(0, 2, 3, 1).numpy()

            # ── Szenen ────────────────────────────────
            for i, scene in enumerate(SCENE_TYPES):
                ax_orig[i].clear()
                orig_img  = (orig_np[i] * 255).astype(np.uint8)
                recon_img = np.clip(recon_np[i]*255, 0, 255).astype(np.uint8)
                # Oben: Original, Unten: Rekonstruktion
                combined = np.concatenate([orig_img, recon_img], axis=0)
                ax_orig[i].imshow(combined, interpolation='nearest')
                mse_i = float(np.mean((orig_np[i]-recon_np[i])**2))
                ax_orig[i].set_title(
                    f'{scene.replace("_"," ")}\nMSE={mse_i:.4f}', fontsize=7
                )
                ax_orig[i].axis('off')
                # Trennlinie
                ax_orig[i].axhline(15.5, color='yellow', linewidth=1.5)
            ax_orig[0].set_ylabel('Orig→\nRecon↓', fontsize=7)

            # ── Free Energy gesamt ─────────────────────
            ax_fe.clear()
            ax_fe.plot(steps_x, history["fe"],
                       color='black', linewidth=2, label='Free Energy')
            if len(history["fe"]) >= 20:
                ma = np.convolve(history["fe"], np.ones(20)/20, mode='valid')
                ax_fe.plot(range(19, len(history["fe"])), ma,
                           color='red', linewidth=1.5, linestyle='--',
                           label='MA-20')
            ax_fe.set_title('Gesamte Free Energy', fontsize=9)
            ax_fe.set_xlabel('Schritt')
            ax_fe.legend(fontsize=7)

            # ── Einzelne Terme ─────────────────────────
            ax_terms.clear()
            term_cfg = [
                ("recon",    "Reconstruction", "steelblue"),
                ("kl",       "KL (×beta)",     "darkorange"),
                ("temporal", "Temporal",        "seagreen"),
                ("action",   "Action",          "purple"),
                ("goal",     "Goal",            "crimson"),
            ]
            for key, label, color in term_cfg:
                if history[key]:
                    vals = history[key]
                    ax_terms.plot(range(len(vals)), vals,
                                  color=color, linewidth=1.3,
                                  label=label, alpha=0.85)
            ax_terms.set_title('Einzelne Verlustterme', fontsize=9)
            ax_terms.set_xlabel('Schritt')
            ax_terms.legend(fontsize=6, ncol=2)

            # ── Beta-Annealing ─────────────────────────
            ax_beta.clear()
            if history["beta"]:
                ax_beta.plot(range(len(history["beta"])), history["beta"],
                             color='darkorange', linewidth=1.5, label='Beta')
                ax_beta.axhline(BETA_MAX, color='red', linestyle='--',
                                linewidth=1, label=f'Max={BETA_MAX}')
            ax_beta.set_title('KL-Annealing\n(Beta)', fontsize=9)
            ax_beta.set_xlabel('Schritt')
            ax_beta.set_ylim(0, BETA_MAX * 1.2)
            ax_beta.legend(fontsize=7)

            # ── Intrinsic Reward ───────────────────────
            ax_rew.clear()
            if history["reward"]:
                ax_rew.plot(range(len(history["reward"])), history["reward"],
                            color='gold', linewidth=1.5, label='Intr. Reward')
                # Gleitender Durchschnitt
                if len(history["reward"]) >= 20:
                    ma = np.convolve(history["reward"],
                                     np.ones(20)/20, mode='valid')
                    ax_rew.plot(range(19, len(history["reward"])), ma,
                                color='orange', linewidth=2, label='MA-20')
            ax_rew.set_title('Intrinsic Reward (Curiosity)\n'
                             '= Vorhersage-Fehler naechster Frame', fontsize=9)
            ax_rew.set_xlabel('Schritt')
            ax_rew.legend(fontsize=7)
            ax_rew.set_facecolor('#1a1a1a')
            ax_rew.tick_params(colors='white')
            ax_rew.title.set_color('white')

            # ── Gewichts-Balken ────────────────────────
            ax_w.clear()
            if history["fe"]:
                last = {k: history[k][-1] if history[k] else 0.0
                        for k in ["recon", "kl", "temporal", "action", "goal"]}
                weights = {
                    "Recon":    criterion.w_recon  * last["recon"],
                    "KL*Beta":  criterion.w_kl * criterion.beta * last["kl"],
                    "Temporal": criterion.w_temp   * last["temporal"],
                    "Action":   criterion.w_action * last["action"],
                    "Goal":     criterion.w_goal   * last["goal"],
                }
                colors_w = ['steelblue','darkorange','seagreen','purple','crimson']
                bars = ax_w.bar(list(weights.keys()),
                                list(weights.values()),
                                color=colors_w)
                for bar, val in zip(bars, weights.values()):
                    ax_w.text(bar.get_x() + bar.get_width()/2,
                              bar.get_height() + 0.0001,
                              f'{val:.4f}', ha='center', va='bottom',
                              fontsize=7)
                total = sum(weights.values())
                ax_w.set_title(
                    f'Gewichtete Verlustterme\n(FE = {total:.4f})', fontsize=9
                )

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            fe_now     = history["fe"][-1]    if history["fe"]     else 0
            recon_now  = history["recon"][-1] if history["recon"]  else 0
            kl_now     = history["kl"][-1]    if history["kl"]     else 0
            temp_now   = history["temporal"][-1] if history["temporal"] else 0
            rew_now    = history["reward"][-1]   if history["reward"]   else 0
            lr_now     = history["lr"][-1]    if history["lr"]     else LR

            lines = [
                "── Free Energy ──────",
                f"FE:      {fe_now:.5f}",
                f"Recon:   {recon_now:.5f}",
                f"KL:      {kl_now:.5f}",
                f"Temporal:{temp_now:.5f}",
                f"Beta:    {criterion.beta:.4f}",
                "",
                "── Gewichte ─────────",
                f"w_recon:  {criterion.w_recon}",
                f"w_kl:     {criterion.w_kl}",
                f"w_temp:   {criterion.w_temp}",
                f"w_action: {criterion.w_action}",
                f"w_goal:   {criterion.w_goal}",
                "",
                "── Training ─────────",
                f"Schritt:  {step+1}/{N_STEPS}",
                f"LR:       {lr_now:.2e}",
                f"Intr.Rew: {rew_now:.5f}",
                "",
                "── Active Inference ─",
                "FE = Complexity",
                "   + Inaccuracy",
                "Minimiere FE durch:",
                "  Perception (KL↓)",
                "  Prediction (Rec↓)",
                "  Action     (FE↓)",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.03)

    print("\nTraining abgeschlossen!")
    print(f"  Free Energy:      {history['fe'][-1]:.5f}")
    print(f"  Reconstruction:   {history['recon'][-1]:.5f}")
    print(f"  KL:               {history['kl'][-1]:.5f}")
    print(f"  Temporal:         {history['temporal'][-1]:.5f}")
    print(f"  Intrinsic Reward: {history['reward'][-1]:.5f}")
    print(f"  Beta final:       {criterion.beta:.4f}")
    print()
    print("Verlust-Hierarchie (Active Inference):")
    print("  Free Energy = Complexity (KL) + Inaccuracy (Recon + Temporal)")
    print("  Intrinsic Reward = Temporal Loss → Curiosity")
    print("  → Kombination mit Gemini-Reward folgt in B15")

    plt.show()


if __name__ == "__main__":
    run_demo()
