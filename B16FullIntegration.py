"""
B16 – Vollintegration
======================
Verbindet alle Bausteine B01–B15 zu einem vollständigen
Online-Learning System.

Architektur-Überblick:
─────────────────────────────────────────────────────────
ROBOTER (low-res 16×16)
    ↓  obs_t
ENCODER (B04b)    →  z_t  (64-dim)
    ↓
TEMPORAL BUFFER (B03)  →  z_hist (5×64), a_hist (5×6)
    ↓
TEMPORAL TRANSFORMER (B07)  ←  goal_emb (CLIP B05)
    ↓  context (128-dim)
    ├──→ DECODER (B08)       →  pred_obs
    └──→ ACTION HEAD (B09)   →  action (6-dim, ACTION_DIM=6)
              ↓
         ROS2 Twist
─────────────────────────────────────────────────────────
ONLINE LEARNING LOOP (das Herzstück):

    Jeder Step:
        r_intrinsic = |pred_obs - real_obs|  (immer, kein API)

    Alle N Steps (adaptiv via B14):
        Gemini ER (high-res Bild) → r_gemini + label
        → Trainings-Batch aus Replay Buffer (B02)
        → Lokales Modell trainiert einen Schritt

    Ziel: Gemini-Interval wächst mit der Zeit
          → Modell wird selbstständiger
─────────────────────────────────────────────────────────
ACTION_DIM = 6:
    [linear_x, angular_z, camera_pan, camera_tilt,
     arc_radius, duration]

Gemini-Modelle:
    gemini-2.5-flash              → Text (B13)
    gemini-robotics-er-1.5-preview → Vision (B15)
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import csv
import json
import base64
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from B10PredictionLoss import combined_recon_loss

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# KONSTANTEN
# ─────────────────────────────────────────────

ACTION_DIM = 6
ACTION_BOUNDS = {
    "linear_x":    (-0.5,  0.5),
    "angular_z":   (-1.0,  1.0),
    "camera_pan":  (-1.57, 1.57),
    "camera_tilt": (-0.79, 0.79),
    "arc_radius":  (-2.0,  2.0),
    "duration":    (0.1,   2.0),
}
LATENT_DIM  = 256   # T16: 64 → 256 (reicherer Zustandsraum)
D_MODEL     = 256   # T16: 128 → 256 (=LATENT_DIM, damit context[:,:LATENT_DIM] passt)
OBS_SHAPE   = (128, 128, 3)

# T13: Szenen-Vokabular für den Beschreibungs-Kopf (scene_head)
# Abbildung von Gemini-training_labels auf feste Klassen-Indizes.
# Letzte Klasse "unknown" fängt alle nicht erkannten Labels ab.
SCENE_VOCAB = [
    "red_box", "yellow_box", "orange_box", "white_box",
    "green_ball", "blue_ball",
    "exploring", "unknown",
]
# Schlüsselwörter (Kleinschreibung) → Vokabular-Index
SCENE_LABEL_MAP = {
    # Objekte (Deutsch + Englisch)
    "red_box":    0, "red box":   0, "rote":   0, "roter":  0, "rotes":  0, "red":    0,
    "yellow_box": 1, "yellow":    1, "gelbe":  1, "gelber": 1, "gelbes": 1,
    "orange_box": 2, "orange":    2,
    "white_box":  3, "white":     3, "weiss":  3, "weiße":  3, "weißer": 3, "weis":   3,
    "green_ball": 4, "green":     4, "grüne":  4, "grüner": 4, "grün":   4,
    "blue_ball":  5, "blue":      5, "blaue":  5, "blauer": 5, "blau":   5,
    # Exploration / kein Ziel sichtbar (häufige Gemini-Labels)
    "exploring":  6, "erkunden":  6, "korridor": 6, "corridor": 6, "ecke": 6,
    "suche":      6, "such":      6, "leer":     6, "empty":    6,
    "wand":       6, "wall":      6, "grau":     6,
    "nicht sichtbar": 6, "nicht erkennbar": 6, "nicht gefunden": 6,
    "unscharf":   6, "kein ziel": 6, "kein_ziel": 6,
}
N_SCENE_CLASSES = len(SCENE_VOCAB)

SCENE_TYPES = ["red_box", "blue_ball", "green_door", "corridor", "corner"]
SCENE_GOALS = {
    "red_box":    "find the red box",
    "blue_ball":  "find the blue ball",
    "green_door": "navigate to the exit door",
    "corridor":   "explore the corridor",
    "corner":     "navigate to the corner",
}
SCENE_ACTIONS = {
    "red_box":    [ 0.6,  0.0,  0.0,  0.1,  0.0, -0.5],
    "blue_ball":  [ 0.4,  0.6, -0.3,  0.2,  0.0, -0.5],
    "green_door": [ 0.8,  0.0,  0.0,  0.0,  0.0, -0.3],
    "corridor":   [ 1.0,  0.0,  0.0,  0.0,  0.4, -0.4],
    "corner":     [ 0.3,  0.8,  0.5,  0.0,  0.0, -0.6],
}


# ─────────────────────────────────────────────
# SZENEN
# ─────────────────────────────────────────────

def draw_scene(scene_type: str, noise: float = 0.0) -> np.ndarray:
    # Basis bei 16x16 zeichnen, dann auf OBS_SHAPE hochskalieren
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(10, 16):
        img[y, :] = [int(60+(y-10)*15)]*3
    img[0:2, :] = [40,40,60]
    img[2:10,1] = img[2:10,14] = [70,70,90]
    for y in range(2,8):
        img[y,2:14] = [100,100,120]
    if scene_type == "red_box":
        img[8:12,5:9]=[200,40,40]; img[6:9,6:10]=[160,30,30]; img[6:12,9]=[120,20,20]
    elif scene_type == "blue_ball":
        for y in range(16):
            for x in range(16):
                d=np.sqrt((x-8)**2+(y-10)**2)
                if d<3.2:
                    b=int(255*max(0,1-d/3.2)); h=int(80*max(0,1-((x-7)**2+(y-9)**2)/4))
                    img[y,x]=[0,b//3,min(255,b+h)]
    elif scene_type == "green_door":
        img[3:8,6:10]=[30,140,50]; img[3:8,7:9]=[20,180,60]; img[5,9]=[200,180,0]
        img[3,6:10]=img[8,6:10]=[20,100,30]; img[3:8,6]=img[3:8,10]=[20,100,30]
    elif scene_type == "corridor":
        img[2:10,2:14]=[90,90,110]
        for y in range(2,10):
            s=max(1,int((y-2)*0.8)); img[y,2:2+s]=[70,70,90]; img[y,14-s:14]=[70,70,90]
        img[4:6,7:9]=[220,220,180]
    elif scene_type == "corner":
        img[2:14,2:8]=[95,90,115]; img[2:14,8:14]=[110,105,130]; img[2:14,8]=[50,45,65]
    img[10,2:14]=[50,50,50]
    if noise > 0:
        img = np.clip(img.astype(int) +
                      (np.random.randn(*img.shape)*noise*255).astype(int),
                      0, 255).astype(np.uint8)
    # Auf OBS_SHAPE skalieren
    target_h, target_w = OBS_SHAPE[0], OBS_SHAPE[1]
    if (target_h, target_w) != (16, 16):
        img = np.repeat(np.repeat(img, target_h // 16, axis=0),
                        target_w // 16, axis=1)
    return img


# ─────────────────────────────────────────────
# MODELLE
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # GroupNorm statt BatchNorm: keine Running Stats → kein Train/Eval-Drift
        # bei wechselnden Szenen und Kamerawinkeln im Online-Learning
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.GroupNorm(8,32), nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GroupNorm(8,64), nn.ReLU(True),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.GroupNorm(8,128), nn.ReLU(True),
            nn.Conv2d(128,128,3,stride=2,padding=1), nn.GroupNorm(8,128), nn.ReLU(True),
            nn.Conv2d(128,128,3,stride=2,padding=1), nn.GroupNorm(8,128), nn.ReLU(True),
        )
        # 128x128 → 64→32→16→8→4  ⇒  flatten = 128*4*4 = 2048
        self.fc_mu      = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)

    def forward(self, x):
        f       = self.conv(x).reshape(x.size(0), -1)
        mu      = self.fc_mu(f)
        log_var = torch.clamp(self.fc_log_var(f), -10, 10)
        std     = torch.exp(0.5 * log_var)
        z       = mu + (torch.randn_like(std) if self.training else 0) * std
        return mu, log_var, z


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        # 2048 = 128*4*4, reshape zu (128,4,4), deconv → 128x128
        # GroupNorm statt BatchNorm (konsistent mit Encoder)
        self.fc = nn.Sequential(nn.Linear(latent_dim, 2048), nn.ReLU(True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,128,3,stride=2,padding=1,output_padding=1),
            nn.GroupNorm(8,128), nn.ReLU(True),
            nn.ConvTranspose2d(128,128,3,stride=2,padding=1,output_padding=1),
            nn.GroupNorm(8,128), nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
            nn.GroupNorm(8,64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
            nn.GroupNorm(8,32), nn.ReLU(True),
            nn.ConvTranspose2d(32,3,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(self.fc(z).reshape(z.size(0),128,4,4))


class TemporalTransformer(nn.Module):
    """Vereinfachter Temporal Transformer (B07)."""
    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                 d_model=D_MODEL, n_heads=4, n_layers=3):
        super().__init__()
        self.d_model = d_model
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj_cur    = nn.Linear(latent_dim, d_model)
        self.proj_goal   = nn.Linear(512, d_model)
        self.proj_hist   = nn.Linear(latent_dim + action_dim + latent_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        # T10: Action-konditioniertes Transitions-Modell P(z_{t+1} | z_t, a_t)
        # Früher: next_z_head = nn.Linear(d_model, latent_dim)  [action-agnostisch]
        # Jetzt:  dynamics_head(cat([context, a_t]))             [explizit konditioniert]
        # Das Modell lernt: "Was sehe ich NACH Aktion a_t?" – echter World-Model-Kern.
        self.dynamics_head = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model * 2),
            nn.ReLU(True),
            nn.Linear(d_model * 2, latent_dim),
        )

    def forward(self, z_cur, goal_emb, z_hist=None, a_hist=None):
        B = z_cur.size(0)
        tokens = [
            self.cls_token.expand(B, -1, -1),
            self.proj_cur(z_cur).unsqueeze(1),
            self.proj_goal(goal_emb).unsqueeze(1),
        ]
        if z_hist is not None and a_hist is not None:
            # Einfache Zeitkodierung
            T = z_hist.size(1)
            time_enc = torch.zeros(B, T, LATENT_DIM)
            for t in range(T):
                time_enc[:, t, :] = t / T
            hist_input = torch.cat([z_hist, a_hist, time_enc], dim=-1)
            tokens.append(self.proj_hist(hist_input))
        seq = torch.cat(tokens, dim=1)
        out = self.transformer(seq)
        return out[:, 0, :]   # CLS token

    def predict_next_z(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        T10: Aktions-konditionierte Zustandsvorhersage.
        P(z_{t+1} | context, a_t) – Kern des generativen World-Models.

        Args:
            context: (B, D_MODEL) – Transformer CLS-Ausgabe
            action:  (B, ACTION_DIM) – aktuelle Aktion (normiert in [-1, 1])
        Returns:
            pred_z_next: (B, LATENT_DIM)
        """
        return self.dynamics_head(torch.cat([context, action], dim=-1))


# ─────────────────────────────────────────────
# T12: RSSM – Rekurrenter Weltzustand (DreamerV3-Stil)
# ─────────────────────────────────────────────

SEQ_LEN = 8   # Sequenzlänge für RSSM-Training (Truncated BPTT)

# T11: EFE (Expected Free Energy) als Aktions-Auswahlprinzip
# Blend zwischen Imitations-Loss und EFE-Loss für den ActionHead.
# EFE_BLEND rampt adaptiv hoch mit der Anzahl verfügbarer Gemini-Samples:
#   efe_blend = EFE_BLEND_MAX * min(gemini_count / EFE_GEMINI_RAMP, 1.0)
# → Kein EFE ohne trainierte reward_head (pragmatischer Term wäre Rauschen).
EFE_BLEND_MAX   = 0.5    # Maximaler EFE-Anteil (Rest = Imitation)
EFE_GEMINI_RAMP = 50     # Ab 50 Gemini-Samples voller EFE-Blend

class RSSM(nn.Module):
    """T12: Rekurrenter Weltzustand (RSSM-Kern, DreamerV3-Stil).

    Ersetzt den TemporalTransformer: GRU-Zelle statt Attention-Fenster.
    Der Hidden-State h_t persistiert über die gesamte Episode und akkumuliert
    Informationen über besuchte Orte, gesehene Objekte und Aktionsfolgen.

    GRU-Input:  cat(z_t, a_{t-1}, goal_proj)  [latent + action + goal]
    GRU-Output: h_t                            [d_model = 256-dim]
    """

    def __init__(self, latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                 d_model=D_MODEL):
        super().__init__()
        self.d_model    = d_model
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # GRU: z_t (256) + a_{t-1} (6) + goal (256) = 518
        gru_input_dim = latent_dim + action_dim + latent_dim
        self.gru = nn.GRUCell(gru_input_dim, d_model)

        # dynamics_head: cat(h_t, a_t) → z_{t+1}  (identisch zur alten Version)
        self.dynamics_head = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model * 2),
            nn.ReLU(True),
            nn.Linear(d_model * 2, latent_dim),
        )

        # Persistent hidden state (für Inferenz, Schritt-für-Schritt)
        self._h = None

    def reset_state(self):
        """Hidden State zurücksetzen (Episode-Grenze)."""
        self._h = None

    def forward(self, z_cur, goal_proj, a_prev=None):
        """
        Ein Schritt: (z_t, goal_projected, a_{t-1}) → context (h_t).

        Args:
            z_cur:     (B, latent_dim)
            goal_proj: (B, latent_dim) – bereits projiziertes Ziel-Embedding
            a_prev:    (B, action_dim) – vorherige Aktion (None = zeros)
        Returns:
            h_t: (B, d_model) – neuer Hidden-State = context
        """
        B = z_cur.size(0)
        if a_prev is None:
            a_prev = torch.zeros(B, self.action_dim, device=z_cur.device)

        x = torch.cat([z_cur, a_prev, goal_proj], dim=-1)

        if self._h is None or self._h.shape[0] != B:
            self._h = torch.zeros(B, self.d_model, device=z_cur.device)

        self._h = self.gru(x, self._h)
        return self._h

    def forward_sequence(self, z_seq, act_seq, goal_proj, h_init=None):
        """
        Sequenz-Forward für Training (Truncated BPTT).

        Args:
            z_seq:     (B, L, latent_dim)
            act_seq:   (B, L, action_dim)
            goal_proj: (B, latent_dim) – gleich für alle Steps
            h_init:    (B, d_model) oder None
        Returns:
            h_seq: (B, L, d_model)
        """
        B, L, _ = z_seq.shape
        h = h_init if h_init is not None else \
            torch.zeros(B, self.d_model, device=z_seq.device)

        h_all = []
        for t in range(L):
            a_prev = torch.zeros(B, self.action_dim, device=z_seq.device) \
                if t == 0 else act_seq[:, t - 1]
            x = torch.cat([z_seq[:, t], a_prev, goal_proj], dim=-1)
            h = self.gru(x, h)
            h_all.append(h)

        return torch.stack(h_all, dim=1)   # (B, L, d_model)

    def predict_next_z(self, context, action):
        """T10: Aktions-konditionierte Zustandsvorhersage P(z_{t+1} | h_t, a_t)."""
        return self.dynamics_head(torch.cat([context, action], dim=-1))


class ActionHead(nn.Module):
    def __init__(self, d_model=D_MODEL, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,256), nn.LayerNorm(256), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(256,128),     nn.LayerNorm(128), nn.ReLU(True),
        )
        self.action_out = nn.Sequential(nn.Linear(128, action_dim), nn.Tanh())
        self.sigma_out  = nn.Sequential(nn.Linear(128, action_dim), nn.Sigmoid())

    def forward(self, ctx):
        f = self.net(ctx)
        return self.action_out(f), self.sigma_out(f)


# ─────────────────────────────────────────────
# REPLAY BUFFER (B02)
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.ptr = self.size = 0
        self.obs      = np.zeros((max_size, *OBS_SHAPE), dtype=np.uint8)
        self.next_obs = np.zeros((max_size, *OBS_SHAPE), dtype=np.uint8)
        self.actions  = np.zeros((max_size, ACTION_DIM), dtype=np.float32)
        self.rewards  = np.zeros(max_size, dtype=np.float32)
        self.goals    = [""] * max_size
        # Gemini ER Labels
        self.gemini_rewards  = np.full(max_size, np.nan, dtype=np.float32)
        self.gemini_labels   = [""] * max_size
        # T12: Episode-Grenze (True = letzter Step vor Reset)
        self.dones = np.zeros(max_size, dtype=bool)

    def add(self, obs, next_obs, action, reward, goal="",
            gemini_reward=np.nan, gemini_label="", done=False):
        i = self.ptr
        self.obs[i]     = obs;     self.next_obs[i]  = next_obs
        self.actions[i] = action;  self.rewards[i]   = reward
        self.goals[i]   = goal
        self.gemini_rewards[i] = gemini_reward
        self.gemini_labels[i]  = gemini_label
        self.dones[i] = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_sequences(self, batch_size, seq_len=SEQ_LEN):
        """T12: Zusammenhängende Sequenzen für RSSM-Training (BPTT).

        Gibt Sequenzen zurück die keine Episode-Grenze enthalten.
        Returns None wenn nicht genug sequenzielle Daten vorhanden.
        """
        if self.size < seq_len + 1:
            return None

        # Gültige Startindizes: keine done-Flag innerhalb der nächsten seq_len-1 Steps
        n = self.size
        valid = np.ones(n, dtype=bool)
        for offset in range(seq_len - 1):
            valid &= ~self.dones[(np.arange(n) + offset) % self.max_size]
        # Nicht zu nahe am Schreibzeiger (Daten könnten überschrieben werden)
        if self.size == self.max_size:
            for offset in range(seq_len):
                valid[(self.ptr - offset) % self.max_size] = False
        valid_idx = np.where(valid[:max(n - seq_len + 1, 1)])[0]

        if len(valid_idx) == 0:
            return None

        chosen = np.random.choice(
            valid_idx, min(batch_size, len(valid_idx)),
            replace=len(valid_idx) < batch_size
        )
        B = len(chosen)
        obs_s   = np.zeros((B, seq_len, *OBS_SHAPE), dtype=np.uint8)
        nobs_s  = np.zeros((B, seq_len, *OBS_SHAPE), dtype=np.uint8)
        act_s   = np.zeros((B, seq_len, ACTION_DIM), dtype=np.float32)
        goals   = []
        for b, start in enumerate(chosen):
            for t in range(seq_len):
                idx = (start + t) % self.max_size
                obs_s[b, t]  = self.obs[idx]
                nobs_s[b, t] = self.next_obs[idx]
                act_s[b, t]  = self.actions[idx]
            goals.append(self.goals[start % self.max_size])

        return {
            "obs":      torch.from_numpy(obs_s).float() / 255.0,
            "next_obs": torch.from_numpy(nobs_s).float() / 255.0,
            "actions":  torch.from_numpy(act_s),
            "goals":    goals,
        }

    def sample(self, batch_size, require_gemini=False):
        if require_gemini:
            valid = np.where(~np.isnan(self.gemini_rewards[:self.size]))[0]
            if len(valid) < batch_size:
                return None
            idx = np.random.choice(valid, batch_size, replace=False)
        else:
            idx = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs":            torch.from_numpy(self.obs[idx]).float()/255.0,
            "next_obs":       torch.from_numpy(self.next_obs[idx]).float()/255.0,
            "actions":        torch.from_numpy(self.actions[idx]),
            "rewards":        torch.from_numpy(self.rewards[idx]),
            "goals":          [self.goals[i] for i in idx],
            "gemini_rewards": torch.from_numpy(
                np.nan_to_num(self.gemini_rewards[idx], nan=0.0)
            ),
            "gemini_labels":  [self.gemini_labels[i] for i in idx],
        }

    @property
    def gemini_count(self):
        return int(np.sum(~np.isnan(self.gemini_rewards[:self.size])))


# ─────────────────────────────────────────────
# ADAPTIVE FREQUENCY (B14)
# ─────────────────────────────────────────────

class AdaptiveController:
    def __init__(self, min_interval=5, max_interval=80,
                 fe_threshold=0.15, fe_low=0.05):
        self.min_interval  = min_interval
        self.max_interval  = max_interval
        self.fe_threshold  = fe_threshold
        self.fe_low        = fe_low
        self.fe_ema        = 0.2
        self.last_call     = -max_interval
        self.total         = 0
        self.calls         = 0
        self.interval_hist = []

    def should_call(self, fe: float, novelty: float = 0.3,
                    force: bool = False) -> bool:
        self.total  += 1
        self.fe_ema  = 0.9 * self.fe_ema + 0.1 * fe
        since   = self.total - self.last_call
        u_fe    = np.clip((self.fe_ema - self.fe_low) /
                          (self.fe_threshold - self.fe_low + 1e-8), 0, 1)
        u_to    = np.clip(since / self.max_interval, 0, 1)
        urgency = 0.6 * u_fe + 0.2 * novelty + 0.2 * u_to
        interval = int(self.max_interval*(1-urgency) + self.min_interval*urgency)
        self.interval_hist.append(interval)
        call = force or (since >= interval)
        if call:
            self.last_call = self.total
            self.calls    += 1
        return call

    @property
    def call_rate(self):
        return self.calls / max(1, self.total)


# ─────────────────────────────────────────────
# GEMINI CLIENTS
# ─────────────────────────────────────────────

class GeminiClients:
    """Beide Gemini-Modelle in einer Klasse."""

    TEXT_MODEL     = "gemini-2.5-flash"
    ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"

    TEXT_SYSTEM = """Übersetze Roboter-Befehle in kurze englische CLIP-Phrasen.
Antworte NUR mit JSON: {"primary_goal": "...", "confidence": 0.0-1.0}"""

    ROBOTICS_SYSTEM = """Du bewertest Kamerabilder eines Roboters.
Antworte NUR mit JSON:
{"reward": 0.0-1.0, "goal_progress": 0.0-1.0,
 "situation": "...", "recommendation": "...",
 "next_action_hint": "vorwärts/links/rechts/stopp/kamera",
 "training_label": "kurzes Label für diesen Zustand"}"""

    def __init__(self, api_key: str = None):
        self.mode = "mock"
        self.client = None

        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")

        if GEMINI_AVAILABLE and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.mode   = "gemini"
                print(f"  Text:    {self.TEXT_MODEL}")
                print(f"  Vision:  {self.ROBOTICS_MODEL}")
            except Exception as e:
                print(f"  Gemini Fehler: {e} – Mock-Modus")
        else:
            print("  Gemini: Mock-Modus")

    def translate_goal(self, user_cmd: str) -> dict:
        """Text: Benutzer-Befehl → CLIP-Phrase."""
        if self.mode == "gemini":
            try:
                resp = self.client.models.generate_content(
                    model=self.TEXT_MODEL,
                    contents=user_cmd,
                    config=types.GenerateContentConfig(
                        system_instruction=self.TEXT_SYSTEM,
                        temperature=0.1,
                    )
                )
                text = resp.text.strip()
                if "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                result = json.loads(text)
                result["source"] = "gemini_text"
                return result
            except Exception as e:
                print(f"  Text-Gemini Fehler: {e}")

        # Mock
        cmd = user_cmd.lower()
        mapping = [
            (["roten box","red box"], "find the red box", 0.95),
            (["blauen ball","blue ball"], "find the blue ball", 0.95),
            (["tür","door","exit"], "navigate to the exit door", 0.92),
            (["korridor","corridor"], "explore the corridor", 0.88),
            (["ecke","corner"], "navigate to the corner", 0.85),
        ]
        for keys, goal, conf in mapping:
            if any(k in cmd for k in keys):
                return {"primary_goal": goal, "confidence": conf,
                        "source": "mock_text"}
        return {"primary_goal": user_cmd[:40], "confidence": 0.5,
                "source": "mock_fallback"}

    def assess_image(self, image_np: np.ndarray, goal: str,
                     action: dict = None) -> dict:
        """Vision: Kamerabild → Reward + Label."""
        if action is None:
            action = {"linear_x": 0, "angular_z": 0,
                      "camera_pan": 0, "camera_tilt": 0}

        if self.mode == "gemini":
            try:
                if PIL_AVAILABLE:
                    img_large = PILImage.fromarray(image_np).resize(
                        (128, 128), PILImage.NEAREST)
                    buf = BytesIO()
                    img_large.save(buf, format="JPEG", quality=95)
                    img_bytes = buf.getvalue()
                else:
                    img_bytes = image_np.tobytes()

                prompt = (f'Ziel: "{goal}"\n'
                          f'Aktion: lx={action["linear_x"]:.2f} '
                          f'az={action["angular_z"]:.2f}\n'
                          f'Bewerte das Bild.')

                resp = self.client.models.generate_content(
                    model=self.ROBOTICS_MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=self.ROBOTICS_SYSTEM,
                        temperature=0.1,
                    )
                )
                text = resp.text.strip()
                if "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                result = json.loads(text)
                result["source"] = "gemini_robotics"
                return result
            except Exception as e:
                print(f"  Robotics-Gemini Fehler: {e}")

        # Mock: Farb-Analyse
        img = image_np.astype(float) / 255.0
        red   = ((img[:,:,0]>0.6)&(img[:,:,1]<0.3)&(img[:,:,2]<0.3)).mean()
        blue  = ((img[:,:,2]>0.5)&(img[:,:,0]<0.2)).mean()
        green = ((img[:,:,1]>0.4)&(img[:,:,0]<0.2)&(img[:,:,2]<0.2)).mean()
        bright= ((img[:,:,0]>0.7)&(img[:,:,1]>0.7)).mean()

        g = goal.lower()
        if "red box"   in g: rew = np.clip(red*15,   0, 1); lbl = "red_box_visible"
        elif "blue"    in g: rew = np.clip(blue*12,  0, 1); lbl = "blue_ball_visible"
        elif "door"    in g: rew = np.clip(green*20, 0, 1); lbl = "green_door_visible"
        elif "corridor"in g: rew = np.clip(bright*10,0, 1); lbl = "in_corridor"
        else:                rew = 0.3;                      lbl = "exploring"

        hint = "vorwärts" if rew > 0.5 else ("kamera" if rew > 0.2 else "links")
        return {
            "reward":          float(rew),
            "goal_progress":   float(rew * 0.8),
            "situation":       f"Mock: {lbl}",
            "recommendation":  "vorwärts" if rew > 0.5 else "erkunden",
            "next_action_hint":hint,
            "training_label":  lbl,
            "source":          "mock_robotics",
        }


# ─────────────────────────────────────────────
# MOCK CLIP
# ─────────────────────────────────────────────

class MockCLIP:
    def encode(self, text: str) -> torch.Tensor:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v   = rng.standard_normal(512).astype(np.float32)
        return torch.from_numpy(v / np.linalg.norm(v)).unsqueeze(0)


# ─────────────────────────────────────────────
# VOLLINTEGRIERTES SYSTEM
# ─────────────────────────────────────────────

class IntegratedSystem:
    """
    Alle Bausteine B01–B15 in einem System.

    Online Learning Loop:
        1. Env-Step → obs, action
        2. Encoder → z
        3. Transformer → context
        4. Decoder → pred_obs
        5. r_intrinsic = MSE(pred, real)
        6. [Adaptiv] Gemini ER → r_gemini, label
        7. Replay Buffer add (mit Gemini-Label wenn verfügbar)
        8. Training Step
    """

    def __init__(self, config: dict, gemini: GeminiClients):
        self.cfg    = config
        self.gemini = gemini
        self.clip   = MockCLIP()

        # Modelle
        self.encoder     = Encoder()
        self.decoder     = Decoder()
        self.rssm        = RSSM()   # T12: ersetzt TemporalTransformer
        self.action_head = ActionHead()
        self.goal_proj   = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, LATENT_DIM),
        )
        # T14: Reward-Prädiktor P(r | z_t, a_t)
        # Pragmatischer EFE-Term: Reward-Schätzung ohne Gemini-Aufruf.
        # Trainiert auf (z, a) → r_gemini aus dem Replay Buffer.
        # Ermöglicht später: Planung in der Imagination (T15).
        self.reward_head = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_DIM, 128), nn.ReLU(True),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        # T13: Semantischer Beschreibungs-Kopf P(label | z_t)
        # Das Modell lernt aus z zu beschreiben, was es sieht.
        # Schwache Supervision durch Gemini training_labels.
        self.scene_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(True),
            nn.Linear(128, N_SCENE_CLASSES),
        )

        # Buffer
        self.replay = ReplayBuffer(max_size=config["buffer_size"])

        # T12: vorherige Aktion für GRU-Input (ersetzt z_hist/a_hist)
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # Adaptive Controller
        self.adaptive = AdaptiveController(
            min_interval=config["min_gemini_interval"],
            max_interval=config["max_gemini_interval"],
        )

        # Aktuelles Ziel
        self.current_goal     = "find the red box"
        self.current_goal_emb = self.clip.encode(self.current_goal)

        # Optimizer
        all_params = (
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.rssm.parameters()) +
                list(self.action_head.parameters()) +
                list(self.goal_proj.parameters()) +
                list(self.reward_head.parameters()) +
                list(self.scene_head.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=config["lr"], weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=80, min_lr=1e-4
        )
        self.beta = 0.0

        # Metriken
        self.metrics = {k: [] for k in [
            "fe", "recon", "pred_img", "kl", "kl_raw", "action",
            "l_sigma", "l_reward", "l_scene",
            "r_intrinsic", "r_gemini", "r_reward_pred", "r_total",
            "goal_progress", "gemini_interval",
            "gemini_call_rate", "lr",
        ]}
        self.last_gemini_result = {}
        self.total_steps        = 0
        self.train_steps        = 0
        self._label_clip_embeddings = None  # wird aus Checkpoint geladen

        # ── Log-Dateien (CSV, ein Timestamp pro Session) ──────
        log_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_ts = ts

        self._log_steps   = open(log_dir / f"steps_{ts}.csv",   "w", newline="")
        self._log_train   = open(log_dir / f"train_{ts}.csv",   "w", newline="")
        self._log_gemini  = open(log_dir / f"gemini_{ts}.csv",  "w", newline="")

        self._csv_steps  = csv.writer(self._log_steps)
        self._csv_train  = csv.writer(self._log_train)
        self._csv_gemini = csv.writer(self._log_gemini)

        # Header
        self._csv_steps.writerow([
            "total_step", "r_intr", "r_gemini", "r_reward_pred", "r_total",
            "sigma_mean", "novelty", "scene_pred", "goal", "scene", "gem_called",
        ])
        self._csv_train.writerow([
            "train_step", "total_step", "fe", "recon", "pred_img",
            "kl", "kl_raw", "action", "l_sigma", "l_reward", "l_scene",
            "goal_loss", "cam_center", "efe_blend", "lr", "beta",
            "grad_enc", "grad_dec", "grad_rssm", "grad_ah", "grad_gp", "grad_rh", "grad_sh",
        ])
        self._csv_gemini.writerow([
            "total_step", "reward", "goal_progress", "situation",
            "recommendation", "action_hint", "training_label", "goal",
        ])

    @staticmethod
    def _label_to_vocab_idx(label: str) -> int:
        """T13: Gemini training_label → SCENE_VOCAB-Index (0..N_SCENE_CLASSES-1)."""
        lower = label.lower()
        for key, idx in SCENE_LABEL_MAP.items():
            if key in lower:
                return idx
        return N_SCENE_CLASSES - 1  # "unknown"

    def reset_hidden_state(self):
        """T12: GRU-State zurücksetzen (Episode-Grenze)."""
        self.rssm.reset_state()
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

    def set_goal(self, user_cmd: str):
        """Neues Ziel via Gemini Text-Interface (B13)."""
        result = self.gemini.translate_goal(user_cmd)
        self.current_goal     = result["primary_goal"]
        self.current_goal_emb = self.clip.encode(self.current_goal)
        return result

    # ─────────────────────────────────────────────
    # B20 – MODELL-PERSISTENZ (Save/Load)
    # ─────────────────────────────────────────────

    def save_checkpoint(self, path: str = None, tag: str = ""):
        """
        Speichert alle Netz-Gewichte + Optimizer + Metriken.

        Args:
            path: Voller Dateipfad. Wenn None → auto-generiert.
            tag:  Optionaler Bezeichner (z.B. "pretrain_vae", "live_step500")

        Returns:
            Pfad der gespeicherten Datei.
        """
        import datetime
        if path is None:
            ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"pwn_{tag}_{ts}.pt" if tag else f"pwn_{ts}.pt"
            path = os.path.join(ckpt_dir, name)

        checkpoint = {
            # Modell-Gewichte
            "encoder":      self.encoder.state_dict(),
            "decoder":      self.decoder.state_dict(),
            "rssm":         self.rssm.state_dict(),
            "action_head":  self.action_head.state_dict(),
            "goal_proj":    self.goal_proj.state_dict(),
            "reward_head":  self.reward_head.state_dict(),   # T14
            "scene_head":   self.scene_head.state_dict(),    # T13
            # Optimizer & Scheduler
            "optimizer":    self.optimizer.state_dict(),
            "scheduler":    self.scheduler.state_dict(),
            # Training-State
            "total_steps":  self.total_steps,
            "train_steps":  self.train_steps,
            "beta":         self.beta,
            "current_goal": self.current_goal,
            # Config (zur Überprüfung beim Laden)
            "config":       self.cfg,
            "constants": {
                "LATENT_DIM": LATENT_DIM,
                "D_MODEL":    D_MODEL,
                "ACTION_DIM": ACTION_DIM,
            },
            "tag":          tag,
        }
        # label_clip_embeddings durchreichen (aus B21)
        if self._label_clip_embeddings is not None:
            checkpoint["label_clip_embeddings"] = self._label_clip_embeddings
        torch.save(checkpoint, path)
        print(f"  Checkpoint gespeichert: {path}")
        print(f"    Tag: {tag or '(ohne)'}  |  "
              f"Steps: {self.total_steps}  |  "
              f"Train: {self.train_steps}")
        return path

    def load_checkpoint(self, path: str, load_optimizer: bool = True,
                        strict: bool = True):
        """
        Lädt Netz-Gewichte (und optional Optimizer-State).

        Args:
            path:           Pfad zur .pt Datei
            load_optimizer: Optimizer/Scheduler-State laden (False für Pre-Training)
            strict:         strict=False erlaubt teilweises Laden
                            (z.B. nur Encoder/Decoder aus Pre-Training)

        Returns:
            dict mit Checkpoint-Metadaten (tag, steps, config)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {path}")

        checkpoint = torch.load(path, weights_only=False)

        # T16: Dimensions-Guard — Checkpoint inkompatibel wenn LATENT_DIM/D_MODEL abweicht
        ckpt_consts = checkpoint.get("constants", {})
        ckpt_latent = ckpt_consts.get("LATENT_DIM", LATENT_DIM)
        ckpt_dmodel = ckpt_consts.get("D_MODEL", D_MODEL)
        if ckpt_latent != LATENT_DIM or ckpt_dmodel != D_MODEL:
            print(f"  ⚠ Checkpoint-Dimensionen ({ckpt_latent}/{ckpt_dmodel}) passen nicht "
                  f"zum Modell ({LATENT_DIM}/{D_MODEL}) — Modellgewichte werden übersprungen.")
            print(f"    Für T16 bitte B20/B21 neu vortrainieren.")
            # Trainings-Metadaten NICHT laden (falscher Checkpoint)
            return {"tag": "", "steps": 0, "config": {}, "constants": ckpt_consts}

        # Modell-Gewichte laden
        self.encoder.load_state_dict(
            checkpoint["encoder"], strict=strict)
        self.decoder.load_state_dict(
            checkpoint["decoder"], strict=strict)

        # T12: RSSM laden (oder Migration von altem Transformer-Checkpoint)
        if "rssm" in checkpoint:
            try:
                self.rssm.load_state_dict(
                    checkpoint["rssm"], strict=strict)
            except RuntimeError as e:
                try:
                    self.rssm.load_state_dict(
                        checkpoint["rssm"], strict=False)
                    print(f"    RSSM: partial load ({e})")
                except Exception as e2:
                    print(f"    RSSM: übersprungen ({e2})")
        elif "transformer" in checkpoint:
            # T12-Migration: dynamics_head Gewichte übertragen, GRU startet frisch
            print("    T12-Migration: Transformer → RSSM (dynamics_head übertragen, GRU frisch)")
            old_state = checkpoint["transformer"]
            rssm_state = self.rssm.state_dict()
            transferred = 0
            for key in old_state:
                if key.startswith("dynamics_head.") and key in rssm_state:
                    if old_state[key].shape == rssm_state[key].shape:
                        rssm_state[key] = old_state[key]
                        transferred += 1
            self.rssm.load_state_dict(rssm_state)
            print(f"    dynamics_head: {transferred} Tensoren übertragen")
        if "action_head" in checkpoint:
            try:
                self.action_head.load_state_dict(
                    checkpoint["action_head"], strict=strict)
            except RuntimeError as e:
                if strict:
                    raise
                print(f"    ActionHead: übersprungen ({e})")
        if "goal_proj" in checkpoint:
            try:
                self.goal_proj.load_state_dict(
                    checkpoint["goal_proj"], strict=strict)
            except RuntimeError as e:
                if strict:
                    raise
                print(f"    GoalProj: übersprungen ({e})")
        if "reward_head" in checkpoint:
            try:
                self.reward_head.load_state_dict(
                    checkpoint["reward_head"], strict=strict)
            except RuntimeError as e:
                if strict:
                    raise
                print(f"    RewardHead: übersprungen ({e})")
        if "scene_head" in checkpoint:
            try:
                self.scene_head.load_state_dict(
                    checkpoint["scene_head"], strict=strict)
            except RuntimeError as e:
                if strict:
                    raise
                print(f"    SceneHead: übersprungen ({e})")

        # Optimizer/Scheduler
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

        # Training-State
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
            self.train_steps = checkpoint.get("train_steps", 0)
            self.beta        = checkpoint.get("beta", 0.0)

        # label_clip_embeddings durchreichen (aus B21)
        if "label_clip_embeddings" in checkpoint:
            self._label_clip_embeddings = checkpoint["label_clip_embeddings"]

        tag   = checkpoint.get("tag", "")
        steps = checkpoint.get("total_steps", 0)
        print(f"  Checkpoint geladen: {path}")
        print(f"    Tag: {tag or '(ohne)'}  |  "
              f"Steps: {steps}  |  "
              f"Train: {checkpoint.get('train_steps', 0)}")

        return {
            "tag":       tag,
            "steps":     steps,
            "config":    checkpoint.get("config", {}),
            "constants": checkpoint.get("constants", {}),
        }

    # ─────────────────────────────────────────────
    # T15 – Mehrstufige Imagination (Planning-as-Inference)
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def plan_action(self, obs_np: np.ndarray,
                    horizon: int = 5, n_candidates: int = 32,
                    discount: float = 0.95) -> dict:
        """T15: Aktions-Planung durch Imagination im latenten Raum.

        Erzeugt N Kandidaten-Aktionssequenzen, rollt jede über H Schritte
        im World-Model (RSSM dynamics + reward_head) aus und wählt die
        erste Aktion der Sequenz mit dem höchsten kumulierten Reward.

        Args:
            obs_np:       (H, W, 3) aktuelle Beobachtung
            horizon:      Planungshorizont (Schritte voraus)
            n_candidates: Anzahl Kandidaten-Sequenzen
            discount:     Diskontfaktor für zukünftige Rewards
        Returns:
            dict mit:
              "action":           (ACTION_DIM,) beste erste Aktion
              "plan_reward_best": bester kumulierter Reward
              "plan_reward_mean": mittlerer kumulierter Reward
              "plan_reward_std":  Streuung der kumulierten Rewards
              "plan_used":        True
        """
        N = n_candidates
        H = horizon

        # Bedingung: reward_head muss trainiert sein (genug Gemini-Daten)
        if self.replay.gemini_count < EFE_GEMINI_RAMP:
            return {"plan_used": False}

        self.encoder.eval()
        self.rssm.eval()
        self.action_head.eval()
        self.reward_head.eval()

        # Aktuellen Zustand encodieren
        x = torch.from_numpy(obs_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        _, _, z_cur = self.encoder(x)         # (1, latent_dim)
        goal_p = F.normalize(
            self.goal_proj(self.current_goal_emb), dim=-1
        )                                      # (1, latent_dim)

        # Aktuellen RSSM-State lesen (NICHT verändern!)
        h_cur = self.rssm._h                   # (1, d_model) oder None
        if h_cur is None:
            h_cur = torch.zeros(1, self.rssm.d_model, device=z_cur.device)

        # Auf N Kandidaten expandieren
        z = z_cur.expand(N, -1).contiguous()    # (N, latent_dim)
        h = h_cur.expand(N, -1).contiguous()    # (N, d_model)
        gp = goal_p.expand(N, -1).contiguous()  # (N, latent_dim)

        # ActionHead-Vorschlag als Mittelwert + Sigma
        mean_action, sigma = self.action_head(h)

        # Akkumulierter Reward pro Kandidat
        cum_reward = torch.zeros(N, device=z.device)
        first_actions = torch.zeros(N, ACTION_DIM, device=z.device)

        for t in range(H):
            if t == 0:
                # Erste Aktion: um ActionHead-Vorschlag herum sampeln
                noise = torch.randn(N, ACTION_DIM, device=z.device)
                actions = torch.clamp(mean_action + sigma * noise, -1.0, 1.0)
                first_actions = actions.clone()
            else:
                # Folge-Aktionen: ActionHead auf imaginierten State anwenden
                a_mean, a_sig = self.action_head(h)
                noise = torch.randn(N, ACTION_DIM, device=z.device)
                actions = torch.clamp(a_mean + 0.3 * a_sig * noise, -1.0, 1.0)

            # Dynamics: z_{t+1} vorhersagen
            pred_z_next = self.rssm.predict_next_z(h, actions)

            # Hidden-State aktualisieren (GRU direkt, nicht self._h!)
            gru_input = torch.cat([pred_z_next, actions, gp], dim=-1)
            h = self.rssm.gru(gru_input, h)

            # Reward bewerten
            r = self.reward_head(
                torch.cat([pred_z_next, actions], dim=-1)
            ).squeeze(-1)
            cum_reward += (discount ** t) * r

        # Beste Sequenz auswählen
        best_idx = cum_reward.argmax().item()
        best_action = first_actions[best_idx].cpu().numpy()

        return {
            "action":           best_action,
            "plan_reward_best": float(cum_reward[best_idx]),
            "plan_reward_mean": float(cum_reward.mean()),
            "plan_reward_std":  float(cum_reward.std()),
            "plan_used":        True,
        }

    def step(self, obs_np: np.ndarray, action_np: np.ndarray,
             next_obs_np: np.ndarray, scene: str):
        """
        Ein vollständiger Online-Learning Step.
        Returns: dict mit allen Metriken
        """
        self.total_steps += 1
        # Cosine Beta-Annealing: langsamer Start, beschleunigt, bremst am Ende
        warmup = self.cfg["beta_warmup"]
        if self.total_steps >= warmup:
            self.beta = self.cfg["beta_max"]
        else:
            t = self.total_steps / warmup
            self.beta = self.cfg["beta_max"] * 0.5 * (1.0 - np.cos(np.pi * t))

        # ── Forward Pass (Inference) ────────────────────────
        self.encoder.eval(); self.decoder.eval()
        self.rssm.eval(); self.action_head.eval()
        self.reward_head.eval(); self.scene_head.eval()

        with torch.no_grad():
            x    = torch.from_numpy(obs_np).float().permute(2,0,1).unsqueeze(0)/255.0
            xn   = torch.from_numpy(next_obs_np).float().permute(2,0,1).unsqueeze(0)/255.0
            _, _, z    = self.encoder(x)
            _, _, z_n  = self.encoder(xn)

            goal_emb = self.current_goal_emb
            goal_p   = F.normalize(self.goal_proj(goal_emb), dim=-1)

            # T12: RSSM – ein Schritt mit persistentem Hidden-State
            a_prev  = torch.from_numpy(self.prev_action).float().unsqueeze(0)
            context = self.rssm(z, goal_p, a_prev)

            a_t         = torch.from_numpy(action_np).float().unsqueeze(0)
            pred_z_next = self.rssm.predict_next_z(context, a_t)
            pred_obs    = self.decoder(pred_z_next)
            pred_action, pred_sigma = self.action_head(context)

            # T14: Reward-Vorhersage ohne Gemini – pragmatischer EFE-Term
            r_reward_pred = float(
                self.reward_head(torch.cat([z, a_t], dim=-1)).squeeze().item()
            )
            # T13: Szenen-Beschreibung aus z – was sieht das Modell gerade?
            scene_logits = self.scene_head(z)
            scene_idx    = int(scene_logits.argmax(dim=-1).item())
            scene_pred   = SCENE_VOCAB[scene_idx]

        # ── Intrinsic Reward: echter Prediction-Error (pred vs. nächstes Bild)
        r_intr = float(F.mse_loss(pred_obs,
                                  xn.clone().detach()).item())

        # ── Novelty (vereinfacht) ───────────────────────────
        novelty = float(np.clip(r_intr * 10, 0, 1))

        # ── Gemini ER Assessment (adaptiv) ─────────────────
        r_gemini    = 0.3   # Fallback
        goal_prog   = 0.0
        gem_called  = False
        gem_label   = ""

        if self.adaptive.should_call(fe=r_intr, novelty=novelty):
            action_dict = {
                "linear_x":   float(action_np[0]),
                "angular_z":  float(action_np[1]),
                "camera_pan": float((action_np[2]+1)/2*180-90),
                "camera_tilt":float((action_np[3]+1)/2*90-45),
            }
            assessment = self.gemini.assess_image(
                obs_np, self.current_goal, action_dict
            )
            r_gemini   = assessment["reward"]
            goal_prog  = assessment["goal_progress"]
            gem_called = True
            gem_label  = assessment.get("training_label", "")
            self.last_gemini_result = assessment

        # ── Replay Buffer ───────────────────────────────────
        self.replay.add(
            obs=obs_np, next_obs=next_obs_np,
            action=action_np,
            reward=r_intr,
            goal=self.current_goal,
            gemini_reward=r_gemini if gem_called else np.nan,
            gemini_label=gem_label,
        )
        # T12: vorherige Aktion für nächsten GRU-Schritt merken
        self.prev_action = action_np.copy()

        # ── Gesamt-Reward (alle Komponenten auf [0,1] normalisiert) ──
        r_intr_norm = min(r_intr, 1.0)
        r_goal_cos  = float(F.cosine_similarity(
            F.normalize(z, dim=-1), goal_p, dim=-1
        ).item())
        r_goal_norm = (r_goal_cos + 1.0) / 2.0   # [-1,1] → [0,1]
        r_sigma     = float(1.0 - pred_sigma.mean().item())

        # Aktions-Effizienz: Kamera-Pan ist billiger als Roboter-Drehung.
        # Bestraft angular_z (Ganzkörper-Drehung), belohnt Pan-Nutzung.
        # Motiviert den Agent, zuerst per Kamera-Schwenk zu suchen.
        angular_cost = abs(float(action_np[1]))          # |angular_z| ∈ [0,1]
        pan_usage    = abs(float(action_np[2]))           # |camera_pan| ∈ [0,1]
        r_efficiency = float(np.clip(
            1.0 - 0.5 * angular_cost + 0.3 * pan_usage, 0.0, 1.0
        ))

        r_total = (0.25 * r_intr_norm + 0.35 * r_gemini +
                   0.2  * r_goal_norm +
                   0.1  * r_sigma +
                   0.1  * r_efficiency)

        # ── Training Step ───────────────────────────────────
        train_info = {}
        if self.replay.size >= self.cfg["batch_size"]:
            train_info = self._train_step()

        # ── Metriken ────────────────────────────────────────
        self.metrics["r_intrinsic"].append(r_intr)
        self.metrics["r_gemini"].append(r_gemini)
        self.metrics["r_total"].append(r_total)
        self.metrics["goal_progress"].append(goal_prog)
        self.metrics["gemini_interval"].append(
            self.adaptive.interval_hist[-1] if self.adaptive.interval_hist else 0
        )
        self.metrics["gemini_call_rate"].append(self.adaptive.call_rate)
        self.metrics["lr"].append(self.optimizer.param_groups[0]["lr"])
        for k in ["fe", "recon", "pred_img", "kl", "kl_raw", "action", "l_sigma", "l_reward", "l_scene"]:
            self.metrics[k].append(train_info.get(k, 0.0))
        self.metrics["r_reward_pred"].append(r_reward_pred)

        # ── CSV-Logging: steps_{ts}.csv ──────────────────────
        sigma_mean = float(pred_sigma.mean().item())
        self._csv_steps.writerow([
            self.total_steps, round(r_intr, 6), round(r_gemini, 4),
            round(r_reward_pred, 4), round(r_total, 4), round(sigma_mean, 4),
            round(novelty, 4), scene_pred, self.current_goal, scene, int(gem_called),
        ])
        self._log_steps.flush()

        # ── CSV-Logging: gemini_{ts}.csv ─────────────────────
        if gem_called and self.last_gemini_result:
            ass = self.last_gemini_result
            self._csv_gemini.writerow([
                self.total_steps,
                round(ass.get("reward", 0), 4),
                round(ass.get("goal_progress", 0), 4),
                ass.get("situation", ""),
                ass.get("recommendation", ""),
                ass.get("next_action_hint", ""),
                ass.get("training_label", ""),
                self.current_goal,
            ])
            self._log_gemini.flush()

        return {
            "r_intr":         r_intr,
            "r_gemini":       r_gemini,
            "r_reward_pred":  r_reward_pred,   # T14: Reward ohne Gemini
            "r_total":        r_total,
            "goal_prog":      goal_prog,
            "gem_called":     gem_called,
            "scene_pred":     scene_pred,       # T13: Szenen-Beschreibung aus z
            "pred_obs":       pred_obs.squeeze(0).permute(1,2,0).detach().numpy(),
            "pred_action":    pred_action.squeeze(0).detach().numpy(),
            "sigma":          pred_sigma.squeeze(0).detach().numpy(),
            "context_norm":   float(context.norm().item()),
            "latent_z":       z.squeeze(0).detach().numpy(),
            **train_info,
        }

    def _train_step(self) -> dict:
        """T12: Sequenz-basiertes Training mit RSSM (Truncated BPTT)."""
        self.train_steps += 1

        # Sequenzen für RSSM-Training (BPTT über SEQ_LEN Steps)
        seq_bs = max(self.cfg["batch_size"] // 2, 4)
        seq_batch = self.replay.sample_sequences(seq_bs, SEQ_LEN)
        if seq_batch is None:
            return {}

        self.encoder.train(); self.decoder.train()
        self.rssm.train(); self.action_head.train()
        self.reward_head.train(); self.scene_head.train()

        B = len(seq_batch["goals"])
        L = SEQ_LEN

        # (B, L, H, W, 3) → (B*L, 3, H, W) für Encoder
        obs_flat  = seq_batch["obs"].reshape(B * L, *OBS_SHAPE).permute(0, 3, 1, 2)
        nobs_flat = seq_batch["next_obs"].reshape(B * L, *OBS_SHAPE).permute(0, 3, 1, 2)
        act_seq   = seq_batch["actions"]   # (B, L, 6)

        # Alle Frames auf einmal encodieren
        mu_all, lv_all, z_all = self.encoder(obs_flat)
        with torch.no_grad():
            _, _, z_next_all = self.encoder(nobs_flat)

        # Reshape → (B, L, latent_dim)
        z_seq      = z_all.reshape(B, L, -1)
        z_next_seq = z_next_all.reshape(B, L, -1)

        # Recon-Loss auf allen Frames
        recon_all = self.decoder(z_all)
        l_recon   = combined_recon_loss(recon_all, obs_flat, ssim_weight=0.3)

        # Free-Bits KL
        kl_per_dim = -0.5 * (1 + lv_all - mu_all.pow(2) - lv_all.exp())
        l_kl       = torch.clamp(kl_per_dim, min=0.5).mean()

        # Goal-Embeddings → projiziert
        goal_embs = torch.cat([
            self.clip.encode(g) for g in seq_batch["goals"]
        ], dim=0)                                          # (B, 512)
        goal_p = F.normalize(self.goal_proj(goal_embs), dim=-1)  # (B, latent_dim)

        # T12: RSSM Forward über Sequenz → h_seq (B, L, d_model)
        h_seq = self.rssm.forward_sequence(z_seq, act_seq, goal_p)

        # Per-Step Losses (über L Steps gemittelt)
        # Gewichte: [linear_x, angular_z, cam_pan, cam_tilt, arc, duration]
        # Pan/Tilt von 0.05 auf 0.3 erhöht: Kamera-Steuerung ist real
        # schneller/billiger als Roboter-Drehung → soll gelernt werden.
        action_weights = torch.tensor(
            [1.0, 1.0, 0.3, 0.3, 0.3, 0.3], device=obs_flat.device
        )
        l_action = l_sigma = l_goal = l_next_z = l_pred_img = l_cam = 0.0

        # T11: EFE-Blend — adaptiv basierend auf Gemini-Daten im Buffer
        gem_count = self.replay.gemini_count
        efe_blend = EFE_BLEND_MAX * min(gem_count / EFE_GEMINI_RAMP, 1.0)

        for t in range(L):
            ctx = h_seq[:, t]            # (B, d_model)
            act = act_seq[:, t]          # (B, 6)

            pa, ps = self.action_head(ctx)

            # T11: Imitation + EFE Blend
            # Imitation: MSE zu ausgeführten Aktionen (Stabilisierung)
            l_imitation_t = (action_weights * (pa - act).pow(2)).mean()
            # EFE pragmatisch: reward_head bewertet die vorhergesagte Aktion
            # Gradient fließt: l_efe → reward_head → pa → ActionHead
            # z ist detached → reward_head bekommt keinen Fehl-Gradienten über z
            r_pred_efe = self.reward_head(
                torch.cat([z_seq[:, t].detach(), pa], dim=-1)
            ).squeeze(-1)
            l_efe_t = -r_pred_efe.mean()  # maximiere vorhergesagten Reward

            l_action += efe_blend * l_efe_t + (1.0 - efe_blend) * l_imitation_t
            l_cam    += pa[:, 2:4].pow(2).mean()

            with torch.no_grad():
                ae = (pa.detach() - act).abs()
            ss = torch.clamp(ps, min=1e-4)
            l_sigma += torch.mean(torch.log(ss) + ae / ss)

            l_goal += torch.clamp(1 - F.cosine_similarity(
                F.normalize(ctx[:, :LATENT_DIM], dim=-1), goal_p, dim=-1
            ), 0.0, 1.0).mean()

            pzn = self.rssm.predict_next_z(ctx, act)
            l_next_z += F.mse_loss(pzn, z_next_seq[:, t].detach())

            # Pred-Image: nobs für Step t (Indices: t, L+t, 2L+t, ...)
            pred_img_t = self.decoder(pzn)
            nobs_t     = nobs_flat[range(t, B * L, L)]
            l_pred_img += combined_recon_loss(pred_img_t, nobs_t, ssim_weight=0.3)

        # Mittel über Sequenzlänge
        l_action   /= L;  l_sigma    /= L;  l_goal   /= L
        l_next_z   /= L;  l_pred_img /= L;  l_cam    /= L
        l_cam_center = l_cam

        # T14: Reward-Prädiktor – eigene Stichprobe aus Gemini-Labels im Buffer
        dev = obs_flat.device
        gem_n = min(self.replay.gemini_count, 8)
        if gem_n >= 2:
            rb = self.replay.sample(gem_n, require_gemini=True)
            if rb is not None:
                with torch.no_grad():
                    _, _, z_rb = self.encoder(rb["obs"].permute(0, 3, 1, 2))
                pred_r   = self.reward_head(
                    torch.cat([z_rb.detach(), rb["actions"]], dim=-1)
                ).squeeze(-1)
                l_reward = F.mse_loss(pred_r, rb["gemini_rewards"])
            else:
                l_reward = torch.tensor(0.0, device=dev)
        else:
            l_reward = torch.tensor(0.0, device=dev)

        # T13: Szenen-Beschreibungs-Loss – schwache Supervision aus Gemini-Labels
        if gem_n >= 2:
            rb_s = self.replay.sample(gem_n, require_gemini=True)
            if rb_s is not None and any(rb_s["gemini_labels"]):
                label_indices = torch.tensor(
                    [self._label_to_vocab_idx(lbl) for lbl in rb_s["gemini_labels"]],
                    dtype=torch.long, device=dev
                )
                with torch.no_grad():
                    _, _, z_s = self.encoder(rb_s["obs"].permute(0, 3, 1, 2))
                l_scene = F.cross_entropy(self.scene_head(z_s.detach()), label_indices)
            else:
                l_scene = torch.tensor(0.0, device=dev)
        else:
            l_scene = torch.tensor(0.0, device=dev)

        fe = (1.0  * l_recon +
              0.5  * l_pred_img +       # Nächst-Frame-Prediction im Bildraum
              self.beta * l_kl +
              0.1  * l_next_z +         # Hilfsziel Latent (reduziert, l_pred_img übernimmt)
              0.2  * l_action +
              0.05 * l_sigma +
              0.1  * l_goal +
              0.05 * l_cam_center +
              0.1  * l_reward +         # T14: Reward-Prädiktor
              0.1  * l_scene)           # T13: Szenen-Beschreibung

        self.optimizer.zero_grad()
        fe.backward()

        # Gradient-Norm pro Modul (Monitoring)
        def _grad_norm(module):
            return sum(p.grad.norm().item()**2
                       for p in module.parameters() if p.grad is not None)**0.5

        grad_norms = {
            "encoder":     _grad_norm(self.encoder),
            "decoder":     _grad_norm(self.decoder),
            "rssm":        _grad_norm(self.rssm),
            "action_head": _grad_norm(self.action_head),
            "goal_proj":   _grad_norm(self.goal_proj),
            "reward_head": _grad_norm(self.reward_head),
            "scene_head":  _grad_norm(self.scene_head),
        }

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.action_head.parameters()) +
            list(self.goal_proj.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.scene_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step(l_recon.detach())

        kl_val = float(l_kl.detach())
        # Mit Free-Bits ist l_kl >= 0.5 (Clamp-Floor). Warnung wenn unerwarteter Collapse.
        # Echter KL-Wert (ohne Free-Bits-Offset) für Monitoring:
        kl_raw = float((-0.5 * (1 + lv_all - mu_all.pow(2) - lv_all.exp())).mean().detach())
        if kl_raw < 0.01 and self.train_steps > 100:
            print(f"  [WARN] KL-Collapse: KL_raw={kl_raw:.4f} nats (TrainStep {self.train_steps})")

        fe_val      = float(fe.detach())
        recon_val   = float(l_recon.detach())
        pred_val    = float(l_pred_img.detach())
        act_val     = float(l_action.detach())
        sigma_val   = float(l_sigma.detach())
        goal_val    = float(l_goal.detach())
        cam_val     = float(l_cam_center.detach())

        reward_val = float(l_reward.detach())
        scene_val  = float(l_scene.detach())

        # CSV-Logging: train_{ts}.csv (jeder Train-Step)
        gn = grad_norms
        self._csv_train.writerow([
            self.train_steps, self.total_steps,
            round(fe_val, 6), round(recon_val, 6), round(pred_val, 6),
            round(kl_val, 6), round(kl_raw, 6), round(act_val, 6),
            round(sigma_val, 6), round(reward_val, 6), round(scene_val, 6),
            round(goal_val, 6), round(cam_val, 6),
            round(efe_blend, 4),
            round(self.optimizer.param_groups[0]["lr"], 8),
            round(self.beta, 6),
            round(gn["encoder"], 4), round(gn["decoder"], 4),
            round(gn["rssm"], 4), round(gn["action_head"], 4),
            round(gn["goal_proj"], 4), round(gn["reward_head"], 4),
            round(gn["scene_head"], 4),
        ])
        if self.train_steps % 20 == 0:
            self._log_train.flush()

        # Kurze Konsolen-Zusammenfassung nur alle 200 Train-Steps
        log_interval = self.cfg.get("log_interval", 200)
        if self.train_steps % log_interval == 0:
            print(
                f"  [T{self.train_steps:5d}] "
                f"FE={fe_val:.4f}  recon={recon_val:.4f}  pred={pred_val:.4f}  "
                f"kl_raw={kl_raw:.4f}  reward={reward_val:.4f}  scene={scene_val:.4f}  "
                f"efe={efe_blend:.2f}  lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

        return {
            "fe":         fe_val,
            "recon":      recon_val,
            "pred_img":   pred_val,
            "kl":         kl_val,
            "kl_raw":     kl_raw,
            "action":     act_val,
            "l_sigma":    sigma_val,
            "l_reward":   reward_val,
            "l_scene":    scene_val,
            "grad_norms": grad_norms,
        }

    def get_ros2_command(self, action_np: np.ndarray) -> dict:
        """Normierte Aktion → physikalische ROS2-Werte."""
        keys   = list(ACTION_BOUNDS.keys())
        bounds = list(ACTION_BOUNDS.values())
        cmd    = {}
        for i, (k, (lo, hi)) in enumerate(zip(keys, bounds)):
            cmd[k] = float((action_np[i]+1)/2*(hi-lo)+lo)
        # Arc-Override
        if abs(cmd["arc_radius"]) > 0.1:
            cmd["angular_z"] = cmd["linear_x"] / cmd["arc_radius"]
        return cmd


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    import sys
    # --headless: kein Matplotlib, nur Konsolen-Ausgabe (für automatisiertes Testen)
    headless = "--headless" in sys.argv
    n_steps  = 300
    for arg in sys.argv[1:]:
        if arg.startswith("--steps="):
            n_steps = int(arg.split("=")[1])

    config = {
        "buffer_size":         1000,
        "batch_size":          16,
        "lr":                  1e-3,
        "beta_max":            0.05,
        "beta_warmup":         200,
        "n_steps":             n_steps,
        "scene_switch_steps":  30,
        "min_gemini_interval": 8,
        "max_gemini_interval": 60,
        "log_interval":        50,   # Konsolen-Log alle N Train-Steps
    }

    api_key = os.environ.get("GEMINI_API_KEY", "")
    print("=" * 60)
    print("B16 – Vollintegration (T10: dynamics_head | T14: reward_head)")
    print("=" * 60)
    print(f"  ACTION_DIM:  {ACTION_DIM}")
    print(f"  LATENT_DIM:  {LATENT_DIM}")
    print(f"  D_MODEL:     {D_MODEL}")
    print(f"  Steps:       {n_steps}")
    print(f"  Headless:    {headless}")
    print()
    print("Gemini:")
    gemini = GeminiClients(api_key=api_key)
    print()

    system = IntegratedSystem(config, gemini)

    total_params = (
            sum(p.numel() for p in system.encoder.parameters()) +
            sum(p.numel() for p in system.decoder.parameters()) +
            sum(p.numel() for p in system.rssm.parameters()) +
            sum(p.numel() for p in system.action_head.parameters())
    )
    # Dynamics-Head Parameter separat anzeigen (T10)
    dyn_params = sum(p.numel() for p in system.rssm.dynamics_head.parameters())
    print(f"Gesamt-Parameter: {total_params:,}  (davon dynamics_head: {dyn_params:,})")
    print(f"Steps: {config['n_steps']}")
    print(f"Logs:  logs/steps_{system._log_ts}.csv")
    print(f"       logs/train_{system._log_ts}.csv")
    print(f"       logs/gemini_{system._log_ts}.csv")
    print()

    if headless:
        fig = None
        ax_obs = ax_recon = ax_action = ax_goal = ax_info = None
        ax_fe = ax_rewards = ax_gemini = None
        ax_prog = ax_online = ax_ros = None
        gem_call_steps = []
    else:
        # ── Matplotlib ────────────────────────────────────────
        fig = plt.figure(figsize=(17, 11))
        fig.suptitle(
            'B16 – Vollintegration: Online Learning mit Gemini ER',
            fontsize=13, fontweight='bold'
        )
        gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

        # Zeile 0: Obs, Recon, Aktion, Ziel, Info
        ax_obs    = fig.add_subplot(gs[0, 0])
        ax_recon  = fig.add_subplot(gs[0, 1])
        ax_action = fig.add_subplot(gs[0, 2])
        ax_goal   = fig.add_subplot(gs[0, 3])
        ax_info   = fig.add_subplot(gs[0, 4]); ax_info.axis('off')

        # Zeile 1: Free Energy, Rewards, Gemini-Interval
        ax_fe      = fig.add_subplot(gs[1, :2])
        ax_rewards = fig.add_subplot(gs[1, 2:4])
        ax_gemini  = fig.add_subplot(gs[1, 4]); ax_gemini.axis('off')

        # Zeile 2: Goal Progress, Online Learning, ROS2
        ax_prog    = fig.add_subplot(gs[2, :2])
        ax_online  = fig.add_subplot(gs[2, 2:4])
        ax_ros     = fig.add_subplot(gs[2, 4]); ax_ros.axis('off')
        gem_call_steps = []

    scene_idx = 0
    scene     = SCENE_TYPES[scene_idx]

    # Initial Goal setzen
    system.set_goal(f"find the {scene.replace('_', ' ')}")

    last_result    = {}

    print(f"Starte Online-Learning Loop: {config['n_steps']} Steps\n")

    for step in range(config["n_steps"]):

        # Szene wechseln
        if step > 0 and step % config["scene_switch_steps"] == 0:
            scene_idx = (scene_idx + 1) % len(SCENE_TYPES)
            scene     = SCENE_TYPES[scene_idx]
            goal_cmd  = f"find the {scene.replace('_', ' ')}"
            res = system.set_goal(goal_cmd)
            print(f"  [Step {step:4d}] Neues Ziel: '{res['primary_goal']}'")

        # Obs + Action
        obs_np      = draw_scene(scene, noise=0.02)
        next_obs_np = draw_scene(scene, noise=0.03)
        base_act    = np.array(SCENE_ACTIONS[scene], dtype=np.float32)
        action_np   = np.clip(base_act + 0.1*np.random.randn(ACTION_DIM).astype(np.float32), -1, 1)

        # System Step
        result = system.step(obs_np, action_np, next_obs_np, scene)
        last_result = result

        if result["gem_called"]:
            gem_call_steps.append(step)
            ass = system.last_gemini_result
            print(f"  [Step {step:4d}] Gemini: r={ass['reward']:.3f}  "
                  f"'{ass.get('situation','')[:50]}'")

        # Visualisierung (nur wenn nicht headless)
        if not headless and (step % 20 == 0 or step == config["n_steps"] - 1):
            m       = system.metrics
            steps_x = list(range(len(m["r_total"])))

            # ── Obs ───────────────────────────────────
            ax_obs.clear()
            ax_obs.imshow(obs_np, interpolation='nearest')
            ax_obs.set_title(f'Obs: {scene}\nGoal: {system.current_goal[:20]}',
                             fontsize=7)
            ax_obs.axis('off')

            # ── Recon ──────────────────────────────────
            ax_recon.clear()
            recon_img = np.clip(result["pred_obs"]*255, 0, 255).astype(np.uint8)
            ax_recon.imshow(recon_img, interpolation='nearest')
            mse = m["recon"][-1] if m["recon"] else 0
            ax_recon.set_title(f'Rekonstruktion\nMSE={mse:.4f}', fontsize=7)
            ax_recon.axis('off')

            # ── Aktion ─────────────────────────────────
            ax_action.clear()
            act = result["pred_action"]
            sig = result["sigma"]
            anames = ["lx","az","pan","tilt","arc","dur"]
            colors = ['steelblue' if v >= 0 else 'tomato' for v in act]
            ax_action.bar(anames, act, color=colors, alpha=0.8)
            ax_action.errorbar(anames, act, yerr=sig,
                               fmt='none', color='black', capsize=3)
            ax_action.axhline(0, color='white', linewidth=0.5)
            ax_action.set_ylim(-1.3, 1.3)
            ax_action.set_title(f'Pred. Aktion (6D)\n± sigma', fontsize=7)
            ax_action.tick_params(labelsize=6)
            ax_action.set_facecolor('#0d0d0d')
            ax_action.tick_params(colors='white')

            # ── Ziel-Panel ─────────────────────────────
            ax_goal.clear(); ax_goal.axis('off')
            g_lines = [
                "── Aktuelles Ziel ──────",
                f'"{system.current_goal}"',
                "",
                f"Szene:  {scene}",
                f"Step:   {step+1}/{config['n_steps']}",
                "",
                "── Gemini Robotics ─────",
            ]
            if system.last_gemini_result:
                ass = system.last_gemini_result
                g_lines += [
                    f"r:      {ass.get('reward',0):.3f}",
                    f"Prog:   {ass.get('goal_progress',0)*100:.0f}%",
                    f"Hint:   {ass.get('next_action_hint','')}",
                    f"Label:  {ass.get('training_label','')}",
                    "",
                    f"'{ass.get('situation','')[:28]}'",
                ]
            ax_goal.text(0.03, 0.98, "\n".join(g_lines),
                         transform=ax_goal.transAxes,
                         fontsize=7.5, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9),
                         color='lightcyan')

            # ── Info ───────────────────────────────────
            ax_info.clear(); ax_info.axis('off')
            i_lines = [
                "── B16 Vollintegration ──",
                "",
                "B02 ✓ Replay Buffer",
                "B03 ✓ Temporal Buffer",
                "B04b✓ VAE Encoder",
                "B05 ✓ CLIP (Mock)",
                "B07 ✓ Transformer",
                "B08 ✓ CNN Decoder",
                "B09 ✓ Action Head 6D",
                "B10 ✓ Free Energy",
                "B11 ✓ Training Loop",
                "B12 ✓ Intrinsic Reward",
                "B13 ✓ Gemini Text",
                "B14 ✓ Adaptive Freq.",
                "B15 ✓ Reward Kombination",
                "",
                f"── Parameter ────────────",
                f"Gesamt: {total_params:,}",
                f"Buffer: {system.replay.size}/{config['buffer_size']}",
                f"Gemini: {system.replay.gemini_count} Labels",
                f"Train:  {system.train_steps} Steps",
                "",
                f"── Online Learning ──────",
                f"Call-Rate: {system.adaptive.call_rate*100:.1f}%",
                f"Interval:  {m['gemini_interval'][-1] if m['gemini_interval'] else 0:.0f} Steps",
                f"LR:        {m['lr'][-1] if m['lr'] else 0:.2e}",
                f"Beta:      {system.beta:.4f}",
            ]
            ax_info.text(0.03, 0.98, "\n".join(i_lines),
                         transform=ax_info.transAxes,
                         fontsize=7, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            # ── Free Energy ─────────────────────────────
            ax_fe.clear()
            if m["fe"]:
                ax_fe.plot(range(len(m["fe"])), m["fe"],
                           color='black', linewidth=1.5, alpha=0.5,
                           label='Free Energy')
                if len(m["fe"]) >= 15:
                    ma = np.convolve(m["fe"], np.ones(15)/15, mode='valid')
                    ax_fe.plot(range(14, len(m["fe"])), ma,
                               color='red', linewidth=2, label='MA-15')
            for cs in gem_call_steps:
                if cs < len(steps_x):
                    ax_fe.axvline(cs, color='cyan', linewidth=0.8, alpha=0.4)
            ax_fe.set_title('Free Energy  |  Cyan = Gemini ER Call', fontsize=9)
            ax_fe.set_facecolor('#0d0d0d')
            ax_fe.tick_params(colors='white')
            ax_fe.legend(fontsize=7)

            # ── Rewards ─────────────────────────────────
            ax_rewards.clear()
            if m["r_intrinsic"]:
                ax_rewards.plot(steps_x, m["r_intrinsic"],
                                color='steelblue', linewidth=1.2, alpha=0.7,
                                label='Intrinsic')
                ax_rewards.plot(steps_x, m["r_gemini"],
                                color='gold', linewidth=1.5,
                                label='Gemini ER')
                ax_rewards.plot(steps_x, m["r_total"],
                                color='white', linewidth=2,
                                label='Total')
                if len(m["r_total"]) >= 15:
                    ma = np.convolve(m["r_total"], np.ones(15)/15, mode='valid')
                    ax_rewards.plot(range(14, len(m["r_total"])), ma,
                                    color='orange', linewidth=2.5, linestyle='--',
                                    label='Total MA-15')
            ax_rewards.set_title('Rewards: Intrinsic + Gemini ER + Goal',
                                 fontsize=9)
            ax_rewards.legend(fontsize=6, ncol=2)
            ax_rewards.set_facecolor('#0d0d0d')
            ax_rewards.tick_params(colors='white')

            # ── Gemini Panel ────────────────────────────
            ax_gemini.clear(); ax_gemini.axis('off')
            gem_lines = [
                "── Gemini-Modelle ───────",
                "",
                "TEXT (B13):",
                f"  gemini-2.5-flash",
                "",
                "VISION (B15):",
                f"  gemini-robotics",
                f"  -er-1.5-preview",
                "",
                "── Online Learning ──────",
                "",
                "Gemini ER generiert:",
                "  reward → Trainings-",
                "           signal",
                "  label  → Annotation",
                "           ohne Mensch",
                "",
                "Lokales Modell lernt",
                "mit 16×16 low-res.",
                "Gemini sieht 128×128.",
                "",
                f"Gemini-Calls: {system.adaptive.calls}",
                f"Labels im Buffer:",
                f"  {system.replay.gemini_count}",
                "",
                "Interval wächst wenn",
                "FE sinkt → Modell",
                "wird selbstständiger.",
            ]
            ax_gemini.text(
                0.03, 0.98, "\n".join(gem_lines),
                transform=ax_gemini.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9),
                color='lightcyan'
            )

            # ── Goal Progress ───────────────────────────
            ax_prog.clear()
            if m["goal_progress"]:
                # Nur Punkte wo Gemini bewertet hat
                prog_vals = [(i, v) for i, v in enumerate(m["goal_progress"]) if v > 0]
                if prog_vals:
                    px, py = zip(*prog_vals)
                    ax_prog.scatter(px, py, c='seagreen', s=40,
                                    zorder=4, label='Gemini Progress')
                    ax_prog.fill_between(steps_x, m["goal_progress"],
                                         alpha=0.3, color='seagreen')
            ax_prog.axhline(1.0, color='gold', linestyle='--',
                            linewidth=1, label='Ziel erreicht')
            ax_prog.set_ylim(0, 1.1)
            ax_prog.set_title('Goal Progress (Gemini ER Bewertung)', fontsize=9)
            ax_prog.legend(fontsize=7)
            ax_prog.set_facecolor('#0d0d0d')
            ax_prog.tick_params(colors='white')

            # ── Online Learning: Interval-Verlauf ───────
            ax_online.clear()
            if m["gemini_interval"]:
                ax_online.plot(steps_x, m["gemini_interval"],
                               color='orange', linewidth=1.5,
                               label='Gemini-Interval (Steps)')
                ax_online2 = ax_online.twinx()
                ax_online2.plot(steps_x,
                                [v*100 for v in m["gemini_call_rate"]],
                                color='lightblue', linewidth=1.5,
                                linestyle='--', alpha=0.8, label='Call-Rate %')
                ax_online2.set_ylabel('Call-Rate %', color='lightblue', fontsize=8)
                ax_online2.tick_params(axis='y', colors='lightblue')
                ax_online2.legend(fontsize=7, loc='upper right')
            ax_online.set_title(
                'Adaptives Gemini-Interval\n'
                '(steigt = Modell wird selbstständiger)', fontsize=9
            )
            ax_online.legend(fontsize=7, loc='upper left')
            ax_online.set_facecolor('#0d0d0d')
            ax_online.tick_params(colors='white')

            # ── ROS2 ────────────────────────────────────
            ax_ros.clear(); ax_ros.axis('off')
            cmd = system.get_ros2_command(result["pred_action"])
            arc_desc = (f"Kurve {'L' if cmd['arc_radius']>0 else 'R'} "
                        f"R={abs(cmd['arc_radius']):.1f}m"
                        if abs(cmd.get("arc_radius", 0)) > 0.1
                        else "geradeaus")
            ros_lines = [
                "── ROS2 Twist ───────────",
                f"linear.x  = {cmd['linear_x']:+.3f} m/s",
                f"angular.z = {cmd['angular_z']:+.3f} rad/s",
                f"duration  = {cmd['duration']:.2f}s",
                "",
                "── Kamera ───────────────",
                f"pan  = {cmd['camera_pan']*180/np.pi:+.0f}°",
                f"tilt = {cmd['camera_tilt']*180/np.pi:+.0f}°",
                "",
                "── Bewegung ─────────────",
                f"{arc_desc}",
                "",
                "── Unsicherheit ─────────",
                *[f"{list(ACTION_BOUNDS.keys())[i][:6]}:"
                  f" {result['sigma'][i]:.3f}"
                  for i in range(ACTION_DIM)],
            ]
            ax_ros.text(
                0.03, 0.98, "\n".join(ros_lines),
                transform=ax_ros.transAxes,
                fontsize=7.5, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
            )

            plt.pause(0.03)

    # ── Finale Konsolen-Ausgabe ───────────────────────────
    m = system.metrics
    print()
    # Log-Dateien schließen
    system._log_steps.close()
    system._log_train.close()
    system._log_gemini.close()

    print()
    print("=" * 60)
    print("Vollintegration abgeschlossen!")
    print("=" * 60)
    print(f"  Steps total:    {system.total_steps}")
    print(f"  Training Steps: {system.train_steps}")
    print(f"  Gemini Calls:   {system.adaptive.calls}  ({system.adaptive.call_rate*100:.1f}%)")
    if m["fe"]:
        print(f"  FE final:       {m['fe'][-1]:.5f}  "
              f"recon={m['recon'][-1]:.5f}  pred={m.get('pred_img',[0])[-1]:.5f}  "
              f"kl_raw={m.get('kl_raw',[0])[-1]:.5f}")
    print(f"  Reward Ø(20):   {np.mean(m['r_total'][-20:]):.4f}")
    print()
    print(f"  Logs: logs/*_{system._log_ts}.csv")

    if not headless:
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    run_demo()
