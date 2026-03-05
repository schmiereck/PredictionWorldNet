"""
B11 – Training Loop
====================
Verbindet alle Bausteine B02–B10 zu einem vollständigen Trainings-Zyklus.

Ablauf pro Episode:
    1. Env-Step: obs, action → nächster obs
    2. Temporal Buffer (B03): (obs, action) speichern
    3. Replay Buffer (B02): Transition speichern
    4. Wenn Buffer bereit:
        a. Batch samplen (B02)
        b. Encoder (B04b): obs → (mu, log_var, z)
        c. CLIP (B05): goal_text → goal_emb  [Mock in B11]
        d. Action Embedding (B06): action → z_action
        e. Temporal Transformer (B07): alles → context
        f. Decoder (B08): context → pred_frame
        g. Action Head (B09): context → pred_action
        h. Prediction Loss (B10): alle Terme → Free Energy
        i. Backward + Update

Gemini-Interface (Ausblick B13):
    goal_text = gemini.parse_command("Geh zur roten Box")
    → gibt "find the red box" zurück
    → CLIP kodiert das zu goal_emb
    → Transformer bekommt goal_emb als Token

Demo:
    Nutzt synthetische Szenen (wie B08) mit Mock-Env.
    Zeigt Live-Training aller Komponenten zusammen.
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# SZENEN & MOCK ENV (aus B08)
# ─────────────────────────────────────────────

def draw_scene(scene_type: str) -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(10, 16):
        img[y, :] = [int(60 + (y-10)*15)]*3
    img[0:2, :] = [40, 40, 60]
    img[2:10, 1] = img[2:10, 14] = [70, 70, 90]
    for y in range(2, 8):
        img[y, 2:14] = [100, 100, 120]
    if scene_type == "red_box":
        img[8:12, 5:9]  = [200, 40, 40]
        img[6:9,  6:10] = [160, 30, 30]
        img[6:12, 9]    = [120, 20, 20]
    elif scene_type == "blue_ball":
        for y in range(16):
            for x in range(16):
                d = np.sqrt((x-8)**2+(y-10)**2)
                if d < 3.2:
                    b = int(255*max(0,1-d/3.2))
                    h = int(80*max(0,1-((x-7)**2+(y-9)**2)/4))
                    img[y,x] = [0,b//3,min(255,b+h)]
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
    return img

SCENE_TYPES  = ["red_box", "blue_ball", "green_door", "corridor", "corner"]
SCENE_LABELS = ["Rote Box", "Blauer Ball", "Gruene Tuer", "Korridor", "Ecke"]

# Ziel-Texte pro Szene (werden später von Gemini erzeugt)
SCENE_GOALS = {
    "red_box":    "find the red box",
    "blue_ball":  "find the blue ball",
    "green_door": "navigate to the exit door",
    "corridor":   "explore the corridor",
    "corner":     "navigate to the corner",
}


class MockEnv:
    """
    Simuliert einen Agenten der durch die 5 Szenen navigiert.
    Alle N_SCENE_STEPS Schritte wechselt die aktive Szene.
    """
    N_SCENE_STEPS = 20
    ACTION_DIM    = 3
    OBS_SHAPE     = (16, 16, 3)
    ACTION_BOUNDS = {"linear_x": (-0.5, 0.5), "angular_z": (-1.0, 1.0), "duration": (0.1, 2.0)}

    # Typische Aktionen pro Szene (normalisiert [-1,1])
    SCENE_ACTIONS = {
        "red_box":    [ 0.6,  0.0, -0.5],
        "blue_ball":  [ 0.4,  0.6, -0.5],
        "green_door": [ 0.8,  0.0, -0.3],
        "corridor":   [ 1.0,  0.0, -0.4],
        "corner":     [ 0.3,  0.8, -0.6],
    }

    def __init__(self):
        self.step_count   = 0
        self.scene_idx    = 0
        self.scene_step   = 0

    @property
    def current_scene(self):
        return SCENE_TYPES[self.scene_idx]

    @property
    def current_goal(self):
        return SCENE_GOALS[self.current_scene]

    def step(self):
        self.step_count += 1
        self.scene_step += 1

        # Szene wechseln
        if self.scene_step >= self.N_SCENE_STEPS:
            self.scene_step = 0
            self.scene_idx  = (self.scene_idx + 1) % len(SCENE_TYPES)

        obs    = draw_scene(self.current_scene)
        # Typische Aktion + Rauschen
        base   = np.array(self.SCENE_ACTIONS[self.current_scene], dtype=np.float32)
        action = np.clip(base + 0.1 * np.random.randn(self.ACTION_DIM).astype(np.float32), -1, 1)
        reward = float(np.random.rand() * 0.1)   # Mock: kleiner zufälliger Reward
        done   = (self.scene_step == self.N_SCENE_STEPS - 1)

        return obs, action, reward, done


# ─────────────────────────────────────────────
# ALLE MODELL-KOMPONENTEN
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
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


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),  nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32,3,3,stride=2,padding=1,output_padding=1),   nn.Sigmoid(),
        )
    def forward(self, z):
        return self.deconv(self.fc(z).reshape(z.size(0),128,2,2))


class ActionHead(nn.Module):
    def __init__(self, d_model=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,128), nn.LayerNorm(128), nn.ReLU(True), nn.Dropout(0.1),
            nn.Linear(128,64),      nn.LayerNorm(64),  nn.ReLU(True),
        )
        self.action_out = nn.Sequential(nn.Linear(64, action_dim), nn.Tanh())
        self.sigma_out  = nn.Sequential(nn.Linear(64, action_dim), nn.Sigmoid())

    def forward(self, z):
        f = self.net(z)
        return self.action_out(f), self.sigma_out(f)


# Mock CLIP-Encoder (ersetzt durch echten CLIP in Produktion)
class MockCLIPEncoder:
    """
    Simuliert CLIP Text-Encoder (B05).
    In B13 wird dieser durch echten CLIP + Gemini ersetzt:
        User: "Geh zur roten Box"
        Gemini: "find the red box"
        CLIP: → 512-dim Embedding
    """
    def __init__(self, dim=512):
        self.dim = dim
        self._cache = {}

    def encode(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            # Deterministisch aus Text-Hash
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            self._cache[text] = torch.from_numpy(vec / np.linalg.norm(vec))
        return self._cache[text].unsqueeze(0)  # (1, 512)


# ─────────────────────────────────────────────
# REPLAY BUFFER (B02)
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, max_size=1000, obs_shape=(16,16,3), action_dim=3):
        self.max_size   = max_size
        self.ptr        = 0
        self.size       = 0
        self.obs        = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.next_obs   = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.actions    = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards    = np.zeros(max_size, dtype=np.float32)
        self.dones      = np.zeros(max_size, dtype=bool)
        self.goals      = [""] * max_size

    def add(self, obs, next_obs, action, reward, done, goal=""):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.goals[self.ptr]    = goal
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs":      torch.from_numpy(self.obs[idx]).float() / 255.0,
            "next_obs": torch.from_numpy(self.next_obs[idx]).float() / 255.0,
            "actions":  torch.from_numpy(self.actions[idx]),
            "rewards":  torch.from_numpy(self.rewards[idx]),
            "goals":    [self.goals[i] for i in idx],
        }

    def is_ready(self, min_size):
        return self.size >= min_size


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

class TrainingLoop:
    """
    Verbindet alle Bausteine zum vollständigen Trainings-Zyklus.

    Architektur-Überblick:
        obs → Encoder → z
        goal_text → MockCLIP → goal_emb
        z + goal_emb → (vereinfachter Kontext, B07 folgt in Vollintegration)
        context → Decoder → pred_obs
        context → ActionHead → pred_action
        alles → PredictionLoss → Free Energy → Backprop
    """

    def __init__(self, config: dict):
        self.cfg = config
        ld = config["latent_dim"]
        ad = config["action_dim"]

        # Modelle
        self.encoder     = Encoder(latent_dim=ld)
        self.decoder     = Decoder(latent_dim=ld)
        self.action_head = ActionHead(d_model=ld, action_dim=ad)
        self.clip        = MockCLIPEncoder(dim=512)

        # Einfache Ziel-Projektion: 512 → latent_dim
        self.goal_proj = nn.Linear(512, ld, bias=False)

        # Buffer
        self.replay_buf = ReplayBuffer(
            max_size=config["buffer_size"],
            obs_shape=(16, 16, 3),
            action_dim=ad,
        )

        # Optimizer
        all_params = (
                list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.action_head.parameters()) +
                list(self.goal_proj.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=config["lr"], weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=100, min_lr=1e-5
        )

        # KL Beta-Annealing
        self.beta     = 0.0
        self.beta_max = config["beta_max"]

        # Metriken
        self.metrics = {k: [] for k in [
            "fe", "recon", "kl", "action",
            "intrinsic_reward", "extrinsic_reward",
            "total_reward", "beta", "lr",
        ]}
        self.episode_rewards = []
        self._ep_reward      = 0.0

    def _anneal_beta(self, step):
        warmup = self.cfg["beta_warmup"]
        self.beta = min(self.beta_max, self.beta_max * step / warmup)

    def collect_step(self, env: MockEnv):
        """Ein Env-Step: Transition sammeln und in Buffer speichern."""
        obs, action, reward, done = env.step()

        # Nächste Obs (gleiche Szene, leicht verrauscht)
        next_obs, _, _, _ = env.step()

        self.replay_buf.add(
            obs, next_obs, action, reward, done,
            goal=env.current_goal
        )
        self._ep_reward += reward
        if done:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0

        return obs, action, reward, done, env.current_goal

    def train_step(self, step: int):
        """Ein Trainings-Schritt auf einem gesampelten Batch."""
        if not self.replay_buf.is_ready(self.cfg["batch_size"]):
            return None

        self._anneal_beta(step)

        batch     = self.replay_buf.sample(self.cfg["batch_size"])
        obs       = batch["obs"].permute(0, 3, 1, 2)       # (B,3,H,W)
        next_obs  = batch["next_obs"].permute(0, 3, 1, 2)
        actions   = batch["actions"]
        rewards   = batch["rewards"]
        goals     = batch["goals"]

        self.encoder.train()
        self.decoder.train()
        self.action_head.train()

        # ── Forward Pass ──────────────────────────────────
        # Encoder: obs → latent
        mu, log_var, z = self.encoder(obs)

        # Ziel-Embedding: text → 512 → latent_dim
        # [Hier kommt in B13 Gemini rein: User-Sprache → CLIP-Text]
        goal_embs = torch.cat([
            self.clip.encode(g) for g in goals
        ], dim=0)                                           # (B, 512)
        goal_proj = F.normalize(self.goal_proj(goal_embs), dim=-1)

        # Kontext: z + goal (vereinfacht, B07 Transformer folgt)
        context = z + 0.1 * goal_proj   # (B, latent_dim)

        # Decoder: context → rekonstruierter Frame
        recon = self.decoder(context)

        # Nächster Frame Vorhersage
        mu_next, _, z_next = self.encoder(next_obs)
        pred_next = self.decoder(z_next)

        # Action Head: context → Aktion
        pred_action, pred_sigma = self.action_head(context)

        # ── Losses ────────────────────────────────────────
        loss_recon  = F.mse_loss(recon, obs)
        loss_kl     = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss_action = F.mse_loss(pred_action, actions)

        # Temporaler Loss: nächster Frame
        loss_temp   = F.mse_loss(pred_next, next_obs)

        # Ziel-Loss: Cosinus-Distanz Kontext ↔ Ziel
        loss_goal   = (1.0 - F.cosine_similarity(
            F.normalize(context, dim=-1),
            goal_proj, dim=-1
        )).mean()

        # Intrinsic Reward (kein Gradient)
        with torch.no_grad():
            intr_rew = F.mse_loss(pred_next, next_obs).item()

        # Free Energy
        fe = (
                1.0  * loss_recon +
                self.beta * loss_kl +
                0.3  * loss_temp +
                0.2  * loss_action +
                0.1  * loss_goal
        )

        # ── Backward ──────────────────────────────────────
        self.optimizer.zero_grad()
        fe.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.action_head.parameters()) +
            list(self.goal_proj.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step(loss_recon.detach())

        # ── Metriken ──────────────────────────────────────
        ext_rew = float(rewards.mean())
        self.metrics["fe"].append(float(fe.detach()))
        self.metrics["recon"].append(float(loss_recon.detach()))
        self.metrics["kl"].append(float(loss_kl.detach()))
        self.metrics["action"].append(float(loss_action.detach()))
        self.metrics["intrinsic_reward"].append(intr_rew)
        self.metrics["extrinsic_reward"].append(ext_rew)
        self.metrics["total_reward"].append(intr_rew + ext_rew)
        self.metrics["beta"].append(self.beta)
        self.metrics["lr"].append(self.optimizer.param_groups[0]["lr"])

        return {
            "fe":     float(fe.detach()),
            "recon":  float(loss_recon.detach()),
            "kl":     float(loss_kl.detach()),
            "action": float(loss_action.detach()),
            "recon_img":  recon.detach(),
            "obs_img":    obs,
            "pred_action": pred_action.detach(),
            "sigma":       pred_sigma.detach(),
            "intr_rew":   intr_rew,
            "scene":      goals[0],
            "goal":       goals[0],
        }

    def get_ros2_action(self, obs_np: np.ndarray, goal_text: str) -> dict:
        """
        Inference: obs + goal_text → ROS2-Aktion.
        Später: goal_text kommt von Gemini-Interface (B13).
        """
        self.encoder.eval()
        self.action_head.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs_np).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0)
            _, _, z    = self.encoder(x)
            goal_emb   = self.clip.encode(goal_text)
            goal_proj  = F.normalize(self.goal_proj(goal_emb), dim=-1)
            context    = z + 0.1 * goal_proj
            action, sigma = self.action_head(context)

        a = action.squeeze(0).numpy()
        s = sigma.squeeze(0).numpy()

        bounds = [(-0.5,0.5), (-1.0,1.0), (0.1,2.0)]
        keys   = ["linear_x", "angular_z", "duration"]
        ros2   = {k: float((a[i]+1)/2*(hi-lo)+lo)
                  for i,(k,(lo,hi)) in enumerate(zip(keys,bounds))}
        ros2["sigma"] = {k: float(s[i]) for i,k in enumerate(keys)}
        return ros2


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    config = {
        "latent_dim":   64,
        "action_dim":   3,
        "buffer_size":  500,
        "batch_size":   16,
        "lr":           1e-3,
        "beta_max":     0.05,
        "beta_warmup":  300,
        "n_steps":      500,
        "collect_steps": 2,    # Env-Steps pro Train-Step
    }

    env    = MockEnv()
    loop   = TrainingLoop(config)

    total_enc = sum(p.numel() for p in loop.encoder.parameters())
    total_dec = sum(p.numel() for p in loop.decoder.parameters())
    total_act = sum(p.numel() for p in loop.action_head.parameters())

    print("B11 – Training Loop")
    print(f"  Encoder:      {total_enc:,}")
    print(f"  Decoder:      {total_dec:,}")
    print(f"  Action Head:  {total_act:,}")
    print(f"  Batch Size:   {config['batch_size']}")
    print(f"  Buffer Size:  {config['buffer_size']}")
    print(f"  Steps:        {config['n_steps']}")
    print()
    print("Gemini-Interface Ausblick (B13):")
    print("  goal_text = gemini.parse('Geh zur roten Box')")
    print("  → 'find the red box'")
    print("  → CLIP → goal_embedding → Transformer\n")

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B11 – Training Loop: Alle Komponenten zusammen',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    # Zeile 0: Live-Szene + Rekonstruktion + Aktion
    ax_obs    = fig.add_subplot(gs[0, 0])
    ax_recon  = fig.add_subplot(gs[0, 1])
    ax_action = fig.add_subplot(gs[0, 2])
    ax_goal   = fig.add_subplot(gs[0, 3])
    ax_info   = fig.add_subplot(gs[0, 4])
    ax_info.axis('off')

    # Zeile 1: Loss-Kurven
    ax_fe     = fig.add_subplot(gs[1, :2])
    ax_terms  = fig.add_subplot(gs[1, 2:4])
    ax_beta   = fig.add_subplot(gs[1, 4])

    # Zeile 2: Rewards + Buffer + ROS2
    ax_rew    = fig.add_subplot(gs[2, :2])
    ax_buf    = fig.add_subplot(gs[2, 2:4])
    ax_ros    = fig.add_subplot(gs[2, 4])
    ax_ros.axis('off')

    last_result = None

    print(f"Starte Training Loop: {config['n_steps']} Schritte\n")

    for step in range(config["n_steps"]):

        # Daten sammeln
        for _ in range(config["collect_steps"]):
            obs, action, reward, done, goal = loop.collect_step(env)

        # Training
        result = loop.train_step(step)

        if result is None:
            print(f"  Step {step:4d}: Buffer füllt sich... "
                  f"({loop.replay_buf.size}/{config['batch_size']})")
            continue

        last_result = result

        if step % 25 == 0 or step == config["n_steps"] - 1:
            m       = loop.metrics
            steps_x = list(range(len(m["fe"])))

            # ── Aktuelle Szene ─────────────────────────
            ax_obs.clear()
            obs_img = (result["obs_img"][0].permute(1,2,0).numpy()*255).astype(np.uint8)
            ax_obs.imshow(obs_img, interpolation='nearest')
            ax_obs.set_title(f'Obs\n{result["scene"]}', fontsize=8)
            ax_obs.axis('off')

            # ── Rekonstruktion ─────────────────────────
            ax_recon.clear()
            recon_img = np.clip(
                result["recon_img"][0].permute(1,2,0).numpy()*255, 0, 255
            ).astype(np.uint8)
            ax_recon.imshow(recon_img, interpolation='nearest')
            ax_recon.set_title(f'Recon\nMSE={result["recon"]:.4f}', fontsize=8)
            ax_recon.axis('off')

            # ── Vorhergesagte Aktion ────────────────────
            ax_action.clear()
            act = result["pred_action"][0].numpy()
            sig = result["sigma"][0].numpy()
            dim_names = ["lin_x", "ang_z", "dur"]
            colors = ['steelblue' if v >= 0 else 'tomato' for v in act]
            bars = ax_action.bar(dim_names, act, color=colors, alpha=0.8)
            ax_action.errorbar(dim_names, act, yerr=sig,
                               fmt='none', color='black', capsize=4)
            ax_action.axhline(0, color='black', linewidth=0.6)
            ax_action.set_ylim(-1.3, 1.3)
            ax_action.set_title('Pred. Aktion\n± Unsicherheit', fontsize=8)
            ax_action.tick_params(labelsize=7)

            # ── Ziel ───────────────────────────────────
            ax_goal.clear()
            ax_goal.axis('off')
            ax_goal.set_facecolor('#0a0a1a')
            goal_lines = [
                "── Aktuelles Ziel ──────",
                "",
                f'"{result["goal"]}"',
                "",
                "← Kommt von:",
                "  User → Gemini (B13)",
                "  Gemini → CLIP (B05)",
                "  CLIP → goal_emb",
                "  → Transformer (B07)",
                "",
                "── ROS2 Aktion ─────────",
            ]
            bounds = [(-0.5,0.5),(-1.0,1.0),(0.1,2.0)]
            keys   = ["linear_x","angular_z","duration"]
            for i,(k,(lo,hi)) in enumerate(zip(keys,bounds)):
                val = float((act[i]+1)/2*(hi-lo)+lo)
                goal_lines.append(f"  {k:10s}: {val:+.3f}")
            ax_goal.text(0.05, 0.95, "\n".join(goal_lines),
                         transform=ax_goal.transAxes,
                         fontsize=7.5, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9),
                         color='lightcyan')

            # ── Info ───────────────────────────────────
            ax_info.clear()
            ax_info.axis('off')
            info_lines = [
                "── B11 Training Loop ──",
                f"Step:     {step+1}/{config['n_steps']}",
                f"Buffer:   {loop.replay_buf.size}/{config['buffer_size']}",
                f"LR:       {m['lr'][-1]:.2e}",
                f"Beta:     {loop.beta:.4f}",
                "",
                "── Losses ─────────────",
                f"FE:       {result['fe']:.5f}",
                f"Recon:    {result['recon']:.5f}",
                f"KL:       {result['kl']:.5f}",
                f"Action:   {result['action']:.5f}",
                "",
                "── Rewards ────────────",
                f"Intr:     {result['intr_rew']:.5f}",
                f"Extr:     {m['extrinsic_reward'][-1]:.5f}",
                "",
                "── Komponenten ────────",
                "B02 ✓ Replay Buffer",
                "B03 ✓ Temporal Buffer",
                "B04b✓ VAE Encoder",
                "B05 ✓ CLIP (Mock)",
                "B08 ✓ CNN Decoder",
                "B09 ✓ Action Head",
                "B10 ✓ Free Energy",
                "B07   Transformer*",
                "",
                "* B07 vereinfacht:",
                "  z + goal_proj",
                "  (Vollintegr. B16)",
            ]
            ax_info.text(0.03, 0.98, "\n".join(info_lines),
                         transform=ax_info.transAxes,
                         fontsize=7, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            # ── Free Energy ────────────────────────────
            ax_fe.clear()
            ax_fe.plot(steps_x, m["fe"], color='black', linewidth=1.5,
                       alpha=0.5, label='Free Energy')
            if len(m["fe"]) >= 20:
                ma = np.convolve(m["fe"], np.ones(20)/20, mode='valid')
                ax_fe.plot(range(19, len(m["fe"])), ma,
                           color='red', linewidth=2, label='MA-20')
            ax_fe.set_title('Gesamte Free Energy', fontsize=9)
            ax_fe.set_xlabel('Schritt')
            ax_fe.legend(fontsize=7)

            # ── Einzelne Terme ─────────────────────────
            ax_terms.clear()
            for key, label, color in [
                ("recon",  "Reconstruction", "steelblue"),
                ("kl",     "KL",             "darkorange"),
                ("action", "Action",          "purple"),
            ]:
                if m[key]:
                    ax_terms.plot(range(len(m[key])), m[key],
                                  color=color, linewidth=1.3,
                                  label=label, alpha=0.8)
            ax_terms.set_title('Verlustterme', fontsize=9)
            ax_terms.set_xlabel('Schritt')
            ax_terms.legend(fontsize=7)

            # ── Beta ───────────────────────────────────
            ax_beta.clear()
            if m["beta"]:
                ax_beta.plot(range(len(m["beta"])), m["beta"],
                             color='darkorange', linewidth=1.5)
                ax_beta.axhline(config["beta_max"], color='red',
                                linestyle='--', linewidth=1,
                                label=f'Max={config["beta_max"]}')
            ax_beta.set_title('KL-Annealing', fontsize=9)
            ax_beta.set_ylim(0, config["beta_max"]*1.2)
            ax_beta.legend(fontsize=7)

            # ── Rewards ────────────────────────────────
            ax_rew.clear()
            ax_rew.plot(steps_x, m["intrinsic_reward"],
                        color='gold', linewidth=1.2, alpha=0.7,
                        label='Intrinsic (Curiosity)')
            ax_rew.plot(steps_x, m["extrinsic_reward"],
                        color='lightgreen', linewidth=1.2, alpha=0.7,
                        label='Extrinsic (Env)')
            ax_rew.plot(steps_x, m["total_reward"],
                        color='white', linewidth=1.8,
                        label='Total')
            if len(m["total_reward"]) >= 20:
                ma = np.convolve(m["total_reward"], np.ones(20)/20, mode='valid')
                ax_rew.plot(range(19, len(m["total_reward"])), ma,
                            color='orange', linewidth=2, linestyle='--',
                            label='Total MA-20')
            ax_rew.set_title('Rewards (Intrinsic + Extrinsic)', fontsize=9)
            ax_rew.set_xlabel('Schritt')
            ax_rew.legend(fontsize=6, ncol=2)
            ax_rew.set_facecolor('#111111')
            ax_rew.tick_params(colors='white')

            # ── Buffer ─────────────────────────────────
            ax_buf.clear()
            buf_fill = loop.replay_buf.size
            ax_buf.barh(["Buffer"], [buf_fill],
                        color='steelblue', alpha=0.8)
            ax_buf.barh(["Buffer"], [config["buffer_size"]],
                        color='lightgray', alpha=0.3)
            ax_buf.set_xlim(0, config["buffer_size"])
            ax_buf.set_title(
                f'Replay Buffer: {buf_fill}/{config["buffer_size"]} '
                f'({100*buf_fill/config["buffer_size"]:.0f}%)',
                fontsize=9
            )
            ax_buf.tick_params(labelsize=8)

            # ── ROS2 Output ────────────────────────────
            ax_ros.clear()
            ax_ros.axis('off')
            # Inferenz für aktuelle Szene
            ros2 = loop.get_ros2_action(obs, env.current_goal)
            dist = abs(ros2["linear_x"]) * ros2["duration"] * 100
            ang  = abs(ros2["angular_z"]) * ros2["duration"] * 180/np.pi
            dir_l = "vor"  if ros2["linear_x"]  > 0.05 else ("zurueck" if ros2["linear_x"]  < -0.05 else "")
            dir_a = "links" if ros2["angular_z"] > 0.05 else ("rechts"  if ros2["angular_z"] < -0.05 else "")
            ros_lines = [
                "── ROS2 Output ─────────",
                f"Ziel: '{env.current_goal}'",
                "",
                f"twist.linear.x  = {ros2['linear_x']:+.3f}",
                f"twist.angular.z = {ros2['angular_z']:+.3f}",
                f"duration        = {ros2['duration']:.2f}s",
                "",
                "── Wirkung ─────────────",
            ]
            if dir_l: ros_lines.append(f"  {dist:.1f}cm {dir_l}waerts")
            if dir_a: ros_lines.append(f"  {ang:.1f}° {dir_a}")
            if not dir_l and not dir_a: ros_lines.append("  Stopp")
            ros_lines += [
                "",
                "── Unsicherheit ────────",
                *[f"  {k:10s}: {v:.3f}"
                  for k,v in ros2["sigma"].items()],
            ]
            ax_ros.text(0.03, 0.98, "\n".join(ros_lines),
                        transform=ax_ros.transAxes,
                        fontsize=7.5, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

            plt.pause(0.03)

    # ── Terminal-Ausgabe ──────────────────────────────────
    m = loop.metrics
    print("\nTraining abgeschlossen!")
    print(f"  Free Energy final:   {m['fe'][-1]:.5f}")
    print(f"  Reconstruction:      {m['recon'][-1]:.5f}")
    print(f"  KL:                  {m['kl'][-1]:.5f}")
    print(f"  Action Loss:         {m['action'][-1]:.5f}")
    print(f"  Intrinsic Reward:    {m['intrinsic_reward'][-1]:.5f}")
    print(f"  LR final:            {m['lr'][-1]:.2e}")
    print()

    # ROS2 Inferenz für alle Szenen
    print("ROS2 Inferenz (alle Szenen):")
    print(f"  {'Szene':15s}  {'lin_x':>7} {'ang_z':>7} {'dur':>6}")
    for scene in SCENE_TYPES:
        test_obs = draw_scene(scene)
        ros2     = loop.get_ros2_action(test_obs, SCENE_GOALS[scene])
        print(f"  {scene:15s}  "
              f"{ros2['linear_x']:+.3f}   "
              f"{ros2['angular_z']:+.3f}   "
              f"{ros2['duration']:.2f}")

    print()
    print("Naechste Schritte:")
    print("  B12 – Intrinsic Reward (Curiosity-Modul)")
    print("  B13 – Gemini API: User-Sprache → CLIP-Text")
    print("        'Geh zur roten Box' → 'find the red box'")

    plt.show()


if __name__ == "__main__":
    run_demo()
