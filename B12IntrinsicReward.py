"""
B12 – Intrinsic Reward (Curiosity-Modul)
==========================================
Berechnet einen intrinsischen Reward aus drei Quellen:

    1. Prediction Error  : Wie falsch war die Vorhersage des nächsten Frames?
                           → Hoher Fehler = unbekannte Region = neugierig!
    2. Epistemic Value   : Unsicherheit des Action Heads (sigma)
                           → Hohe Unsicherheit = explizit erkunden
    3. Novelty Score     : Wie verschieden ist dieser Zustand von bekannten?
                           → Via k-NN im Latent Space

Kombination:
    r_intrinsic = w_pred * Prediction_Error
                + w_epist * Epistemic_Value
                + w_novel * Novelty_Score

Active Inference Bedeutung:
    Epistemic Value = Erwartete Reduktion der Unsicherheit
    "Wenn ich diese Aktion ausführe, lerne ich mehr über die Welt"
    → Treibt Exploration ohne extrinsischen Reward

Kamera-Aktionen (NEU ab B12):
    Der Roboter hat eine bewegliche Kamera ("Kopf"):
        camera_pan:  [-90°, +90°]  – links/rechts
        camera_tilt: [-45°, +45°]  – oben/unten

    Arc-Movement (NEU ab B12):
        arc_radius:  [-2.0m, +2.0m]  – Kurvenradius
        → arc_radius=0    : geradeaus (wie bisher)
        → arc_radius=+1.0 : Linkskurve mit R=1m
        → arc_radius=-0.5 : scharfe Rechtskurve mit R=0.5m
        → ROS2: angular_z = linear_x / arc_radius

Aktualisiertes ACTION_DIM = 6:
    [linear_x, angular_z, camera_pan, camera_tilt, arc_radius, duration]

Hinweis: B06 und B09 werden in B16 (Vollintegration) auf ACTION_DIM=6 aktualisiert.
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
# ERWEITERTER AKTIONS-RAUM (ab B12)
# ─────────────────────────────────────────────

ACTION_DIM = 6
ACTION_BOUNDS = {
    "linear_x":    (-0.5,  0.5),    # m/s
    "angular_z":   (-1.0,  1.0),    # rad/s  (bei arc_radius=0)
    "camera_pan":  (-1.57, 1.57),   # rad  (-90° … +90°)
    "camera_tilt": (-0.79, 0.79),   # rad  (-45° … +45°)
    "arc_radius":  (-2.0,  2.0),    # m    (0=gerade, ±=Kurve)
    "duration":    (0.1,   2.0),    # s
}
ACTION_NAMES = list(ACTION_BOUNDS.keys())


def denormalize_action(action_norm: np.ndarray) -> dict:
    result = {}
    for i, (key, (lo, hi)) in enumerate(ACTION_BOUNDS.items()):
        result[key] = float((action_norm[i] + 1.0) / 2.0 * (hi - lo) + lo)
    return result


def to_ros2_twist(action_norm: np.ndarray) -> dict:
    """
    Konvertiert normierten Aktions-Vektor → ROS2-Kommandos.

    Arc-Movement:
        Wenn |arc_radius| > 0.1m:
            angular_z = linear_x / arc_radius   (Kreisbogen-Kinematik)
        Sonst:
            angular_z wie angegeben
    """
    p = denormalize_action(action_norm)
    twist = {
        "linear_x":     p["linear_x"],
        "angular_z":    p["angular_z"],
        "duration":     p["duration"],
        "camera_pan":   p["camera_pan"],
        "camera_tilt":  p["camera_tilt"],
    }
    # Arc-Override
    if abs(p["arc_radius"]) > 0.1:
        twist["angular_z"] = p["linear_x"] / p["arc_radius"]
        twist["arc_radius"] = p["arc_radius"]
        arc_deg = abs(p["linear_x"] / p["arc_radius"] * p["duration"] * 180 / np.pi)
        arc_dist = abs(p["linear_x"] * p["duration"])
        twist["arc_description"] = (
            f"Kurve {'links' if p['arc_radius']>0 else 'rechts'}: "
            f"R={abs(p['arc_radius']):.1f}m, "
            f"{arc_dist*100:.0f}cm, {arc_deg:.0f}°"
        )
    else:
        twist["arc_description"] = "geradeaus"

    twist["camera_description"] = (
        f"Pan={p['camera_pan']*180/np.pi:.0f}°  "
        f"Tilt={p['camera_tilt']*180/np.pi:.0f}°"
    )
    return twist


# ─────────────────────────────────────────────
# INTRINSIC REWARD MODUL
# ─────────────────────────────────────────────

class IntrinsicReward(nn.Module):
    """
    Berechnet intrinsischen Reward aus Prediction Error, Epistemic Value
    und Novelty Score.

    Intern wird ein kleines Forward-Model trainiert:
        (z_t, action) → z_t+1_predicted
    Der Fehler dieses Modells ist der Prediction-Error-Reward.

    Novelty:
        k-NN im Latent Space: Distanz zu den k ähnlichsten bekannten Zuständen.
        Viele ähnliche Nachbarn → bekannter Zustand → geringe Novelty.
    """

    def __init__(
            self,
            latent_dim:    int   = 64,
            action_dim:    int   = ACTION_DIM,
            memory_size:   int   = 1000,
            k_neighbors:   int   = 5,
            w_pred:        float = 1.0,
            w_epist:       float = 0.5,
            w_novel:       float = 0.5,
            pred_lr:       float = 1e-3,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.action_dim  = action_dim
        self.memory_size = memory_size
        self.k_neighbors = k_neighbors
        self.w_pred      = w_pred
        self.w_epist     = w_epist
        self.w_novel     = w_novel

        # Forward-Model: (z_t, action) → z_t+1
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(True),
            nn.Linear(64, latent_dim),
        )

        # Optimizer für das Forward-Model (separates Training)
        self.optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=pred_lr
        )

        # Episodischer Speicher für Novelty (k-NN)
        self._memory = deque(maxlen=memory_size)

    def update_forward_model(
            self,
            z_t:      torch.Tensor,    # (B, latent_dim)
            action:   torch.Tensor,    # (B, action_dim)
            z_t1:     torch.Tensor,    # (B, latent_dim) – tatsächlicher nächster Zustand
    ) -> float:
        """
        Trainiert das Forward-Model einen Schritt.
        Returns: Forward-Model Loss
        """
        self.forward_model.train()
        inp       = torch.cat([z_t, action], dim=-1)
        z_pred    = self.forward_model(inp)
        loss      = F.mse_loss(z_pred, z_t1.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def prediction_error(
            self,
            z_t:    torch.Tensor,
            action: torch.Tensor,
            z_t1:   torch.Tensor,
    ) -> torch.Tensor:
        """Prediction Error: wie falsch war die Vorhersage? (B,)"""
        self.forward_model.eval()
        with torch.no_grad():
            inp    = torch.cat([z_t, action], dim=-1)
            z_pred = self.forward_model(inp)
            error  = F.mse_loss(z_pred, z_t1, reduction='none').mean(dim=-1)
        return error

    def epistemic_value(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Epistemic Value aus Action-Head Unsicherheit.
        sigma: (B, action_dim) → mittlere Unsicherheit (B,)
        """
        return sigma.mean(dim=-1)

    def novelty_score(self, z: torch.Tensor) -> torch.Tensor:
        """
        Novelty via k-NN im Latent Space.
        Weiter entfernte Nachbarn → novelty Score.
        z: (B, latent_dim)
        """
        z_np  = z.detach().cpu().numpy()
        scores = []

        for z_i in z_np:
            if len(self._memory) < self.k_neighbors:
                # Noch wenig Daten → hohe Novelty
                scores.append(1.0)
                continue

            mem = np.stack(list(self._memory))              # (M, latent_dim)
            # Cosinus-Ähnlichkeit zu allen gespeicherten Zuständen
            z_norm   = z_i / (np.linalg.norm(z_i) + 1e-8)
            mem_norm = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)
            sims     = mem_norm @ z_norm                    # (M,)
            # k nächste Nachbarn
            top_k    = np.sort(sims)[-self.k_neighbors:]
            # Hohe Ähnlichkeit = bekannt = niedrige Novelty
            novelty  = 1.0 - float(top_k.mean())
            scores.append(np.clip(novelty, 0, 1))

        # Aktuellen Zustand speichern
        for z_i in z_np:
            self._memory.append(z_i.copy())

        return torch.tensor(scores, dtype=torch.float32)

    def forward(
            self,
            z_t:    torch.Tensor,    # (B, latent_dim)
            action: torch.Tensor,    # (B, action_dim)
            z_t1:   torch.Tensor,    # (B, latent_dim)
            sigma:  torch.Tensor,    # (B, action_dim)
    ) -> dict:
        """
        Berechnet alle Komponenten des intrinsischen Rewards.

        Returns dict mit:
            total:       (B,)  – Gesamt-Reward
            pred_error:  (B,)  – Prediction Error
            epistemic:   (B,)  – Epistemic Value
            novelty:     (B,)  – Novelty Score
        """
        pred_err  = self.prediction_error(z_t, action, z_t1)
        epistemic = self.epistemic_value(sigma)
        novelty   = self.novelty_score(z_t)

        total = (
                self.w_pred  * pred_err  +
                self.w_epist * epistemic +
                self.w_novel * novelty
        )
        return {
            "total":      total,
            "pred_error": pred_err,
            "epistemic":  epistemic,
            "novelty":    novelty,
        }

    def summary(self) -> dict:
        return {
            "latent_dim":  self.latent_dim,
            "action_dim":  self.action_dim,
            "memory_size": self.memory_size,
            "k_neighbors": self.k_neighbors,
            "weights":     f"pred={self.w_pred} epist={self.w_epist} novel={self.w_novel}",
            "params":      sum(p.numel() for p in self.forward_model.parameters()),
        }


# ─────────────────────────────────────────────
# SZENEN & MOCK (aus B08/B11)
# ─────────────────────────────────────────────

def draw_scene(scene_type: str) -> np.ndarray:
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
    return img

SCENE_TYPES = ["red_box", "blue_ball", "green_door", "corridor", "corner"]

# Typische Aktionen mit vollem ACTION_DIM=6
SCENE_ACTIONS_6D = {
    "red_box":    [ 0.6,  0.0,  0.0,  0.1,  0.0, -0.5],  # geradeaus, Kamera leicht hoch
    "blue_ball":  [ 0.4,  0.6, -0.3,  0.2,  0.0, -0.5],  # Kurve links, Kamera links
    "green_door": [ 0.8,  0.0,  0.0,  0.0,  0.0, -0.3],  # schnell geradeaus
    "corridor":   [ 1.0,  0.0,  0.0,  0.0,  0.4, -0.4],  # geradeaus, Arc leicht
    "corner":     [ 0.3,  0.8,  0.5,  0.0,  0.0, -0.6],  # Kurve + Kamera rechts
}


class MockEncoder(nn.Module):
    """Vereinfachter Encoder für B12-Demo."""
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1), nn.ReLU(True),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU(True),
            nn.Conv2d(64,128,3,stride=2,padding=1), nn.ReLU(True),
        )
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x):
        return F.normalize(self.fc(self.conv(x).reshape(x.size(0),-1)), dim=-1)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    LATENT_DIM  = 64
    N_STEPS     = 300
    BATCH_SIZE  = 8

    encoder   = MockEncoder(latent_dim=LATENT_DIM)
    curiosity = IntrinsicReward(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        memory_size=1000,
        k_neighbors=5,
        w_pred=1.0, w_epist=0.5, w_novel=0.8,
    )

    info = curiosity.summary()
    print("B12 – Intrinsic Reward (Curiosity)")
    for k, v in info.items():
        print(f"  {k:12s}: {v}")
    print()
    print(f"Erweiterter Aktions-Raum (ACTION_DIM={ACTION_DIM}):")
    for i, (name, (lo, hi)) in enumerate(ACTION_BOUNDS.items()):
        print(f"  [{i}] {name:12s}: [{lo:+.2f}, {hi:+.2f}]")
    print()

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(
        f'B12 – Intrinsic Reward (Curiosity)  |  ACTION_DIM={ACTION_DIM}',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    # Zeile 0: Szenen + Reward-Komponenten
    ax_scenes = [fig.add_subplot(gs[0, i]) for i in range(len(SCENE_TYPES))]

    # Zeile 1: Reward-Kurven
    ax_total   = fig.add_subplot(gs[1, :2])
    ax_comps   = fig.add_subplot(gs[1, 2:4])
    ax_novelty = fig.add_subplot(gs[1, 4])

    # Zeile 2: Kamera + Arc + ROS2 + Stats
    ax_camera  = fig.add_subplot(gs[2, 0:2])
    ax_arc     = fig.add_subplot(gs[2, 2:4])
    ax_stats   = fig.add_subplot(gs[2, 4])
    ax_stats.axis('off')

    # Tracking
    history = {k: [] for k in [
        "total", "pred_error", "epistemic", "novelty",
        "fwd_loss", "camera_pan", "camera_tilt", "arc_radius"
    ]}
    scene_reward_history = {s: [] for s in SCENE_TYPES}

    print(f"Starte Demo: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        # Szene auswählen (rotiert)
        scene   = SCENE_TYPES[step % len(SCENE_TYPES)]
        scene2  = SCENE_TYPES[(step + 1) % len(SCENE_TYPES)]

        # Bilder → Latent
        img1 = draw_scene(scene)
        img2 = draw_scene(scene2)

        def to_tensor(img):
            return torch.from_numpy(img).float()/255.0 \
                .permute(2,0,1) if False else \
                torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0

        with torch.no_grad():
            x1 = to_tensor(img1)
            x2 = to_tensor(img2)
            z1 = encoder(x1).expand(BATCH_SIZE, -1)
            z2 = encoder(x2).expand(BATCH_SIZE, -1)

        # Zufällige Aktionen (6D)
        base_action = np.array(SCENE_ACTIONS_6D[scene], dtype=np.float32)
        actions_np  = np.clip(
            base_action + 0.15 * np.random.randn(BATCH_SIZE, ACTION_DIM).astype(np.float32),
            -1, 1
        )
        actions = torch.from_numpy(actions_np)

        # Mock sigma (Action-Head Unsicherheit)
        sigma = torch.sigmoid(torch.randn(BATCH_SIZE, ACTION_DIM) * 0.5)

        # Forward-Model trainieren
        fwd_loss = curiosity.update_forward_model(z1, actions, z2)

        # Intrinsic Reward berechnen
        with torch.no_grad():
            rewards = curiosity(z1, actions, z2, sigma)

        # Tracking
        history["total"].append(float(rewards["total"].mean()))
        history["pred_error"].append(float(rewards["pred_error"].mean()))
        history["epistemic"].append(float(rewards["epistemic"].mean()))
        history["novelty"].append(float(rewards["novelty"].mean()))
        history["fwd_loss"].append(fwd_loss)

        # Kamera & Arc Tracking
        mean_action = actions_np.mean(axis=0)
        pan_deg   = (mean_action[2]+1)/2*(1.57-(-1.57))+(-1.57)
        tilt_deg  = (mean_action[3]+1)/2*(0.79-(-0.79))+(-0.79)
        arc_m     = (mean_action[4]+1)/2*(2.0-(-2.0))+(-2.0)
        history["camera_pan"].append(pan_deg * 180/np.pi)
        history["camera_tilt"].append(tilt_deg * 180/np.pi)
        history["arc_radius"].append(arc_m)

        scene_reward_history[scene].append(float(rewards["total"].mean()))

        if step % 15 == 0 or step == N_STEPS - 1:
            steps_x = list(range(len(history["total"])))

            # ── Szenen + Reward ────────────────────────
            for i, s in enumerate(SCENE_TYPES):
                ax_scenes[i].clear()
                img = draw_scene(s)
                ax_scenes[i].imshow(img, interpolation='nearest')
                mean_rew = np.mean(scene_reward_history[s]) \
                    if scene_reward_history[s] else 0
                last_rew = scene_reward_history[s][-1] \
                    if scene_reward_history[s] else 0
                ax_scenes[i].set_title(
                    f'{s.replace("_"," ")}\n'
                    f'r={last_rew:.3f}',
                    fontsize=7,
                    color='yellow' if s == scene else 'white'
                )
                ax_scenes[i].axis('off')
                ax_scenes[i].set_facecolor(
                    '#2a1a0a' if s == scene else '#0a0a0a'
                )
                # Aktive Szene markieren
                for spine in ax_scenes[i].spines.values():
                    spine.set_edgecolor('orange' if s==scene else 'gray')
                    spine.set_linewidth(2 if s==scene else 0.5)

            # ── Gesamt-Reward ──────────────────────────
            ax_total.clear()
            ax_total.plot(steps_x, history["total"],
                          color='gold', linewidth=1.2, alpha=0.6,
                          label='Intrinsic Total')
            if len(history["total"]) >= 20:
                ma = np.convolve(history["total"], np.ones(20)/20, mode='valid')
                ax_total.plot(range(19, len(history["total"])), ma,
                              color='orange', linewidth=2, label='MA-20')
            ax_total.set_title('Intrinsic Reward (gesamt)', fontsize=9)
            ax_total.set_xlabel('Schritt')
            ax_total.legend(fontsize=7)
            ax_total.set_facecolor('#111111')
            ax_total.tick_params(colors='white')
            ax_total.title.set_color('white')

            # ── Komponenten ────────────────────────────
            ax_comps.clear()
            comp_cfg = [
                ("pred_error", "Prediction Error", "steelblue"),
                ("epistemic",  "Epistemic Value",  "seagreen"),
                ("novelty",    "Novelty Score",    "mediumpurple"),
                ("fwd_loss",   "Forward Model Loss","tomato"),
            ]
            for key, label, color in comp_cfg:
                if history[key]:
                    ax_comps.plot(range(len(history[key])), history[key],
                                  color=color, linewidth=1.3,
                                  label=label, alpha=0.85)
            ax_comps.set_title('Reward-Komponenten', fontsize=9)
            ax_comps.set_xlabel('Schritt')
            ax_comps.legend(fontsize=6)

            # ── Novelty pro Szene ─────────────────────
            ax_novelty.clear()
            scene_means = []
            for s in SCENE_TYPES:
                vals = scene_reward_history[s]
                scene_means.append(np.mean(vals[-20:]) if len(vals)>=5 else 0)
            colors_n = ['tomato','steelblue','seagreen','gold','mediumpurple']
            bars = ax_novelty.bar(
                [s.replace("_","\n") for s in SCENE_TYPES],
                scene_means, color=colors_n, alpha=0.8
            )
            for bar, val in zip(bars, scene_means):
                ax_novelty.text(
                    bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6
                )
            ax_novelty.set_title('Ø Reward\nletzte 20 Steps\npro Szene', fontsize=8)
            ax_novelty.tick_params(axis='x', labelsize=6)

            # ── Kamera-Visualisierung ──────────────────
            ax_camera.clear()
            ax_camera.set_facecolor('#0d1b2a')

            # Kamera-Feld als Kreis + Zeiger
            cam_circle = plt.Circle((0.5, 0.5), 0.35, color='#1a3a5c',
                                    zorder=2)
            ax_camera.add_patch(cam_circle)

            if history["camera_pan"] and history["camera_tilt"]:
                pan_last  = history["camera_pan"][-1] * np.pi/180
                tilt_last = history["camera_tilt"][-1] * np.pi/180
                # Projektion: pan=horizontal, tilt=vertikal
                px = 0.5 + 0.3 * np.sin(pan_last)
                py = 0.5 + 0.3 * np.sin(tilt_last)
                ax_camera.plot([0.5, px], [0.5, py],
                               color='cyan', linewidth=3, zorder=4)
                ax_camera.scatter([px], [py], color='cyan', s=80, zorder=5)

            # Gitter
            for a in np.linspace(-90, 90, 7):
                r = a * np.pi/180
                x2l = 0.5 + 0.38*np.sin(r)
                y2l = 0.5 + 0.38*np.cos(r)*0.2
                ax_camera.plot([0.5, x2l], [0.5, y2l],
                               color='gray', linewidth=0.5, alpha=0.4)

            pan_now  = history["camera_pan"][-1]  if history["camera_pan"]  else 0
            tilt_now = history["camera_tilt"][-1] if history["camera_tilt"] else 0
            ax_camera.set_title(
                f'Kamera-Kopf\nPan={pan_now:+.0f}°  Tilt={tilt_now:+.0f}°',
                fontsize=9, color='white'
            )
            ax_camera.set_xlim(0,1); ax_camera.set_ylim(0,1)
            ax_camera.set_aspect('equal'); ax_camera.axis('off')

            # Kamera-Verlauf
            if len(history["camera_pan"]) > 1:
                ax2 = ax_camera.inset_axes([0.0, -0.55, 1.0, 0.45])
                ax2.plot(range(len(history["camera_pan"])),
                         history["camera_pan"],
                         color='cyan', linewidth=1.2, label='Pan°')
                ax2.plot(range(len(history["camera_tilt"])),
                         history["camera_tilt"],
                         color='lightblue', linewidth=1.2,
                         linestyle='--', label='Tilt°')
                ax2.axhline(0, color='gray', linewidth=0.5)
                ax2.set_ylim(-100, 100)
                ax2.legend(fontsize=6)
                ax2.set_facecolor('#0d1b2a')
                ax2.tick_params(colors='white', labelsize=6)

            # ── Arc-Movement Visualisierung ────────────
            ax_arc.clear()
            ax_arc.set_facecolor('#0d1b2a')

            if history["arc_radius"]:
                arc_now = history["arc_radius"][-1]
                # Roboter in der Mitte
                robot = plt.Circle((0.5, 0.3), 0.06,
                                   color='steelblue', zorder=4)
                ax_arc.add_patch(robot)

                if abs(arc_now) > 0.1:
                    # Kreisbogen zeichnen
                    r_norm = np.clip(arc_now / 2.0, -1, 1)
                    cx_arc = 0.5 + r_norm * 0.3
                    theta_range = np.linspace(-np.pi/2, np.pi/6, 40)
                    arc_x = cx_arc + abs(r_norm*0.3) * np.cos(theta_range)
                    arc_y = 0.3    + abs(r_norm*0.3) * np.sin(theta_range)
                    ax_arc.plot(arc_x, arc_y, color='orange',
                                linewidth=2.5, zorder=3)
                    ax_arc.scatter([arc_x[-1]], [arc_y[-1]],
                                   color='orange', s=60, zorder=5)
                    label = (f"Arc R={arc_now:+.2f}m\n"
                             f"{'Linkskurve' if arc_now>0 else 'Rechtskurve'}")
                else:
                    # Gerade
                    ax_arc.annotate(
                        "", xy=(0.5, 0.75), xytext=(0.5, 0.3),
                        arrowprops=dict(arrowstyle='->', color='lime',
                                        lw=2.5, mutation_scale=15)
                    )
                    label = "Geradeaus"

                ax_arc.set_title(
                    f'Arc-Movement\n{label}',
                    fontsize=9, color='white'
                )

                # Arc-Radius Verlauf
                ax3 = ax_arc.inset_axes([0.0, -0.55, 1.0, 0.45])
                ax3.plot(range(len(history["arc_radius"])),
                         history["arc_radius"],
                         color='orange', linewidth=1.2, label='Arc R [m]')
                ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--')
                ax3.fill_between(range(len(history["arc_radius"])),
                                 history["arc_radius"], 0,
                                 alpha=0.3, color='orange')
                ax3.set_ylim(-2.2, 2.2)
                ax3.legend(fontsize=6)
                ax3.set_facecolor('#0d1b2a')
                ax3.tick_params(colors='white', labelsize=6)

            ax_arc.set_xlim(0,1); ax_arc.set_ylim(0,1)
            ax_arc.set_aspect('equal'); ax_arc.axis('off')

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            total_now = history["total"][-1]  if history["total"]  else 0
            pred_now  = history["pred_error"][-1] if history["pred_error"] else 0
            epist_now = history["epistemic"][-1]  if history["epistemic"]  else 0
            novel_now = history["novelty"][-1]    if history["novelty"]    else 0
            fwd_now   = history["fwd_loss"][-1]   if history["fwd_loss"]   else 0

            lines = [
                "── Intrinsic Reward ──",
                f"Total:    {total_now:.4f}",
                f"Pred Err: {pred_now:.4f}",
                f"Epistemic:{epist_now:.4f}",
                f"Novelty:  {novel_now:.4f}",
                f"FwdLoss:  {fwd_now:.4f}",
                "",
                "── Gewichte ──────────",
                f"w_pred:   {curiosity.w_pred}",
                f"w_epist:  {curiosity.w_epist}",
                f"w_novel:  {curiosity.w_novel}",
                "",
                "── Kamera (NEU) ──────",
                f"Pan:  {history['camera_pan'][-1] if history['camera_pan'] else 0:+.0f}°",
                f"Tilt: {history['camera_tilt'][-1] if history['camera_tilt'] else 0:+.0f}°",
                "",
                "── Arc (NEU) ─────────",
                f"Radius: {history['arc_radius'][-1] if history['arc_radius'] else 0:+.2f}m",
                "",
                "── ACTION_DIM=6 ──────",
                "0: linear_x",
                "1: angular_z",
                "2: camera_pan",
                "3: camera_tilt",
                "4: arc_radius",
                "5: duration",
                "",
                f"Memory: {len(curiosity._memory)}/{curiosity.memory_size}",
                f"Step:   {step+1}/{N_STEPS}",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.03)

    # ── Terminal-Ausgabe ──────────────────────────────────
    print("\nDemo abgeschlossen!")
    print(f"  Intrinsic Reward final: {history['total'][-1]:.4f}")
    print(f"  Forward Model Loss:     {history['fwd_loss'][-1]:.4f}")
    print(f"  Memory gefüllt:         {len(curiosity._memory)}/{curiosity.memory_size}")
    print()
    print("ROS2 Beispiel-Befehle mit vollem ACTION_DIM=6:")
    for scene, act_6d in SCENE_ACTIONS_6D.items():
        act_np = np.array(act_6d, dtype=np.float32)
        twist  = to_ros2_twist(act_np)
        print(f"\n  [{scene}]  Ziel: '{twist.get('arc_description','')}'")
        print(f"    twist.linear.x  = {twist['linear_x']:+.3f} m/s")
        print(f"    twist.angular.z = {twist['angular_z']:+.3f} rad/s")
        print(f"    camera.pan      = {twist['camera_pan']*180/np.pi:+.0f}°")
        print(f"    camera.tilt     = {twist['camera_tilt']*180/np.pi:+.0f}°")
        print(f"    duration        = {twist['duration']:.2f}s")
        print(f"    → {twist['arc_description']}  |  Kamera: {twist['camera_description']}")

    print()
    print("Naechste Schritte:")
    print("  B13 – Gemini API: 'Geh zur roten Box' → CLIP-Text")
    print("  B06/B09 Update: ACTION_DIM 3→6 (in B16 Vollintegration)")

    plt.show()


if __name__ == "__main__":
    run_demo()
