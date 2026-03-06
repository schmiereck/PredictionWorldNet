"""
B09 – Action Head Demo
=======================
Berechnet aus dem Kontext-Vektor (B07) die nächste ROS2-Aktion.

Output: [linear_x, angular_z, duration] normalisiert auf [-1, 1]
→ wird direkt als ROS2 geometry_msgs/Twist verwendet:
    twist.linear.x  = linear_x  * MAX_LINEAR   (max 0.5 m/s)
    twist.angular.z = angular_z * MAX_ANGULAR  (max 1.0 rad/s)
    duration        = (duration + 1) / 2 * 1.9 + 0.1  (0.1 – 2.0 s)

Architektur:
    Input:  context (B, d_model=128)  ← Temporal Transformer B07
    FC1:    (B, 256) + LayerNorm + ReLU + Dropout
    FC2:    (B, 128) + LayerNorm + ReLU + Dropout
    FC3:    (B, action_dim=3)
    Output: tanh → [-1, 1]

Warum tanh am Ausgang?
    tanh begrenzt den Output auf [-1, 1] – passend zur normierten
    ROS2-Aktion. Im Gegensatz zu Sigmoid (nur [0,1]) können damit
    auch negative Aktionen (rückwärts, rechts drehen) kodiert werden.

Zusatz: Uncertainty Head
    Neben der Aktion selbst schätzt der Action Head auch die
    Unsicherheit (sigma) für jede Aktionsdimension:
    → Hohe Unsicherheit bei linear_x → Agent ist unsicher wie weit er fahren soll
    → Kann als Explorations-Signal verwendet werden (Active Inference!)
    → Gibt dem Operator Feedback über Konfidenz der Entscheidung

Active Inference Bedeutung:
    Aktion = Teil des Markov Blankets (active boundary)
    Der Agent wählt die Aktion die seine Free Energy minimiert:
    "Welche Aktion führt zu dem Bild das ich erwarte?"
    → context enthält bereits Ziel-Information (aus B05/B07)
    → Action Head lernt: "Wie bewege ich mich zum Ziel?"
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
# ROS2 AKTION (aus B06)
# ─────────────────────────────────────────────

ACTION_DIM = 3
ACTION_BOUNDS = {
    "linear_x":  (-0.5,  0.5),   # m/s
    "angular_z": (-1.0,  1.0),   # rad/s
    "duration":  ( 0.1,  2.0),   # s
}
ACTION_NAMES = list(ACTION_BOUNDS.keys())
MAX_VALUES   = [0.5, 1.0, 2.0]


def denormalize_action(action_norm: np.ndarray) -> dict:
    """[-1,1] → physikalische Einheiten für ROS2."""
    result = {}
    for i, (key, (lo, hi)) in enumerate(ACTION_BOUNDS.items()):
        result[key] = float((action_norm[i] + 1.0) / 2.0 * (hi - lo) + lo)
    return result


def format_ros2_command(action_norm: np.ndarray) -> list:
    """Formatiert die Aktion als ROS2-Befehl für die Anzeige."""
    p = denormalize_action(action_norm)
    dist_cm = abs(p["linear_x"]) * p["duration"] * 100
    angle_deg = abs(p["angular_z"]) * p["duration"] * 180 / np.pi

    lines = [
        "── ROS2 Twist ───────────",
        f"linear.x  = {p['linear_x']:+.3f} m/s",
        f"angular.z = {p['angular_z']:+.3f} rad/s",
        f"duration  = {p['duration']:.2f} s",
        "",
        "── Wirkung ──────────────",
    ]
    if abs(p["linear_x"]) > 0.05:
        direction = "vorwaerts" if p["linear_x"] > 0 else "rueckwaerts"
        lines.append(f"Fahre {dist_cm:.1f}cm {direction}")
    if abs(p["angular_z"]) > 0.05:
        direction = "links" if p["angular_z"] > 0 else "rechts"
        lines.append(f"Drehe {angle_deg:.1f}° {direction}")
    if abs(p["linear_x"]) <= 0.05 and abs(p["angular_z"]) <= 0.05:
        lines.append("Stopp")

    return lines


# ─────────────────────────────────────────────
# ACTION HEAD
# ─────────────────────────────────────────────

class ActionHead(nn.Module):
    """
    Kontext-Vektor → ROS2-Aktion + Unsicherheit.

    Input:  (B, d_model)
    Output:
        action : (B, action_dim)  – tanh normalisiert [-1, 1]
        sigma  : (B, action_dim)  – Unsicherheit [0, 1] pro Dimension

    Zwei Köpfe:
        action_head : FC → tanh   → Aktion
        sigma_head  : FC → sigmoid → Unsicherheit (0=sicher, 1=unsicher)

    Dropout nur im Training → deterministisch im Eval-Modus.
    """

    def __init__(
            self,
            d_model:    int   = 128,
            action_dim: int   = ACTION_DIM,
            hidden_dim: int   = 256,
            dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model    = d_model
        self.action_dim = action_dim

        # Geteilter Backbone
        self.backbone = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Aktions-Kopf: tanh → [-1, 1]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

        # Unsicherheits-Kopf: sigmoid → [0, 1]
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor):
        """
        Args:
            context: (B, d_model)
        Returns:
            action : (B, action_dim) in [-1, 1]
            sigma  : (B, action_dim) in [0, 1]
        """
        features = self.backbone(context)
        action   = self.action_head(features)
        sigma    = self.sigma_head(features)
        return action, sigma

    def predict_numpy(self, context_np: np.ndarray):
        """numpy (d_model,) → (action, sigma) als numpy"""
        self.eval()
        with torch.no_grad():
            ctx = torch.from_numpy(context_np).float().unsqueeze(0)
            action, sigma = self.forward(ctx)
            return action.squeeze(0).numpy(), sigma.squeeze(0).numpy()

    def summary(self) -> dict:
        return {
            "d_model":    self.d_model,
            "action_dim": self.action_dim,
            "params":     sum(p.numel() for p in self.parameters()),
            "output":     "action [-1,1] + sigma [0,1]",
        }


# ─────────────────────────────────────────────
# MOCK: ZIEL-ABHÄNGIGE AKTION
# ─────────────────────────────────────────────

# Für die Demo: verschiedene Ziele → verschiedene Ground-Truth Aktionen
GOAL_SCENARIOS = {
    "Vorwaerts fahren":    np.array([ 0.8,  0.0,  0.0], dtype=np.float32),
    "Links abbiegen":      np.array([ 0.4,  0.8,  0.0], dtype=np.float32),
    "Rechts abbiegen":     np.array([ 0.4, -0.8,  0.0], dtype=np.float32),
    "Zurueck fahren":      np.array([-0.6,  0.0,  0.2], dtype=np.float32),
    "Kurze Drehung links": np.array([ 0.0,  1.0, -0.5], dtype=np.float32),
    "Stopp":               np.array([ 0.0,  0.0, -1.0], dtype=np.float32),
}


def make_mock_context(goal_idx: int, d_model: int, noise: float = 0.3) -> torch.Tensor:
    """
    Simuliert einen Kontext-Vektor der das Ziel kodiert.
    Gleiche Ziel-Indizes → ähnliche Kontexte.
    """
    torch.manual_seed(goal_idx * 100)
    base    = torch.randn(d_model)
    context = base + noise * torch.randn(d_model)
    return F.normalize(context, dim=0) * np.sqrt(d_model)


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def draw_robot_action(ax, action_norm: np.ndarray, sigma: np.ndarray,
                      title: str = ""):
    """Visualisiert eine ROS2-Aktion als Pfeil + Balkendiagramm."""
    ax.clear()

    lin_x  = float(action_norm[0])
    ang_z  = float(action_norm[1])
    dur    = float(action_norm[2])
    sig_l  = float(sigma[0])
    sig_a  = float(sigma[1])

    # Roboter als Kreis mit Richtungspfeil
    robot = plt.Circle((0.5, 0.5), 0.15, color='steelblue', zorder=3)
    ax.add_patch(robot)

    # Vorwärts-Pfeil
    arrow_len = lin_x * 0.35
    ax.annotate("", xy=(0.5, 0.5 + arrow_len),
                xytext=(0.5, 0.5),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=2.5, mutation_scale=15))

    # Rotations-Bogen
    if abs(ang_z) > 0.05:
        theta = np.linspace(0, ang_z * np.pi * 0.6, 30)
        arc_x = 0.5 + 0.22 * np.sin(theta)
        arc_y = 0.5 + 0.22 * np.cos(theta)
        ax.plot(arc_x, arc_y, color='orange', linewidth=2.5, zorder=4)
        ax.annotate("", xy=(arc_x[-1], arc_y[-1]),
                    xytext=(arc_x[-2], arc_y[-2]),
                    arrowprops=dict(arrowstyle='->', color='orange',
                                    lw=1.5, mutation_scale=12))

    # Unsicherheits-Ellipse
    ell = plt.matplotlib.patches.Ellipse(
        (0.5, 0.5),
        width=0.3 + sig_l * 0.4,
        height=0.3 + sig_a * 0.4,
        color='lightblue', alpha=0.3, zorder=2
    )
    ax.add_patch(ell)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')
    p = denormalize_action(action_norm)
    ax.set_title(
        f"{title}\n"
        f"lx={p['linear_x']:+.2f}m/s  "
        f"az={p['angular_z']:+.2f}rad/s  "
        f"t={p['duration']:.1f}s",
        fontsize=7
    )


def run_demo():
    D_MODEL    = 128
    ACTION_DIM = 3
    N_STEPS    = 400
    LR         = 1e-3

    action_head = ActionHead(d_model=D_MODEL, action_dim=ACTION_DIM)
    optimizer   = torch.optim.AdamW(
        action_head.parameters(), lr=LR, weight_decay=1e-3
    )
    # LR-Scheduler: reduziert LR wenn Loss plateauiert
    # → verhindert dass der Optimizer nach gutem Training wieder destabilisiert
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5
    )

    info = action_head.summary()
    print("Action Head initialisiert:")
    for k, v in info.items():
        print(f"  {k:12s}: {v}")
    print()

    goals       = list(GOAL_SCENARIOS.keys())
    gt_actions  = list(GOAL_SCENARIOS.values())
    n_goals     = len(goals)

    loss_history   = []
    sigma_history  = []   # mittlere Unsicherheit über Zeit

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B09 – Action Head: Context → ROS2 Aktion',
                 fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, n_goals, figure=fig, hspace=0.6, wspace=0.4)

    ax_robots = [fig.add_subplot(gs[0, i]) for i in range(n_goals)]

    gs2 = gridspec.GridSpec(3, 4, figure=fig, hspace=0.6, wspace=0.4)
    ax_loss   = fig.add_subplot(gs2[1, :2])
    ax_sigma  = fig.add_subplot(gs2[1, 2:])
    ax_bars   = fig.add_subplot(gs2[2, :2])
    ax_ros    = fig.add_subplot(gs2[2, 2:])
    ax_ros.axis('off')

    # gs und gs2 überlappen sich – gs nur für Zeile 0 (Roboter)
    # gs2 für Zeilen 1 und 2 (Plots)

    print(f"Starte Training: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        action_head.train()

        # Alle Ziele im Batch trainieren
        contexts   = torch.stack([
            make_mock_context(i, D_MODEL) for i in range(n_goals)
        ])                                              # (n_goals, D_MODEL)
        gt_tensor  = torch.tensor(np.stack(gt_actions))  # (n_goals, 3)

        pred_actions, pred_sigma = action_head(contexts)

        # Haupt-Loss: MSE zwischen vorhergesagter und Ground-Truth Aktion
        loss_action = F.mse_loss(pred_actions, gt_tensor)

        # Unsicherheits-Loss: NLL für kalibrierte Unsicherheit
        # loss = log(sigma) + |error| / sigma
        with torch.no_grad():
            errors   = (pred_actions.detach() - gt_tensor).abs()

        sigma_safe = torch.clamp(pred_sigma, min=1e-4)
        loss_sigma = torch.mean(torch.log(sigma_safe) + errors / sigma_safe)

        loss = loss_action + 0.05 * loss_sigma

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(action_head.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss_action.detach())

        loss_history.append(float(loss.detach()))
        sigma_history.append(float(pred_sigma.detach().mean()))

        if step % 20 == 0 or step == N_STEPS - 1:
            action_head.eval()
            steps_x = list(range(len(loss_history)))

            # Alle Ziele evaluieren
            all_actions = []
            all_sigmas  = []
            for i in range(n_goals):
                ctx_np  = make_mock_context(i, D_MODEL).numpy()
                act, sig = action_head.predict_numpy(ctx_np)
                all_actions.append(act)
                all_sigmas.append(sig)

            # ── Roboter-Visualisierung ──────────────────
            for i in range(n_goals):
                draw_robot_action(
                    ax_robots[i],
                    all_actions[i], all_sigmas[i],
                    title=goals[i]
                )

            # ── Loss ───────────────────────────────────
            ax_loss.clear()
            ax_loss.plot(steps_x, loss_history,
                         color='steelblue', linewidth=1.5)
            if len(loss_history) >= 20:
                ma = np.convolve(loss_history, np.ones(20)/20, mode='valid')
                ax_loss.plot(range(19, len(loss_history)), ma,
                             color='darkblue', linewidth=2, label='MA-20')
                ax_loss.legend(fontsize=7)
            ax_loss.set_title('Training Loss (Action MSE + Sigma)', fontsize=9)
            ax_loss.set_xlabel('Schritt')

            # ── Sigma ──────────────────────────────────
            ax_sigma.clear()
            ax_sigma.plot(steps_x, sigma_history,
                          color='darkorange', linewidth=1.5)
            ax_sigma.set_title('Mittlere Unsicherheit (sigma) über Zeit', fontsize=9)
            ax_sigma.set_xlabel('Schritt')
            ax_sigma.set_ylim(0, 1)
            ax_sigma.axhline(0.5, color='gray', linestyle='--',
                             linewidth=0.8, label='Mitte')
            ax_sigma.legend(fontsize=7)

            # ── Balken: Pred vs GT ──────────────────────
            ax_bars.clear()
            x_pos    = np.arange(n_goals)
            width    = 0.12
            dim_colors = ['steelblue', 'seagreen', 'tomato']

            for dim in range(ACTION_DIM):
                pred_vals = [all_actions[i][dim] for i in range(n_goals)]
                gt_vals   = [gt_actions[i][dim]   for i in range(n_goals)]
                sigma_vals = [all_sigmas[i][dim]  for i in range(n_goals)]

                offset = (dim - 1) * width * 2.5
                bars = ax_bars.bar(x_pos + offset - width/2, pred_vals,
                                   width, color=dim_colors[dim],
                                   alpha=0.8, label=ACTION_NAMES[dim])
                ax_bars.bar(x_pos + offset + width/2, gt_vals,
                            width, color=dim_colors[dim],
                            alpha=0.3, hatch='//')

                # Unsicherheits-Fehlerbalken
                ax_bars.errorbar(x_pos + offset - width/2, pred_vals,
                                 yerr=sigma_vals,
                                 fmt='none', color='black',
                                 capsize=3, linewidth=1)

            ax_bars.set_xticks(x_pos)
            ax_bars.set_xticklabels(
                [g.replace(" ", "\n") for g in goals],
                fontsize=6
            )
            ax_bars.axhline(0, color='black', linewidth=0.8)
            ax_bars.set_ylim(-1.3, 1.3)
            ax_bars.set_title(
                'Vorhergesagte (solid) vs. GT-Aktionen (schraffiert)\n'
                'Fehlerbalken = Unsicherheit sigma',
                fontsize=8
            )
            ax_bars.legend(fontsize=7, loc='upper right')

            # ── ROS2 Ausgabe für bestes Ziel ───────────
            ax_ros.clear()
            ax_ros.axis('off')
            # Zeige ROS2-Befehl für "Vorwaerts fahren"
            act_best = all_actions[0]
            sig_best = all_sigmas[0]
            ros_lines = format_ros2_command(act_best)
            ros_lines += [
                "",
                "── Unsicherheit ─────────",
                *[f"{ACTION_NAMES[d]:10s}: sigma={sig_best[d]:.3f}  "
                  f"{'[sicher]' if sig_best[d] < 0.3 else '[unsicher]'}"
                  for d in range(ACTION_DIM)],
                "",
                "── B09 Summary ──────────",
                f"d_model:    {D_MODEL}",
                f"action_dim: {ACTION_DIM}",
                f"params:     {info['params']:,}",
                f"Loss:       {loss_history[-1]:.4f}",
                f"LR:         {optimizer.param_groups[0]['lr']:.2e}",
                f"Step:       {step+1}/{N_STEPS}",
            ]
            ax_ros.text(
                0.03, 0.98, "\n".join(ros_lines),
                transform=ax_ros.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
            )

            plt.pause(0.05)

    # Terminal-Ausgabe
    print("\nTraining abgeschlossen!")
    print(f"  Loss final: {loss_history[-1]:.4f}\n")
    print("Finale Aktionen vs. Ground-Truth:")
    print(f"  {'Ziel':22s}  {'lin_x':>6} {'ang_z':>6} {'dur':>6}  "
          f"{'GT_lx':>6} {'GT_az':>6} {'GT_d':>6}")
    for i, goal in enumerate(goals):
        act_np = make_mock_context(i, D_MODEL).numpy()
        act, sig = action_head.predict_numpy(act_np)
        gt  = gt_actions[i]
        print(f"  {goal:22s}  "
              f"{act[0]:+.3f} {act[1]:+.3f} {act[2]:+.3f}  "
              f"{gt[0]:+.3f} {gt[1]:+.3f} {gt[2]:+.3f}")

    plt.show()
    print("\nDemo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
