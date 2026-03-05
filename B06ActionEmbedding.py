"""
B06 – Action Embedding Demo (v2)
=================================
Kontinuierliche ROS2-Aktionen statt diskreter Indizes.

ROS2 Twist-Aktion:
    linear_x  : Vorwärts-Geschwindigkeit  [m/s]  z.B. 0.0 – 0.5
    angular_z : Drehgeschwindigkeit       [rad/s] z.B. -1.0 – +1.0
    duration  : Ausführungsdauer          [s]     z.B. 0.1 – 2.0

Warum kontinuierlich?
    "Bewege dich 15cm geradeaus" = linear_x=0.3, angular_z=0.0, duration=0.5s
    → Präzise, physikalisch sinnvoll, direkt als ROS2 Twist-Message verwendbar
    → Kein "starte Vorwärts" + "stoppe Vorwärts" mehr nötig

Architektur-Änderung gegenüber v1:
    Diskret:        nn.Embedding(n_actions, embed_dim)  → Lookup-Tabelle
    Kontinuierlich: nn.Linear(action_dim, embed_dim)    → Projektion

    Input:  (B, action_dim)   – Float-Tensor [linear_x, angular_z, duration]
    Output: (B, embed_dim)    – Embedding-Vektor

Mode Collapse Fix (v1 hatte alle Embeddings = +1.0):
    Problem: Loss nur zwischen konsekutiven Aktionen → alles kollabiert
    Lösung:  Triplet Loss + explizite Abstoßung zwischen unähnlichen Aktionen

ROS2 Integration (Ausblick B11):
    Das Action Head (B09) gibt [linear_x, angular_z, duration] aus
    → wird direkt als geometry_msgs/Twist + duration publiziert:
        twist.linear.x  = action[0] * MAX_LINEAR
        twist.angular.z = action[1] * MAX_ANGULAR
        rospy.sleep(action[2] * MAX_DURATION)
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
# ROS2 AKTION DEFINITION
# ─────────────────────────────────────────────

ACTION_DIM = 3   # [linear_x, angular_z, duration]

ACTION_BOUNDS = {
    "linear_x":  (-0.5,  0.5),   # m/s    (negativ = rückwärts)
    "angular_z": (-1.0,  1.0),   # rad/s  (negativ = rechts)
    "duration":  ( 0.1,  2.0),   # s
}

NAMED_ACTIONS = {
    "vorwaerts_kurz":  [ 0.3,  0.0,  0.5],
    "vorwaerts_lang":  [ 0.3,  0.0,  1.5],
    "links_drehen":    [ 0.0,  0.8,  0.5],
    "rechts_drehen":   [ 0.0, -0.8,  0.5],
    "kurve_links":     [ 0.2,  0.4,  0.8],
    "kurve_rechts":    [ 0.2, -0.4,  0.8],
    "rueckwaerts":     [-0.2,  0.0,  0.5],
    "stopp":           [ 0.0,  0.0,  0.1],
}


def normalize_action(action_raw: list) -> np.ndarray:
    """Normalisiert eine Aktion auf [-1, 1]."""
    bounds = list(ACTION_BOUNDS.values())
    norm   = []
    for val, (lo, hi) in zip(action_raw, bounds):
        norm.append(2.0 * (val - lo) / (hi - lo) - 1.0)
    return np.array(norm, dtype=np.float32)


def denormalize_action(action_norm: np.ndarray) -> dict:
    """Konvertiert normierten Vektor zurück in physikalische Einheiten."""
    bounds = list(ACTION_BOUNDS.values())
    keys   = list(ACTION_BOUNDS.keys())
    return {key: float((action_norm[i] + 1.0) / 2.0 * (hi - lo) + lo)
            for i, (key, (lo, hi)) in enumerate(zip(keys, bounds))}


# ─────────────────────────────────────────────
# KONTINUIERLICHES ACTION EMBEDDING
# ─────────────────────────────────────────────

class ContinuousActionEmbedding(nn.Module):
    """
    Kontinuierliche Aktion → Embedding-Vektor.

    Input:  (B, action_dim)  – normalisierter Aktions-Vektor [-1, 1]
    Output: (B, embed_dim)
    """

    def __init__(self, action_dim: int = ACTION_DIM, embed_dim: int = 32,
                 hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim  = embed_dim

        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)

    def embed_numpy(self, action_raw: list) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            norm = normalize_action(action_raw)
            x    = torch.from_numpy(norm).unsqueeze(0)
            return self.forward(x).squeeze(0).numpy()

    def embed_all_named(self) -> dict:
        return {name: self.embed_numpy(action)
                for name, action in NAMED_ACTIONS.items()}

    def summary(self) -> dict:
        return {
            "action_dim": self.action_dim,
            "embed_dim":  self.embed_dim,
            "params":     sum(p.numel() for p in self.parameters()),
            "type":       "continuous (ROS2 Twist)",
        }


# ─────────────────────────────────────────────
# TRAINING MIT TRIPLET LOSS
# ─────────────────────────────────────────────

def action_similarity_gt(a1: np.ndarray, a2: np.ndarray) -> float:
    """Ground-Truth Ähnlichkeit: exp(-||a1-a2||)"""
    return float(np.exp(-np.linalg.norm(a1 - a2)))


def triplet_loss(anchor, positive, negative, margin: float = 0.3):
    d_pos = 1.0 - F.cosine_similarity(anchor, positive)
    d_neg = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()


def run_training(embedding: ContinuousActionEmbedding, n_steps: int = 500):
    optimizer    = torch.optim.Adam(embedding.parameters(), lr=1e-3)
    loss_history = []

    action_list = [normalize_action(a) for a in NAMED_ACTIONS.values()]
    n           = len(action_list)

    sim_matrix = np.array([
        [action_similarity_gt(action_list[i], action_list[j]) for j in range(n)]
        for i in range(n)
    ])

    for step in range(n_steps):
        anchor_idx = np.random.randint(n)
        sims       = [(sim_matrix[anchor_idx][i], i)
                      for i in range(n) if i != anchor_idx]
        pos_idx    = max(sims, key=lambda x: x[0])[1]
        neg_idx    = min(sims, key=lambda x: x[0])[1]

        a_t = torch.tensor(action_list[anchor_idx]).unsqueeze(0)
        p_t = torch.tensor(action_list[pos_idx]).unsqueeze(0)
        n_t = torch.tensor(action_list[neg_idx]).unsqueeze(0)

        loss = triplet_loss(embedding(a_t), embedding(p_t), embedding(n_t))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))

    return loss_history


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    EMBED_DIM = 32
    N_TRAIN   = 500

    embedding = ContinuousActionEmbedding(action_dim=ACTION_DIM, embed_dim=EMBED_DIM)
    info      = embedding.summary()

    print("Continuous Action Embedding initialisiert:")
    for k, v in info.items():
        print(f"  {k:12s}: {v}")
    print()

    print("Normalisierte Aktionen [-1, 1]:")
    print(f"  {'Name':20s}  {'lin_x':>6}  {'ang_z':>6}  {'dur':>6}")
    for name, raw in NAMED_ACTIONS.items():
        n = normalize_action(raw)
        print(f"  {name:20s}  {n[0]:+.3f}  {n[1]:+.3f}  {n[2]:+.3f}")

    print(f"\nTrainiere {N_TRAIN} Schritte mit Triplet Loss...")
    loss_history = run_training(embedding, n_steps=N_TRAIN)
    print(f"Training abgeschlossen. Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}\n")

    embeddings_dict = embedding.embed_all_named()
    names    = list(embeddings_dict.keys())
    emb_mat  = np.stack(list(embeddings_dict.values()))
    emb_norm = emb_mat / (np.linalg.norm(emb_mat, axis=1, keepdims=True) + 1e-8)
    sim_mat  = emb_norm @ emb_norm.T

    action_list = [normalize_action(a) for a in NAMED_ACTIONS.values()]
    n = len(action_list)
    gt_sim = np.array([
        [action_similarity_gt(action_list[i], action_list[j]) for j in range(n)]
        for i in range(n)
    ])

    print("Cosinus-Aehnlichkeit (gelernt) vs. Ground-Truth:")
    print(f"  {'Paar':37s}  {'Gelernt':>8}  {'GT':>8}")
    for i in range(n):
        for j in range(i+1, n):
            print(f"  {names[i]:17s} <-> {names[j]:17s}  "
                  f"{sim_mat[i,j]:+.4f}  {gt_sim[i,j]:+.4f}")

    # ── Matplotlib ────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10))
    fig.suptitle('B06 – Continuous Action Embedding (ROS2 Twist)', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45)

    ax_gt     = fig.add_subplot(gs[0, 0])
    ax_sim    = fig.add_subplot(gs[0, 1])
    ax_heat   = fig.add_subplot(gs[0, 2])
    ax_stats  = fig.add_subplot(gs[0, 3])
    ax_loss   = fig.add_subplot(gs[1, 0])
    ax_action = fig.add_subplot(gs[1, 1])
    ax_ros    = fig.add_subplot(gs[1, 2:])
    ax_stats.axis('off')
    ax_ros.axis('off')

    short_names = [name.replace("_", "\n") for name in names]

    # GT Ähnlichkeit
    im1 = ax_gt.imshow(gt_sim, cmap='YlOrRd', vmin=0, vmax=1, interpolation='nearest')
    ax_gt.set_xticks(range(n))
    ax_gt.set_yticks(range(n))
    ax_gt.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax_gt.set_yticklabels(short_names, fontsize=6)
    ax_gt.set_title('Ground-Truth\nAehnlichkeit', fontsize=9)
    fig.colorbar(im1, ax=ax_gt, fraction=0.046)

    # Gelernte Ähnlichkeit
    im2 = ax_sim.imshow(sim_mat, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
    ax_sim.set_xticks(range(n))
    ax_sim.set_yticks(range(n))
    ax_sim.set_xticklabels(short_names, rotation=45, ha='right', fontsize=6)
    ax_sim.set_yticklabels(short_names, fontsize=6)
    ax_sim.set_title('Gelernte Cosinus-\nAehnlichkeit', fontsize=9)
    fig.colorbar(im2, ax=ax_sim, fraction=0.046)

    # Embedding Heatmap
    im3 = ax_heat.imshow(emb_mat, cmap='coolwarm', aspect='auto',
                         vmin=-2, vmax=2, interpolation='nearest')
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(short_names, fontsize=6)
    ax_heat.set_xlabel(f'Dim ({EMBED_DIM})', fontsize=8)
    ax_heat.set_title('Embedding-Vektoren\nHeatmap', fontsize=9)
    fig.colorbar(im3, ax=ax_heat, fraction=0.046)

    # Statistiken
    lines = [
        "── Continuous Action Emb. ──",
        f"action_dim:  {ACTION_DIM}",
        f"embed_dim:   {EMBED_DIM}",
        f"params:      {info['params']}",
        "",
        "── Aktions-Dimensionen ─────",
        "linear_x:  m/s   [-0.5, 0.5]",
        "angular_z: rad/s [-1.0, 1.0]",
        "duration:  s     [ 0.1, 2.0]",
        "",
        "── Training ────────────────",
        "Methode: Triplet Loss",
        f"Schritte: {N_TRAIN}",
        f"Start:   {loss_history[0]:.4f}",
        f"Ende:    {loss_history[-1]:.4f}",
        "",
        "── Mode Collapse Fix ───────",
        "v1: Cosine Loss → Collapse",
        "v2: Triplet Loss → OK",
    ]
    ax_stats.text(0.03, 0.98, "\n".join(lines),
                  transform=ax_stats.transAxes, fontsize=8,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Loss
    ax_loss.plot(loss_history, color='steelblue', linewidth=1, alpha=0.5)
    if len(loss_history) >= 20:
        ma = np.convolve(loss_history, np.ones(20)/20, mode='valid')
        ax_loss.plot(range(19, len(loss_history)), ma,
                     color='darkblue', linewidth=2, label='MA-20')
        ax_loss.legend(fontsize=7)
    ax_loss.set_title('Triplet Loss\n(kein Mode Collapse)', fontsize=9)
    ax_loss.set_xlabel('Schritt')

    # Aktions-Raum 2D
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (name, raw) in enumerate(NAMED_ACTIONS.items()):
        ax_action.scatter(raw[0], raw[1], color=colors[i], s=120, zorder=3)
        ax_action.annotate(name.replace("_", "\n"), (raw[0], raw[1]),
                           fontsize=6, ha='center', va='bottom',
                           xytext=(0, 6), textcoords='offset points')
    ax_action.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_action.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_action.set_xlabel('linear_x [m/s]')
    ax_action.set_ylabel('angular_z [rad/s]')
    ax_action.set_title('Aktions-Raum\nlinear_x vs angular_z', fontsize=9)

    # ROS2 Ausblick
    ros_lines = [
        "── ROS2 Integration (Ausblick B09/B11) ──────────────────────────",
        "",
        "  Action Head (B09) Output → ROS2 Twist Message:",
        "",
        "  action = model.predict(obs, goal)       # [-1,1] normalisiert",
        "  params = denormalize_action(action)",
        "",
        "  twist = Twist()",
        "  twist.linear.x  = params['linear_x']   # z.B. 0.30 m/s",
        "  twist.angular.z = params['angular_z']  # z.B. 0.00 rad/s",
        "  publisher.publish(twist)                # Bewegung starten",
        "  time.sleep(params['duration'])          # z.B. 0.50 s warten",
        "  publisher.publish(Twist())              # Stopp",
        "",
        "  Ergebnis: 'Bewege dich 15cm geradeaus'",
        "  statt:    'starte Vorwaerts' ... 'stoppe Vorwaerts'",
        "",
        "  Vorteil fuer das World Model:",
        "  Das Netz lernt: linear_x=0.3, duration=0.5 → naechstes Bild",
        "  ist um ~15cm verschoben – physikalisch praezise vorhersagbar.",
    ]
    ax_ros.text(0.01, 0.98, "\n".join(ros_lines),
                transform=ax_ros.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.show()
    print("\nDemo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
