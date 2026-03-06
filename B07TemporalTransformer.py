"""
B07 – Temporal Transformer Demo
=================================
Fusioniert die 5 historischen (Frame, Aktion)-Paare aus B03
mit dem aktuellen Frame (B04b) und dem Ziel-Embedding (B05)
zu einem gemeinsamen Kontext-Vektor für den Decoder (B08).

Architektur:
    Für jeden der 5 historischen Slots:
        z_image  (latent_dim)   ← CNN-Encoder B04b
        z_action (action_dim)   ← Action Embedding B06
        z_time   (latent_dim)   ← Sinusförmiges Zeit-Encoding B03
        → konkateniert → Linear → Token (d_model)

    Zusätzliche Tokens:
        z_current  (latent_dim)  ← aktueller Frame (B04b)
        z_goal     (clip_dim)    ← CLIP Text-Embedding (B05)
        → jeweils per Linear auf d_model projiziert

    Alle Tokens zusammen → TransformerEncoder (Self-Attention)
    → [CLS]-Token als Kontext-Vektor für B08/B09

Warum Transformer?
    - Self-Attention lernt WELCHE vergangenen Frames relevant sind
    - z.B.: "t-8 war wichtig weil dort der Ball sichtbar war"
    - Keine feste Gewichtung – vollständig datengetrieben

Active Inference Bedeutung:
    Der Transformer = Generatives Modell
    Input  = Sensorische Geschichte (Markov Blanket Vergangenheit)
    Output = Komprimierte Belief über den aktuellen Weltzustand

Token-Sequenz:
    [CLS] [current] [goal] [t-1] [t-2] [t-4] [t-8] [t-16]
      ↑       ↑       ↑      ↑     ↑     ↑     ↑      ↑
    Output  Frame   Text  (Frame+Action+Zeit) × 5
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
# ZEITENCODING (aus B03)
# ─────────────────────────────────────────────

def sinusoidal_time_encoding(time_steps: list, dim: int) -> torch.Tensor:
    """
    Sinusförmiges Encoding für nicht-lineare Zeitabstände.
    Kodiert log(1+t) für bessere Verteilung bei [1,2,4,8,16].
    Returns: (len(time_steps), dim)
    """
    T   = len(time_steps)
    enc = torch.zeros(T, dim)
    log_t = torch.log1p(torch.tensor(time_steps, dtype=torch.float32))

    for i in range(0, dim, 2):
        freq = 1.0 / (10000 ** (i / dim))
        enc[:, i] = torch.sin(log_t * freq)
        if i + 1 < dim:
            enc[:, i+1] = torch.cos(log_t * freq)
    return enc


# ─────────────────────────────────────────────
# TEMPORAL TRANSFORMER
# ─────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """
    Fusioniert historische (Frame, Aktion)-Paare + aktuellen Frame + Ziel
    zu einem Kontext-Vektor via Self-Attention.

    Input:
        z_current  : (B, latent_dim)          – aktueller Frame-Embedding
        z_goal     : (B, clip_dim)            – CLIP Text-Embedding
        z_frames   : (B, n_slots, latent_dim) – historische Frame-Embeddings
        z_actions  : (B, n_slots, action_dim) – historische Aktions-Vektoren
        time_steps : Liste [1, 2, 4, 8, 16]

    Output:
        context    : (B, d_model) – fusionierter Kontext-Vektor
        attn_maps  : Liste von Attention-Gewichten (für Visualisierung)

    Token-Sequenz (Länge = 2 + n_slots + 1):
        [CLS] [current] [goal] [t-1] [t-2] [t-4] [t-8] [t-16]
    """

    def __init__(
            self,
            latent_dim:  int = 64,
            clip_dim:    int = 512,
            action_dim:  int = 3,
            d_model:     int = 128,
            n_heads:     int = 4,
            n_layers:    int = 3,
            dropout:     float = 0.1,
            time_steps:  list = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_dim   = clip_dim
        self.action_dim = action_dim
        self.d_model    = d_model
        self.time_steps = time_steps or [1, 2, 4, 8, 16]
        self.n_slots    = len(self.time_steps)

        # ── Token-Projektionen ────────────────────────
        # CLS-Token (gelernter Vektor)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Aktueller Frame → Token
        self.proj_current = nn.Linear(latent_dim, d_model)

        # Ziel-Embedding → Token
        self.proj_goal = nn.Linear(clip_dim, d_model)

        # Historischer Slot: Frame + Aktion + Zeit → Token
        # Eingabe-Dim = latent_dim + action_dim + latent_dim (Zeit-Encoding)
        slot_input_dim = latent_dim + action_dim + latent_dim
        self.proj_slot = nn.Linear(slot_input_dim, d_model)

        # ── Zeit-Encoding (fest, nicht trainierbar) ───
        time_enc = sinusoidal_time_encoding(self.time_steps, latent_dim)
        self.register_buffer("time_enc", time_enc)  # (n_slots, latent_dim)

        # ── Transformer Encoder ───────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,    # (B, T, d_model)
            norm_first=True,     # Pre-Norm für stabileres Training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # ── Output Norm ───────────────────────────────
        self.output_norm = nn.LayerNorm(d_model)

        # ── Next-Latent-Prediction Head ───────────────
        self.next_z_head = nn.Linear(d_model, latent_dim)

    def forward(
            self,
            z_current:  torch.Tensor,           # (B, latent_dim)
            z_goal:     torch.Tensor,           # (B, clip_dim)
            z_frames:   torch.Tensor,           # (B, n_slots, latent_dim)
            z_actions:  torch.Tensor,           # (B, n_slots, action_dim)
            valid_mask: torch.Tensor = None,    # (B, n_slots) bool – False = ungültiger Slot
    ):
        B = z_current.size(0)

        # ── CLS-Token ─────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)

        # ── Aktueller Frame ───────────────────────────
        tok_current = self.proj_current(z_current).unsqueeze(1)  # (B, 1, d_model)

        # ── Ziel ──────────────────────────────────────
        tok_goal = self.proj_goal(z_goal).unsqueeze(1)            # (B, 1, d_model)

        # ── Historische Slots ─────────────────────────
        # Zeit-Encoding auf Batch-Dim erweitern
        time_enc = self.time_enc.unsqueeze(0).expand(B, -1, -1)  # (B, n_slots, latent_dim)

        # Konkateniere: Frame + Aktion + Zeit
        slot_input = torch.cat([z_frames, z_actions, time_enc], dim=-1)  # (B, n_slots, slot_input_dim)
        tok_slots  = self.proj_slot(slot_input)                           # (B, n_slots, d_model)

        # ── Alle Tokens zusammen ──────────────────────
        # Sequenz: [CLS, current, goal, t-1, t-2, t-4, t-8, t-16]
        tokens = torch.cat([cls, tok_current, tok_goal, tok_slots], dim=1)
        # → (B, 3 + n_slots, d_model)

        # ── Attention Mask für ungültige Slots ────────
        # Ungültige Slots werden maskiert (Attention ignoriert sie)
        key_padding_mask = None
        if valid_mask is not None:
            # True = ignorieren in Attention
            # CLS, current, goal sind immer gültig → False
            prefix_valid = torch.zeros(B, 3, dtype=torch.bool, device=z_current.device)
            slot_invalid = ~valid_mask                                    # (B, n_slots)
            key_padding_mask = torch.cat([prefix_valid, slot_invalid], dim=1)

        # ── Transformer ───────────────────────────────
        out = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        out = self.output_norm(out)

        # ── CLS-Token als Kontext-Vektor ──────────────
        context = out[:, 0, :]    # (B, d_model)

        return context, out       # context + alle Token-Outputs für Visualisierung

    def summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        seq_len = 3 + self.n_slots
        return {
            "latent_dim":  self.latent_dim,
            "clip_dim":    self.clip_dim,
            "action_dim":  self.action_dim,
            "d_model":     self.d_model,
            "n_slots":     self.n_slots,
            "seq_len":     seq_len,
            "params":      total,
        }


# ─────────────────────────────────────────────
# MOCK INPUTS (alle Vorgänger-Bausteine)
# ─────────────────────────────────────────────

def make_mock_inputs(
        batch_size: int,
        latent_dim: int,
        clip_dim:   int,
        action_dim: int,
        n_slots:    int,
        step:       int = 0,
):
    """
    Simuliert die Outputs von B04b, B05, B06 und B03.
    Später werden diese durch die echten Bausteine ersetzt.
    """
    # Aktueller Frame-Embedding (B04b) – leicht variiert pro Step
    z_current = F.normalize(
        torch.randn(batch_size, latent_dim) + 0.1 * step / 100,
        dim=-1
    )

    # Ziel-Embedding (B05) – konstant (gleiches Ziel)
    torch.manual_seed(42)
    z_goal = F.normalize(torch.randn(batch_size, clip_dim), dim=-1)
    torch.manual_seed(step + 100)

    # Historische Frame-Embeddings (B04b × 5)
    z_frames = F.normalize(
        torch.randn(batch_size, n_slots, latent_dim),
        dim=-1
    )

    # Historische Aktions-Vektoren (B06 × 5) – normalisiert [-1, 1]
    z_actions = torch.tanh(torch.randn(batch_size, n_slots, action_dim))

    # Validity Mask (B03) – erste Slots sind bei wenig Daten ungültig
    valid = torch.ones(batch_size, n_slots, dtype=torch.bool)
    if step < 16:
        for i, t in enumerate([1, 2, 4, 8, 16]):
            if step < t:
                valid[:, i] = False

    return z_current, z_goal, z_frames, z_actions, valid


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    LATENT_DIM  = 64
    CLIP_DIM    = 512
    ACTION_DIM  = 3
    D_MODEL     = 128
    N_HEADS     = 4
    N_LAYERS    = 2
    BATCH_SIZE  = 4
    N_STEPS     = 80
    TIME_STEPS  = [1, 2, 4, 8, 16]

    transformer = TemporalTransformer(
        latent_dim=LATENT_DIM,
        clip_dim=CLIP_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        time_steps=TIME_STEPS,
    )

    info = transformer.summary()
    print("Temporal Transformer initialisiert:")
    for k, v in info.items():
        print(f"  {k:12s}: {v}")
    print()
    print(f"Token-Sequenz: [CLS] [current] [goal] + {len(TIME_STEPS)} Slots")
    print(f"Sequenz-Laenge: {info['seq_len']} Tokens à {D_MODEL} dim\n")

    # AdamW mit Weight Decay verhindert Gewichts-Explosion
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_history   = []
    context_history = []   # CLS-Vektor über Zeit

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('B07 – Temporal Transformer', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    ax_arch    = fig.add_subplot(gs[0, 0])    # Architektur-Diagramm
    ax_tokens  = fig.add_subplot(gs[0, 1:3])  # Token-Heatmap
    ax_stats   = fig.add_subplot(gs[0, 3])    # Statistiken
    ax_attn    = fig.add_subplot(gs[1, :2])   # Attention-Gewichte
    ax_context = fig.add_subplot(gs[1, 2:])   # Kontext-Vektor
    ax_loss    = fig.add_subplot(gs[2, :2])   # Loss
    ax_cosine  = fig.add_subplot(gs[2, 2:])   # Cosinus-Stabilität
    ax_arch.axis('off')
    ax_stats.axis('off')

    print(f"Starte Demo: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        transformer.train()

        z_current, z_goal, z_frames, z_actions, valid = make_mock_inputs(
            BATCH_SIZE, LATENT_DIM, CLIP_DIM, ACTION_DIM, len(TIME_STEPS), step
        )

        context, all_tokens = transformer(
            z_current, z_goal, z_frames, z_actions, valid
        )

        # Loss: Next-Latent-Prediction
        # Der Transformer soll aus dem Kontext z_{t+1} vorhersagen
        z_next_target = torch.randn(BATCH_SIZE, LATENT_DIM) * 0.3 + z_current * 0.7
        pred_z_next   = transformer.next_z_head(context)
        loss = F.mse_loss(pred_z_next, z_next_target.detach())

        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping: verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(float(loss.detach()))

        # Grad-Norm nach Clipping berechnen (für Monitoring)
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in transformer.parameters() if p.grad is not None
        ) ** 0.5

        # Eval für Visualisierung
        transformer.eval()
        with torch.no_grad():
            ctx, tokens_out = transformer(
                z_current, z_goal, z_frames, z_actions, valid
            )
        context_np = ctx[0].numpy()           # Erster Batch-Eintrag
        tokens_np  = tokens_out[0].numpy()    # (seq_len, d_model)
        context_history.append(context_np.copy())

        if step % 8 == 0 or step == N_STEPS - 1:

            # ── Architektur ────────────────────────────
            ax_arch.clear()
            ax_arch.axis('off')
            arch_lines = [
                "Token-Sequenz:",
                "",
                " [CLS]      ← Output",
                " [current]  ← B04b",
                " [goal]     ← B05",
                " [t-1 ]     ← B04b+B06",
                " [t-2 ]     ← B04b+B06",
                " [t-4 ]     ← B04b+B06",
                " [t-8 ]     ← B04b+B06",
                " [t-16]     ← B04b+B06",
                "",
                "    ↓ Self-Attention",
                f"   {N_LAYERS}x TransformerLayer",
                f"   {N_HEADS} Heads, d={D_MODEL}",
                "",
                "    ↓",
                " context (d_model)",
                " → B08 Decoder",
                " → B09 Action Head",
            ]
            ax_arch.text(
                0.05, 0.97, "\n".join(arch_lines),
                transform=ax_arch.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            # ── Token Heatmap ──────────────────────────
            ax_tokens.clear()
            token_labels = (
                    ["CLS", "current", "goal"] +
                    [f"t-{t}" for t in TIME_STEPS]
            )
            # Zeige erste 64 Dimensionen
            show_d = min(64, D_MODEL)
            im = ax_tokens.imshow(
                tokens_np[:, :show_d].T,
                cmap='coolwarm', aspect='auto',
                vmin=-2, vmax=2, interpolation='nearest'
            )
            ax_tokens.set_xticks(range(len(token_labels)))
            ax_tokens.set_xticklabels(token_labels, rotation=30, ha='right', fontsize=8)
            ax_tokens.set_ylabel(f'Dim (erste {show_d})', fontsize=8)
            ax_tokens.set_title(
                f'Transformer Output – alle Tokens (Step {step+1})', fontsize=9
            )
            fig.colorbar(im, ax=ax_tokens, fraction=0.02)

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            n_valid = int(valid[0].sum())
            lines = [
                "── Transformer ──────",
                f"d_model:    {D_MODEL}",
                f"n_heads:    {N_HEADS}",
                f"n_layers:   {N_LAYERS}",
                f"seq_len:    {info['seq_len']}",
                f"params:     {info['params']:,}",
                "",
                "── Laufzeit ─────────",
                f"Schritt:    {step + 1}",
                f"Loss:       {loss_history[-1]:.4f}",
                f"Grad-Norm:  {grad_norm:.4f}",
                f"Guelt.Slots:{n_valid}/{len(TIME_STEPS)}",
                f"Ctx-Norm:   {np.linalg.norm(context_np):.4f}",
                "",
                "── Inputs ───────────",
                f"z_current:  ({LATENT_DIM},)",
                f"z_goal:     ({CLIP_DIM},)",
                f"z_frames:   ({len(TIME_STEPS)},{LATENT_DIM})",
                f"z_actions:  ({len(TIME_STEPS)},{ACTION_DIM})",
                "",
                "── Output ───────────",
                f"context:    ({D_MODEL},)",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
            )

            # ── Attention Simulation ────────────────────
            # Zeigt wie stark jeder Token auf den CLS-Token "aufmerksam" ist
            # (echte Attention weights bräuchten Hook – hier approximiert via Dot-Product)
            ax_attn.clear()
            cls_vec    = tokens_np[0]                     # CLS-Token Output
            attn_proxy = tokens_np @ cls_vec              # Dot-Product mit allen Tokens
            attn_proxy = np.exp(attn_proxy - attn_proxy.max())
            attn_proxy = attn_proxy / attn_proxy.sum()

            colors_attn = ['gold'] + ['steelblue'] + ['seagreen'] + \
                          [plt.cm.plasma(i / len(TIME_STEPS))
                           for i in range(len(TIME_STEPS))]
            bars = ax_attn.bar(
                token_labels, attn_proxy,
                color=colors_attn[:len(token_labels)]
            )
            ax_attn.set_title(
                'Approximierte Attention auf CLS-Token\n'
                '(Dot-Product der Output-Tokens mit CLS)',
                fontsize=9
            )
            ax_attn.set_ylabel('Gewicht (softmax)')
            ax_attn.tick_params(axis='x', rotation=20, labelsize=8)

            # Werte über Balken
            for bar, val in zip(bars, attn_proxy):
                ax_attn.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=7
                )

            # ── Kontext-Vektor ─────────────────────────
            ax_context.clear()
            colors_ctx = ['steelblue' if v >= 0 else 'tomato' for v in context_np]
            ax_context.bar(range(D_MODEL), context_np, color=colors_ctx, width=1.0)
            ax_context.axhline(0, color='black', linewidth=0.5)
            ax_context.set_title(
                f'Kontext-Vektor (CLS-Token Output, d={D_MODEL})\n'
                f'Norm: {np.linalg.norm(context_np):.4f}',
                fontsize=9
            )
            ax_context.set_xlabel('Dimension')
            ax_context.set_ylim(-3, 3)

            # ── Loss ──────────────────────────────────
            ax_loss.clear()
            ax_loss.plot(loss_history, color='steelblue', linewidth=1, alpha=0.6)
            if len(loss_history) >= 10:
                ma = np.convolve(loss_history, np.ones(10)/10, mode='valid')
                ax_loss.plot(range(9, len(loss_history)), ma,
                             color='darkblue', linewidth=2, label='MA-10')
                ax_loss.legend(fontsize=7)
            ax_loss.set_title('Training Loss (Next-Latent-Prediction)', fontsize=9)
            ax_loss.set_xlabel('Schritt')

            # ── Cosinus-Stabilität ─────────────────────
            ax_cosine.clear()
            if len(context_history) >= 2:
                cos_sims = []
                for i in range(1, len(context_history)):
                    a = context_history[i-1]
                    b = context_history[i]
                    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                    cos_sims.append(cos)
                ax_cosine.plot(cos_sims, color='purple', linewidth=1.5)
                ax_cosine.axhline(1.0, color='green', linestyle='--',
                                  linewidth=1, label='Max=1.0')
                ax_cosine.axhline(0.0, color='gray', linestyle='--',
                                  linewidth=0.8, label='Unkorreliert')
            ax_cosine.set_title('Cosinus-Stabilitaet des Kontext-Vektors\n'
                                '(aufeinanderfolgende Steps)', fontsize=9)
            ax_cosine.set_xlabel('Schritt')
            ax_cosine.set_ylim(-0.2, 1.2)
            if len(context_history) >= 2:
                ax_cosine.legend(fontsize=7)

            plt.pause(0.05)

    print("\nDemo abgeschlossen!")
    print(f"  Loss final  : {loss_history[-1]:.4f}")
    print(f"  Ctx-Norm    : {np.linalg.norm(context_history[-1]):.4f}")
    print(f"  Grad-Norm   : {grad_norm:.4f}  (Clipping bei 1.0)")
    print()
    print("Fixes angewendet:")
    print("  3 Transformer-Layer          → mehr Kapazität")
    print("  Next-Latent-Prediction Loss  → task-aligned statt Norm-Konsistenz")
    print("  Pre-Norm (norm_first=True)   → stabileres Training")
    print()
    print("Naechste Schritte:")
    print("  B08 – CNN Decoder: context → predicted frame")
    print("  B09 – Action Head: context → next action")

    plt.show()


if __name__ == "__main__":
    run_demo()
