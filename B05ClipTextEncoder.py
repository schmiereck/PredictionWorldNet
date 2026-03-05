"""
B05 – CLIP Text-Encoder Demo
=============================
Kodiert ein Ziel (z.B. "finde den roten Ball") in einen Embedding-Vektor,
der im selben Latent Space liegt wie Bild-Embeddings.

Warum CLIP?
    CLIP (Contrastive Language-Image Pretraining, OpenAI) wurde auf
    400 Millionen Bild-Text-Paaren trainiert. Es versteht den Zusammenhang
    zwischen Sprache und visuellen Konzepten.

    → Text "roter Ball" und ein Bild mit rotem Ball haben hohe Cosinus-Ähnlichkeit
    → Perfekt als Ziel-Konditionierung für unser World Model

Rolle im Gesamtsystem (Active Inference):
    Text-Embedding = Prior Belief
    "Was der Agent in der Zukunft zu sehen erwartet, wenn er das Ziel erreicht"
    → Wird per Cross-Attention in den Temporal Transformer (B07) injiziert

Input:  Text-String  (z.B. "find the red ball")
Output: Embedding-Vektor (512-dim für ViT-B/32)

Installation:
    pip install git+https://github.com/openai/CLIP.git
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP nicht installiert. Starte im Mock-Modus.")
    print("Installation: pip install git+https://github.com/openai/CLIP.git\n")


# ─────────────────────────────────────────────
# CLIP TEXT ENCODER WRAPPER
# ─────────────────────────────────────────────

class CLIPTextEncoder:
    """
    Wrapper um den CLIP Text-Encoder.

    Gibt für einen Text-String einen L2-normierten Embedding-Vektor zurück.
    Der Vektor liegt im gemeinsamen CLIP Latent Space (Text + Bild).

    Modell-Optionen (Größe vs. Qualität):
        "ViT-B/32"  → 512-dim,  kleinst, schnell   ← Empfehlung für uns
        "ViT-B/16"  → 512-dim,  besser
        "ViT-L/14"  → 768-dim,  groß, langsam

    Warum ViT-B/32?
        Unser World Model arbeitet mit 16x16 Bildern – die semantische
        Qualität des größeren Modells würde keinen Mehrwert bringen.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.model_name = model_name
        self.dim        = 512  # ViT-B/32 Output-Dimension

        if CLIP_AVAILABLE:
            print(f"Lade CLIP Modell '{model_name}' auf {device}...")
            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            # Tatsächliche Dimension aus dem Modell lesen
            self.dim = self.model.text_projection.shape[1]
            print(f"CLIP geladen. Embedding-Dim: {self.dim}\n")
        else:
            print(f"Mock-Modus: Simuliere CLIP '{model_name}', dim={self.dim}\n")

    def encode_text(self, text: str) -> np.ndarray:
        """
        Text → L2-normierter Embedding-Vektor.

        Args:
            text: Ziel-Beschreibung (z.B. "find the red ball")
        Returns:
            embedding: (dim,) numpy float32, L2-normiert
        """
        if CLIP_AVAILABLE:
            with torch.no_grad():
                tokens    = clip.tokenize([text]).to(self.device)
                embedding = self.model.encode_text(tokens)
                embedding = F.normalize(embedding, dim=-1)
                return embedding.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            # Mock: deterministischer Pseudo-Embedding aus Text-Hash
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            return vec / np.linalg.norm(vec)

    def encode_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Bild (H,W,3 uint8) → L2-normierter CLIP Bild-Embedding.
        Für den Vergleich Text vs. Bild im selben Latent Space.
        """
        if CLIP_AVAILABLE:
            from PIL import Image
            pil_img = Image.fromarray(image_rgb).resize((224, 224))
            with torch.no_grad():
                x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(x)
                embedding = F.normalize(embedding, dim=-1)
                return embedding.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            rng = np.random.default_rng(int(image_rgb.mean() * 1000))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            return vec / np.linalg.norm(vec)

    def similarity(self, text: str, other_text: str) -> float:
        """Cosinus-Ähnlichkeit zwischen zwei Text-Embeddings."""
        a = self.encode_text(text)
        b = self.encode_text(other_text)
        return float(np.dot(a, b))

    def summary(self) -> dict:
        return {
            "model":    self.model_name,
            "device":   self.device,
            "dim":      self.dim,
            "mode":     "CLIP" if CLIP_AVAILABLE else "Mock",
        }


# ─────────────────────────────────────────────
# ZIEL-DEFINITIONEN FÜR MINIWORLD
# ─────────────────────────────────────────────

# Verschiedene Ziele die unser Roboter verfolgen könnte
GOALS = {
    "Primärziel":    "find the red box",
    "Alternative 1": "find the blue ball",
    "Alternative 2": "navigate to the exit door",
    "Alternative 3": "explore the room",
    "Alternative 4": "avoid obstacles",
    "Negativ":       "find the green triangle",   # Nicht in unserer Env
    "Ähnlich":       "locate the red cube",       # Semantisch ähnlich zu Primärziel
    "Unrelated":     "the weather is sunny today",# Irrelevant
}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    encoder = CLIPTextEncoder(model_name="ViT-B/32")
    info    = encoder.summary()

    print("Kodiere alle Ziele...")
    embeddings = {}
    for name, text in GOALS.items():
        embeddings[name] = encoder.encode_text(text)
        print(f"  [{name:12s}] '{text}'  → dim={embeddings[name].shape[0]}, "
              f"norm={np.linalg.norm(embeddings[name]):.4f}")

    # Ähnlichkeitsmatrix berechnen
    names   = list(GOALS.keys())
    n       = len(names)
    sim_mat = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            sim_mat[i, j] = float(np.dot(embeddings[ni], embeddings[nj]))

    print(f"\nModell: {info['mode']} '{info['model']}'")
    print(f"Embedding-Dim: {info['dim']}")

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"B05 – CLIP Text-Encoder  [{info['mode']}: {info['model']}, dim={info['dim']}]",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    ax_sim      = fig.add_subplot(gs[0, :2])   # Ähnlichkeitsmatrix
    ax_stats    = fig.add_subplot(gs[0, 2])    # Statistiken
    ax_emb      = fig.add_subplot(gs[1, :2])   # Embedding-Vektoren als Heatmap
    ax_bar      = fig.add_subplot(gs[1, 2])    # Ähnlichkeit zum Primärziel

    # ── Ähnlichkeitsmatrix ─────────────────────────────────
    im = ax_sim.imshow(sim_mat, cmap='RdYlGn', vmin=-0.3, vmax=1.0,
                       interpolation='nearest')
    ax_sim.set_xticks(range(n))
    ax_sim.set_yticks(range(n))
    ax_sim.set_xticklabels(
        [f"{k}\n'{v[:20]}...'" if len(v) > 20 else f"{k}\n'{v}'"
         for k, v in GOALS.items()],
        rotation=30, ha='right', fontsize=7
    )
    ax_sim.set_yticklabels(list(GOALS.keys()), fontsize=7)
    ax_sim.set_title('Cosinus-Aehnlichkeitsmatrix aller Ziele', fontsize=10)
    fig.colorbar(im, ax=ax_sim, fraction=0.03)

    # Werte in die Zellen schreiben
    for i in range(n):
        for j in range(n):
            ax_sim.text(j, i, f"{sim_mat[i,j]:.2f}",
                        ha='center', va='center', fontsize=7,
                        color='black' if abs(sim_mat[i,j]) < 0.7 else 'white')

    # ── Statistiken ────────────────────────────────────────
    ax_stats.axis('off')
    primary = "Primärziel"
    sims_to_primary = {k: float(np.dot(embeddings[primary], v))
                       for k, v in embeddings.items() if k != primary}
    most_similar  = max(sims_to_primary, key=sims_to_primary.get)
    least_similar = min(sims_to_primary, key=sims_to_primary.get)

    lines = [
        "── CLIP Text-Encoder ──────",
        f"Modell:  {info['model']}",
        f"Modus:   {info['mode']}",
        f"Dim:     {info['dim']}",
        f"Device:  {info['device']}",
        "",
        f"── Primärziel ─────────────",
        f"'{GOALS[primary]}'",
        "",
        "Aehnlichkeit zu anderen:",
        *[f"  {k[:12]:12s}: {v:.3f}"
          for k, v in sorted(sims_to_primary.items(),
                             key=lambda x: x[1], reverse=True)],
        "",
        f"Aehnlichste: {most_similar}",
        f"Unaehnlichste: {least_similar}",
        "",
        "── Active Inference ───────",
        "Text-Embedding = Prior",
        "Hohe Cos-Sim   = Ziel nah",
        "Niedrige Sim   = Ziel fern",
    ]
    ax_stats.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_stats.transAxes,
        fontsize=8, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    # ── Embedding-Heatmap ──────────────────────────────────
    emb_matrix = np.stack([embeddings[k] for k in names])  # (n, dim)
    # Erste 128 Dimensionen zeigen
    show_dim = min(128, info['dim'])
    im2 = ax_emb.imshow(emb_matrix[:, :show_dim], cmap='coolwarm',
                        aspect='auto', vmin=-0.15, vmax=0.15,
                        interpolation='nearest')
    ax_emb.set_yticks(range(n))
    ax_emb.set_yticklabels(list(GOALS.keys()), fontsize=8)
    ax_emb.set_xlabel(f'Embedding-Dimension (erste {show_dim} von {info["dim"]})')
    ax_emb.set_title('Embedding-Vektoren als Heatmap', fontsize=10)
    fig.colorbar(im2, ax=ax_emb, fraction=0.02)

    # ── Ähnlichkeiten zum Primärziel ───────────────────────
    ax_bar.clear()
    other_names = [k for k in names if k != primary]
    other_sims  = [sims_to_primary[k] for k in other_names]
    colors      = ['steelblue' if s > 0 else 'tomato' for s in other_sims]
    bars = ax_bar.barh(other_names, other_sims, color=colors)
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.set_xlim(-0.3, 1.0)
    ax_bar.set_title(f"Aehnlichkeit zu\n'{GOALS[primary]}'", fontsize=9)
    ax_bar.set_xlabel('Cosinus-Aehnlichkeit')

    # Werte an Balken
    for bar, val in zip(bars, other_sims):
        ax_bar.text(
            val + 0.02 if val >= 0 else val - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va='center', fontsize=7,
            ha='left' if val >= 0 else 'right'
        )

    # ── Terminal: Ähnlichkeits-Tabelle ─────────────────────
    print(f"\nAehnlichkeit zu Primaerziel '{GOALS[primary]}':")
    for name, sim in sorted(sims_to_primary.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(max(0, sim) * 30)
        print(f"  {name:12s}  {sim:+.4f}  {bar}")

    print(f"\n{'='*50}")
    print("Hinweis fuer das Gesamtsystem:")
    print("  Das Text-Embedding wird spaeter per Cross-Attention")
    print("  in den Temporal Transformer (B07) injiziert.")
    print("  Hohe Cosinus-Aehnlichkeit = Agent ist nah am Ziel")
    print("  → kann als Teil des Reward-Signals verwendet werden")

    plt.show()
    print("\nDemo abgeschlossen!")


if __name__ == "__main__":
    run_demo()
