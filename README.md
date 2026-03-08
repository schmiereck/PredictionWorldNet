# 🧠 PredictionWorldNet

**Predictive World Model basierend auf Active Inference — eine Alternative zu klassischem Reinforcement Learning**

> *Intelligenz bedeutet, die eigenen Vorhersagen über die Welt zu bestätigen.*

---

## Schnellstart

```bash
# Voraussetzungen: Python 3.11, GOOGLE_API_KEY als Umgebungsvariable

# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. (Optional) VAE vortrainieren — Bildkompression lernen
python B20PreTrainVAE.py --source miniworld --epochs 50

# 3. (Optional) CLIP Goal-Projection trainieren
python B21PreTrainCLIP.py --vae-checkpoint checkpoints/pwn_*.pt --epochs 60

# 4. (Optional) Dynamics-Head vortrainieren — Zustandsübergänge lernen
python B24PreTrainDynamics.py --checkpoint checkpoints/pwn_*.pt --epochs 30

# 5. Simulation starten (Hauptprogramm)
python B19OrchestratorModeMiniworld.py
```

### Optionale Parameter

| Skript | Parameter | Beschreibung |
|--------|-----------|-------------|
| `B20PreTrainVAE.py` | `--source miniworld\|synthetic` | Datenquelle |
| | `--epochs N` | Anzahl Epochen (Standard: 50) |
| `B21PreTrainCLIP.py` | `--vae-checkpoint PATH` | VAE-Checkpoint laden |
| | `--epochs N` | Anzahl Epochen (Standard: 60) |
| `B24PreTrainDynamics.py` | `--checkpoint PATH` | Checkpoint laden |
| | `--epochs N` | Anzahl Epochen (Standard: 30) |
| `B16FullIntegration.py` | `--headless` | Ohne GUI (nur Konsole + CSV) |
| | `--steps=N` | Anzahl Schritte (Standard: 300) |

### Reihenfolge

```
B20 (VAE) → B21 (CLIP) → B24 (Dynamics) → B19 (Live-Training)
     ↑           ↑             ↑                    ↑
  optional    optional      optional           Hauptprogramm
```

Alle Pre-Training-Schritte sind optional — das System lernt auch ohne Vortraining,
aber die Konvergenz ist mit Pre-Training deutlich schneller.

---

## Überblick

PredictionWorldNet implementiert ein **Predictive World Model**, das auf den Prinzipien der **Active Inference** basiert.
Statt passiv auf Belohnungen zu reagieren, antizipiert der Agent aktiv die Realität und minimiert die
**Variational Free Energy** — den Unterschied zwischen seinen Vorhersagen und eingehenden Sensordaten.

Das System kombiniert ein schnelles, niedrig aufgelöstes neuronales Netz (lokales Lernen) mit
hochauflösenden **Gemini Vision-Bewertungen** (semantische Belohnungen) für effizientes, erklärbares Lernen.

### Kernkonzepte

| Konzept | Beschreibung |
|---------|-------------|
| **Active Inference** | Der Agent antizipiert aktiv die Realität statt passiv auf Rewards zu reagieren |
| **Generatives Modell** | Explizites Weltmodell, das durch Vorhersage lernt — nicht durch Reward-Chasing |
| **Variational Free Energy** | Einziges Optimierungsprinzip: Vorhersagefehler minimieren |
| **Expected Free Energy (EFE)** | Aktionen minimieren erwartete FE: epistemisch (Unsicherheit reduzieren) + pragmatisch (Ziel erreichen) |
| **Planning-as-Inference** | Aktions-Planung durch mehrstufige Imagination im latenten Raum |

---

## Architektur-Übersicht

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PredictionWorldNet                           │
│                                                                      │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ MiniWorld│───▶│  Encoder  │───▶│    RSSM      │───▶│  Decoder  │  │
│  │   Env   │    │   (VAE)   │    │ (GRU-World)  │    │   (CNN)   │  │
│  └────┬────┘    └─────┬─────┘    └──────┬───────┘    └─────┬─────┘  │
│       │               │                 │                   │        │
│       │          z (256-dim)       h_t (256-dim)       pred_obs      │
│       │               │                 │                            │
│       │               │    ┌────────────┼────────────┐               │
│       │               │    │            │            │               │
│       │               │    ▼            ▼            ▼               │
│       │               │  Action     dynamics    reward_head          │
│       │               │  Head       _head       (T14)                │
│       │               │  (6D+σ)    (z_{t+1})    (r_pred)            │
│       │               │    │            │            │               │
│       │               │    ▼            ▼            ▼               │
│       │               │  ┌──────────────────────────────┐            │
│       │               │  │  T15: Imagination Rollout    │            │
│       │               │  │  N=32 Kandidaten, H=5 Steps  │            │
│       │               │  │  → Beste Aktion auswählen    │            │
│       │               │  └──────────────────────────────┘            │
│       │               │                                              │
│       ▼               ▼                                              │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐                   │
│  │ Gemini  │    │   CLIP    │    │  Strategy    │                   │
│  │ ER API  │    │  Encoder  │    │  Executor    │                   │
│  └─────────┘    └───────────┘    └──────────────┘                   │
│                                                                      │
│  ┌─────────────────────┐  ┌──────────────────────┐                  │
│  │   Dashboard (B18)   │  │  Overhead Map View   │                  │
│  └─────────────────────┘  └──────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Neuronales Netz — Detailarchitektur

### Dimensionen

| Parameter | Wert | Beschreibung |
|-----------|------|-------------|
| LATENT_DIM | 256 | Komprimierte Repräsentation (z) |
| ACTION_DIM | 6 | Aktionsvektor-Dimensionen |
| D_MODEL | 256 | RSSM Hidden-State Dimension (h_t) |
| OBS_SHAPE | 128×128×3 | Beobachtungs-Auflösung |

### Encoder (VAE)

```
Input: (B, 3, 128, 128) — RGB-Bild

Conv2d(3→32, 3×3, stride=2)  + GroupNorm(8) + ReLU    → (B, 32, 64, 64)
Conv2d(32→64, 3×3, stride=2) + GroupNorm(8) + ReLU    → (B, 64, 32, 32)
Conv2d(64→128, 3×3, stride=2)+ GroupNorm(8) + ReLU    → (B, 128, 16, 16)
Conv2d(128→128, 3×3, stride=2)+ GroupNorm(8) + ReLU   → (B, 128, 8, 8)
Conv2d(128→128, 3×3, stride=2)+ GroupNorm(8) + ReLU   → (B, 128, 4, 4)

Flatten → 2048-dim
├── fc_mu:      Linear(2048, 256)  → μ
└── fc_log_var: Linear(2048, 256)  → log(σ²)

Reparametrisierung: z = μ + ε·σ,  ε ~ N(0, I)
```

GroupNorm statt BatchNorm: kein Train/Eval-Drift bei wechselnden Szenen im Online-Learning.

### Decoder (CNN)

```
Input: z (B, 256)

Linear(256, 2048) + ReLU → Reshape (B, 128, 4, 4)

ConvTranspose2d(128→128) + GroupNorm(8) + ReLU  → (B, 128, 8, 8)
ConvTranspose2d(128→128) + GroupNorm(8) + ReLU  → (B, 128, 16, 16)
ConvTranspose2d(128→64)  + GroupNorm(8) + ReLU  → (B, 64, 32, 32)
ConvTranspose2d(64→32)   + GroupNorm(8) + ReLU  → (B, 32, 64, 64)
ConvTranspose2d(32→3)    + Sigmoid              → (B, 3, 128, 128)
```

### RSSM — Recurrent State-Space Model (T12, DreamerV3-Stil)

```
Ersetzt den früheren Temporal Transformer.
Persistenter Hidden-State h_t über die gesamte Episode.

GRUCell:
  Input:  cat(z_t, a_{t-1}, goal_proj) = 256 + 6 + 256 = 518-dim
  Hidden: h_t (256-dim)
  Output: h_{t+1} (256-dim) = neuer context

dynamics_head (T10):
  Input:  cat(h_t, a_t) = 256 + 6 = 262-dim
  → Linear(262, 512) + ReLU + Linear(512, 256)
  Output: z_{t+1} (256-dim) = vorhergesagter nächster Zustand

Training: Truncated BPTT über Sequenzen (SEQ_LEN = 8)
Episode-Grenze: reset_hidden_state() → GRU + prev_action auf Null
```

### Action Head

```
Input: context h_t (256-dim)

Linear(256, 256) + LayerNorm + ReLU + Dropout(0.1)
Linear(256, 128) + LayerNorm + ReLU

├── action_out: Linear(128, 6) + Tanh    → Aktion [-1, 1]
└── sigma_out:  Linear(128, 6) + Sigmoid → Unsicherheit [0, 1]
```

### Zusätzliche Köpfe

| Kopf | Input | Output | Zweck |
|------|-------|--------|-------|
| **reward_head** (T14) | cat(z, a) = 262-dim | Skalar [0,1] | Reward-Vorhersage ohne Gemini |
| **scene_head** (T13) | z = 256-dim | 8 Klassen | Szenen-Beschreibung (was sieht das Modell?) |
| **goal_proj** | CLIP 512-dim | 256-dim | Ziel-Embedding in Latent-Space projizieren |

### CLIP Text-Encoder (B05)

```
OpenAI CLIP (ViT-B/32)
Input:  Ziel-Text (z.B. "find the red box")
Output: 512-dim L2-normalisiertes Embedding

goal_proj: Linear(512→128) + ReLU + Linear(128→256) — Two-Stage Projektion
```

---

## Aktionsraum (6D kontinuierlich)

| Dimension | Bereich | Beschreibung |
|-----------|---------|-------------|
| linear_x | [-0.5, 0.5] m/s | Vorwärts/Rückwärts |
| angular_z | [-1.0, 1.0] rad/s | Drehung links/rechts |
| camera_pan | [-90°, +90°] | Kamera-Schwenk horizontal |
| camera_tilt | [-45°, +45°] | Kamera-Neigung vertikal |
| arc_radius | [-2.0, 2.0] m | Kurvenradius (0 = geradeaus) |
| duration | [0.1, 2.0] s | Aktionsdauer |

---

## Training Loop (Online Learning)

### Pro Schritt

```
1. obs_t → Encoder → (μ, log_var, z)
2. z + goal_proj + a_{t-1} → RSSM (GRU) → h_t (context)
3. h_t → dynamics_head(a_t) → pred_z_next → Decoder → pred_obs
4. h_t → Action Head → (action, σ)
5. r_intrinsic = MSE(pred_obs, actual_obs)
6. [Adaptiv] Gemini ER auf High-Res Bild → r_gemini
7. [T15] plan_action() → Imagination-Rollout → beste Aktion
8. ReplayBuffer.add(obs, action, reward, done, ...)
9. Wenn Buffer voll: _train_step() (Sequenz-basiert, BPTT über 8 Steps)
```

### Variational Free Energy (Verlustfunktion)

```
FE = 1.0  · L_recon              Rekonstruktionsfehler (MSE + SSIM)
   + β    · L_KL                 KL-Divergenz mit Free Bits (cosine annealing: 0 → 0.05)
   + 0.5  · L_pred_img           Nächst-Frame Prediction im Bildraum
   + 0.1  · L_next_z             Nächst-Latent Vorhersage (Hilfsziel)
   + 0.2  · L_action             T11: EFE-Blend(Imitation, -reward_pred)
   + 0.05 · L_sigma              NLL-kalibrierte Unsicherheit
   + 0.1  · L_goal               Ziel-Alignment (Cosinus-Ähnlichkeit)
   + 0.05 · L_cam_center         Kamera-Pan/Tilt Regularisierung
   + 0.1  · L_reward             T14: Reward-Prädiktor (MSE zu Gemini-Rewards)
   + 0.1  · L_scene              T13: Szenen-Beschreibung (Cross-Entropy)

Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
Scheduler: ReduceLROnPlateau (factor=0.5, patience=80, min_lr=1e-4)
Gradient Clipping: max_norm=1.0
```

### T11: EFE-basierte Aktionswahl

```
Statt reiner Imitation (MSE zu ausgeführten Aktionen):

efe_blend = 0.5 · min(gemini_count / 50, 1.0)    ← ramp-up mit Gemini-Daten

L_action = efe_blend · L_efe + (1 - efe_blend) · L_imitation
         = efe_blend · (-reward_head(z, predicted_a))
         + (1 - efe_blend) · MSE(predicted_a, executed_a)

→ ActionHead lernt: Aktionen wählen die vorhergesagten Reward maximieren
→ Imitation bleibt als Stabilisierungs-Term (Bootstrap)
```

### T15: Imagination (Planning-as-Inference)

```
Für jeden Aktions-Step (wenn reward_head trainiert):

1. Aktuelle Beobachtung → z_cur, h_cur (RSSM Hidden-State)
2. 32 Kandidaten-Aktionssequenzen sampeln (±σ um ActionHead-Vorschlag)
3. Für 5 Schritte vorausplanen:
   z_{t+1} = dynamics_head(h_t, a_t)         # Transition im Latent-Space
   h_{t+1} = GRU(z_{t+1}, a_t, goal)         # Hidden-State Update
   r_{t+1} = reward_head(z_{t+1}, a_t)       # Reward-Vorhersage
   cum_r  += 0.95^t · r_{t+1}                # Diskontierter Reward
4. Beste erste Aktion der Sequenz mit höchstem cum_r zurückgeben

Wichtig: RSSM._h wird NICHT verändert — GRU direkt mit kopiertem State.
Erst aktiv ab ≥50 Gemini-Samples (reward_head muss valide sein).
```

### Reward-Kombination

```
r_total = 0.3 · r_intrinsic           Vorhersagefehler (Neugier)
        + 0.4 · r_gemini              Semantische Bewertung
        + 0.2 · cos_sim(z, goal)      Ziel-Ähnlichkeit im Latent-Space
        + 0.1 · (1 - mean(σ))         Aktions-Konfidenz
```

---

## Gemini API Integration

### Drei Gemini-Komponenten

| Modul | Modell | Aufgabe |
|-------|--------|---------|
| **B13** Text | gemini-2.5-flash | Natürliche Sprache → CLIP-kompatibles Ziel |
| **B14** Adaptiv | — | Intelligente Aufruf-Frequenz (5–80 Steps) |
| **B15** Vision ER | gemini-robotics-er | Semantische Bild-Bewertung + Belohnung |

### Adaptive Aufruf-Steuerung (B14)

```
Dringlichkeit = 0.6 · u_fe + 0.2 · u_novelty + 0.2 · u_timeout

Intervall = max_interval · (1 - urgency) + min_interval · urgency

→ Früh: häufige Aufrufe (hoher Fehler)
→ Spät: seltene Aufrufe (Modell sicher) → 8-16× Kostenreduktion
```

### Wand-Filter

Einheitliche graue Flächen (Varianz < 200) werden nicht an Gemini gesendet —
spart API-Kosten und verhindert "unscharf"-Bewertungen.

---

## Strategie-System

### Generator (B22)
Gemini generiert regelbasierte Explorations-Strategien:

```
P100: target_close    → stop              (1 Step)
P95:  stuck           → move_backward     (3 Steps)
P93:  target_below    → tilt_down         (2 Steps)
P90:  target_left     → pan_right         (1 Step)
P90:  target_right    → pan_left          (1 Step)
P80:  target_centered → move_forward      (1 Step)
P75:  timeout         → move_forward      (10 Steps)
P50:  no_target       → turn_left         (4 Steps)
P60:  pan_done        → random_turn       (2 Steps)
```

### Executor (B23) — Sigma-basiertes Blending

```
mean_σ = mean(pred_sigma)        ← Unsicherheit des Action Head
blend  = sigmoid((mean_σ - 0.4) · 8.0)

final_action = blend · strategy_action + (1-blend) · nn_action

Hohe σ (>0.4) → Strategie dominiert (frühes Training)
Niedrige σ    → NN/Imagination übernimmt (nach Lernen)
```

Bei aktiver Imagination (T15) wird `nn_action` durch die geplante Aktion ersetzt.

---

## Visualisierung

### Training Dashboard (B18)
Echtzeit-Dashboard mit:
- **Live-Kamerabild** (NN-Input, 60×80) + letztes Gemini-Bild (128×128)
- **Vorhersage** des Decoders + Differenzbild
- **Kurven**: Free Energy, Rekonstruktion, KL-Divergenz
- **Rewards**: Intrinsisch, Gemini, Gesamt, Ziel-Fortschritt
- **Latent-Space**: PCA-Visualisierung der Encoder-Ausgaben
- **Gemini-Timeline**: Wann und wo semantische Bewertungen stattfanden
- **Aktions-Balken**: Aktuelle Aktion + Unsicherheit (σ)
- **Strategie-Info**: Aktive Strategie, Blend-Faktor, sigma-Mittelwert

### Overhead Map (OverheadMapView)
2D-Draufsicht mit:
- **Roboter-Position** als Pfeil (Richtung = Heading)
- **Bewegungs-Trail** mit Szenen-Färbung
- **Kamera-FOV**: Sichtfeld-Kegel (Pan + Tilt)
- **Szenen-Objekte**: Farbcodierte Marker (Boxen, Kugeln)
- **Gemini-Calls**: Gold-Diamanten an Aufruf-Positionen

---

## MiniWorld Umgebung

### PredictionWorld-OneRoom-v0

Eigene Gymnasium-Umgebung mit 6 Objekten:

| Objekt | Typ | Farbe |
|--------|-----|-------|
| Rote Box | Box | rot |
| Gelbe Box | Box | gelb |
| Weiße Box | Box | weiß |
| Orange Box | Box | orange |
| Grüne Kugel | Ball | grün |
| Blaue Kugel | Ball | blau |

---

## Logging

Drei CSV-Dateien pro Session in `logs/` (Timestamp-verknüpft):

| Datei | Inhalt | Frequenz |
|-------|--------|----------|
| `steps_{ts}.csv` | r_intr, r_gemini, sigma_mean, scene_pred, goal | Jeden Step |
| `train_{ts}.csv` | Alle Losses, Gradienten, efe_blend, lr, beta | Jeden Train-Step |
| `gemini_{ts}.csv` | reward, situation, recommendation, label | Jeden Gemini-Call |

Konsole: Nur Gemini-Events + alle 200 Train-Steps eine kompakte Zeile.

---

## Robot Interface (B17)

| Modus | Beschreibung |
|-------|-------------|
| **MiniWorld** | 3D-Simulation (Standard) |
| **ROS2** | Echte Roboter via `/cmd_vel` + `/camera/image_raw` |
| **Mock** | Synthetische Szenen zum Testen |

---

## Projektstruktur

| Datei | Beschreibung |
|-------|-------------|
| `B16FullIntegration.py` | **Vollständiges ML-System** — Encoder, Decoder, RSSM, ActionHead, reward/scene_head, EFE, Imagination |
| `B17RobotInterfaces.py` | Abstrakte Robot-Interfaces (Mock/MiniWorld/ROS2) |
| `B18Dashboard.py` | Echtzeit Training-Dashboard (matplotlib) |
| `B19Orchestrator.py` | Zentraler Orchestrator (ML + I/O + Visualisierung) |
| `B19OrchestratorModeMiniworld.py` | **Einstiegspunkt** — MiniWorld-Modus |
| `B20PreTrainVAE.py` | Offline VAE Pre-Training |
| `B21PreTrainCLIP.py` | Offline CLIP Goal-Projection Training |
| `B22StrategyGenerator.py` | Gemini-gesteuerte Strategie-Generierung |
| `B23StrategyExecutor.py` | Regelbasierter Strategie-Executor mit σ-Blending |
| `B24PreTrainDynamics.py` | Offline Dynamics-Head Pre-Training |
| `OverheadMapView.py` | 2D Overhead Map Visualisierung |
| `B10PredictionLoss.py` | Combined Reconstruction Loss (MSE + SSIM) |

### Standalone-Demos (B02–B09)

Einzelne Bausteine mit eigenen Demos (16×16 Bilder). Dienen zur Dokumentation,
die Runtime-Implementierungen sind in B16 integriert (128×128).

---

## Active Inference — Architektur-Mapping

```
Generatives Modell:    P(o_t | s_t)          ← Decoder (rekonstruiert/prediziert Bild)
                       P(s_t | s_{t-1}, a_t) ← RSSM dynamics_head (T10/T12)
                       P(a_t)                 ← Prior auf Aktionen (via Gemini)

Erkennungsmodell:      Q(s_t | o_t)           ← Encoder (VAE posterior)

Freie Energie (FE):    KL[Q(s)||P(s)]  +  E_Q[-log P(o|s)]
                       ↑ Complexity        ↑ Inaccuracy

Erwartete FE (EFE):    Epistemisch (σ-Kalibrierung) + Pragmatisch (reward_head)
                       → T11: ActionHead minimiert EFE statt nur Imitation

Imagination:           T15: Mehrstufige Rollouts im Latent-Space
                       → 32 Kandidaten × 5 Schritte → beste Aktion
```

| Komponente | Active Inference Begriff | Status |
|------------|------------------------|--------|
| Encoder (VAE) | Recognition model Q(s\|o) | ✅ |
| Decoder (Recon) | Generative model P(o\|s) | ✅ |
| Next-Frame Prediction | P(o_{t+1}\|s_t) | ✅ |
| RSSM (GRU) | Q(s_t\|o_{0:t}, a_{0:t}) | ✅ T12 |
| dynamics_head | P(s_{t+1}\|s_t, a_t) | ✅ T10 |
| KL + Free Bits | Complexity (FE-Term) | ✅ |
| Gemini-Reward | Pragmatischer EFE-Term | ✅ |
| Sigma-Unsicherheit | Epistemischer EFE-Term | ✅ |
| EFE-Aktionswahl | Aktionen minimieren EFE | ✅ T11 |
| Reward-Prädiktor | P(r\|s, a) | ✅ T14 |
| Semantik-Kopf | P(label\|s) | ✅ T13 |
| Imagination | Planning-as-Inference | ✅ T15 |

---

## Inspiriert von

- [Active Inference — This physics idea might be the next generation of ML](https://www.youtube.com/watch?v=MqDdYybN8o0)
- [Gemini Robotics ER (Embodied Reasoning)](https://deepmind.google/technologies/gemini/robotics-er/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [MiniWorld](https://github.com/maximecb/gym-miniworld)
- [DreamerV3](https://arxiv.org/abs/2301.04104) — RSSM World Model Architektur

---

## Lizenz

Dieses Projekt ist ein Forschungsprojekt.
