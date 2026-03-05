# 🧠 PredictionWorldNet

**Predictive World Model basierend auf Active Inference — eine Alternative zu klassischem Reinforcement Learning**

> *Intelligenz bedeutet, die eigenen Vorhersagen über die Welt zu bestätigen.*

PredictionWorldNet implementiert ein **Predictive World Model**, das auf den Prinzipien der **Active Inference** basiert.
Statt passiv auf Belohnungen zu reagieren, antizipiert der Agent aktiv die Realität und minimiert die
**Variational Free Energy** — den Unterschied zwischen seinen Vorhersagen und eingehenden Sensordaten.

Das System kombiniert ein schnelles, niedrig aufgelöstes neuronales Netz (lokales Lernen) mit
hochauflösenden **Gemini Vision-Bewertungen** (semantische Belohnungen) für effizientes, erklärbares Lernen.

---

## 🎯 Kernkonzepte

| Konzept | Beschreibung |
|---------|-------------|
| **Active Inference** | Der Agent antizipiert aktiv die Realität statt passiv auf Rewards zu reagieren |
| **Markov Blanket** | Klare Grenze zwischen Wahrnehmung (Sensoren) und Aktion (Motoren) |
| **Generatives Modell** | Explizites Weltmodell, das durch Vorhersage lernt — nicht durch Reward-Chasing |
| **Variational Free Energy** | Einziges Optimierungsprinzip: Vorhersagefehler minimieren |
| **Self-Evidencing** | Intelligenz = Fähigkeit, eigene Vorhersagen über die Welt zu bestätigen |

---

## 🏗️ Architektur-Übersicht

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PredictionWorldNet                           │
│                                                                      │
│  ┌─────────┐    ┌───────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ MiniWorld│───▶│  Encoder  │───▶│  Transformer │───▶│  Decoder  │  │
│  │   Env   │    │   (VAE)   │    │  (Temporal)  │    │   (CNN)   │  │
│  └────┬────┘    └─────┬─────┘    └──────┬───────┘    └─────┬─────┘  │
│       │               │                 │                   │        │
│       │          z (64-dim)        context (128-dim)   pred_obs      │
│       │               │                 │                            │
│       │               │          ┌──────┴───────┐                    │
│       │               │          │  Action Head │                    │
│       │               │          │  (6D + σ)    │                    │
│       │               │          └──────┬───────┘                    │
│       │               │                 │                            │
│       ▼               ▼                 ▼                            │
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

## 🧬 Neuronales Netz — Detailarchitektur

### Dimensionen

| Parameter | Wert | Beschreibung |
|-----------|------|-------------|
| LATENT_DIM | 64 | Komprimierte Repräsentation |
| ACTION_DIM | 6 | Aktionsvektor-Dimensionen |
| D_MODEL | 128 | Transformer Embedding-Dimension |
| OBS_SHAPE | 128×128×3 | Beobachtungs-Auflösung |

### Encoder (B04b — Variational Autoencoder)

```
Input: (B, 3, 128, 128) — RGB-Bild

Conv2d(3→32, 4×4, stride=2)  + BatchNorm + ReLU    → (B, 32, 64, 64)
Conv2d(32→64, 4×4, stride=2) + BatchNorm + ReLU    → (B, 64, 32, 32)
Conv2d(64→128, 4×4, stride=2)+ BatchNorm + ReLU    → (B, 128, 16, 16)
Conv2d(128→128, 4×4, stride=2)+ BatchNorm + ReLU   → (B, 128, 8, 8)
Conv2d(128→128, 4×4, stride=2)+ BatchNorm + ReLU   → (B, 128, 4, 4)

Flatten → 2048-dim
├── fc_mu:      Linear(2048, 64)  → μ
└── fc_log_var: Linear(2048, 64)  → log(σ²)

Reparametrisierung: z = μ + ε·σ,  ε ~ N(0, I)
```

### Decoder (B08 — CNN)

```
Input: z (B, 64)

Linear(64, 2048) + ReLU → Reshape (B, 128, 4, 4)

ConvTranspose2d(128→128) + BatchNorm + ReLU  → (B, 128, 8, 8)
ConvTranspose2d(128→128) + BatchNorm + ReLU  → (B, 128, 16, 16)
ConvTranspose2d(128→64)  + BatchNorm + ReLU  → (B, 64, 32, 32)
ConvTranspose2d(64→32)   + BatchNorm + ReLU  → (B, 32, 64, 64)
ConvTranspose2d(32→3)    + Sigmoid           → (B, 3, 128, 128)
```

### Temporal Transformer (B07)

```
Tokens:
  [CLS]          — lernbarer Summary-Token (1, 128)
  z_current      — Linear(64, 128)
  goal_embedding — Linear(512, 128) via CLIP
  z_history[T]   — Linear(64+6+64, 128) mit Positional Encoding

4 Attention Heads, 2 Encoder Layers
dim_feedforward = 256, dropout = 0.1

Output: CLS-Token → context (128-dim)
```

### Action Head (B09)

```
Input: context (128-dim)

Linear(128, 256) + LayerNorm + ReLU + Dropout(0.1)
Linear(256, 128) + LayerNorm + ReLU

├── action_out: Linear(128, 6) + Tanh    → Aktion [-1, 1]
└── sigma_out:  Linear(128, 6) + Sigmoid → Unsicherheit [0, 1]
```

### CLIP Text-Encoder (B05)

```
OpenAI CLIP (ViT-B/32)
Input:  Ziel-Text (z.B. "find the red box")
Output: 512-dim L2-normalisiertes Embedding

goal_proj: Linear(512, 64) — projiziert in Latent-Space
```

---

## ⚡ Aktionsraum (6D kontinuierlich)

| Dimension | Bereich | Beschreibung |
|-----------|---------|-------------|
| linear_x | [-0.5, 0.5] m/s | Vorwärts/Rückwärts |
| ngular_z | [-1.0, 1.0] rad/s | Drehung links/rechts |
| camera_pan | [-90°, +90°] | Kamera-Schwenk horizontal |
| camera_tilt | [-45°, +45°] | Kamera-Neigung vertikal |
| rc_radius | [-2.0, 2.0] m | Kurvenradius (0 = geradeaus) |
| duration | [0.1, 2.0] s | Aktionsdauer |

---

## 🔄 Training Loop (Online Learning)

### Pro Schritt

```
1. obs_t → Encoder → (μ, log_var, z)
2. z + goal_emb + history → Transformer → context
3. context → Decoder → pred_obs
4. context → Action Head → (action, σ)
5. r_intrinsic = MSE(pred_obs, actual_obs)
6. [Adaptiv] Gemini ER auf High-Res Bild → r_gemini
7. ReplayBuffer.add(obs, action, reward, ...)
8. Wenn Buffer voll: _train_step()
```

### Variational Free Energy (Verlustfunktion)

```
FE = L_recon                          Rekonstruktionsfehler
   + β · L_KL                         KL-Divergenz (annealing: 0 → 0.05)
   + 0.3 · L_temporal                 Zeitliche Konsistenz
   + 0.2 · L_action                   Aktionsvorhersage
   + 0.1 · L_goal                     Ziel-Alignment (Cosinus-Ähnlichkeit)
   + 0.2 · L_gemini                   Gemini-gewichtete Rekonstruktion

Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
Scheduler: ReduceLROnPlateau (factor=0.5, patience=80)
Gradient Clipping: max_norm=1.0
```

### Reward-Kombination

```
r_total = 0.3 · r_intrinsic           Vorhersagefehler (Neugier)
        + 0.4 · r_gemini              Semantische Bewertung
        + 0.2 · cos_sim(z, goal)      Ziel-Ähnlichkeit im Latent-Space
        + 0.1 · (1 - mean(σ))         Aktions-Konfidenz
```

---

## 🤖 Gemini API Integration

### Drei Gemini-Komponenten

| Modul | Modell | Aufgabe |
|-------|--------|---------|
| **B13** Text | gemini-2.5-flash | Natürliche Sprache → CLIP-kompatibles Ziel |
| **B14** Adaptiv | — | Intelligente Aufruf-Frequenz (5–80 Steps) |
| **B15** Vision ER | gemini-robotics-er | Semantische Bild-Bewertung + Belohnung |

### Adaptive Aufruf-Steuerung (B14)

```
Dringlichkeit = 0.6 · u_fe + 0.2 · u_novelty + 0.2 · u_timeout

u_fe      = clip((FE_ema - FE_low) / (FE_thresh - FE_low))
u_novelty = clip(novelty / novelty_thresh)
u_timeout = clip(steps_since_call / max_interval)

Intervall = max_interval · (1 - urgency) + min_interval · urgency

→ Früh: häufige Aufrufe (hoher Fehler)
→ Spät: seltene Aufrufe (Modell sicher) → 8-16× Kostenreduktion
```

### Gemini ER Ausgabe

```json
{
  "reward": 0.0-1.0,
  "goal_progress": 0.0-1.0,
  "situation": "Rote Box links sichtbar",
  "recommendation": "Nach links drehen und vorwärts",
  "training_label": "red_box_visible"
}
```

---

## 🎮 Strategie-System

### Generator (B22)
Gemini generiert regelbasierte Explorations-Strategien:

```
P100: target_close    → stop              (1 Step)
P95:  stuck           → move_backward     (3 Steps)
P90:  target_left     → pan_right         (1 Step)
P90:  target_right    → pan_left          (1 Step)
P80:  target_centered → move_forward      (1 Step)
P70:  no_target       → scan_panorama     (1 Step)
P60:  pan_done        → random_turn       (2 Steps)
```

### Executor (B23) — Sigma-basiertes Blending

```
mean_σ = mean(pred_sigma)        ← Unsicherheit des Action Head
blend  = sigmoid((mean_σ - threshold) · steepness)

final_action = blend · strategy_action + (1-blend) · nn_action

Hohe σ → Strategie dominiert (Anfang)
Niedrige σ → NN übernimmt (nach Lernen)
```

---

## 🗺️ Visualisierung

### Training Dashboard (B18)
Echtzeit-Dashboard mit:
- **Live-Kamerabild** (NN-Input, 60×80) + letztes Gemini-Bild (128×128)
- **Vorhersage** des Decoders + Differenzbild
- **Kurven**: Free Energy, Rekonstruktion, KL-Divergenz
- **Rewards**: Intrinsisch, Gemini, Gesamt, Ziel-Fortschritt
- **Latent-Space**: PCA-Visualisierung der Encoder-Ausgaben
- **Gemini-Timeline**: Wann und wo semantische Bewertungen stattfanden
- **Aktions-Balken**: Aktuelle Aktion + Unsicherheit (σ)

### Overhead Map (OverheadMapView)
2D-Draufsicht mit:
- **Roboter-Position** als Pfeil (Richtung = Heading)
- **Bewegungs-Trail** mit Szenen-Färbung
- **Kamera-FOV**: Sichtfeld-Kegel (Pan + Tilt)
- **Szenen-Objekte**: Farbcodierte Marker (Boxen ■, Kugeln ●)
- **Gemini-Calls**: Gold-Diamanten an Aufruf-Positionen
- **Statistiken**: Distanz, Rotation, Heading-Kurve

---

## 🌍 MiniWorld Umgebung

### PredictionWorld-OneRoom-v0

Eigene Gymnasium-Umgebung mit 6 Objekten:

| Objekt | Typ | Farbe |
|--------|-----|-------|
| 🟥 Rote Box | Box | rot |
| 🟨 Gelbe Box | Box | gelb |
| ⬜ Weiße Box | Box | weiß |
| 🟧 Orange Box | Box | orange |
| 🟢 Grüne Kugel | Ball | grün |
| 🔵 Blaue Kugel | Ball | blau |

Alle Objekte werden zufällig im Raum platziert und sind im Pretraining gelabelt.

---

## 📚 Pretraining Pipeline

### Schritt 1: VAE Pre-Training (B20)

```bash
python B20PreTrainVAE.py --source miniworld --epochs 50
```

Lernt Bildkompression (Encoder + Decoder) auf zufälligen MiniWorld-Frames.

### Schritt 2: CLIP Goal-Projection (B21)

```bash
python B21PreTrainCLIP.py --vae-checkpoint checkpoints/pwn_*.pt --epochs 60
```

Trainiert `goal_proj` (512→64) via kontrastivem Loss:
- Auto-Labels aus Farbheuristiken (rot/grün/blau/gelb/orange/weiß + Wand + leer)
- Aligniert CLIP-Text-Embeddings mit VAE-Latent-Space

### Schritt 3: Live-Training (B19)

```bash
python B19OrchestratorModeMiniworld.py
```

Lädt Checkpoint und startet Online-Learning mit Dashboard + Overhead Map.

---

## 🔌 Robot Interface (B17)

| Modus | Beschreibung |
|-------|-------------|
| **MiniWorld** | 3D-Simulation (Standard) |
| **ROS2** | Echte Roboter via `/cmd_vel` + `/camera/image_raw` |
| **Mock** | Synthetische Szenen zum Testen |

```
ObsSource (abstrakt)          ActionSink (abstrakt)
├── MiniWorldObsSource        ├── MiniWorldActionSink
├── ROS2ObsSource             ├── ROS2ActionSink
└── MockObsSource             └── MockActionSink
```

---

## 📁 Projektstruktur

| Datei | Beschreibung |
|-------|-------------|
| `B02ReplayBuffer.py` | Zirkulärer Replay Buffer für Experience Replay |
| `B03TemporalBuffer.py` | Zeitliche Pufferung von Beobachtungs-Sequenzen |
| `B04bVariationalEncoder.py` | VAE Encoder (CNN → μ, log_var, z) |
| `B05ClipTextEncoder.py` | CLIP Text-Encoder für Ziel-Embeddings |
| `B06ActionEmbedding.py` | Aktions-Einbettung für Transformer-Input |
| `B07TemporalTransformer.py` | Multi-Head Attention mit temporaler Historie |
| `B08CnnDecoder.py` | CNN Decoder (z → rekonstruiertes Bild) |
| `B09ActionHead.py` | Aktions-Vorhersage + Unsicherheits-Schätzung (σ) |
| `B10PredictionLoss.py` | Variational Free Energy Verlustfunktion |
| `B11TrainingLoop.py` | Online-Learning Trainingsschleife |
| `B12IntrinsicReward.py` | Curiosity-basierte intrinsische Belohnung |
| `B13GeminiApi.py` | Gemini API Wrapper (Text + Vision) |
| `B14AdaptiveGemini.py` | Adaptive Aufruf-Frequenz-Steuerung |
| `B15RewardCombination.py` | Multi-Reward Kombination + Gemini-Gewichtung |
| `B16FullIntegration.py` | Vollständiges ML-System (alle Komponenten) |
| `B17RobotInterfaces.py` | Abstrakte Robot-Interfaces (Mock/MiniWorld/ROS2) |
| `B18Dashboard.py` | Echtzeit Training-Dashboard (matplotlib) |
| `B19Orchestrator.py` | Zentraler Orchestrator (ML + I/O + Visualisierung) |
| `B19OrchestratorModeMiniworld.py` | **Einstiegspunkt** — MiniWorld-Modus |
| `B20PreTrainVAE.py` | Offline VAE Pre-Training |
| `B21PreTrainCLIP.py` | Offline CLIP Goal-Projection Training |
| `B22StrategyGenerator.py` | Gemini-gesteuerte Strategie-Generierung |
| `B23StrategyExecutor.py` | Regelbasierter Strategie-Executor mit σ-Blending |
| `OverheadMapView.py` | 2D Overhead Map Visualisierung |

---

## 🚀 Schnellstart

```bash
# 1. Abhängigkeiten installieren
pip install -r requirements.txt

# 2. (Optional) VAE vortrainieren
python B20PreTrainVAE.py --source miniworld --epochs 50

# 3. (Optional) CLIP Goal-Projection trainieren
python B21PreTrainCLIP.py --vae-checkpoint checkpoints/pwn_*.pt --epochs 60

# 4. Simulation starten
python B19OrchestratorModeMiniworld.py
```

---

## 💡 Inspiriert von

- [Active Inference — This physics idea might be the next generation of ML](https://www.youtube.com/watch?v=MqDdYybN8o0)
- [Gemini Robotics ER (Embodied Reasoning)](https://deepmind.google/technologies/gemini/robotics-er/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [MiniWorld](https://github.com/maximecb/gym-miniworld)

---

## �� Lizenz

Dieses Projekt ist ein Forschungsprojekt.

