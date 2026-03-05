# AGENTS.md – Technische Referenz für PredictionWorldNet

> Diese Datei dient als Wissensbasis für AI-Agenten (Copilot, Cursor, etc.),
> um das Projekt schnell zu verstehen ohne alles neu zu erkunden.

---

## Projektübersicht

**PredictionWorldNet** ist ein Predictive World Model basierend auf Active Inference.
Ein neuronales Netz lernt vorherzusagen, was ein Roboter als nächstes sieht,
und leitet daraus Aktionen ab. Trainiert wird in einer MiniWorld-3D-Simulation
mit intrinsischer Neugier und semantischem Gemini-API-Feedback.

**Einstiegspunkt:** `B19OrchestratorModeMiniworld.py`
**Umgebung:** Python 3.11, Windows, PyTorch, MiniWorld + Gymnasium

---

## Globale Konstanten (B16FullIntegration.py)

| Konstante    | Wert           | Beschreibung                          |
|-------------|----------------|---------------------------------------|
| OBS_SHAPE   | (128, 128, 3)  | RGB-Bild des Roboters                 |
| LATENT_DIM  | 64             | Dimension des VAE-Latent-Raums        |
| D_MODEL     | 128            | Transformer Hidden Dimension          |
| ACTION_DIM  | 6              | Aktionsvektor-Dimension               |
| CLIP_DIM    | 512            | CLIP-Embedding-Dimension              |

**Aktionsvektor-Layout (6D):**
`[linear_x, angular_z, cam_pan, cam_tilt, arc_radius, duration]`

---

## NN-Architektur (alles in B16FullIntegration.py re-definiert)

### Encoder (VAE, 128×128 Input)

5× Conv2d stride-2: 128→64→32→16→8→4
Flatten: 128×4×4 = 2048 → fc_mu(64), fc_log_var(64) → z(64)

| Layer    | Channels | Output Spatial |
|----------|----------|---------------|
| Conv2d-1 | 3→32     | 64×64         |
| Conv2d-2 | 32→64    | 32×32         |
| Conv2d-3 | 64→128   | 16×16         |
| Conv2d-4 | 128→128  | 8×8           |
| Conv2d-5 | 128→128  | 4×4           |

Alle: kernel=3, stride=2, padding=1, BatchNorm, ReLU.
log_var wird auf [-10, 10] geclampt.

**Achtung:** B04bVariationalEncoder.py hat nur 3 Conv-Layer (für 16×16 Input).
B16 definiert seinen eigenen Encoder mit 5 Layers für 128×128.
Pretraining (B20/B21) nutzt den B16-Encoder.

### Decoder (128×128 Output)

Linear(64→2048) → Reshape(128,4,4)
5× ConvTranspose2d stride-2: 4→8→16→32→64→128, letztes Layer → Sigmoid

| Layer           | Channels | Output Spatial |
|-----------------|----------|---------------|
| ConvTranspose-1 | 128→128  | 8×8           |
| ConvTranspose-2 | 128→128  | 16×16         |
| ConvTranspose-3 | 128→64   | 32×32         |
| ConvTranspose-4 | 64→32    | 64×64         |
| ConvTranspose-5 | 32→3     | 128×128       |

### Temporal Transformer

Token-Sequenz: [CLS, current_z, goal_emb, history_slots...]
- CLS: Learnable Parameter (1, D_MODEL)
- current_z: proj_cur Linear(64→128)
- goal_emb: proj_goal Linear(512→128)
- History: proj_hist Linear(64+6+64=134→128) pro Zeitslot

Transformer: n_heads=4, n_layers=2, dim_ff=256, dropout=0.1, batch_first=True
Output: CLS-Token → context (128-dim)

Zeitkodierung: Einfach `t/T` als Skalar auf alle Latent-Dims.

### Action Head

context(128) → Linear(128→256)+LN+ReLU+Dropout → Linear(256→128)+LN+ReLU
→ action_out: Linear(128→6)+Tanh → action ∈ [-1, 1]
→ sigma_out:  Linear(128→6)+Sigmoid → sigma ∈ [0, 1]

### Goal Projection

`goal_proj = nn.Linear(512, 64, bias=False)`
Projiziert CLIP-Text-Embedding (512-dim) in den Latent-Raum (64-dim).

---

## Datenpipeline

```
MiniWorld Env (128×128 RGB)
    ↓
Encoder → z(64), mu(64), log_var(64)
    ↓
TemporalBuffer → z_hist(B,T,64), a_hist(B,T,6)
    ↓
Transformer(z_cur, goal_emb, z_hist, a_hist) → context(128)
    ↓
    ├→ Decoder(z) → pred_obs(128×128)  [Rekonstruktion]
    ├→ ActionHead(context) → action(6), sigma(6)
    └→ goal_progress = cos_sim(context[:,:64], goal_proj(clip_goal))
    
    ↓ Strategy-Blending (sigmoid auf sigma)
    ↓
Final Action → MiniWorld Step oder ROS2 Twist
```
---

## Reward-System

| Komponente      | Quelle          | Bereich | Gewicht (default) |
|----------------|-----------------|---------|-------------------|
| r_intrinsic    | Prediction Error (B12) | [0,1] | 0.3 |
| r_visual       | Decoder MSE     | [0,1]   | 0.4               |
| r_goal         | 1 - cos_sim     | [0,2]!  | 0.2               |
| r_action       | Action Smoothness | [0,1]  | 0.1               |
| r_gemini       | Gemini API      | [0,1]   | separat            |

**Bekannter Bug:** r_goal Bereich ist [0,2] statt [0,1] → siehe TODO T01.

### Gemini-Integration

- AdaptiveController (B14) berechnet Urgency aus: FE(0.5) + Novelty(0.3) + Timeout(0.2)
- Intervall: min=5, max=80 Steps (dynamisch via Urgency)
- Hochauflösendes Bild wird an Gemini gesendet, Antwort als Reward [0,1]
- Gemini gibt Situation + Empfehlung als Text zurück (deutsch konfiguriert)
- **Gemini-Sprache:** Deutsch. CLIP versteht aber nur Englisch → Labels in B21 sind auf Englisch.

---

## Strategy-System (B22/B23)

```
Gemini-Bewertung → StrategyGenerator.generate(goal) → Strategy(rules[])
    → StrategyExecutor.evaluate(obs) → matching Rule → strategy_action(6D)
    → blend(strategy_action, nn_action, sigma)
    → final_action
```
**Blending:** `factor = sigmoid((sigma - 0.4) * 8.0)`, clipped [0.1, 0.9]
- Hohe Unsicherheit (sigma>0.4) → mehr Strategie
- Niedrige Unsicherheit → mehr NN

**Conditions:** no_target, target_left/right/centered, target_close/far,
pan_done, stuck, timeout, always

---

## Dateien und Verantwortlichkeiten

| Datei | Rolle |
|-------|-------|
| B00/B01 | Render-Demos (nicht für Produktion) |
| B02 | ReplayBuffer (in B16 re-implementiert) |
| B03 | TemporalBuffer (deque-basiert) |
| B04b | VAE Encoder (Standalone, 16×16) |
| B05 | CLIP Text-Encoder (OpenAI CLIP ViT-B/32) |
| B06 | Action Embedding (Linear 6→128) |
| B07 | Temporal Transformer (Standalone) |
| B08 | CNN Decoder (Standalone, 16×16) |
| B09 | Action Head (Standalone) |
| B10 | PredictionLoss + Free Energy |
| B11 | TrainingLoop (veraltet, Mock) |
| B12 | IntrinsicReward (Prediction Error + k-NN Novelty) |
| B13 | GeminiApi (Wrapper für google-genai) |
| B14 | AdaptiveGemini (Urgency-basierte Call-Frequenz) |
| B15 | RewardCombination (gewichtete Summe) |
| **B16** | **FullIntegration** (eigene Encoder/Decoder/Transformer/ActionHead für 128×128) |
| B17 | RobotInterfaces (Abstraction für MiniWorld/ROS2/Mock) |
| **B18** | **Dashboard** (matplotlib TkAgg, 3×6 Grid) |
| **B19** | **Orchestrator** (verbindet alles) |
| **B19...Miniworld** | **Entry Point** (MiniWorld-Modus) |
| B20 | PreTrain VAE (sammelt Frames, trainiert Encoder+Decoder) |
| B21 | PreTrain CLIP (trainiert goal_proj mit Bild-Text-Paaren) |
| B22 | StrategyGenerator (Gemini-basiert oder Mock) |
| B23 | StrategyExecutor (Regel-Matching + Sigma-Blending) |
| OverheadMapView | 2D-Draufsicht mit Trail, Entities, Heading-Kurve |

**Wichtig:** B04b–B11 sind Standalone-Module mit eigenen Demos.
B16 re-implementiert die Kernkomponenten für 128×128 Input.
Zur Laufzeit werden die B16-Versionen verwendet.

---

## MiniWorld-Koordinatensystem

- `dir_vec = (cos(dir), 0, -sin(dir))` → Vorwärtsrichtung
- Map-Mapping: `map_x = MW.x`, `map_y = -MW.z` (Z negiert!)
- `agent.dir` steigend = CCW = nach links schauen
- Camera-Pan: `agent.dir = original_dir - cam_pan` (Subtraktion!)
- Heading wird normalisiert: `agent.dir % (2π)`

---

## MiniWorld Entity-Farbsystem

- `Box.color` ist ein **String** (z.B. "red", "yellow") — kein RGB-Tupel
- `Ball` hat **kein** `.color` und **kein** `.mesh_name` Attribut
- Farbe bei Ball nur über `ObjMesh.cache`-Keys rekonstruierbar
  (Dateipfad enthält z.B. `ball_green.obj`)
- Farbauflösungs-Reihenfolge in OverheadMapView:
  1. String-Attribut → `_mw_colors`-Dict Lookup
  2. RGB-Array → Hex-Konvertierung
  3. `ObjMesh.cache`-Key → Farbnamen-Suche
  4. Fallback: #ffffff

---

## Custom Environment: PredictionWorld-OneRoom-v0

Registriert in `_register_pw_env()` (existiert in B19Orchestrator, B19...Miniworld, B20, B21).
**Achtung:** Wird 4× dupliziert definiert — Refactoring in gemeinsames Modul ausstehend.

**Objekte im Raum:**
- Box(color="red") — Hauptziel
- Box(color="yellow")
- Box(color="white")
- Box(color="orange")
- Ball(color="green")
- Ball(color="blue")

Zusätzliche Farben registriert: `COLORS["orange"]`, `COLORS["white"]`

---

## OverheadMapView Trail-Struktur

4-Tupel in deque: `(x, y, heading, scene)`
- Index 0,1: Position
- Index 2: Heading in Radians [0, 2π]
- Index 3: Scene-String

---

## Dashboard Update-Signatur (B18)

```python
dashboard.update(
    obs=np.ndarray,           # (128,128,3) oder (60,80,3)
    pred=np.ndarray,          # (128,128,3) Predicted Frame
    metrics=dict,             # Keys: fe, recon, kl, r_intrinsic, r_gemini, r_total,
                              #        goal_progress, beta, lr, gemini_interval
    gemini_event=dict|None,   # reward, goal_progress, situation, recommendation, image
    goal=str,                 # Aktuelles Ziel
    scene=str,                # Aktueller Szenentyp
    latent_z=np.ndarray|None, # (64,) Latent-Embedding für PCA
)
```
PCA-Visualisierung braucht ≥5 Datenpunkte bevor sie rendert.

---

## Checkpoint-Format (.pt)

```python
{
    "encoder":      state_dict,     # B16 Encoder
    "decoder":      state_dict,     # B16 Decoder
    "goal_proj":    state_dict,     # Linear(512→64), nach CLIP-Training
    "total_steps":  int,
    "train_steps":  int,
    "beta":         float,
    "current_goal": str,            # "pretrain_vae" | "pretrain_clip" | ...
    "tag":          str,
    "config":       dict,
    "constants":    {"LATENT_DIM": 64, "D_MODEL": 128, "ACTION_DIM": 6},
    "result":       dict,           # VAE Training-Ergebnisse
    "result_clip":  dict,           # CLIP Training-Ergebnisse
}
```
B20 und B21 laden bestehenden Checkpoint und aktualisieren nur ihre Keys.
Reihenfolge: B20 (VAE) → B21 (CLIP) → B19 (Live).

---

## Dependencies (requirements.txt)

```
gymnasium, miniworld (git), matplotlib, numpy
torch, torchvision, timm
CLIP (git+https://github.com/openai/CLIP.git)
google-genai, Pillow, pyopengl
```
**API-Key:** `GOOGLE_API_KEY` für Gemini.

---

## Bekannte Probleme / TODOs

Siehe `TODO.md` für priorisierte Verbesserungsliste (T01–T11).
Wichtigste offene Punkte:
- T01: r_goal Bereich [0,2] statt [0,1] (Reward-Imbalance)
- T03: Transformer-Loss nicht task-aligned
- `_register_pw_env()` 4× dupliziert
- B04b/B07/B08/B09 Standalone-Module divergieren von B16-Versionen
