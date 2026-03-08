# DONE – Verbesserungsplan PredictionWorldNet

Trage hier erledigte Punkte aus der TODO.md ein.

### T01 – Reward-Normalisierung (B15) ✅ ERLEDIGT
**Problem:** Der Goal-Reward basiert auf `1 - cos_sim`, Wertebereich [0, 2].
Die gewichtete Summe in `RewardCombinator` (Z. 321–326) behandelt aber alle
Rewards als [0, 1]. Dadurch dominiert der Goal-Reward unbeabsichtigt.

**Lösung:**
- [x] `r_goal` auf [0, 1] gemappt: `(cos_sim + 1) / 2` in B16
- [x] `r_intr` auf [0, 1] geclampt: `min(r_intr, 1.0)` in B16
- [x] `l_goal` Training-Loss mit `torch.clamp(..., 0, 1)` in B16 + B10
- [x] Alle Reward-Inputs in B15 `combine()` auf [0,1] geclampt

**Dateien:** `B16FullIntegration.py`, `B15RewardCombination.py`, `B10PredictionLoss.py`

---

### T02 – Adaptive Gemini-Frequenz durchgängig nutzen (B14 ↔ B16) ✅ BEREITS IMPLEMENTIERT
**Problem:** `AdaptiveController` (B14) berechnet Urgency-basierte Intervalle
(Z. 148–159), aber der Orchestrator nutzt teilweise noch feste Intervalle.

**Befund:** B16 hat bereits einen eigenen `AdaptiveController` der alle
Gemini-Entscheidungen trifft. Es gibt keine festen Intervalle. B14 wird
als Standalone-Modul nicht importiert — B16 re-implementiert eine
vereinfachte Version. Kein Code-Änderung nötig.

**Dateien:** Keine Änderungen

---

### T03 – Temporal Transformer Loss ersetzen (B07) ✅ ERLEDIGT
**Problem:** Die Loss-Funktion (Norm + Cosine-Consistency) hat keinen
direkten Bezug zur eigentlichen Aufgabe (Action-Prediction / Next-Frame).
Nur 2 Layer × 4 Heads ist flach für 8 Token-Slots.

**Lösung:**
- [x] Loss auf Next-Latent-Prediction umgestellt:
  `l_next_z = MSE(next_z_head(context), z_next.detach())`
- [x] Layer von 2 → 3 erhöht
- [x] `norm_first=True` für stabileres Training (Pre-Norm)
- [x] `next_z_head = Linear(d_model → latent_dim)` zum Transformer hinzugefügt

**Dateien:** `B16FullIntegration.py`, `B07TemporalTransformer.py`

---

## 🟡 Priorität 2 – Mittel

### T04 – Action Head: Sigma-Loss kalibrieren (B09) ✅ ERLEDIGT
**Problem:** Sigma-Loss nutzt MSE gegen `err_norm` (Z. 364–365).
Korrekte NLL-Formulierung: `-log σ + |error| / σ` liefert kalibrierte
Unsicherheitsschätzung.

**Lösung:**
- [x] Sigma-Loss auf NLL umgestellt:
  `loss_sigma = torch.mean(torch.log(sigma) + err / sigma)`
- [x] B16: Sigma-Loss (0.05 Gewicht) in Free-Energy-Term hinzugefügt
  (war vorher gar nicht im Training!)
- [x] B09: Demo-Loss ebenfalls auf NLL umgestellt
- [x] `sigma_safe = torch.clamp(pred_sigma, min=1e-4)` für numerische Stabilität

**Dateien:** `B16FullIntegration.py`, `B09ActionHead.py`

---

### T05 – Beta-Annealing verbessern (B10) ✅ ERLEDIGT
**Problem:** Lineares Annealing (Z. 183–188) kann bei kleinen Modellen
zu schnell ansteigen → Posterior Collapse. Beta startet bei 0.0, aber
der Warmup ist linear.

**Lösung:**
- [x] Cosine-Annealing implementiert:
  `beta = beta_max * 0.5 * (1 - cos(π * t / warmup))`
- [x] In B10 `anneal_beta()` und B16 `step()` gleichzeitig umgestellt
- [x] KL-Collapse-Warnung bei KL < 0.1 nats nach Step 50

**Dateien:** `B10PredictionLoss.py`, `B16FullIntegration.py`

---

### T06 – Decoder: Perceptual Loss ergänzen (B08) ✅ ERLEDIGT
**Problem:** Reiner MSE-Loss auf Pixel-Ebene ist perceptuell blind —
verschwommene Rekonstruktionen werden akzeptiert, solange der
Durchschnittsfehler stimmt.

**Lösung:**
- [x] SSIM-Loss als zweite Komponente in B10 implementiert
  (`ssim()` + `combined_recon_loss()`)
- [x] Gewichtung: `L = 0.7 * MSE + 0.3 * (1 - SSIM)`
- [x] B16 `l_recon` nutzt jetzt `combined_recon_loss()`
- [x] B08 `free_energy_loss()` ebenfalls aktualisiert

**Dateien:** `B10PredictionLoss.py`, `B16FullIntegration.py`, `B08CnnDecoder.py`

---

### T07 – Strategy-Executor: Farberkennung robuster (B23) ✅ ERLEDIGT
**Problem:** Hardcodierte RGB-Schwellen für Farberkennung funktionieren nur
in synthetischen Szenen ohne Beleuchtungsvariation.

**Lösung:**
- [x] HSV-Farbraum statt RGB: robuster gegen Helligkeit
- [x] Konfigurierbare `_HSV_RANGES` Dict pro Farbe
- [x] Orange und White als explizite Farben hinzugefügt
- [x] `set_target_color()` um orange/white/weiß erweitert

**Dateien:** `B23StrategyExecutor.py`

---

## 🟢 Priorität 3 – Nice-to-Have

### T08 – Intrinsic Reward: Memory vergrößern (B12) ✅ ERLEDIGT
**Problem:** k-NN mit 200 Samples in 64-dim Raum ist statistisch dünn.
Novelty-Schätzungen haben hohe Varianz.

**Lösung:**
- [x] Memory von 200 → 1000 erhöht (Default + Demo)
- [ ] Zeitliches Decay einführen (ältere Einträge abwerten) — optional
- [ ] Optional: Locality-Sensitive Hashing für O(1) — nicht nötig bei 1000

**Dateien:** `B12IntrinsicReward.py`

---

### T09 – Goal-Projektion: CLIP-Bottleneck reduzieren ✅ ERLEDIGT
**Problem:** `goal_proj = Linear(512 → 64)` ist ein extremer
Informations-Bottleneck. CLIP-Embeddings verlieren semantische Nuancen.

**Lösung:**
- [x] Zweistufige Projektion: `Sequential(Linear(512→128), ReLU, Linear(128→64))`
- [x] In B16 und B10 gleichzeitig umgestellt
- [x] Checkpoint-Loader fängt Inkompatibilität ab (try/except)

**Dateien:** `B16FullIntegration.py`, `B10PredictionLoss.py`

---

### T10 – Gradient Monitoring & Loss-Balancing ✅ ERLEDIGT
**Problem:** Keine Überwachung der Gradientenmagnitude.
Loss-Gewichte sind fest — können zu Imbalance führen wenn eine
Komponente dominiert.

**Lösung:**
- [x] Gradient-Norm pro Modul loggen (Encoder, Decoder, Transformer,
  ActionHead, GoalProj) — in `_train_step()` Rückgabe als `grad_norms`
- [ ] Automatisches Loss-Balancing: `w_i = 1 / var(L_i)` — optional, später
- [x] Gradient Clipping bei max_norm=1.0 bereits vorhanden und validiert

**Dateien:** `B16FullIntegration.py`

---

### T11 – Decoder: Skip-Connections (U-Net Stil) ⛔ ÜBERSPRUNGEN
**Problem:** Ohne Skip-Connections gehen räumliche Details
im Bottleneck verloren.

**Bewertung:** Übersprungen — höchstes Architektur-Risiko:
- Bricht alle Encoder/Decoder-Aufrufe (B19, B20, B21)
- VAE soll abstrahieren — Skip-Connections konterkarieren das
- T06 (SSIM-Loss) adressiert das Problem der unscharfen Rekonstruktionen
  bereits auf sanftere Weise

**Dateien:** Keine Änderungen

---

---

### T10 (neu) – Aktions-konditioniertes Transitions-Modell ✅ ERLEDIGT
**Problem:** `pred_z_next = next_z_head(context)` war action-agnostisch.
Der Gradient wusste nicht, wie verschiedene Aktionen unterschiedliche
Zukünfte erzeugen.

**Lösung:**
- [x] `next_z_head` durch `dynamics_head` ersetzt:
  ```python
  dynamics_head = nn.Sequential(
      nn.Linear(D_MODEL + ACTION_DIM, 256), nn.ReLU(True),
      nn.Linear(256, LATENT_DIM)
  )
  ```
- [x] Neue Methode `predict_next_z(context, action)` im `TemporalTransformer`
- [x] In `step()`: echte Aktion `action_np` übergeben (action-conditioned inference)
- [x] In `_train_step()`: Buffer-Aktion `acts` übergeben (action-conditioned training)
- [x] Checkpoint-Migration: partial-load wenn alter Checkpoint `next_z_head` hat
- [x] Strukturiertes CSV-Logging: `logs/steps_*.csv`, `logs/train_*.csv`, `logs/gemini_*.csv`
  (über `total_step` verknüpfbar, kein langer Konsolen-Spam)
- [x] `run_demo()` Headless-Modus: `python B16FullIntegration.py --headless --steps=N`

**Ergebnis:** Modell lernt P(z_{t+1} | z_t, a_t) – echter World-Model-Kern.
dynamics_head hat 51.008 Parameter (D_MODEL+ACTION_DIM=134 → 256 → LATENT_DIM).

**Dateien:** `B16FullIntegration.py`

---

---

### T14 (neu) – Reward-Prädiktor im latenten Raum ✅ ERLEDIGT

**Lösung:**
- [x] `reward_head = Sequential(Linear(LATENT_DIM+ACTION_DIM, 128), ReLU, Linear(128,1), Sigmoid)`
- [x] Separate Gemini-Stichprobe: `sample(gem_n, require_gemini=True)` – unabhängig von
  der Gemini-Dichte im Haupt-Batch, zuverlässig ab 2 Gemini-Samples im Buffer
- [x] `l_reward = F.mse_loss(reward_head(cat([z_rb.detach(), acts])), gemini_rewards)`
  z.detach(): kein Gradient zurück durch Encoder (Reward-Head nutzt den Latent-Raum, verbiegt ihn nicht)
- [x] `r_reward_pred` in `step()` (Inference ohne Gemini-Aufruf)
- [x] Gewicht: `0.1 * l_reward` in FE-Summe
- [x] Checkpoint: `reward_head` im state_dict; Migrations-Load für ältere Checkpoints
- [x] CSV-Logging: `l_reward` in `train_*.csv`, `r_reward_pred` in `steps_*.csv`

**Ergebnis:** Modell schätzt Reward für beliebige (z, a)-Paare.
Baustein für T11 (EFE) und T15 (Imagination).

**Dateien:** `B16FullIntegration.py`

---

### T13 (neu) – Semantischer Selbstbeschreibungs-Kopf ✅ ERLEDIGT

**Lösung:**
- [x] `SCENE_VOCAB` (8 Klassen) + `SCENE_LABEL_MAP` (Gemini-Labels → Index) als Konstanten
- [x] `scene_head = Sequential(Linear(64→128), ReLU, Linear(128→8))`
- [x] `_label_to_vocab_idx(label)` Helfer: Schlüsselwort-Matching (Deutsch + Englisch)
- [x] Training: zufällige Stichprobe aus Gemini-gelabelten Buffer-Einträgen (`require_gemini=True`)
- [x] `l_scene = cross_entropy(scene_head(z.detach()), label_indices)` – z.detach() schützt Encoder
- [x] Gewicht: `0.1 * l_scene` in FE-Summe
- [x] `scene_pred` (Vokabular-String) in `step()` + in `steps_*.csv`
- [x] Checkpoint: `scene_head` im state_dict

**Verhalten:** Braucht viele Gemini-Calls zum Konvergieren (jede Szene mehrfach besucht).
In 120-Step-Demo: l_scene fällt von 2.05 auf ~1.3 mit nur 3 Gemini-Samples.

**Dateien:** `B16FullIntegration.py`

---

---

### T16 (neu) – LATENT_DIM/D_MODEL 64/128 → 256/256 ✅ ERLEDIGT

**Problem:** 64-dim Latent-Raum ist zu schmal für Szenen mit 6 Objekten,
Kamera-Orientierung, Ziel-Embedding und Dynamics.

**Lösung:**
- [x] `LATENT_DIM = 256` in B16FullIntegration.py (war 64)
- [x] `D_MODEL = 256` gleichzeitig (war 128) — muss = LATENT_DIM sein wegen `context[:, :LATENT_DIM]` im Goal-Loss
- [x] `dim_feedforward = d_model * 2 = 512` (skaliert mit D_MODEL, war hardcoded 256)
- [x] `dynamics_head`: Intermediate `d_model * 2 = 512` (war hardcoded 256)
- [x] B20/B21 importieren LATENT_DIM aus B16 → automatisch aktualisiert
- [x] Dimensions-Guard in `load_checkpoint()`: prüft `constants.LATENT_DIM/D_MODEL`,
  warnt bei Mismatch und überspringt Gewichte gracefully (kein Crash)

**Konsequenz:** Alte Checkpoints (LATENT_DIM=64) werden automatisch erkannt und
übersprungen. Neues Pre-Training mit B20 → B21 nötig.

**Parameteranzahl (neu):**
- Encoder: 2048→256 (fc_mu/log_var je +100% Parameter)
- dynamics_head: Linear(262→512→256) statt Linear(134→256→64)
- ActionHead: Linear(256→256→128) statt Linear(128→256→128)

**Dateien:** `B16FullIntegration.py`

---

### T12 (neu) – Rekurrenter Weltzustand (RSSM-Kern, DreamerV3-Stil) ✅ ERLEDIGT

**Lösung:**
- [x] `RSSM` Klasse: GRUCell(518→256) ersetzt TemporalTransformer
  - GRU-Input: cat(z_t, a_{t-1}, goal_proj) = 256+6+256 = 518-dim
  - Hidden-State h_t persistiert über gesamte Episode (statt 4-Step-Fenster)
  - `forward()`: Schritt-für-Schritt (Inferenz), aktualisiert self._h
  - `forward_sequence()`: Sequenz-Forward (Training, Truncated BPTT)
  - `predict_next_z()`: dynamics_head (identisch zur alten Version)
- [x] `ReplayBuffer`: `done` Flag + `sample_sequences(batch_size, seq_len=8)`
  - Episode-Grenzen markiert, nur zusammenhängende Sequenzen gesampelt
  - Alter `sample()` für reward_head/scene_head unverändert
- [x] `_train_step()`: Sequenz-basiert statt Einzel-Samples
  - B*L Frames auf einmal encodiert, GRU unrollt über L=8 Steps
  - Losses pro Step berechnet, über Sequenz gemittelt
  - Truncated BPTT via PyTorch autograd
- [x] `reset_hidden_state()`: GRU-State + prev_action zurücksetzen
  - B19Orchestrator ruft bei Episode-Reset auf
- [x] Checkpoint-Migration: dynamics_head Gewichte von Transformer übertragen, GRU startet frisch
- [x] 862k Parameter (vs. Transformer ~600k): GRUCell(518,256) + dynamics_head

**Parameteranzahl:** 861.952 (GRU: 595.200, dynamics_head: 266.752)

**Dateien:** `B16FullIntegration.py`, `B19Orchestrator.py`

---

### T17 (neu) – Offline-Vortraining der Dynamics ✅ ERLEDIGT

**Lösung:**
- [x] Neues Script `B24PreTrainDynamics.py`
- [x] Sammelt (obs, action, next_obs)-Paare aus MiniWorld (400 Episoden × 25 Steps)
- [x] Encoder eingefroren; nur `dynamics_head` wird trainiert
- [x] Loss: `MSE(dynamics_head(cat([z_cur, action])), z_next.detach())`
- [x] `z_cur` (256-dim) als Context-Ersatz valide, da LATENT_DIM == D_MODEL == 256 (T16)
- [x] Defaults: 40 Epochs, lr=3e-4; 60% move_forward / 20% turn_left / 20% turn_right
- [x] Speichert aktualisierten Checkpoint (transformer/dynamics_head state)
- [x] Pre-Training-Pipeline: B20 → B21 → B24 → B19

**Dateien:** `B24PreTrainDynamics.py` (neu)

---

### T18 (neu) – Free Energy Dashboard-Erweiterung ✅ ERLEDIGT

**Lösung:**
- [x] `ax_fe` Panel: gestapelte Flächen (stacked areas) statt einfacher FE-Kurve
  - Complexity (β·KL, darkorange)
  - Inaccuracy (Recon+Pred, steelblue)
  - Residual (mediumpurple)
  - Total FE weiße Linie + MA-20 rot gestrichelt
- [x] EFE-Proxy in `ax_stats`: Epistemic (sigma_mean) vs. Pragmatisch (r_reward_pred)
  farbiges Label: ERKUNDEN (σ>0.5) / ZIEL NÄHERN (r_rp>0.6) / AUSGEWOGEN
- [x] `r_reward_pred` als gepunktete violette Linie im Rewards-Panel
- [x] `scene_pred` (T13) im `ax_recog`-Titel angezeigt
- [x] hist-Keys ergänzt: `r_reward_pred`, `l_pred_img`, `l_reward`, `l_scene`, `complexity`, `inaccuracy`
- [x] `run_demo()` aktualisiert: latent 256-dim, neue Mock-Metriken

**Dateien:** `B18Dashboard.py`

---

## Abhängigkeiten

```
T01 ──→ T10  (Reward-Normalisierung vor Loss-Balancing)
T03 ──→ T04  (Transformer-Loss vor Action-Head-Sigma)
T05 ──→ T11  (Beta-Annealing vor Skip-Connections)
T06 ──→ T11  (Perceptual Loss vor Skip-Connections)
```

---

### T11 – Expected Free Energy (EFE) als Aktions-Auswahlprinzip ✅ ERLEDIGT

**Problem:** Der ActionHead lernte via reiner Imitation (`l_action = MSE` zu ausgeführten
Aktionen). In Active Inference soll der Agent Aktionen wählen die **EFE minimieren**:
epistemisch (Unsicherheit reduzieren) und pragmatisch (Ziel erreichen).

**Lösung:** Adaptiver Blend zwischen Imitation und EFE im `_train_step()`:

```python
# T11: EFE-Blend — adaptiv basierend auf Gemini-Daten im Buffer
efe_blend = EFE_BLEND_MAX * min(gemini_count / EFE_GEMINI_RAMP, 1.0)

# Pro Zeitschritt t:
l_imitation_t = (action_weights * (pa - act).pow(2)).mean()  # Stabilisierung
r_pred_efe = reward_head(cat([z.detach(), pa], dim=-1))      # Pragmatischer EFE
l_efe_t = -r_pred_efe.mean()                                 # Maximiere Reward

l_action += efe_blend * l_efe_t + (1 - efe_blend) * l_imitation_t
```

**Konstanten:**
- `EFE_BLEND_MAX = 0.5` — maximaler EFE-Anteil (Rest = Imitation)
- `EFE_GEMINI_RAMP = 50` — ab 50 Gemini-Samples voller Blend

**Design-Entscheidungen:**
- z ist detached → reward_head bekommt keinen Fehl-Gradienten über z
- Gradient fließt: l_efe → r_pred → pa → ActionHead (korrekt!)
- Imitation bleibt als Stabilisierungs-Term (Bootstrap + Regularisierung)
- Epistemischer Term separat über existierenden l_sigma (NLL-Kalibrierung)
- Blend ramp-up verhindert EFE-Rauschen bevor reward_head trainiert ist

**Logging:** `efe_blend` Spalte in train-CSV; `grad_rssm` statt `grad_tr`.

**Dateien:** `B16FullIntegration.py` (Zeilen ~298, ~1260-1280, ~1400)

---

## Empfohlene Reihenfolge

1. **T01** – Reward-Normalisierung (schneller Fix, großer Effekt)
2. **T02** – Adaptive Gemini durchgängig
3. **T05** – Beta-Annealing (schützt vor Collapse)
4. **T04** – Sigma-Loss NLL
5. **T03** – Transformer-Loss (größter Umbau)
6. **T06** – Perceptual Loss
7. **T07–T11** – nach Bedarf
