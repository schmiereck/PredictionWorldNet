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

## Abhängigkeiten

```
T01 ──→ T10  (Reward-Normalisierung vor Loss-Balancing)
T03 ──→ T04  (Transformer-Loss vor Action-Head-Sigma)
T05 ──→ T11  (Beta-Annealing vor Skip-Connections)
T06 ──→ T11  (Perceptual Loss vor Skip-Connections)
```

## Empfohlene Reihenfolge

1. **T01** – Reward-Normalisierung (schneller Fix, großer Effekt)
2. **T02** – Adaptive Gemini durchgängig
3. **T05** – Beta-Annealing (schützt vor Collapse)
4. **T04** – Sigma-Loss NLL
5. **T03** – Transformer-Loss (größter Umbau)
6. **T06** – Perceptual Loss
7. **T07–T11** – nach Bedarf
