# TODO – Verbesserungsplan PredictionWorldNet

Priorisierte Liste der NN-, Reward- und Training-Verbesserungen.

---

## 🔴 Priorität 1 – Kritisch

### T01 – Reward-Normalisierung (B15)
**Problem:** Der Goal-Reward basiert auf `1 - cos_sim`, Wertebereich [0, 2].
Die gewichtete Summe in `RewardCombinator` (Z. 321–326) behandelt aber alle
Rewards als [0, 1]. Dadurch dominiert der Goal-Reward unbeabsichtigt.

**Lösung:**
- [ ] `r_goal` auf [0, 1] clampen: `r_goal = torch.clamp(1 - cos_sim, 0, 1)`
- [ ] Alle Reward-Komponenten explizit normalisieren (Running-Mean/Std)
- [ ] Reward-Statistiken im Dashboard anzeigen (Verteilung pro Komponente)

**Dateien:** `B15RewardCombination.py` Z. 321–326, `B10PredictionLoss.py`

---

### T02 – Adaptive Gemini-Frequenz durchgängig nutzen (B14 ↔ B16)
**Problem:** `AdaptiveController` (B14) berechnet Urgency-basierte Intervalle
(Z. 148–159), aber der Orchestrator nutzt teilweise noch feste Intervalle.

**Lösung:**
- [ ] Sicherstellen, dass `adaptive.should_call()` überall die Entscheidung trifft
- [ ] Festes `gemini_interval` aus Orchestrator/Config entfernen
- [ ] Urgency-Score + aktuelles Intervall im Dashboard anzeigen

**Dateien:** `B14AdaptiveGemini.py`, `B16FullIntegration.py` Z. 329,
`B19Orchestrator.py`

---

### T03 – Temporal Transformer Loss ersetzen (B07)
**Problem:** Die Loss-Funktion (Norm + Cosine-Consistency) hat keinen
direkten Bezug zur eigentlichen Aufgabe (Action-Prediction / Next-Frame).
Nur 2 Layer × 4 Heads ist flach für 8 Token-Slots.

**Lösung:**
- [ ] Loss auf Next-Frame-Prediction umstellen:
      `L = -log p(z_{t+1} | context)` statt Norm-Regularisierung
- [ ] Zeitliches Decay für ältere Frames:
      `weight_t = exp(-t / tau)` statt gleiche Gewichtung
- [ ] Layer von 2 → 4 erhöhen (oder 3 + größeres FFN testen)
- [ ] Cross-Attention zum Goal-Embedding statt simpler Konkatenation

**Dateien:** `B07TemporalTransformer.py` Z. 98–105, 137–149

---

## 🟡 Priorität 2 – Mittel

### T04 – Action Head: Sigma-Loss kalibrieren (B09)
**Problem:** Sigma-Loss nutzt MSE gegen `err_norm` (Z. 364–365).
Korrekte NLL-Formulierung: `-log σ + |error| / σ` liefert kalibrierte
Unsicherheitsschätzung.

**Lösung:**
- [ ] Sigma-Loss auf NLL umstellen:
      `loss_sigma = torch.mean(torch.log(sigma) + err / sigma)`
- [ ] Sigma-Kalibrierung validieren (predicted vs actual error)
- [ ] Separate Initialisierung für Action- und Sigma-Head

**Dateien:** `B09ActionHead.py` Z. 149–151, 364–365

---

### T05 – Beta-Annealing verbessern (B10)
**Problem:** Lineares Annealing (Z. 183–188) kann bei kleinen Modellen
zu schnell ansteigen → Posterior Collapse. Beta startet bei 0.0, aber
der Warmup ist linear.

**Lösung:**
- [ ] Sigmoid- oder Cosine-Annealing implementieren:
      `beta = beta_max * 0.5 * (1 + cos(pi * (1 - t/warmup)))`
- [ ] Cyclical Annealing testen (β steigt und fällt periodisch)
- [ ] KL-Divergenz monitoren — bei < 0.1 nats: Collapse-Warnung

**Dateien:** `B10PredictionLoss.py` Z. 178–188

---

### T06 – Decoder: Perceptual Loss ergänzen (B08)
**Problem:** Reiner MSE-Loss auf Pixel-Ebene ist perceptuell blind —
verschwommene Rekonstruktionen werden akzeptiert, solange der
Durchschnittsfehler stimmt.

**Lösung:**
- [ ] SSIM-Loss als zweite Komponente hinzufügen
- [ ] Optional: Lightweight VGG-Feature-Loss (erste 3 Layer)
- [ ] Gewichtung: `L = 0.7 * MSE + 0.3 * (1 - SSIM)` als Startpunkt

**Dateien:** `B08CnnDecoder.py`, `B10PredictionLoss.py`

---

### T07 – Strategy-Executor: Farberkennung robuster (B23)
**Problem:** Hardcodierte RGB-Schwellen für Farberkennung funktionieren nur
in synthetischen Szenen ohne Beleuchtungsvariation.

**Lösung:**
- [ ] HSV-Farbraum statt RGB verwenden (robuster gegen Helligkeit)
- [ ] Alternativ: CLIP-basierte Objekterkennung nutzen
      (CLIP-Embedding des Bildausschnitts vs. Text-Label)
- [ ] Farbmasken-Schwellen konfigurierbar machen (nicht hardcoded)

**Dateien:** `B23StrategyExecutor.py`

---

## 🟢 Priorität 3 – Nice-to-Have

### T08 – Intrinsic Reward: Memory vergrößern (B12)
**Problem:** k-NN mit 200 Samples in 64-dim Raum ist statistisch dünn.
Novelty-Schätzungen haben hohe Varianz.

**Lösung:**
- [ ] Memory auf 1000–2000 erhöhen
- [ ] Zeitliches Decay einführen (ältere Einträge abwerten)
- [ ] Optional: Locality-Sensitive Hashing für O(1) statt O(n) Lookup

**Dateien:** `B12IntrinsicReward.py` Z. 138–139, 171

---

### T09 – Goal-Projektion: CLIP-Bottleneck reduzieren
**Problem:** `goal_proj = Linear(512 → 64)` ist ein extremer
Informations-Bottleneck. CLIP-Embeddings verlieren semantische Nuancen.

**Lösung:**
- [ ] Zweistufige Projektion: `512 → 128 → 64` mit ReLU dazwischen
- [ ] Oder: Attention-basiertes Alignment statt linearer Projektion
- [ ] Kosinus-Ähnlichkeit in 128-dim statt 64-dim berechnen

**Dateien:** `B05ClipTextEncoder.py`, `B10PredictionLoss.py`

---

### T10 – Gradient Monitoring & Loss-Balancing
**Problem:** Keine Überwachung der Gradientenmagnitude.
Loss-Gewichte sind fest — können zu Imbalance führen wenn eine
Komponente dominiert.

**Lösung:**
- [ ] Gradient-Norm pro Modul loggen (Encoder, Decoder, Transformer)
- [ ] Automatisches Loss-Balancing: `w_i = 1 / var(L_i)`
- [ ] Gradient Clipping validieren (aktueller Wert angemessen?)

**Dateien:** `B16FullIntegration.py`, `B11TrainingLoop.py`

---

### T11 – Decoder: Skip-Connections (U-Net Stil)
**Problem:** Ohne Skip-Connections gehen räumliche Details
im Bottleneck verloren.

**Lösung:**
- [ ] Encoder-Feature-Maps an Decoder weiterreichen (concat + 1×1 Conv)
- [ ] Vorsicht: Nicht zu viel Detail — VAE soll abstrahieren
- [ ] Progressive Resolution als Alternative testen

**Dateien:** `B04bVariationalEncoder.py`, `B08CnnDecoder.py`

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
