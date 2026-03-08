# TODO – Verbesserungsplan PredictionWorldNet

Priorisierte Liste der NN-, Reward- und Training-Verbesserungen.

---
## Anmerkungen
Jeweils überlegen ob der Vorschlag Deiner Meinung nach noch sinnvoll und Zielführend ist.
Wenn sinnvoll die betroffenen Demo-Block-Scripte anpassen und mir sagen, wenn ich etwas testen soll.

Verschiebe erledigte Punkte aus der "TODO.md" in die "DONE.md".

---
## Leitprinzip: Active Inference

Alle Architektur-Entscheidungen folgen dem Active Inference Rahmen:

```
Generatives Modell:    P(o_t | s_t)          ← Decoder (rekonstruiert/prediziert Bild)
                       P(s_t | s_{t-1}, a_t) ← Transitions-Modell (fehlt noch: T10, T12)
                       P(a_t)                 ← Prior auf Aktionen (via Gemini-Präferenzen)

Erkennungsmodell:      Q(s_t | o_t)           ← Encoder (VAE posterior)

Freie Energie (FE):    KL[Q(s)||P(s)]  +  E_Q[-log P(o|s)]
                       ↑ Complexity        ↑ Inaccuracy
                       (schon vorhanden: l_kl + l_recon + l_pred_img)

Erwartete FE (EFE):    Epistemic  +  Pragmatisch
   Epistemic:          Unsicherheits-Reduktion (sigma, novelty) ← schon vorhanden
   Pragmatisch:        Ziel-Erreichung (Gemini-Reward) ← schon vorhanden, aber nicht
                       direkt mit Aktionswahl verbunden (Ziel: T11)
```

Der Agent wählt Aktionen nicht um Reward zu maximieren, sondern um **EFE zu minimieren**:
exploriert wenn unsicher (epistemisch), handelt zielgerichtet wenn sicher (pragmatisch).
Gemini liefert den pragmatischen Term als externe semantische Bewertung.

---

## Stand der Architektur (aktuell implementiert)

```
Encoder      VAE, GroupNorm, 128×128 → z (64-dim)                 ✅
Decoder      Rekonstruktion UND Next-Frame-Prediction aus pred_z   ✅
Transformer  z + goal + a_hist → context                           ✅
ActionHead   context → action (6D) + sigma (Unsicherheit)          ✅
dynamics_head context+a_t → z_{t+1}  (action-konditioniert)        ✅ T10
FE-Loss      l_recon + l_pred_img + l_kl(FreeBits) + l_action...  ✅
Gemini       Semantischer Reward adaptiv                            ✅
Strategie    B22/B23 Sigma-Blending                                ✅
```

---

## 🔴 Priorität 1 – Kern des World-Models

### T10 – Aktions-konditioniertes Transitions-Modell ✅ ERLEDIGT → DONE.md

`dynamics_head(cat([context, a_t])) → z_{t+1}` implementiert.
Headless-Testmodus + CSV-Logging als Nebenprodukt.

---

### T11 – Expected Free Energy (EFE) als Aktions-Auswahlprinzip
**Warum:** Derzeit lernt der ActionHead via Imitation (l_action = MSE zu ausgeführten
Aktionen). In echtem Active Inference wählt der Agent Aktionen die die EFE minimieren:

```
EFE(a_t) = Epistemisch  +  Pragmatisch
           = E[H[Q(z|o)]] - E[log P(o|bevorzugt)]
           ≈ −sigma_pred  +  (−r_gemini)
```

**Lösung:** Statt l_action als Imitation-Loss: l_efe als Aktions-Verlust formulieren.
Der ActionHead minimiert direkt:
```
l_efe = −α * r_gemini_pred  +  β * sigma.mean()
```
Dabei kommt `r_gemini_pred` aus einem Reward-Prädiktor (T14).

**Achtung:** Dies ist ein größerer konzeptueller Schritt. Erst nach T10 und T14 angehen.

**Dateien:** `B16FullIntegration.py` (ActionHead, _train_step)
**Aufwand:** Mittel | **Nutzen:** Sehr Hoch (zentral für Active Inference)

---

## 🟡 Priorität 2 – Weltzustand und Semantik

### T12 – Rekurrenter Weltzustand (RSSM-Kern, DreamerV3-Stil) ✅ ERLEDIGT → DONE.md

---

### T13 – Semantischer Selbstbeschreibungs-Kopf ✅ ERLEDIGT → DONE.md

`scene_head(z) → N_SCENE_CLASSES` mit SCENE_VOCAB + SCENE_LABEL_MAP implementiert.
`scene_pred` in steps-CSV; l_scene in train-CSV.

---

### T14 – Reward-Prädiktor im latenten Raum ✅ ERLEDIGT → DONE.md

`reward_head(cat([z, a])) → r_gemini` implementiert.
Eigene Gemini-Stichprobe (require_gemini=True); `r_reward_pred` in steps-CSV.

---

## 🟢 Priorität 3 – Skalierung und Planung

### T15 – Mehrstufige Imagination (Planning-as-Inference)
**Warum:** Das Herzstück des Active Inference Ansatzes: Der Agent plant nicht
durch echte Aktionen in der Welt, sondern durch **Simulation im Weltzustand**.

```
Für k Schritte vorausplanen:
z_{t+1} = dynamics(z_t, a_t)         # Transition (T10)
r_{t+1} = reward_head(z_{t+1}, a_t)  # Reward-Vorhersage (T14)
EFE(a_t) += r_{t+1} + sigma(z_{t+1}) # EFE akkumulieren

→ Wähle Aktionssequenz mit minimalem EFE
```

**Voraussetzungen:** T10 + T12 (RSSM) + T14 (Reward-Prädiktor)
**Dateien:** Neues Modul `B24ImaginaryRollout.py`
**Aufwand:** Hoch | **Nutzen:** Sehr Hoch (echter World-Model-Planer)

---

### T16 – LATENT_DIM 64 → 256 ✅ ERLEDIGT → DONE.md

---

### T17 – Offline-Vortraining der Dynamics ✅ ERLEDIGT → DONE.md

---

### T18 – Free Energy Dashboard-Erweiterung ✅ ERLEDIGT → DONE.md

---

## Empfohlene Reihenfolge

```
T10  Aktions-konditionierte Dynamik          ← Jetzt (Fundament)
 ↓
T14  Reward-Prädiktor                        ← Klein, sofort nach T10
 ↓
T13  Semantischer Beschreibungs-Kopf         ← Parallel zu T14
 ↓
T16  LATENT_DIM 64 → 256                     ← Eigener Milestone (Pre-Training)
 ↓
T12  RSSM Rekurrenter Weltzustand            ← Größter Schritt, nach T10 stabil
 ↓
T11  EFE als Aktionsprinzip                  ← Nach T12 + T14
 ↓
T17  Offline Dynamics-Vortraining            ← Parallel zu T12
 ↓
T15  Mehrstufige Imagination                 ← Finale Stufe
 ↓
T18  FE-Dashboard                            ← Begleitend zu T11/T15
```

---

## Zusammenhang: Active Inference – Was wir haben / Was wir wollen

| Komponente              | Active Inference Begriff           | Status     | TODO |
|-------------------------|------------------------------------|------------|------|
| Encoder (VAE)           | Recognition model Q(s\|o)          | ✅ vorhanden |      |
| Decoder (Recon)         | Generative model P(o\|s)           | ✅ vorhanden |      |
| Next-Frame Prediction   | P(o_{t+1}\|s_t)                    | ✅ neu       |      |
| KL + Free Bits          | Complexity (FE-Term)               | ✅ vorhanden |      |
| Gemini-Reward           | Pragmatischer EFE-Term             | ✅ vorhanden |      |
| Sigma-Unsicherheit      | Epistemischer EFE-Term (Proxy)     | ✅ vorhanden |      |
| Transition-Modell       | P(s_{t+1}\|s_t, **a_t**)          | ✅ T10       |      |
| EFE-Aktionswahl         | Aktionen minimieren EFE            | ❌ fehlt     | T11  |
| GRU-Weltzustand         | Q(s_t\|o_{0:t}, a_{0:t}) (RSSM)   | ❌ fehlt     | T12  |
| Semantik-Kopf           | P(label\|s) – Szene beschreiben   | ✅ T13       |      |
| Reward-Prädiktor        | Pragmatischer Prior P(o\|bevorzugt)| ✅ T14       |      |
| Imagination             | Planning-as-Inference              | ❌ fehlt     | T15  |
| Größerer Latent         | Reicherer Zustandsraum             | ✅ T16       |      |
| Offline Dynamics        | Vortraining dynamics_head          | ✅ T17       |      |
| FE Dashboard            | Complexity/Inaccuracy sichtbar     | ✅ T18       |      |
