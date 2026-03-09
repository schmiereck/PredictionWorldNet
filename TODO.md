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
                       P(s_t | s_{t-1}, a_t) ← Transitions-Modell (T10 + T12 ✅)
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
Encoder      VAE, GroupNorm, 128×128 → z (256-dim)                ✅
Decoder      Rekonstruktion UND Next-Frame-Prediction aus pred_z   ✅
Transformer  z + goal + a_hist → context                           ✅
ActionHead   context → action (6D) + sigma (Unsicherheit)          ✅
dynamics_head context+a_t → z_{t+1}  (action-konditioniert)        ✅ T10
FE-Loss      l_recon + l_pred_img + l_kl(FreeBits) + l_action...  ✅
Gemini       Semantischer Reward adaptiv                            ✅
Strategie    B22/B23 Sigma-Blending                                ✅
```

---

## 🔴 Priorität 1 – Imagination & Planning skalieren

### T19 – Actor-Critic Training in der Imagination (Dreamer-Stil)
**Problem:** `plan_action()` nutzt "Random Shooting" (N=32, H=5). Das ist eine Brute-Force-Suche zur Laufzeit. Es skaliert sehr schlecht, wenn der Horizont größer wird, und verlangsamt die Inferenz.
**Lösung:** Statt zur Laufzeit zu suchen, trainieren wir einen "Actor" und einen "Value/Critic" **ausschließlich auf den imaginierten Trajektorien**. 
  - Der `Value-Head` lernt, den erwarteten zukünftigen Reward (bzw. die negative EFE) für einen Zustand `h_t` vorherzusagen.
  - Der `Action-Head` (Actor) wird so trainiert, dass seine vorgeschlagenen Aktionen den Wert des Value-Heads maximieren.
**Vorteil:** Zur Laufzeit muss das Modell nicht mehr aufwendig Trajektorien berechnen (`plan_action` entfällt). Der Action-Head liefert direkt in O(1) die optimale, langfristig geplante Aktion.

### T20 – Echte Epistemische Unsicherheit durch ein Dynamics-Ensemble
**Problem:** Aktuell ist `sigma` einfach ein Output des ActionHeads, trainiert via NLL. Das misst eher die "Rauschigkeit" der Aktion, aber nicht die echte Unsicherheit des Weltmodells über die Umgebung (epistemische Unsicherheit).
**Lösung:** Wir instanziieren ein kleines Ensemble von `dynamics_head` Modellen (z.B. 3 bis 5 Stück), die alle parallel trainiert werden. Die Varianz zwischen ihren Vorhersagen (`Var[z_{t+1}]`) ist ein echtes mathematisches Maß für "Neugier" (Information Gain).
**Vorteil:** Der Agent wird extrem zielgerichtet Räume und Objekte ansteuern, bei denen sich die Dynamics-Heads "uneinig" sind. Das ist pure Active Inference.

---

## 🟡 Priorität 2 – Repräsentation & Semantik

### T21 – Contrastive Learning für den VAE-Encoder (SimCLR/BYOL)
**Problem:** Der Encoder wird aktuell fast nur über den Reconstruction Loss (MSE + SSIM) trainiert. Er muss also Pixel genau abspeichern, was Kapazität kostet und ihn anfällig für irrelevante Details (Licht, Rauschen).
**Lösung:** Ein "Contrastive Loss" wird hinzugefügt. Ein Bild wird zweifach mit leichtem Rauschen/Farbverschiebungen versehen. Der Encoder muss lernen, dass beide Bilder denselben Latent-Vektor `z` erzeugen sollen.
**Vorteil:** Der Latent Space ordnet sich stärker nach Semantik (Objekte) als nach Pixeln. Die Kopplung mit dem CLIP-Goal-Embedding (B05) wird dadurch massiv besser und robuster.

### T22 – Hierarchische Ziele (Gemini → Sub-Goals)
**Problem:** Gemini gibt uns ein globales Ziel (`a red box`). Die Strategie (B22/B23) ist aktuell eher regelbasiert.
**Lösung:** Gemini wird angewiesen, konkrete textuelle Sub-Goals zu generieren (z.B. "Drehe dich, bis die Wand weg ist", "Gehe auf den blauen Blob zu"). Diese Sub-Goals werden on-the-fly durch den CLIP-Encoder gejagt und überschreiben temporär das `goal_proj` des Modells. 
**Vorteil:** Das neuronale Netz muss nur noch einfache Micro-Tasks lösen, der LLM-Planer behält die Makro-Kontrolle.

---

## 🟢 Priorität 3 – Effizienz im Training

### T23 – Prioritized Experience Replay (PER)
**Problem:** Der Buffer samplet uniform. Eine Sequenz, bei der der Roboter nur gegen eine Wand starrt, wird genauso oft trainiert wie der seltene Moment, in dem er die Rote Box entdeckt.
**Lösung:** Sequenzen mit einem hohen Prediction Error (FE) oder hohen Gemini-Rewards bekommen ein höheres Gewicht beim Sampling aus dem Ringpuffer.
**Vorteil:** Drastisch schnellere Konvergenz, gerade für die seltenen, aber wertvollen Gemini-gestützten Trajektorien.

---

## Zusammenhang: Active Inference – Was wir haben / Was wir wollen

| Komponente              | Active Inference Begriff           | Status     | TODO |
|-------------------------|------------------------------------|------------|------|
| Encoder (VAE)           | Recognition model Q(s\|o)          | ✅ vorhanden |      |
| Decoder (Recon)         | Generative model P(o\|s)           | ✅ vorhanden |      |
| Next-Frame Prediction   | P(o_{t+1}\|s_t)                    | ✅ neu       |      |
| KL + Free Bits          | Complexity (FE-Term)               | ✅ vorhanden |      |
| Gemini-Reward           | Pragmatischer EFE-Term             | ✅ vorhanden |      |
| Sigma-Unsicherheit      | Epistemischer EFE-Term (Proxy)     | ✅ vorhanden | T20  |
| Transition-Modell       | P(s_{t+1}\|s_t, **a_t**)          | ✅ T10       |      |
| EFE-Aktionswahl         | Aktionen minimieren EFE            | ✅ T11       | T19  |
| GRU-Weltzustand         | Q(s_t\|o_{0:t}, a_{0:t}) (RSSM)   | ✅ T12       |      |
| Semantik-Kopf           | P(label\|s) – Szene beschreiben   | ✅ T13       |      |
| Reward-Prädiktor        | Pragmatischer Prior P(o\|bevorzugt)| ✅ T14       |      |
| Imagination             | Planning-as-Inference              | ✅ T15       | T19  |
| Größerer Latent         | Reicherer Zustandsraum             | ✅ T16       |      |
| Offline Dynamics        | Vortraining dynamics_head          | ✅ T17       |      |
| FE Dashboard            | Complexity/Inaccuracy sichtbar     | ✅ T18       |      |
