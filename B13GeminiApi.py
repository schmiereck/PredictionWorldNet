"""
B13 – Gemini API: Natürlichsprachiges Ziel-Interface
======================================================
Übersetzt natürlichsprachige Benutzer-Befehle in CLIP-kompatible
Ziel-Texte für den Roboter.

Pipeline:
    User: "Geh zur roten Box und weiche dem Hindernis aus"
         ↓  Gemini (structured prompt)
    "navigate to the red box, avoid obstacles"
         ↓  CLIP (B05)
    goal_embedding (512-dim)
         ↓  Temporal Transformer (B07)
    Agent handelt

Warum Gemini statt direkt CLIP?
    CLIP versteht kurze, präzise englische Phrasen am besten.
    Gemini übersetzt:
        - Deutsch → Englisch
        - Umgangssprache → präzise CLIP-Phrasen
        - Komplexe Befehle → strukturierte Ziel-Liste
        - Mehrdeutigkeiten → konkrete Interpretationen

Gemini Prompt-Strategie:
    System: "Du bist ein Roboter-Assistent. Übersetze Befehle in
             kurze englische CLIP-Ziel-Phrasen."
    User:   "Geh zur roten Box"
    Output: { "primary_goal": "find the red box",
               "secondary_goals": ["navigate forward"],
               "avoid": [],
               "confidence": 0.95 }

Installation:
    pip install google-generativeai

API-Key:
    Umgebungsvariable: GEMINI_API_KEY
    Oder direkt: GeminiInterface(api_key="...")
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("google-genai nicht installiert.")
    print("Installation: pip install google-genai\n")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ─────────────────────────────────────────────
# GEMINI INTERFACE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Du bist ein KI-Assistent für einen mobilen Roboter mit Kamera.
Der Roboter kann:
- Vorwärts/Rückwärts fahren
- Links/Rechts drehen
- In Kurven fahren (Arc-Movement)
- Die Kamera schwenken (Pan: -90° bis +90°) und neigen (Tilt: -45° bis +45°)

Deine Aufgabe: Übersetze Benutzer-Befehle in präzise englische CLIP-Ziel-Phrasen.

Antworte NUR mit validem JSON in diesem Format:
{
  "primary_goal": "kurze englische CLIP-Phrase (max 8 Wörter)",
  "secondary_goals": ["weitere Ziele als Liste"],
  "avoid": ["was vermieden werden soll"],
  "camera_hint": "was die Kamera beobachten soll (optional)",
  "arc_hint": "ob Kurvenfahrt sinnvoll ist (optional)",
  "confidence": 0.0-1.0,
  "interpretation": "kurze deutsche Erklärung der Interpretation"
}

Beispiele:
- "Geh zur roten Box" → primary_goal: "find the red box"
- "Erkunde den Raum" → primary_goal: "explore the room"
- "Geh zur Tür aber weiche Hindernissen aus" → primary_goal: "navigate to the door", avoid: ["obstacles"]
- "Schau nach rechts" → primary_goal: "look right", camera_hint: "pan camera right"
- "Fahr eine Kurve links zur grünen Box" → primary_goal: "find the green box", arc_hint: "left arc movement"
"""


class GeminiInterface:
    """
    Natürlichsprachiges Interface für Roboter-Befehle via Gemini.

    Modi:
        "gemini"  : Echter Gemini API Aufruf
        "mock"    : Regelbasierte Übersetzung (kein API-Key nötig)
    """

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.mode       = "mock"
        self.call_count = 0
        self.last_call  = 0.0
        self.min_interval = 1.0   # Mindestabstand zwischen API-Calls (Rate Limiting)

        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")

        if GEMINI_AVAILABLE and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                # Verbindung testen
                _ = self.client.models.list()
                self.mode = "gemini"
                print(f"Gemini API verbunden: {model}")
            except Exception as e:
                print(f"Gemini API Fehler: {e}")
                print("Fallback auf Mock-Modus.\n")
        else:
            print("Gemini läuft im Mock-Modus (kein API-Key).")
            print("Setze GEMINI_API_KEY Umgebungsvariable für echte API.\n")

    def translate(self, user_command: str) -> dict:
        """
        Übersetzt einen Benutzer-Befehl in strukturiertes Ziel-Dict.

        Args:
            user_command: Natürlichsprachiger Befehl (Deutsch oder Englisch)
        Returns:
            dict mit primary_goal, secondary_goals, avoid, confidence, ...
        """
        self.call_count += 1

        if self.mode == "gemini":
            return self._call_gemini(user_command)
        else:
            return self._mock_translate(user_command)

    def _call_gemini(self, command: str) -> dict:
        """Echter Gemini API Aufruf mit Rate Limiting."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=command,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                )
            )
            text = response.text.strip()

            # JSON aus Response extrahieren
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            result["source"] = "gemini"
            result["raw"]    = response.text
            return result

        except Exception as e:
            print(f"Gemini Fehler: {e} – dauerhafter Fallback auf Mock")
            self.mode = "mock"   # ← Nicht mehr erneut versuchen
            return self._mock_translate(command)

    def _mock_translate(self, command: str) -> dict:
        """
        Regelbasierte Mock-Übersetzung ohne API.
        Erkennt Keywords und gibt passende CLIP-Phrasen zurück.
        """
        cmd = command.lower()

        # Keyword-Mapping Deutsch → CLIP-Englisch
        mappings = [
            # Objekte – alle deutschen Deklinationsformen
            (["rote box", "roten box", "roter box", "rote kiste", "red box"],
             "find the red box", 0.95),
            (["blauen ball", "blauer ball", "blaue ball", "blue ball"],
             "find the blue ball", 0.95),
            (["grüne tür", "grünen tür", "grüner tür", "green door",
              "tür", "ausgang", "exit"],
             "navigate to the exit door", 0.92),
            (["korridor", "flur", "gang", "corridor", "passage"],
             "explore the corridor", 0.88),
            (["ecke", "corner", "winkel"],
             "navigate to the corner", 0.85),
            # Aktionen
            (["erkunde", "explore", "schau dich um", "suche", "such"],
             "explore the room", 0.80),
            (["stopp", "stop", "steh", "halt", "anhalten"],
             "stop moving", 0.99),
            (["zurück", "rückwärts", "back", "rueckwaerts"],
             "move backwards", 0.90),
            (["geradeaus", "vorwärts", "vorwaerts", "forward", "geh weiter"],
             "move forward", 0.90),
            # Drehen / Rotation
            (["dreh", "drehe", "drehen", "rotation", "wende", "umdrehen",
              "links drehen", "rechts drehen", "rotate", "turn around"],
             "rotate and explore", 0.85),
            # Sonne / Ausrichten
            (["sonne", "sun", "ausrichten", "align", "richte"],
             "align with light source", 0.78),
            # Kamera
            (["schau rechts", "kamera rechts", "look right", "nach rechts"],
             "look right", 0.88),
            (["schau links", "kamera links", "look left", "nach links schauen"],
             "look left", 0.88),
            (["schau hoch", "nach oben", "look up", "schau nach oben"],
             "look up", 0.85),
            (["schau runter", "nach unten", "look down", "schau nach unten"],
             "look down", 0.85),
            # Kurve
            (["kurve links", "links abbiegen", "arc left", "linkskurve"],
             "turn left arc movement", 0.87),
            (["kurve rechts", "rechts abbiegen", "arc right", "rechtskurve"],
             "turn right arc movement", 0.87),
        ]

        for keywords, clip_phrase, conf in mappings:
            if any(kw in cmd for kw in keywords):
                # Vermeidungs-Keywords
                avoid = []
                if any(w in cmd for w in ["hindernis", "obstacle", "wand", "wall"]):
                    avoid.append("obstacles")
                if any(w in cmd for w in ["objekt", "object"]):
                    avoid.append("objects")

                # Kamera-Hint
                camera_hint = None
                if any(w in cmd for w in ["schau", "kamera", "look"]):
                    if "rechts" in cmd or "right" in cmd:
                        camera_hint = "pan camera right 90 degrees"
                    elif "links" in cmd or "left" in cmd:
                        camera_hint = "pan camera left 90 degrees"
                    elif "hoch" in cmd or "oben" in cmd or "up" in cmd:
                        camera_hint = "tilt camera up 45 degrees"
                    elif "runter" in cmd or "unten" in cmd or "down" in cmd:
                        camera_hint = "tilt camera down 45 degrees"

                # Arc-Hint
                arc_hint = None
                if any(w in cmd for w in ["kurve", "arc", "bogen"]):
                    arc_hint = "use arc movement"

                return {
                    "primary_goal":    clip_phrase,
                    "secondary_goals": [],
                    "avoid":           avoid,
                    "camera_hint":     camera_hint,
                    "arc_hint":        arc_hint,
                    "confidence":      conf,
                    "interpretation":  f"Erkannt: '{clip_phrase}'",
                    "source":          "mock",
                }

        # Fallback: direkter Befehl
        return {
            "primary_goal":    command[:50],
            "secondary_goals": [],
            "avoid":           [],
            "camera_hint":     None,
            "arc_hint":        None,
            "confidence":      0.5,
            "interpretation":  "Unbekannter Befehl – direkt verwendet",
            "source":          "mock_fallback",
        }

    def get_clip_embedding(self, goal_dict: dict,
                           clip_encoder=None) -> np.ndarray:
        """
        Konvertiert das Ziel-Dict → CLIP-Embedding.

        Kombiniert primary_goal und secondary_goals gewichtet:
            embedding = 0.7 * primary + 0.3 * Ø(secondary)
        """
        primary = goal_dict["primary_goal"]

        if clip_encoder is not None and CLIP_AVAILABLE:
            # Echter CLIP
            import clip as clip_module
            device = "cpu"
            with torch.no_grad():
                tokens    = clip_module.tokenize([primary]).to(device)
                emb       = clip_encoder.encode_text(tokens)
                emb       = F.normalize(emb, dim=-1)
                return emb.squeeze(0).cpu().numpy()
        else:
            # Mock-Embedding
            rng = np.random.default_rng(abs(hash(primary)) % (2**32))
            vec = rng.standard_normal(512).astype(np.float32)
            return vec / np.linalg.norm(vec)

    def summary(self) -> dict:
        return {
            "mode":       self.mode,
            "model":      self.model_name,
            "calls":      self.call_count,
        }


# ─────────────────────────────────────────────
# TEST-BEFEHLE
# ─────────────────────────────────────────────

TEST_COMMANDS = [
    # Deutsch
    "Geh zur roten Box",
    "Finde den blauen Ball und weiche Hindernissen aus",
    "Erkunde den Korridor",
    "Navigiere zur grünen Tür",
    "Schau nach rechts und such die rote Box",
    "Fahr eine Kurve links zur Ecke",
    "Schau nach oben",
    "Stopp",
    # Englisch
    "find the red box",
    "explore the room carefully",
    # Komplex
    "Dreh dich um und schau ob hinter dir ein Hindernis ist",
    "Geh geradeaus bis du die Tür siehst",
]


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    # API-Key aus Umgebungsvariable oder direkt setzen
    api_key = os.environ.get("GEMINI_API_KEY", "")
    # api_key = "DEIN_KEY_HIER"  # ← Alternativ direkt eintragen

    gemini = GeminiInterface(api_key=api_key)

    print(f"Modus: {gemini.mode.upper()}")
    print(f"Verarbeite {len(TEST_COMMANDS)} Test-Befehle...\n")

    # Alle Befehle übersetzen
    results = []
    for cmd in TEST_COMMANDS:
        result = gemini.translate(cmd)
        result["original"] = cmd
        result["embedding"] = gemini.get_clip_embedding(result)
        results.append(result)
        print(f"  [{result['source']:12s}] '{cmd}'")
        print(f"    → '{result['primary_goal']}'  (conf={result['confidence']:.2f})")
        if result.get("camera_hint"):
            print(f"    📷 {result['camera_hint']}")
        if result.get("arc_hint"):
            print(f"    🔄 {result['arc_hint']}")
        if result.get("avoid"):
            print(f"    ⚠ Vermeiden: {result['avoid']}")
        print()

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(
        f'B13 – Gemini Interface: Natürlichsprachige Roboter-Befehle  '
        f'[{gemini.mode.upper()}]',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.6, wspace=0.4)

    ax_pipeline = fig.add_subplot(gs[0, :2])   # Pipeline-Diagramm
    ax_conf     = fig.add_subplot(gs[0, 2:])   # Konfidenz-Balken
    ax_emb      = fig.add_subplot(gs[1, :3])   # Embedding-Heatmap
    ax_sim      = fig.add_subplot(gs[1, 3])    # Ähnlichkeits-Matrix (klein)
    ax_detail   = fig.add_subplot(gs[2, :2])   # Detail-Ansicht
    ax_ros      = fig.add_subplot(gs[2, 2:])   # ROS2-Ausgabe
    ax_pipeline.axis('off')
    ax_detail.axis('off')
    ax_ros.axis('off')

    # ── Pipeline-Diagramm ─────────────────────────────────
    pipeline_text = [
        "── Gemini Interface Pipeline ──────────────────────",
        "",
        f"  Modus: {gemini.mode.upper()}",
        "",
        "  User (Deutsch/Englisch)",
        '  "Geh zur roten Box und weiche Hindernissen aus"',
        "           ↓",
        "  Gemini API  (google-generativeai)",
        '  → { "primary_goal": "find the red box",',
        '      "avoid": ["obstacles"],',
        '      "confidence": 0.95 }',
        "           ↓",
        "  CLIP Text-Encoder (B05)",
        "  → goal_embedding (512-dim)",
        "           ↓",
        "  Temporal Transformer (B07)",
        "  [CLS][current][goal][t-1]...[t-16]",
        "           ↓",
        "  Action Head (B09) → ROS2 Twist",
    ]
    ax_pipeline.text(
        0.02, 0.98, "\n".join(pipeline_text),
        transform=ax_pipeline.transAxes,
        fontsize=8.5, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9),
        color='lightcyan'
    )

    # ── Konfidenz-Balken ──────────────────────────────────
    short_cmds = [r["original"][:30] + ("…" if len(r["original"])>30 else "")
                  for r in results]
    confs      = [r["confidence"] for r in results]
    colors_c   = ['seagreen' if c>=0.85 else
                  'gold'     if c>=0.70 else
                  'tomato'   for c in confs]

    ax_conf.barh(range(len(results)), confs,
                 color=colors_c, alpha=0.85)
    ax_conf.set_yticks(range(len(results)))
    ax_conf.set_yticklabels(short_cmds, fontsize=6.5)
    ax_conf.set_xlim(0, 1.15)
    ax_conf.axvline(0.85, color='seagreen', linestyle='--',
                    linewidth=1, alpha=0.7, label='Hoch ≥0.85')
    ax_conf.axvline(0.70, color='gold',    linestyle='--',
                    linewidth=1, alpha=0.7, label='Mittel ≥0.70')
    for i, (conf, r) in enumerate(zip(confs, results)):
        ax_conf.text(conf + 0.01, i,
                     f"{conf:.2f}  '{r['primary_goal'][:25]}'",
                     va='center', fontsize=6)
    ax_conf.set_title('Übersetzungs-Konfidenz', fontsize=9)
    ax_conf.set_xlabel('Konfidenz')
    ax_conf.legend(fontsize=7, loc='lower right')
    ax_conf.invert_yaxis()

    # ── Embedding Heatmap ─────────────────────────────────
    emb_matrix = np.stack([r["embedding"] for r in results])  # (N, 512)
    show_dim   = 128
    im = ax_emb.imshow(
        emb_matrix[:, :show_dim], cmap='coolwarm',
        aspect='auto', vmin=-0.15, vmax=0.15,
        interpolation='nearest'
    )
    ax_emb.set_yticks(range(len(results)))
    ax_emb.set_yticklabels(
        [f"[{i}] {r['primary_goal'][:35]}" for i, r in enumerate(results)],
        fontsize=6.5
    )
    ax_emb.set_xlabel(f'CLIP Embedding Dim (erste {show_dim} von 512)')
    ax_emb.set_title('CLIP Goal-Embeddings (Mock)', fontsize=9)
    fig.colorbar(im, ax=ax_emb, fraction=0.02)

    # ── Ähnlichkeits-Matrix ───────────────────────────────
    n   = len(results)
    sim = emb_matrix @ emb_matrix.T
    im2 = ax_sim.imshow(sim, cmap='RdYlGn', vmin=0.6, vmax=1.0,
                        interpolation='nearest')
    ax_sim.set_xticks(range(n))
    ax_sim.set_yticks(range(n))
    ax_sim.set_xticklabels(range(n), fontsize=6)
    ax_sim.set_yticklabels(range(n), fontsize=6)
    ax_sim.set_title('Cos-Ähnlichkeit\nder Embeddings', fontsize=8)
    fig.colorbar(im2, ax=ax_sim, fraction=0.05)

    # ── Detail-Ansicht: Erstes Ergebnis ───────────────────
    r0 = results[0]
    detail_lines = [
        "── Beispiel: Detailliertes Ergebnis ─────────────",
        "",
        f"Eingabe:    '{r0['original']}'",
        "",
        f"primary_goal:    '{r0['primary_goal']}'",
    ]
    if r0.get("secondary_goals"):
        for sg in r0["secondary_goals"]:
            detail_lines.append(f"secondary:       '{sg}'")
    if r0.get("avoid"):
        detail_lines.append(f"avoid:           {r0['avoid']}")
    if r0.get("camera_hint"):
        detail_lines.append(f"camera_hint:     '{r0['camera_hint']}'")
    if r0.get("arc_hint"):
        detail_lines.append(f"arc_hint:        '{r0['arc_hint']}'")
    detail_lines += [
        f"confidence:      {r0['confidence']:.2f}",
        f"interpretation:  '{r0.get('interpretation', '')}'",
        f"source:          {r0['source']}",
        "",
        f"CLIP Embedding: shape=(512,)",
        f"  norm={np.linalg.norm(r0['embedding']):.4f}",
        f"  mean={r0['embedding'].mean():.4f}",
        f"  std= {r0['embedding'].std():.4f}",
    ]
    ax_detail.text(
        0.02, 0.98, "\n".join(detail_lines),
        transform=ax_detail.transAxes,
        fontsize=8, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    # ── ROS2 Ausgabe ──────────────────────────────────────
    ros_lines = [
        "── ROS2 Integration (Ausblick) ──────────────────",
        "",
        "# B13 Ausgabe → B05 → B07 → B09 → ROS2:",
        "",
    ]
    # Zeige ersten 4 Befehle mit ihrer Wirkung
    action_hints = {
        "find the red box":           "linear_x=+0.3, duration=1.0s",
        "find the blue ball":         "linear_x=+0.2, angular_z=+0.6",
        "explore the corridor":       "linear_x=+0.5, duration=1.5s",
        "navigate to the exit door":  "linear_x=+0.4, duration=0.8s",
        "look right":                 "camera_pan=+1.57 (90°)",
        "turn left arc movement":     "arc_radius=+1.0m",
        "look up":                    "camera_tilt=+0.79 (45°)",
        "stop moving":                "linear_x=0, angular_z=0",
    }
    for r in results[:6]:
        hint = action_hints.get(r["primary_goal"], "→ action_head output")
        ros_lines.append(f"  '{r['original'][:28]}'")
        ros_lines.append(f"  → '{r['primary_goal']}'")
        ros_lines.append(f"  → {hint}")
        ros_lines.append("")

    ros_lines += [
        "── Gemini Adaptive Frequenz (B14) ───────────────",
        "",
        "  Hoher FE-Loss → Gemini oft fragen",
        "  Niedriger FE-Loss → Gemini selten",
        "  → Kostenoptimierung API-Calls",
    ]
    ax_ros.text(
        0.02, 0.98, "\n".join(ros_lines),
        transform=ax_ros.transAxes,
        fontsize=7.5, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8)
    )

    plt.show()

    # ── Interaktiver Modus ────────────────────────────────
    print("\n" + "="*55)
    print("Interaktiver Modus – eigene Befehle eingeben")
    print("(leer lassen oder 'q' zum Beenden)")
    print("="*55 + "\n")

    while True:
        try:
            cmd = input("Befehl: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd or cmd.lower() in ("q", "quit", "exit"):
            break

        result = gemini.translate(cmd)
        emb    = gemini.get_clip_embedding(result)

        print(f"\n  primary_goal:   '{result['primary_goal']}'")
        print(f"  confidence:     {result['confidence']:.2f}")
        print(f"  interpretation: {result.get('interpretation','')}")
        if result.get("camera_hint"):
            print(f"  camera_hint:    {result['camera_hint']}")
        if result.get("arc_hint"):
            print(f"  arc_hint:       {result['arc_hint']}")
        if result.get("avoid"):
            print(f"  avoid:          {result['avoid']}")
        print(f"  embedding:      shape=(512,) norm={np.linalg.norm(emb):.4f}")
        print(f"  source:         {result['source']}\n")

    print(f"\nGemini Zusammenfassung:")
    for k, v in gemini.summary().items():
        print(f"  {k:8s}: {v}")


if __name__ == "__main__":
    run_demo()
