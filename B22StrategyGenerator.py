"""
B22 – Strategy Generator
=========================
Generiert Explorations-Strategien für Roboter-Ziele.

Zwei Implementierungen (austauschbar):
    - GeminiStrategyGenerator: Gemini generiert Regeln dynamisch
    - MockStrategyGenerator:   Fest codierte Such-Strategie als Fallback

Eine Strategy besteht aus einer Liste von Rules.
Jede Rule hat eine Bedingung (Condition) und eine Aktion.

Beispiel:
    goal = "find the red box"
    → Rules:
        1. target_centered + close → stop (Ziel erreicht)
        2. target_centered → move_forward
        3. target_left     → turn_left
        4. target_right    → turn_right
        5. no_target       → pan_camera (systematisch suchen)
        6. pan_done        → turn_left (weiterdrehen, neuer Blickwinkel)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import json
import os
import time

# ─────────────────────────────────────────────
# DATENSTRUKTUREN
# ─────────────────────────────────────────────

@dataclass
class Rule:
    """Eine einzelne Verhaltensregel."""
    condition: str          # z.B. "no_target", "target_left", "target_centered"
    action: str             # z.B. "pan_left", "turn_left", "move_forward"
    duration: int = 3       # Wie viele Steps diese Aktion ausführen
    priority: int = 0       # Höher = wird zuerst geprüft

    def to_dict(self) -> dict:
        return {"condition": self.condition, "action": self.action,
                "duration": self.duration, "priority": self.priority}

    @staticmethod
    def from_dict(d: dict) -> Rule:
        return Rule(
            condition=d["condition"],
            action=d["action"],
            duration=d.get("duration", 3),
            priority=d.get("priority", 0),
        )


@dataclass
class Strategy:
    """Eine vollständige Explorations-Strategie für ein Ziel."""
    goal: str
    rules: List[Rule] = field(default_factory=list)
    description: str = ""
    source: str = "unknown"       # "gemini" oder "mock"

    def sorted_rules(self) -> List[Rule]:
        """Rules nach Priorität sortiert (höchste zuerst)."""
        return sorted(self.rules, key=lambda r: r.priority, reverse=True)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "rules": [r.to_dict() for r in self.rules],
            "description": self.description,
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> Strategy:
        return Strategy(
            goal=d["goal"],
            rules=[Rule.from_dict(r) for r in d.get("rules", [])],
            description=d.get("description", ""),
            source=d.get("source", "unknown"),
        )

    def __str__(self):
        lines = [f"Strategy: {self.goal} ({self.source})"]
        if self.description:
            lines.append(f"  {self.description}")
        for r in self.sorted_rules():
            lines.append(f"  P{r.priority}: {r.condition} → {r.action} ({r.duration} steps)")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# BEKANNTE CONDITIONS
# ─────────────────────────────────────────────

KNOWN_CONDITIONS = {
    "no_target",          # Ziel nicht im Bild sichtbar
    "target_left",        # Ziel in linker Bildhälfte
    "target_right",       # Ziel in rechter Bildhälfte
    "target_centered",    # Ziel in Bildmitte
    "target_close",       # Ziel nah (großer Anteil im Bild)
    "target_far",         # Ziel weit weg (kleiner Anteil)
    "target_below",       # Ziel im unteren Bilddrittel sichtbar (nah, Kamera zu hoch)
    "pan_done",           # Kamera hat vollen Schwenk abgeschlossen
    "stuck",              # Agent bewegt sich nicht (keine Veränderung)
    "wall_stuck",         # Feststecken AN einer Wand (uniform + stuck)
    "boring_scene",       # r_intr lange tief: Szene ändert sich kaum
    "timeout",            # N Steps ohne Fortschritt
    "always",             # Immer wahr (niedrigste Priorität)
}

# ─────────────────────────────────────────────
# BEKANNTE ACTIONS
# ─────────────────────────────────────────────

KNOWN_ACTIONS = {
    "move_forward",       # Vorwärts fahren
    "move_backward",      # Rückwärts fahren
    "turn_left",          # Roboter nach links drehen
    "turn_right",         # Roboter nach rechts drehen
    "pan_left",           # Kamera nach links schwenken
    "pan_right",          # Kamera nach rechts schwenken
    "center_camera",      # Kamera zurück zur Mitte
    "stop",               # Anhalten
    "random_turn",        # Zufällige Drehung (Exploration)
    "scan_panorama",      # Volle 360° Kamera-Schwenk-Sequenz
    "tilt_down",          # Kamera leicht nach unten schwenken (Objekt am Boden)
    "escape_wall",        # Wandflucht: Kamera zentrieren → rückwärts → drehen
}

# Action → 6D Array Mapping [linear_x, angular_z, cam_pan, cam_tilt, arc_radius, duration]
ACTION_VECTORS = {
    "move_forward":   [ 0.8,  0.0,  0.0,  0.0, 0.0, 0.0],
    "move_backward":  [-0.5,  0.0,  0.0,  0.0, 0.0, 0.0],
    "turn_left":      [ 0.0,  0.7,  0.0,  0.0, 0.0, 0.0],
    "turn_right":     [ 0.0, -0.7,  0.0,  0.0, 0.0, 0.0],
    "pan_left":       [ 0.0,  0.0, -0.6,  0.0, 0.0, 0.0],
    "pan_right":      [ 0.0,  0.0,  0.6,  0.0, 0.0, 0.0],
    "center_camera":  [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
    "tilt_down":      [ 0.0,  0.0,  0.0, -0.35, 0.0, 0.0],  # cam_tilt: ~-16° (Objekt unten)
    "stop":           [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
    "random_turn":    [ 0.0,  0.5,  0.0,  0.0, 0.0, 0.0],  # wird randomisiert
    "scan_panorama":  [ 0.0,  0.0, -0.8,  0.0, 0.0, 0.0],  # sequenziell
    "escape_wall":    [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0],  # sequenziell (3 Phasen)
}

# Wandflucht-Sequenz (escape_wall): 3 Phasen, Gesamtdauer 22 Steps
# Phase 1 (Steps 0-2):   Stopp + Kamera zentrieren
# Phase 2 (Steps 3-7):   Rückwärts fahren (weg von der Wand)
# Phase 3 (Steps 8-21):  Zufällig drehen bis r_intr wieder steigt
ESCAPE_WALL_DURATION    = 22
ESCAPE_WALL_PHASE1_END  = 3
ESCAPE_WALL_PHASE2_END  = 8


# ─────────────────────────────────────────────
# ABSTRACT BASE: StrategyGenerator
# ─────────────────────────────────────────────

class StrategyGenerator(ABC):
    """Interface für Strategy-Generierung."""

    @abstractmethod
    def generate(self, goal: str) -> Strategy:
        """Erzeugt eine Explorations-Strategie für das gegebene Ziel."""
        ...


# ─────────────────────────────────────────────
# MOCK STRATEGY GENERATOR
# ─────────────────────────────────────────────

class MockStrategyGenerator(StrategyGenerator):
    """
    Fest codierte Such-Strategie.
    Universell einsetzbar für "finde Objekt X"-Aufgaben.
    """

    def generate(self, goal: str) -> Strategy:
        rules = [
            # Höchste Priorität: Ziel nah + zentriert → Geschafft
            Rule("target_close",    "stop",          duration=1,  priority=100),
            # Ziel am unteren Bildrand: Kamera leicht runter (Hinweis, kein Zwang)
            # Niedrige duration (2) + Sigma-Blending → NN lernt selbst den richtigen Tilt
            Rule("target_below",    "tilt_down",     duration=2,  priority=93),
            # Wandflucht: Kamera zentrieren → rückwärts → drehen (höher als stuck!)
            Rule("wall_stuck",      "escape_wall",   duration=ESCAPE_WALL_DURATION, priority=97),
            # Langweilige Szene (gegen Wand / r_intr tief) → ebenfalls flüchten
            Rule("boring_scene",    "escape_wall",   duration=ESCAPE_WALL_DURATION, priority=96),
            # Feststecken (ohne Wand) → rückwärts
            Rule("stuck",           "move_backward", duration=5,  priority=95),
            # Ziel sichtbar und zentriert → drauf zu
            Rule("target_centered", "move_forward",  duration=5,  priority=90),
            # Ziel links → nach links drehen
            Rule("target_left",     "turn_left",     duration=3,  priority=80),
            # Ziel rechts → nach rechts drehen
            Rule("target_right",    "turn_right",    duration=3,  priority=80),
            # Timeout → Vorwärts fahren (höher als no_target, bricht Dreh-Schleife auf)
            # Ohne diese Regel dreht der Roboter ewig (no_target immer wahr, stuck erkennt Drehen nicht)
            Rule("timeout",         "move_forward",  duration=10, priority=75),
            # Nichts sichtbar → Roboter kurz drehen (Kamera bleibt vorne = kohärenter Bildstrom)
            Rule("no_target",       "turn_left",     duration=4,  priority=50),
            # Kamera zentrieren falls sie versehentlich abgewichen ist
            # duration=6: Servo braucht bei 0.35 rad/Step ~5 Steps von ±90° auf 0°
            Rule("pan_done",        "center_camera", duration=6,  priority=45),
            # Fallback: Weiterdrehen wenn sonst nichts greift
            Rule("always",          "turn_left",     duration=4,  priority=25),
        ]

        return Strategy(
            goal=goal,
            rules=rules,
            description="Standard-Suchstrategie: Drehen → Annähern (Kamera vorne)",
            source="mock",
        )


# ─────────────────────────────────────────────
# GEMINI STRATEGY GENERATOR
# ─────────────────────────────────────────────

GEMINI_STRATEGY_PROMPT = """Du bist ein Roboter-Strategie-Planer.
Der Roboter hat eine schwenkbare Kamera und kann sich drehen und fahren.

Erstelle eine Explorations-Strategie als Liste von Regeln für das Ziel: "{goal}"

Verfügbare Bedingungen (conditions):
{conditions}

Verfügbare Aktionen (actions):
{actions}

Antworte NUR mit JSON:
{{
  "description": "Kurze Beschreibung der Strategie",
  "rules": [
    {{"condition": "...", "action": "...", "duration": N, "priority": N}},
    ...
  ]
}}

Regeln mit höherer Priorität werden zuerst geprüft.
Typisch: target_close (P100) > target_centered (P90) > target_left/right (P80) > no_target (P50) > pan_done (P40)

Wichtiger Hinweis: Für no_target bevorzuge turn_left/turn_right statt scan_panorama.
scan_panorama bewegt die Kamera schnell hin und her und erzeugt unzusammenhängende Bilder.
Roboter-Drehung hält die Kamera vorne und gibt dem NN einen kohärenten Bildstrom.
Erstelle 6-10 sinnvolle Regeln."""


class GeminiStrategyGenerator(StrategyGenerator):
    """
    Gemini generiert dynamisch Regeln basierend auf dem Ziel.
    Fällt auf MockStrategyGenerator zurück bei Fehlern.
    """

    def __init__(self, client=None, model: str = "gemini-2.5-flash"):
        self._client = client
        self._model  = model
        self._mock   = MockStrategyGenerator()
        self._cache  = {}  # goal → Strategy (einmal generieren, wiederverwenden)

    def generate(self, goal: str) -> Strategy:
        # Cache prüfen
        if goal in self._cache:
            print(f"  Strategie aus Cache: {goal}")
            return self._cache[goal]

        if self._client is None:
            strategy = self._mock.generate(goal)
            self._cache[goal] = strategy
            return strategy

        try:
            prompt = GEMINI_STRATEGY_PROMPT.format(
                goal=goal,
                conditions=", ".join(sorted(KNOWN_CONDITIONS)),
                actions=", ".join(sorted(KNOWN_ACTIONS)),
            )

            # Import hier um zirkuläre Abhängigkeiten zu vermeiden
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                return self._mock.generate(goal)

            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                )
            )
            text = resp.text.strip()

            # JSON extrahieren
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            # Rules validieren
            rules = []
            for rd in data.get("rules", []):
                cond = rd.get("condition", "")
                act  = rd.get("action", "")
                if cond in KNOWN_CONDITIONS and act in KNOWN_ACTIONS:
                    rules.append(Rule.from_dict(rd))
                else:
                    print(f"  Unbekannte Rule ignoriert: {cond} → {act}")

            if not rules:
                print("  Gemini gab keine gültigen Rules → Fallback auf Mock")
                return self._mock.generate(goal)

            strategy = Strategy(
                goal=goal,
                rules=rules,
                description=data.get("description", "Gemini-generiert"),
                source="gemini",
            )

            print(f"  Gemini-Strategie generiert: {len(rules)} Regeln")
            print(f"  {strategy.description}")

            self._cache[goal] = strategy
            return strategy

        except Exception as e:
            print(f"  Gemini Strategy Fehler: {e} → Fallback auf Mock")
            strategy = self._mock.generate(goal)
            self._cache[goal] = strategy
            return strategy


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MockStrategyGenerator ===\n")
    mock = MockStrategyGenerator()
    s = mock.generate("find the red box")
    print(s)
    print()

    print("=== Strategy als Dict ===\n")
    print(json.dumps(s.to_dict(), indent=2))
