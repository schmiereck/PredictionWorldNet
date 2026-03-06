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
    "pan_done",           # Kamera hat vollen Schwenk abgeschlossen
    "stuck",              # Agent bewegt sich nicht (keine Veränderung)
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
}

# Action → 6D Array Mapping [linear_x, angular_z, cam_pan, cam_tilt, grip, aux]
ACTION_VECTORS = {
    "move_forward":   [ 0.8,  0.0,  0.0,  0.0, 0.0, 0.0],
    "move_backward":  [-0.5,  0.0,  0.0,  0.0, 0.0, 0.0],
    "turn_left":      [ 0.0,  0.7,  0.0,  0.0, 0.0, 0.0],
    "turn_right":     [ 0.0, -0.7,  0.0,  0.0, 0.0, 0.0],
    "pan_left":       [ 0.0,  0.0, -0.6,  0.0, 0.0, 0.0],
    "pan_right":      [ 0.0,  0.0,  0.6,  0.0, 0.0, 0.0],
    "center_camera":  [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
    "stop":           [ 0.0,  0.0,  0.0,  0.0, 0.0, 0.0],
    "random_turn":    [ 0.0,  0.5,  0.0,  0.0, 0.0, 0.0],  # wird randomisiert
    "scan_panorama":  [ 0.0,  0.0, -0.8,  0.0, 0.0, 0.0],  # sequenziell
}


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
            Rule("target_close",    "stop",          duration=1, priority=100),
            # Ziel sichtbar und zentriert → drauf zu
            Rule("target_centered", "move_forward",  duration=5, priority=90),
            # Ziel links → nach links drehen
            Rule("target_left",     "turn_left",     duration=3, priority=80),
            # Ziel rechts → nach rechts drehen
            Rule("target_right",    "turn_right",    duration=3, priority=80),
            # Nichts sichtbar → Kamera schwenken (kompletter Sweep: links→rechts→mitte)
            Rule("no_target",       "scan_panorama", duration=16, priority=50),
            # Kamera-Schwenk fertig → erst Kamera zentrieren
            Rule("pan_done",        "center_camera", duration=3, priority=45),
            # Feststecken → rückwärts + drehen
            Rule("stuck",           "move_backward", duration=3, priority=95),
            # Timeout → zufällig drehen
            Rule("timeout",         "random_turn",   duration=4, priority=30),
            # Fallback: Wenn nichts anderes greift → Roboter langsam drehen
            Rule("always",          "turn_left",     duration=5, priority=25),
        ]

        return Strategy(
            goal=goal,
            rules=rules,
            description=f"Standard-Suchstrategie: Scannen → Drehen → Annähern",
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
