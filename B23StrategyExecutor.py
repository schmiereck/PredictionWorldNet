"""
B23 – Strategy Executor
========================
Führt die von B22 generierten Strategien aus.

Kernaufgaben:
    1. Condition-Erkennung aus Bilddaten (obs_info)
    2. Regelauswertung (höchste Priorität gewinnt)
    3. Action-Vektor erzeugen
    4. Sigma-basiertes Blending mit NN-Aktionen

Blending-Logik:
    blend = sigmoid((mean_sigma - threshold) * steepness)
    final = blend * strategy_action + (1 - blend) * nn_action

    Hohe Sigma (NN unsicher) → Strategie dominiert
    Niedrige Sigma (NN sicher) → NN dominiert
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass
from collections import deque

from B22StrategyGenerator import (
    Strategy, Rule, ACTION_VECTORS, KNOWN_CONDITIONS,
    ESCAPE_WALL_DURATION, ESCAPE_WALL_PHASE1_END, ESCAPE_WALL_PHASE2_END,
)


# ─────────────────────────────────────────────
# CONDITION EVALUATOR
# ─────────────────────────────────────────────

class ConditionEvaluator:
    """
    Wertet Bedingungen anhand von Bilddaten und Zustand aus.
    Austauschbar: Kann durch ML-basierte Erkennung ersetzt werden.
    """

    def __init__(self,
                 target_color: str = "red",
                 stuck_threshold: int = 15,
                 timeout_steps: int = 20):
        self.target_color     = target_color
        self.stuck_threshold  = stuck_threshold
        self.timeout_steps    = timeout_steps

        # State-Tracking
        self._no_progress_count = 0
        self._last_obs_hash     = 0
        self._stuck_count       = 0
        self._pan_position      = 0.0   # -1.0 (links) bis 1.0 (rechts)
        self._pan_direction     = -1     # -1 = nach links, +1 = nach rechts
        self._pan_steps         = 0
        self._last_reward       = 0.0
        self._boring_count      = 0     # Steps mit niedrigem r_intr

    def evaluate(self, obs_info: dict) -> Dict[str, bool]:
        """
        Wertet alle bekannten Conditions aus.

        obs_info Keys:
            image_nn: np.ndarray (128,128,3) uint8
            reward: float (letzter Gemini-Reward)
            r_intr: float (intrinsischer Reward / Novelty)
            sigma: float (NN-Unsicherheit)
            cam_pan: float (-1..1, aktuelle Kamera-Position)
            step: int
        """
        img = obs_info.get("image_nn")
        reward = obs_info.get("reward", 0.0)
        r_intr = obs_info.get("r_intr", 0.05)
        cam_pan = obs_info.get("cam_pan", 0.0)

        # Farb-Erkennung
        target_mask = self._detect_target(img)
        target_ratio = np.mean(target_mask) if target_mask is not None else 0.0

        # Position im Bild (links/rechts/zentriert)
        target_pos = self._target_position(target_mask) if target_mask is not None else "none"

        # Stuck-Erkennung
        if img is not None:
            obs_hash = hash(img.tobytes()) % (2**32)
            if obs_hash == self._last_obs_hash:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
            self._last_obs_hash = obs_hash

        # wall_stuck: feststecken UND Bild gleichförmig (Wand füllt Frame)
        if img is not None:
            img_variance = float(np.var(img.astype(np.float32)))
        else:
            img_variance = 9999.0
        wall_stuck = (self._stuck_count >= self.stuck_threshold) and (img_variance < 200.0)

        # boring_scene: r_intr dauerhaft niedrig (Szene ändert sich kaum)
        if r_intr < 0.005:
            self._boring_count += 1
        else:
            self._boring_count = 0
        boring_scene = self._boring_count >= 30

        # Fortschritts-Tracking
        if reward <= self._last_reward + 0.01:
            self._no_progress_count += 1
        else:
            self._no_progress_count = 0
        self._last_reward = reward

        # Pan-Tracking
        self._pan_position = cam_pan
        pan_done = abs(cam_pan) > 0.8 and self._pan_steps > 5
        self._pan_steps += 1

        # target_below: Zielpixel konzentriert im unteren Bilddrittel
        # → Objekt ist nah und am unteren Bildrand (Kamera sollte runterschwenken)
        if target_mask is not None and img is not None:
            h = img.shape[0]
            lower_third = target_mask[h * 2 // 3:, :]
            target_below = (np.mean(lower_third) > 0.05) and (target_ratio < 0.12)
        else:
            target_below = False

        conditions = {
            "no_target":       target_ratio < 0.02,
            "target_left":     target_pos == "left",
            "target_right":    target_pos == "right",
            "target_centered": target_pos == "center",
            "target_close":    target_ratio > 0.15,
            "target_far":      0.02 <= target_ratio <= 0.15,
            "target_below":    target_below,
            "pan_done":        pan_done,
            "stuck":           self._stuck_count >= self.stuck_threshold,
            "wall_stuck":      wall_stuck,
            "boring_scene":    boring_scene,
            "timeout":         self._no_progress_count >= self.timeout_steps,
            "always":          True,
        }

        return conditions

    def reset_pan(self):
        """Wird aufgerufen wenn ein neuer Scan startet."""
        self._pan_steps = 0

    # HSV-Farbbereiche: (H_min, H_max, S_min, V_min)
    # H in [0, 360], S und V in [0, 1]
    _HSV_RANGES = {
        "red":    [(0, 15, 0.3, 0.25), (345, 360, 0.3, 0.25)],
        "green":  [(80, 160, 0.25, 0.2)],
        "blue":   [(200, 260, 0.25, 0.2)],
        "yellow": [(45, 75, 0.3, 0.3)],
        "orange": [(15, 45, 0.4, 0.3)],
        "white":  [(0, 360, 0.0, 0.6)],  # Spezialfall: S < 0.15
    }

    def _detect_target(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Erkennt Ziel-Pixel im Bild basierend auf HSV-Farbraum."""
        if img is None:
            return None

        f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
        r, g, b = f[:,:,0], f[:,:,1], f[:,:,2]

        # RGB → HSV
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-8

        # Hue [0, 360]
        h = np.zeros_like(r)
        rm = (cmax == r)
        gm = (cmax == g) & ~rm
        bm = ~rm & ~gm
        h[rm] = 60.0 * (((g[rm] - b[rm]) / delta[rm]) % 6)
        h[gm] = 60.0 * (((b[gm] - r[gm]) / delta[gm]) + 2)
        h[bm] = 60.0 * (((r[bm] - g[bm]) / delta[bm]) + 4)

        # Saturation [0, 1]
        s = np.where(cmax > 1e-8, delta / cmax, 0.0)
        v = cmax

        color = self.target_color.lower()
        ranges = self._HSV_RANGES.get(color)

        if ranges is None:
            # Unbekannte Farbe: "etwas Auffälliges" suchen
            gray = (r + g + b) / 3.0
            deviation = np.abs(r - gray) + np.abs(g - gray) + np.abs(b - gray)
            return deviation > 0.3

        if color == "white":
            return (s < 0.15) & (v > 0.6)

        mask = np.zeros_like(r, dtype=bool)
        for h_min, h_max, s_min, v_min in ranges:
            mask |= (h >= h_min) & (h <= h_max) & (s >= s_min) & (v >= v_min)
        return mask

    def _target_position(self, mask: np.ndarray) -> str:
        """Bestimmt ob das Ziel links, rechts oder zentriert ist."""
        if mask is None or np.sum(mask) < 3:
            return "none"

        h, w = mask.shape
        third = w // 3

        left_count   = np.sum(mask[:, :third])
        center_count = np.sum(mask[:, third:2*third])
        right_count  = np.sum(mask[:, 2*third:])

        total = left_count + center_count + right_count
        if total < 3:
            return "none"

        if center_count >= left_count and center_count >= right_count:
            return "center"
        elif left_count > right_count:
            return "left"
        else:
            return "right"

    def set_target_color(self, goal: str):
        """Extrahiert die Zielfarbe aus dem Goal-String."""
        g = goal.lower()
        for color in ["red", "green", "blue", "yellow", "orange", "white", "purple"]:
            if color in g:
                self.target_color = color
                return
        # Deutsche Farbnamen
        for de, en in [("rot", "red"), ("grün", "green"), ("blau", "blue"),
                       ("gelb", "yellow"), ("orange", "orange"), ("weiß", "white")]:
            if de in g:
                self.target_color = en
                return


# ─────────────────────────────────────────────
# STRATEGY EXECUTOR
# ─────────────────────────────────────────────

class StrategyExecutor:
    """
    Führt eine Strategy aus und liefert Action-Vektoren.

    Nutzung:
        executor = StrategyExecutor()
        executor.set_strategy(strategy)

        # Im Loop:
        action = executor.get_action(obs_info)
        if action is not None:
            blended = executor.blend(action, nn_action, sigma)
    """

    def __init__(self,
                 sigma_threshold: float = 0.4,
                 sigma_steepness: float = 8.0,
                 min_blend: float = 0.1,
                 max_blend: float = 0.9):
        self._strategy       = None
        self._evaluator      = ConditionEvaluator()
        self._sigma_threshold = sigma_threshold
        self._sigma_steepness = sigma_steepness
        self._min_blend       = min_blend
        self._max_blend       = max_blend

        # Aktuelle Aktion (wird für duration Steps beibehalten)
        self._current_action_name  = None
        self._current_action_vec   = None
        self._remaining_steps      = 0
        self._last_matched_rule    = None
        self._escape_turn_dir      = 1.0   # +1 oder -1, wird bei escape_wall gesetzt

        # Statistiken
        self.stats = {
            "strategy_steps":  0,
            "nn_steps":        0,
            "blended_steps":   0,
            "rules_matched":   {},
        }

    def set_strategy(self, strategy: Strategy):
        """Setzt eine neue Strategie."""
        self._strategy = strategy
        self._current_action_name = None
        self._remaining_steps = 0
        self._evaluator.set_target_color(strategy.goal)
        self._evaluator.reset_pan()
        print(f"  Executor: Strategie gesetzt")
        print(f"  {strategy}")

    @property
    def has_strategy(self) -> bool:
        return self._strategy is not None

    @property
    def active_rule(self) -> Optional[str]:
        """Gibt die aktuell aktive Regel zurück."""
        if self._last_matched_rule:
            return f"{self._last_matched_rule.condition} → {self._last_matched_rule.action}"
        return None

    def get_action(self, obs_info: dict) -> Optional[np.ndarray]:
        """
        Bestimmt die Strategie-Aktion basierend auf Conditions.

        Returns:
            np.ndarray (6,) mit Action-Werten, oder None wenn keine Regel greift.
        """
        if self._strategy is None:
            return None

        # Laufende Aktion fortsetzen?
        if self._remaining_steps > 0:
            self._remaining_steps -= 1
            # Spezialbehandlung: scan_panorama ist eine Sequenz
            if self._current_action_name == "scan_panorama":
                return self._compute_scan_panorama_action()
            # Spezialbehandlung: escape_wall ist eine 3-Phasen-Sequenz
            if self._current_action_name == "escape_wall":
                return self._compute_escape_wall_action()
            return self._current_action_vec.copy()

        # Conditions auswerten
        conditions = self._evaluator.evaluate(obs_info)

        # Regeln nach Priorität prüfen
        for rule in self._strategy.sorted_rules():
            if conditions.get(rule.condition, False):
                self._last_matched_rule = rule
                self._current_action_name = rule.action
                self._remaining_steps = rule.duration - 1

                # Action-Vektor holen
                vec = np.array(ACTION_VECTORS.get(
                    rule.action, [0, 0, 0, 0, 0, 0]
                ), dtype=np.float32)

                # Spezialbehandlung: random_turn
                if rule.action == "random_turn":
                    vec[1] = np.random.choice([-0.7, 0.7])

                # Spezialbehandlung: scan_panorama (alternierend links/rechts)
                if rule.action == "scan_panorama":
                    self._evaluator.reset_pan()

                # Spezialbehandlung: escape_wall (Drehrichtung zufällig wählen)
                if rule.action == "escape_wall":
                    self._escape_turn_dir = float(np.random.choice([-1.0, 1.0]))

                # Spezialbehandlung: timeout – Counter zurücksetzen damit keine Endlosschleife
                if rule.condition == "timeout":
                    self._evaluator._no_progress_count = 0

                self._current_action_vec = vec

                # Statistik
                key = f"{rule.condition}→{rule.action}"
                self.stats["rules_matched"][key] = \
                    self.stats["rules_matched"].get(key, 0) + 1

                return vec.copy()

        return None  # Keine Regel hat gepasst → NN entscheidet

    def _compute_scan_panorama_action(self) -> np.ndarray:
        """
        Berechnet scan_panorama als Full Sweep: links → Mitte → rechts → Mitte.

        Duration: 16 Steps
        - Steps 0-4:   Links schwenken (cam_pan = -0.7)
        - Steps 5-7:   Zurück zur Mitte (cam_pan = +0.7)
        - Steps 8-12:  Rechts schwenken (cam_pan = +0.7)
        - Steps 13-15: Zurück zur Mitte (cam_pan = -0.7)

        Resultat: Kamera oszilliert von -70° bis +70° und endet bei 0°
        """
        # Aktueller Step in der Sequenz (0 = Start, 15 = Ende)
        total_duration = 16
        current_step = total_duration - 1 - self._remaining_steps

        if current_step < 5:
            # Phase 1: Links schwenken bis ca. -70°
            cam_pan = -0.7
        elif current_step < 8:
            # Phase 2: Zurück zur Mitte (0°)
            cam_pan = 0.7
        elif current_step < 13:
            # Phase 3: Rechts schwenken bis ca. +70°
            cam_pan = 0.7
        else:
            # Phase 4: Zurück zur Mitte (0°)
            cam_pan = -0.7

        return np.array([0.0, 0.0, cam_pan, 0.0, 0.0, 0.0], dtype=np.float32)

    def _compute_escape_wall_action(self) -> np.ndarray:
        """
        Wandflucht-Sequenz (3 Phasen, Gesamtdauer 22 Steps):
          Phase 1 (Steps 0-2):   Stopp + Kamera zentrieren
          Phase 2 (Steps 3-7):   Rückwärts fahren (weg von der Wand)
          Phase 3 (Steps 8-21):  Zufällig drehen (Richtung einmal gewählt)
        """
        current_step = ESCAPE_WALL_DURATION - 1 - self._remaining_steps

        if current_step < ESCAPE_WALL_PHASE1_END:
            # Phase 1: Anhalten und Kamera geradeaus
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif current_step < ESCAPE_WALL_PHASE2_END:
            # Phase 2: Rückwärts
            return np.array([-0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Phase 3: Drehen in gewählter Richtung
            return np.array([0.0, 0.8 * self._escape_turn_dir, 0.0, 0.0, 0.0, 0.0],
                            dtype=np.float32)

    def blend(self, strategy_action: np.ndarray,
              nn_action: np.ndarray,
              sigma: float) -> np.ndarray:
        """
        Blendet Strategie- und NN-Aktion basierend auf Sigma.

        sigma hoch → mehr Strategie (NN unsicher)
        sigma niedrig → mehr NN (NN sicher)
        """
        factor = self.blend_factor(sigma)

        if factor > 0.95:
            self.stats["strategy_steps"] += 1
        elif factor < 0.05:
            self.stats["nn_steps"] += 1
        else:
            self.stats["blended_steps"] += 1

        return factor * strategy_action + (1 - factor) * nn_action

    def blend_factor(self, sigma: float) -> float:
        """
        Berechnet den Blend-Faktor (0 = nur NN, 1 = nur Strategie).
        Sigmoid-Funktion für smooth Übergang.
        """
        x = (sigma - self._sigma_threshold) * self._sigma_steepness
        raw = 1.0 / (1.0 + np.exp(-x))
        return np.clip(raw, self._min_blend, self._max_blend)

    def summary(self) -> dict:
        """Statistik-Zusammenfassung."""
        total = (self.stats["strategy_steps"] +
                 self.stats["nn_steps"] +
                 self.stats["blended_steps"])
        return {
            "total_steps":     total,
            "strategy_pct":    self.stats["strategy_steps"] / max(1, total) * 100,
            "nn_pct":          self.stats["nn_steps"] / max(1, total) * 100,
            "blended_pct":     self.stats["blended_steps"] / max(1, total) * 100,
            "top_rules":       sorted(self.stats["rules_matched"].items(),
                                      key=lambda x: x[1], reverse=True)[:5],
            "active_rule":     self.active_rule,
        }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from B22StrategyGenerator import MockStrategyGenerator

    print("=== StrategyExecutor Demo ===\n")

    gen = MockStrategyGenerator()
    strategy = gen.generate("find the red box")

    executor = StrategyExecutor()
    executor.set_strategy(strategy)

    # Simuliere einige Steps
    for i in range(20):
        # Fake obs_info
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        # Ab Step 10: rotes Objekt links einblenden
        if i >= 10:
            img[:, :5, 0] = 200
            img[:, :5, 1] = 30
            img[:, :5, 2] = 30

        obs_info = {
            "image_nn": img,
            "reward": 0.1 if i < 10 else 0.5,
            "sigma": 0.6 if i < 15 else 0.2,
            "cam_pan": 0.0,
            "step": i,
        }

        action = executor.get_action(obs_info)
        if action is not None:
            nn_action = np.random.randn(6).astype(np.float32) * 0.3
            blended = executor.blend(action, nn_action, obs_info["sigma"])
            rule = executor.active_rule
            bf = executor.blend_factor(obs_info["sigma"])
            print(f"  Step {i:2d}: {rule:30s} blend={bf:.2f}")
        else:
            print(f"  Step {i:2d}: NN entscheidet")

    print()
    print("Statistik:", executor.summary())
