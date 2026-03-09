"""
B15 – Reward-Kombination
==========================
Kombiniert alle Reward-Quellen zu einem finalen Trainings-Signal.

Zwei Gemini-Modelle (klar getrennt):
    gemini-2.5-flash              → Text-Interface (B13)
                                    "Geh zur roten Box" → CLIP-Phrase
    gemini-robotics-er-1.5-preview → Vision-Interface (B15)
                                    Kamerabild → Reward + Situation + Tipps

Reward-Quellen:
    1. r_intrinsic   : Curiosity / Prediction Error (B12)
    2. r_gemini_vis  : Gemini Robotics visuelles Assessment
                       → Wie gut sieht das Bild aus? Ziel erreicht?
    3. r_goal        : Cosinus-Ähnlichkeit Context ↔ Goal-Embedding (B10)
    4. r_action      : Aktions-Qualität (sigma-gewichtet)

Gesamt-Reward:
    r_total = w_int  * r_intrinsic
            + w_vis  * r_gemini_vis
            + w_goal * r_goal
            + w_act  * r_action

Gemini Robotics Prompt:
    Input:  Kamerabild (base64) + Ziel-Text + letzte Aktion
    Output: {
        "reward": 0.0-1.0,
        "goal_progress": 0.0-1.0,
        "situation": "kurze Beschreibung",
        "recommendation": "Trainings-Empfehlung",
        "obstacles": ["Liste"],
        "next_action_hint": "was der Roboter tun sollte"
    }

Adaptive Frequenz (B14):
    Gemini Robotics wird NUR aufgerufen wenn:
        - Hohe Free Energy (Agent verwirrt)
        - Hohe Novelty (neue Situation)
        - Timeout (periodisches Update)
        → Kosten-Kontrolle wie in B14
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO

import torch
import torch.nn.functional as F

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────
# GEMINI ROBOTICS VISUAL INTERFACE
# ─────────────────────────────────────────────

ROBOTICS_SYSTEM_PROMPT = """You are an AI assistant for a mobile robot equipped with a camera.
You evaluate camera images and provide training feedback.

Respond ONLY with valid JSON:
{
  "reward": 0.0-1.0,
  "goal_progress": 0.0-1.0,
  "situation": "brief description of the situation",
  "recommendation": "specific recommendation for the next action",
  "obstacles": ["detected obstacles"],
  "next_action_hint": "forward/left/right/stop/camera_down/camera_up/avoid_left/avoid_right",
  "confidence": 0.0-1.0
}

IMPORTANT BEHAVIOR WHEN APPROACHING:
Since objects are lower than the camera, they disappear at the bottom edge of the image when the robot gets very close!
When a target object is close, the robot MUST tilt the camera down ("camera_down") to avoid losing sight of it.
Pay close attention to objects at the bottom edge of the image, which are often only partially visible. 
In such cases, recommend driving back a bit and tilting the camera down so the target becomes fully visible again.
The robot must then navigate around obstacles, even if the target temporarily leaves the field of view.

Evaluation Rules:
- reward=1.0: Target clearly visible, centered, and close (success).
- reward=0.8: Target close, camera_down recommended.
- reward=0.6: Target clearly visible and centered, path is clear.
- reward=0.1: Target not visible or blocked.
"""

ROBOTICS_USER_TEMPLATE = """Current Goal: "{goal}"
Last Action: linear_x={linear_x:.2f} m/s, angular_z={angular_z:.2f} rad/s,
             camera_pan={camera_pan:.0f}°, camera_tilt={camera_tilt:.0f}°

Please evaluate the camera image and provide training feedback."""

class GeminiRoboticsInterface:
    """
    Gemini Robotics für visuelle Situation-Bewertung.

    Modell: gemini-robotics-er-1.5-preview
    Input:  Kamerabild (numpy uint8) + Ziel-Text + Aktion
    Output: reward, situation, recommendation, next_action_hint
    """

    #ROBOTICS_MODEL = "gemini-2.5-flash"
    ROBOTICS_MODEL = "gemini-robotics-er-1.5-preview"
    # Zukünftig: "models/gemini-robotics-er-1.5-preview"
    # Sobald das Modell öffentlich verfügbar ist, hier ersetzen

    def __init__(self, api_key: str = None):
        self.mode       = "mock"
        self.call_count = 0

        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")

        if GEMINI_AVAILABLE and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                self.mode   = "gemini_robotics"
                print(f"Gemini Robotics verbunden: {self.ROBOTICS_MODEL}")
            except Exception as e:
                print(f"Gemini Robotics Fehler: {e} – Mock-Modus")
        else:
            print("Gemini Robotics: Mock-Modus")

    def _image_to_base64(self, image_np: np.ndarray) -> str:
        """numpy uint8 (H,W,3) → base64 JPEG string"""
        if PIL_AVAILABLE:
            # Hochskalieren für bessere Gemini-Erkennung
            img_large = PILImage.fromarray(image_np).resize(
                (128, 128), PILImage.NEAREST
            )
            buf = BytesIO()
            img_large.save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode()
        else:
            # Fallback: direkt
            buf = BytesIO()
            np.save(buf, image_np)
            return base64.b64encode(buf.getvalue()).decode()

    def assess(
            self,
            image_np:   np.ndarray,
            goal_text:  str,
            last_action: dict = None,
    ) -> dict:
        """
        Bewertet Kamerabild und gibt Reward + Empfehlungen zurück.

        Args:
            image_np:    (H, W, 3) uint8
            goal_text:   Ziel-Text (aus B13/Gemini)
            last_action: dict mit linear_x, angular_z, camera_pan, camera_tilt

        Returns:
            dict mit reward, goal_progress, situation, recommendation, ...
        """
        self.call_count += 1

        if last_action is None:
            last_action = {"linear_x": 0, "angular_z": 0,
                           "camera_pan": 0, "camera_tilt": 0}

        if self.mode == "gemini_robotics":
            return self._call_gemini_robotics(image_np, goal_text, last_action)
        else:
            return self._mock_assess(image_np, goal_text, last_action)

    def _call_gemini_robotics(self, image_np, goal_text, last_action):
        img_b64  = self._image_to_base64(image_np)
        user_msg = ROBOTICS_USER_TEMPLATE.format(
            goal=goal_text, **last_action
        )

        try:
            response = self.client.models.generate_content(
                model=self.ROBOTICS_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=base64.b64decode(img_b64),
                        mime_type="image/jpeg"
                    ),
                    user_msg,
                ],
                config=types.GenerateContentConfig(
                    system_instruction=ROBOTICS_SYSTEM_PROMPT,
                    temperature=0.1,
                )
            )
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            result["source"] = "gemini_robotics"
            return result

        except Exception as e:
            print(f"Gemini Robotics Fehler: {e}")
            return self._mock_assess(image_np, goal_text, last_action)

    def _mock_assess(self, image_np, goal_text, last_action):
        """
        Regelbasiertes Mock-Assessment ohne API.
        Analysiert Farben im Bild um Objekte zu erkennen.
        """
        img = image_np.astype(float) / 255.0

        # Farb-Analyse für 6-Objekt-Umgebung:
        # 4 Boxen (red, yellow, orange, white) + 2 Bälle (green, blue)
        red_ratio    = ((img[:,:,0] > 0.6) & (img[:,:,1] < 0.3) & (img[:,:,2] < 0.3)).mean()
        blue_ratio   = ((img[:,:,2] > 0.5) & (img[:,:,0] < 0.2)).mean()
        green_ratio  = ((img[:,:,1] > 0.4) & (img[:,:,0] < 0.2) & (img[:,:,2] < 0.2)).mean()
        yellow_ratio = ((img[:,:,0] > 0.7) & (img[:,:,1] > 0.7) & (img[:,:,2] < 0.3)).mean()
        orange_ratio = ((img[:,:,0] > 0.7) & (img[:,:,1] > 0.3) & (img[:,:,1] < 0.6) & (img[:,:,2] < 0.2)).mean()
        white_ratio  = ((img[:,:,0] > 0.8) & (img[:,:,1] > 0.8) & (img[:,:,2] > 0.8)).mean()

        goal_lower = goal_text.lower()

        # Reward basierend auf Ziel-Übereinstimmung
        if "red box" in goal_lower or "rote" in goal_lower:
            reward = float(np.clip(red_ratio * 15, 0, 1))
            found  = red_ratio > 0.02
            situation = "Rote Box sichtbar" if found else "Rote Box nicht sichtbar"
        elif "yellow box" in goal_lower or "gelbe" in goal_lower:
            reward = float(np.clip(yellow_ratio * 15, 0, 1))
            found  = yellow_ratio > 0.02
            situation = "Gelbe Box sichtbar" if found else "Gelbe Box nicht sichtbar"
        elif "orange box" in goal_lower or "orange" in goal_lower:
            reward = float(np.clip(orange_ratio * 15, 0, 1))
            found  = orange_ratio > 0.02
            situation = "Orange Box sichtbar" if found else "Orange Box nicht sichtbar"
        elif "white box" in goal_lower or "weiße" in goal_lower or "weiss" in goal_lower:
            reward = float(np.clip(white_ratio * 12, 0, 1))
            found  = white_ratio > 0.02
            situation = "Weiße Box sichtbar" if found else "Weiße Box nicht sichtbar"
        elif "green ball" in goal_lower or "grüne" in goal_lower:
            reward = float(np.clip(green_ratio * 15, 0, 1))
            found  = green_ratio > 0.02
            situation = "Grüner Ball sichtbar" if found else "Grüner Ball nicht sichtbar"
        elif "blue ball" in goal_lower or "blau" in goal_lower:
            reward = float(np.clip(blue_ratio * 12, 0, 1))
            found  = blue_ratio > 0.02
            situation = "Blauer Ball sichtbar" if found else "Blauer Ball nicht sichtbar"
        else:
            reward    = 0.3
            situation = "Allgemeine Exploration"
            found     = False

        # Empfehlung
        if reward > 0.7:
            rec  = "Ziel nah – weiter vorwärts"
            hint = "vorwärts"
        elif reward > 0.3:
            rec  = "Ziel teilweise sichtbar – ausrichten"
            hint = "kamera_pan"
        else:
            rec  = "Ziel nicht sichtbar – erkunden"
            hint = "links" if np.random.rand() > 0.5 else "rechts"

        return {
            "reward":           reward,
            "goal_progress":    float(np.clip(reward * 0.8, 0, 1)),
            "situation":        situation,
            "recommendation":   rec,
            "obstacles":        [],
            "next_action_hint": hint,
            "confidence":       0.7 if found else 0.4,
            "source":           "mock_robotics",
        }


# ─────────────────────────────────────────────
# REWARD COMBINER
# ─────────────────────────────────────────────

class RewardCombiner:
    """
    Kombiniert alle Reward-Quellen zu einem finalen Signal.

    r_total = w_int  * r_intrinsic   (B12: Curiosity)
            + w_vis  * r_gemini_vis  (B15: Gemini Robotics)
            + w_goal * r_goal        (B10: Ziel-Nähe)
            + w_act  * r_action      (B09: Aktions-Qualität)
    """

    def __init__(
            self,
            w_intrinsic: float = 0.3,
            w_visual:    float = 0.4,
            w_goal:      float = 0.2,
            w_action:    float = 0.1,
    ):
        assert abs(w_intrinsic + w_visual + w_goal + w_action - 1.0) < 1e-6, \
            "Gewichte müssen sich zu 1.0 addieren!"

        self.w_intrinsic = w_intrinsic
        self.w_visual    = w_visual
        self.w_goal      = w_goal
        self.w_action    = w_action

        self.history = {k: [] for k in [
            "total", "intrinsic", "visual", "goal", "action",
            "goal_progress", "gemini_called",
        ]}

    def combine(
            self,
            r_intrinsic:  float,
            r_visual:     float,
            r_goal:       float,
            r_action:     float,
            goal_progress: float = 0.0,
            gemini_called: bool  = False,
    ) -> dict:
        # Alle Reward-Komponenten auf [0, 1] normalisieren
        r_intrinsic = float(np.clip(r_intrinsic, 0.0, 1.0))
        r_visual    = float(np.clip(r_visual,    0.0, 1.0))
        r_goal      = float(np.clip(r_goal,      0.0, 1.0))
        r_action    = float(np.clip(r_action,    0.0, 1.0))

        r_total = (
                self.w_intrinsic * r_intrinsic +
                self.w_visual    * r_visual    +
                self.w_goal      * r_goal      +
                self.w_action    * r_action
        )

        self.history["total"].append(r_total)
        self.history["intrinsic"].append(r_intrinsic)
        self.history["visual"].append(r_visual)
        self.history["goal"].append(r_goal)
        self.history["action"].append(r_action)
        self.history["goal_progress"].append(goal_progress)
        self.history["gemini_called"].append(1.0 if gemini_called else 0.0)

        return {
            "total":         r_total,
            "intrinsic":     r_intrinsic,
            "visual":        r_visual,
            "goal":          r_goal,
            "action":        r_action,
            "goal_progress": goal_progress,
        }

    def weighted_breakdown(self) -> dict:
        if not self.history["total"]:
            return {}
        return {
            "w_intrinsic * r": self.w_intrinsic * np.mean(self.history["intrinsic"][-20:]),
            "w_visual    * r": self.w_visual    * np.mean(self.history["visual"][-20:]),
            "w_goal      * r": self.w_goal      * np.mean(self.history["goal"][-20:]),
            "w_action    * r": self.w_action    * np.mean(self.history["action"][-20:]),
        }


# ─────────────────────────────────────────────
# SZENEN (aus B08)
# ─────────────────────────────────────────────

def draw_scene(scene_type: str) -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(10, 16):
        img[y, :] = [int(60+(y-10)*15)]*3
    img[0:2, :] = [40,40,60]
    img[2:10,1] = img[2:10,14] = [70,70,90]
    for y in range(2,8):
        img[y,2:14] = [100,100,120]
    if scene_type == "red_box":
        img[8:12,5:9]=[200,40,40]; img[6:9,6:10]=[160,30,30]; img[6:12,9]=[120,20,20]
    elif scene_type == "yellow_box":
        img[8:12,5:9]=[220,220,30]; img[6:9,6:10]=[180,180,20]; img[6:12,9]=[140,140,10]
    elif scene_type == "orange_box":
        img[8:12,5:9]=[220,130,20]; img[6:9,6:10]=[180,100,15]; img[6:12,9]=[140,80,10]
    elif scene_type == "white_box":
        img[8:12,5:9]=[230,230,230]; img[6:9,6:10]=[200,200,200]; img[6:12,9]=[170,170,170]
    elif scene_type == "green_ball":
        for y in range(16):
            for x in range(16):
                d=np.sqrt((x-8)**2+(y-10)**2)
                if d<3.2:
                    g=int(255*max(0,1-d/3.2)); h=int(80*max(0,1-((x-7)**2+(y-9)**2)/4))
                    img[y,x]=[0,min(255,g+h),0]
    elif scene_type == "blue_ball":
        for y in range(16):
            for x in range(16):
                d=np.sqrt((x-8)**2+(y-10)**2)
                if d<3.2:
                    b=int(255*max(0,1-d/3.2)); h=int(80*max(0,1-((x-7)**2+(y-9)**2)/4))
                    img[y,x]=[0,b//3,min(255,b+h)]
    img[10,2:14]=[50,50,50]
    return img

SCENE_TYPES = ["red_box", "green_ball", "blue_ball", "orange_box", "yellow_box", "white_box"]
SCENE_GOALS = {
    "red_box":    "find the red box",
    "green_ball": "find the green ball",
    "blue_ball":  "find the blue ball",
    "orange_box": "find the orange box",
    "yellow_box": "find the yellow box",
    "white_box":  "find the white box",
}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    api_key  = os.environ.get("GEMINI_API_KEY", "")
    N_STEPS  = 200

    robotics = GeminiRoboticsInterface(api_key=api_key)
    combiner = RewardCombiner(
        w_intrinsic=0.3,
        w_visual=0.4,
        w_goal=0.2,
        w_action=0.1,
    )

    # Adaptive Frequenz für Gemini Robotics (aus B14)
    from collections import deque
    last_vis_call = -20
    vis_interval  = 20   # Alle 20 Steps visuell bewerten

    print("\nB15 – Reward-Kombination")
    print("  Gemini 2.5 Flash:          Text-Interface  (B13)")
    print(f"  {robotics.ROBOTICS_MODEL}:")
    print("                             Vision-Interface (B15)")
    print()
    print("  Reward-Gewichte:")
    print(f"    w_intrinsic = {combiner.w_intrinsic}  (Curiosity B12)")
    print(f"    w_visual    = {combiner.w_visual}  (Gemini Robotics)")
    print(f"    w_goal      = {combiner.w_goal}  (Ziel-Nähe B10)")
    print(f"    w_action    = {combiner.w_action}  (Aktions-Qualität B09)")
    print()

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B15 – Reward-Kombination: Intrinsic + Gemini Robotics + Goal',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    ax_scenes  = [fig.add_subplot(gs[0, i]) for i in range(5)]
    ax_total   = fig.add_subplot(gs[1, :2])
    ax_stack   = fig.add_subplot(gs[1, 2:4])
    ax_gemini  = fig.add_subplot(gs[1, 4])
    ax_prog    = fig.add_subplot(gs[2, :2])
    ax_break   = fig.add_subplot(gs[2, 2:4])
    ax_stats   = fig.add_subplot(gs[2, 4])
    ax_gemini.axis('off')
    ax_stats.axis('off')

    last_assessment = {}
    print(f"Starte Demo: {N_STEPS} Schritte\n")

    for step in range(N_STEPS):
        scene_idx = (step // 20) % len(SCENE_TYPES)
        scene = SCENE_TYPES[scene_idx]
        goal  = SCENE_GOALS[scene]
        img   = draw_scene(scene)

        # ── Mock Reward-Quellen ────────────────────────────
        # r_intrinsic (B12): sinkt über Zeit
        r_intr = float(np.clip(0.3 * np.exp(-step/80) + 0.05 * np.random.rand(), 0, 1))

        # r_goal (B10): steigt über Zeit (Agent lernt Richtung Ziel)
        r_goal = float(np.clip(0.1 + 0.6 * (1 - np.exp(-step/60)) +
                               0.05 * np.random.randn(), 0, 1))

        # r_action (B09): stabil mit Rauschen
        r_act  = float(np.clip(0.5 + 0.1 * np.random.randn(), 0, 1))

        # ── Gemini Robotics (adaptiv) ──────────────────────
        gemini_called = False
        if step - last_vis_call >= vis_interval:
            last_action = {
                "linear_x":   float(np.random.uniform(0.1, 0.5)),
                "angular_z":  float(np.random.uniform(-0.3, 0.3)),
                "camera_pan": float(np.random.uniform(-45, 45)),
                "camera_tilt": float(np.random.uniform(-20, 20)),
            }
            assessment    = robotics.assess(img, goal, last_action)
            last_vis_call = step
            gemini_called = True
            last_assessment = assessment
            print(f"  Step {step:4d} [{scene:12s}] "
                  f"r_vis={assessment['reward']:.3f}  "
                  f"'{assessment['situation']}'")
            print(f"           → {assessment['recommendation']}")

        r_vis = last_assessment.get("reward", 0.3)
        g_prog = last_assessment.get("goal_progress", 0.0)

        # ── Reward kombinieren ─────────────────────────────
        reward = combiner.combine(
            r_intrinsic=r_intr,
            r_visual=r_vis,
            r_goal=r_goal,
            r_action=r_act,
            goal_progress=g_prog,
            gemini_called=gemini_called,
        )

        if step % 15 == 0 or step == N_STEPS - 1:
            h       = combiner.history
            steps_x = list(range(len(h["total"])))

            # ── Szenen ────────────────────────────────
            for i, s in enumerate(SCENE_TYPES):
                ax_scenes[i].clear()
                ax_scenes[i].imshow(draw_scene(s), interpolation='nearest')
                ax_scenes[i].axis('off')
                ax_scenes[i].set_facecolor('#0a0a0a')
                for spine in ax_scenes[i].spines.values():
                    spine.set_edgecolor('orange' if s==scene else 'gray')
                    spine.set_linewidth(2.5 if s==scene else 0.5)
                ax_scenes[i].set_title(
                    f'{s.replace("_"," ")}\n'
                    f'r={combiner.history["visual"][-1] if s==scene else "--":.3f}'
                    if s==scene else s.replace("_","\n"),
                    fontsize=7,
                    color='orange' if s==scene else 'white'
                )

            # ── Gesamt-Reward ──────────────────────────
            ax_total.clear()
            ax_total.plot(steps_x, h["total"],
                          color='white', linewidth=2, label='Total', alpha=0.7)
            if len(h["total"]) >= 15:
                ma = np.convolve(h["total"], np.ones(15)/15, mode='valid')
                ax_total.plot(range(14, len(h["total"])), ma,
                              color='gold', linewidth=2.5, label='MA-15')
            # Gemini-Call Markierungen
            for i, gc in enumerate(h["gemini_called"]):
                if gc > 0:
                    ax_total.axvline(i, color='cyan', linewidth=1,
                                     alpha=0.5)
            ax_total.set_title('Gesamt-Reward  |  Cyan = Gemini Robotics Call',
                               fontsize=9)
            ax_total.set_facecolor('#0d0d0d')
            ax_total.legend(fontsize=7)
            ax_total.tick_params(colors='white')

            # ── Stacked Area ───────────────────────────
            ax_stack.clear()
            n = len(h["total"])
            contrib = np.array([
                [combiner.w_intrinsic * v for v in h["intrinsic"]],
                [combiner.w_visual    * v for v in h["visual"]],
                [combiner.w_goal      * v for v in h["goal"]],
                [combiner.w_action    * v for v in h["action"]],
            ])
            ax_stack.stackplot(
                steps_x, contrib,
                labels=[f'Intrinsic (×{combiner.w_intrinsic})',
                        f'Visual   (×{combiner.w_visual})',
                        f'Goal     (×{combiner.w_goal})',
                        f'Action   (×{combiner.w_action})'],
                colors=['steelblue','gold','seagreen','mediumpurple'],
                alpha=0.8
            )
            ax_stack.set_title('Reward-Anteile (gewichtet, gestapelt)', fontsize=9)
            ax_stack.legend(fontsize=6, loc='upper left')
            ax_stack.set_facecolor('#0d0d0d')
            ax_stack.tick_params(colors='white')

            # ── Letztes Gemini Robotics Assessment ─────
            ax_gemini.clear()
            ax_gemini.axis('off')
            if last_assessment:
                src = last_assessment.get('source', 'mock')
                model_label = "gemini-robotics\n(mock)" if "mock" in src \
                    else "gemini-robotics\n✓ live"
                lines = [
                    "── Gemini Robotics ──────",
                    f"Modell: {robotics.ROBOTICS_MODEL[:22]}",
                    f"Calls:  {robotics.call_count}",
                    f"Quelle: {src}",
                    "",
                    "── Letztes Assessment ───",
                    f"Reward:   {last_assessment.get('reward',0):.3f}",
                    f"Progress: {last_assessment.get('goal_progress',0)*100:.0f}%",
                    f"Konfidenz:{last_assessment.get('confidence',0):.2f}",
                    "",
                    f"Situation:",
                    f"  {last_assessment.get('situation','')[:30]}",
                    "",
                    f"Empfehlung:",
                    f"  {last_assessment.get('recommendation','')[:30]}",
                    "",
                    f"Hint: {last_assessment.get('next_action_hint','')}",
                ]
                if last_assessment.get("obstacles"):
                    lines += ["", f"⚠ {last_assessment['obstacles']}"]
            else:
                lines = ["Noch kein Assessment..."]

            ax_gemini.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_gemini.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9),
                color='lightcyan'
            )

            # ── Goal Progress ──────────────────────────
            ax_prog.clear()
            ax_prog.plot(steps_x, h["goal_progress"],
                         color='seagreen', linewidth=2,
                         label='Goal Progress')
            ax_prog.fill_between(steps_x, 0, h["goal_progress"],
                                 color='seagreen', alpha=0.3)
            ax_prog.axhline(1.0, color='gold', linestyle='--',
                            linewidth=1, label='Ziel erreicht')
            ax_prog.set_title('Ziel-Fortschritt (Goal Progress)', fontsize=9)
            ax_prog.set_ylim(0, 1.1)
            ax_prog.legend(fontsize=7)
            ax_prog.set_facecolor('#0d0d0d')
            ax_prog.tick_params(colors='white')

            # ── Reward Breakdown Balken ────────────────
            ax_break.clear()
            breakdown = combiner.weighted_breakdown()
            if breakdown:
                colors_b = ['steelblue','gold','seagreen','mediumpurple']
                bars = ax_break.bar(
                    list(breakdown.keys()),
                    list(breakdown.values()),
                    color=colors_b, alpha=0.85
                )
                total_b = sum(breakdown.values())
                for bar, val in zip(bars, breakdown.values()):
                    pct = val/total_b*100 if total_b > 0 else 0
                    ax_break.text(
                        bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.001,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=8
                    )
                ax_break.set_title(
                    f'Reward-Beitrag (Ø letzte 20 Steps)\n'
                    f'Total Ø = {total_b:.4f}',
                    fontsize=9
                )
                ax_break.tick_params(axis='x', labelsize=6.5)

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            total_now = h["total"][-1] if h["total"] else 0
            lines = [
                "── Reward Combiner ──────",
                f"w_intrinsic: {combiner.w_intrinsic}",
                f"w_visual:    {combiner.w_visual}",
                f"w_goal:      {combiner.w_goal}",
                f"w_action:    {combiner.w_action}",
                "",
                "── Aktuell ──────────────",
                f"Total:    {total_now:.4f}",
                f"Intr:     {h['intrinsic'][-1] if h['intrinsic'] else 0:.4f}",
                f"Visual:   {h['visual'][-1] if h['visual'] else 0:.4f}",
                f"Goal:     {h['goal'][-1] if h['goal'] else 0:.4f}",
                f"Action:   {h['action'][-1] if h['action'] else 0:.4f}",
                f"Progress: {h['goal_progress'][-1] if h['goal_progress'] else 0:.2%}",
                "",
                "── Gemini Modelle ───────",
                "Text (B13):",
                "  gemini-2.5-flash",
                "",
                "Vision (B15):",
                f"  {robotics.ROBOTICS_MODEL[:22]}",
                f"  Calls: {robotics.call_count}",
                f"  Interval: {vis_interval} Steps",
                "",
                "── Active Inference ─────",
                "r_total → Policy Update",
                "r_intr  → Exploration",
                "r_vis   → Gemini Prior",
                "r_goal  → Convergence",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.03)

    print("\nDemo abgeschlossen!")
    bd = combiner.weighted_breakdown()
    print("Reward-Breakdown (Ø letzte 20 Steps):")
    for k, v in bd.items():
        print(f"  {k}: {v:.5f}")
    print(f"  Gesamt:          {sum(bd.values()):.5f}")
    print()
    print("Gemini Modelle:")
    print("  Text  : gemini-2.5-flash  (B13 – natürliche Sprache)")
    print(f"  Vision: {robotics.ROBOTICS_MODEL}")
    print(f"         (B15 – visuelle Bewertung, {robotics.call_count} Calls)")
    print()
    print("Naechste Schritte:")
    print("  B16 – Vollintegration aller Bausteine")
    print("  B17 – MiniWorld Env live")
    print("  B18 – Visualisierungs-Dashboard")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_demo()
