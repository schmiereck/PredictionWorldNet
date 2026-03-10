"""
B19 – Vollintegration: Dirigent
=================================
Verbindet B16 (ML-Kern) + B17 (I/O) + B18 (Dashboard)
zu einem vollständigen, lauffähigen System.

Importiert:
    B16FullIntegration  → IntegratedSystem
    B17RobotInterfaces  → RobotInterface, ObsSource, ActionSink
    B18Dashboard        → TrainingDashboard

Neue Komponente:
    MiniWorldObsSource  → gym MiniWorld als ObservationSource
                          (fällt auf Mock zurück wenn nicht installiert)

Start-Modi:
    python B19Orchestrator.py               → Mock (immer verfügbar)
    python B19Orchestrator.py --miniworld   → MiniWorld Gym
    python B19Orchestrator.py --ros2        → ROS2 (Platzhalter)

Architektur:
    ┌─────────────────────────────────────────────────────┐
    │  B19 Orchestrator                                   │
    │                                                     │
    │  ObsSource  →  RobotInterface  →  ActionSink        │
    │      ↓                                              │
    │  Observation (low-res + high-res)                   │
    │      ↓                 ↓                            │
    │  B16 IntegratedSystem  Gemini ER (high-res)         │
    │      ↓                                              │
    │  action_array (6D)                                  │
    │      ↓                                              │
    │  RobotInterface.step()                              │
    │      ↓                                              │
    │  B18 Dashboard.update()                             │
    └─────────────────────────────────────────────────────┘
"""

import sys
import os
import argparse
import time
import numpy as np

# ─────────────────────────────────────────────
# IMPORTS AUS B16 / B17 / B18
# (Pfad-unabhängig: sucht im gleichen Ordner)
# ─────────────────────────────────────────────

import importlib.util

def _load(filename: str, classname: str = None):
    """Lädt ein Modul aus dem gleichen Verzeichnis."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nicht gefunden: {path}")
    mod_name = filename[:-3]
    spec   = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # nötig für @dataclass
    spec.loader.exec_module(module)
    return module

print("Lade Bausteine...")
try:
    _b16 = _load("B16FullIntegration.py")
    IntegratedSystem = _b16.IntegratedSystem
    GeminiClients    = _b16.GeminiClients
    SCENE_TYPES      = _b16.SCENE_TYPES
    SCENE_GOALS      = _b16.SCENE_GOALS
    SCENE_ACTIONS    = _b16.SCENE_ACTIONS
    print("  B16 IntegratedSystem      ✓")
except Exception as e:
    print(f"  B16 FEHLER: {e}")
    sys.exit(1)

try:
    _b17 = _load("B17RobotInterfaces.py")
    RobotInterface   = _b17.RobotInterface
    MockObsSource    = _b17.MockObsSource
    MockActionSink   = _b17.MockActionSink
    ROS2ObsSource    = _b17.ROS2ObsSource
    ROS2ActionSink   = _b17.ROS2ActionSink
    Action           = _b17.Action
    print("  B17 RobotInterface        ✓")
except Exception as e:
    print(f"  B17 FEHLER: {e}")
    sys.exit(1)

try:
    _b18 = _load("B18Dashboard.py")
    TrainingDashboard = _b18.TrainingDashboard
    print("  B18 TrainingDashboard     ✓")
except Exception as e:
    print(f"  B18 FEHLER: {e}")
    sys.exit(1)

try:
    _ovm = _load("OverheadMapView.py")
    OverheadMapView = _ovm.OverheadMapView
    print("  OverheadMapView           ✓")
except Exception as e:
    print(f"  OverheadMapView nicht gefunden: {e}")
    OverheadMapView = None

try:
    _b22 = _load("B22StrategyGenerator.py")
    _b23 = _load("B23StrategyExecutor.py")
    MockStrategyGenerator  = _b22.MockStrategyGenerator
    GeminiStrategyGenerator = _b22.GeminiStrategyGenerator
    StrategyExecutor       = _b23.StrategyExecutor
    print("  B22+B23 Strategie         ✓")
except Exception as e:
    print(f"  B22/B23 nicht gefunden: {e}")
    MockStrategyGenerator = None
    GeminiStrategyGenerator = None
    StrategyExecutor = None

print()

import matplotlib
matplotlib.use('TkAgg')

# ─────────────────────────────────────────────
# CUSTOM MINIWORLD ENVIRONMENT
# ─────────────────────────────────────────────

def _register_prediction_world_env(gym):
    """Registriert PredictionWorld-OneRoom mit mehreren Objekten."""
    env_id = "PredictionWorld-OneRoom-v0"
    if env_id in gym.envs.registry:
        return

    from miniworld.envs.oneroom import OneRoom
    from miniworld.entity import Box, Ball, COLORS, COLOR_NAMES

    # Fehlende Farben nachtragen
    if "orange" not in COLORS:
        COLORS["orange"] = np.array([1.0, 0.5, 0.0])
    if "white" not in COLORS:
        COLORS["white"] = np.array([1.0, 1.0, 1.0])
    for c in ("orange", "white"):
        if c not in COLOR_NAMES:
            COLOR_NAMES.append(c)

    class PredictionWorldRoom(OneRoom):
        """OneRoom mit mehreren farbigen Objekten."""

        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            # Boxen
            self.box        = self.place_entity(Box(color="red"))
            self.box_yellow = self.place_entity(Box(color="yellow"))
            self.box_white  = self.place_entity(Box(color="white"))
            self.box_orange = self.place_entity(Box(color="orange"))

            # Kugeln
            self.ball_green = self.place_entity(Ball(color="green"))
            self.ball_blue  = self.place_entity(Ball(color="blue"))

            self.place_agent()
            # Kamera tiefer setzen (Hexapod-Perspektive)
            from B16FullIntegration import CAM_HEIGHT
            self.agent.cam_height = CAM_HEIGHT

    gym.register(
        id=env_id,
        entry_point=lambda **kw: PredictionWorldRoom(**kw),
        max_episode_steps=300,
    )


# ─────────────────────────────────────────────
# MINIWORLD OBSERVATION SOURCE
# ─────────────────────────────────────────────

class MiniWorldObsSource(_b17.ObservationSource):
    """
    MiniWorld Gym als ObservationSource.
    Fällt auf MockObsSource zurück wenn miniworld nicht installiert.

    Installation:
        pip install miniworld

    Verfügbare Envs:
        PredictionWorld-OneRoom-v0  (mehrere Objekte)
        MiniWorld-Hallway-v0
        MiniWorld-OneRoom-v0
        MiniWorld-FourRooms-v0
        MiniWorld-TMaze-v0
        MiniWorld-Maze-v0
    """

    def __init__(self, env_name: str = "PredictionWorld-OneRoom-v0",
                 low_res=(128,128), high_res=(256,256),
                 render_mode: str = "rgb_array"):
        self._env_name  = env_name
        self._low_res   = low_res
        self._high_res  = high_res
        self._env       = None
        self._obs       = None
        self._frame     = 0
        self._available = False
        self._cam_pan        = 0.0    # Kamera-Pan aktuell in rad (-1.57 .. +1.57)
        self._cam_pan_target = 0.0    # Zielwinkel (Servo-Simulation)
        self._cam_tilt       = 0.0    # Kamera-Tilt (für spätere Nutzung)
        self._CAM_PAN_SPEED  = 0.35   # rad pro Step
        self.episode_reset   = False
        self._needs_reset    = False

        try:
            import gymnasium as gym
            import miniworld
            import miniworld.entity as _mw_entity
            self._gym = gym

            # Farben "orange" und "white" ergänzen (fehlen in MiniWorld)
            if "orange" not in _mw_entity.COLORS:
                _mw_entity.COLORS["orange"] = np.array([1.0, 0.5, 0.0])
            if "white" not in _mw_entity.COLORS:
                _mw_entity.COLORS["white"] = np.array([1.0, 1.0, 1.0])

            # Custom Environment registrieren (falls noch nicht geschehen)
            _register_prediction_world_env(gym)

            self._env = gym.make(env_name, render_mode=render_mode,
                                 view="agent",
                                 obs_width=low_res[0],
                                 obs_height=low_res[1])
            obs, _    = self._env.reset()
            self._obs = obs
            self._available = True

            # High-Res Framebuffer für Gemini (nativ gerendert, kein Upscale)
            from miniworld.opengl import FrameBuffer as _FB
            self._highres_fb = _FB(high_res[0], high_res[1], 8)

            print(f"  MiniWorld: {env_name}  ✓")
            print(f"    Obs shape: {obs.shape}  |  High-Res FB: {high_res}")
        except ImportError:
            print(f"  MiniWorld nicht installiert → Mock-Fallback")
            print(f"    pip install miniworld gymnasium")
        except Exception as e:
            print(f"  MiniWorld Fehler: {e} → Mock-Fallback")

        if not self._available:
            self._mock = MockObsSource(low_res=low_res, high_res=high_res)

    def _resize(self, img: np.ndarray, size: tuple) -> np.ndarray:
        try:
            from PIL import Image as PILImage
            return np.array(
                PILImage.fromarray(img).resize(
                    (size[1], size[0]), PILImage.BILINEAR)
            )
        except ImportError:
            # Grob skalieren ohne PIL
            h, w = size
            return img[::img.shape[0]//h, ::img.shape[1]//w, :][:h,:w,:]

    def _render_with_pan(self, frame_buffer=None):
        """Rendert Bild mit simuliertem Kamera-Pan und Kamera-Tilt.

        Pan:  Dreht agent.dir temporär um den Pan-Winkel (horizontal).
        Tilt: Setzt agent.cam_pitch temporär (vertikal, in Grad).
              Positiv = nach oben, negativ = nach unten.
        Beide Werte werden nach dem Rendern wiederhergestellt.
        """
        import math
        agent = self._env.unwrapped.agent
        original_dir   = agent.dir
        original_pitch = agent.cam_pitch
        
        # MiniWorld: dir ist Winkel in rad (0 = +x, pi/2 = +z, pi = -x)
        # Pan ist bei uns: >0 = rechts schauen, <0 = links schauen
        # Wir addieren den Pan, damit sich die Blickrichtung verschiebt
        agent.dir       = original_dir + self._cam_pan
        agent.cam_pitch = self._cam_tilt * 180.0 / math.pi     # rad → Grad (MiniWorld erwartet Grad)
        obs = self._env.unwrapped.render_obs(frame_buffer=frame_buffer)
        agent.dir       = original_dir
        agent.cam_pitch = original_pitch
        return obs

    def get_observation(self) -> _b17.Observation:
        if not self._available:
            return self._mock.get_observation()

        self._frame += 1
        # Kamera-Bild mit Pan-Offset rendern
        raw = self._render_with_pan()
        img_low = self._resize(raw, self._low_res)
        return _b17.Observation(
            image=img_low.astype(np.uint8),
            timestamp=time.time(),
            frame_id=self._frame,
            source="miniworld",
            metadata={"env": self._env_name,
                      "cam_pan": self._cam_pan},
        )

    def get_high_res(self) -> _b17.Observation:
        if not self._available:
            return self._mock.get_high_res()

        # Nativ in 256×256 rendern (eigener Framebuffer, kein Upscale)
        img_high = self._render_with_pan(frame_buffer=self._highres_fb)
        return _b17.Observation(
            image=img_high.astype(np.uint8),
            timestamp=time.time(),
            frame_id=self._frame,
            source="miniworld_highres",
            metadata={"env": self._env_name,
                      "cam_pan": self._cam_pan},
        )

    def apply_action(self, action: _b17.Action, current_goal: str = ""):
        """
        Überträgt eine Aktion an MiniWorld.
        """
        if not self._available:
            return

        if self._needs_reset:
            obs, _ = self._env.reset()
            self._obs = obs
            self._cam_pan        = 0.0
            self._cam_pan_target = 0.0
            self._needs_reset    = False
            self.episode_reset   = True
        else:
            self.episode_reset   = False

        ros2 = action.to_ros2()
        lx   = ros2["twist"]["linear"]["x"]
        az   = ros2["twist"]["angular"]["z"]

        self._cam_pan_target = float(ros2["camera"]["pan"])
        delta = self._cam_pan_target - self._cam_pan
        step  = min(abs(delta), self._CAM_PAN_SPEED)
        self._cam_pan += step if delta >= 0 else -step
        self._cam_tilt = float(ros2["camera"]["tilt"])

        if abs(az) > abs(lx):
            gym_action = 0 if az > 0 else 1   # links / rechts
        elif lx > 0.05:
            gym_action = 2                     # vorwärts
        else:
            gym_action = 2                     # default: vorwärts

        obs, reward, terminated, truncated, info = self._env.step(gym_action)
        self._obs = obs

        # Eigene Ziel-Erkennung mit Farbabgleich
        agent_pos = self._env.unwrapped.agent.pos
        self._terminal_reward = None
        
        for ent in self._env.unwrapped.entities:
            dist = np.linalg.norm(agent_pos - ent.pos)
            if dist < (self._env.unwrapped.agent.radius + ent.radius + 0.1): # 0.1m Toleranz
                terminated = True
                
                # Zielabgleich: "find the red box" -> "red" und "box"
                ent_type = ent.__class__.__name__.lower() # z.B. "box" oder "ball"
                ent_color = getattr(ent, 'color', '').lower()
                
                if ent_color in current_goal and ent_type in current_goal:
                    # Richtiges Objekt berührt! Harter Reward 1.0.
                    self._terminal_reward = 1.0
                else:
                    # Falsches Objekt berührt! (Hindernis/Wand)
                    self._terminal_reward = 0.05
                break

        if terminated or truncated:
            self._needs_reset = True

    @property
    def obs_shape(self):
        if self._available:
            return (*self._low_res, 3)
        return self._mock.obs_shape

    @property
    def is_miniworld(self) -> bool:
        return self._available

    def close(self):
        if self._env is not None:
            self._env.close()


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

class Orchestrator:
    """
    Dirigent: verbindet B16 + B17 + B18.

    config:
        mode:           "mock" | "miniworld" | "ros2"
        n_steps:        Anzahl Training-Steps
        scene_switch:   Steps pro Szene (Mock-Modus)
        update_display: Dashboard alle N Steps updaten
        gemini_api_key: API-Key (oder GEMINI_API_KEY env var)
    """

    DEFAULT_CONFIG = {
        "mode":              "mock",
        "n_steps":           500,
        "scene_switch":      40,
        "update_display":    8,
        "miniworld_env":     "PredictionWorld-OneRoom-v0",
        "buffer_size":       1000,
        "batch_size":        16,
        "lr":                1e-3,
        "beta_max":          0.05,
        "beta_warmup":       200,
        "min_gemini_interval": 8,
        "max_gemini_interval": 80,
    }

    def __init__(self, config: dict = None):
        self.cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.api_key = self.cfg.get(
            "gemini_api_key",
            os.environ.get("GEMINI_API_KEY", "")
        )
        self._step        = 0
        self._scene_idx   = 0
        self._scene       = SCENE_TYPES[0]
        self._goal        = SCENE_GOALS[SCENE_TYPES[0]]

        # Komponenten (werden in setup() erstellt)
        self.obs_source   = None
        self.action_sink  = None
        self.robot        = None
        self.ml_system    = None
        self.dashboard    = None
        self.overhead     = None
        self.gemini       = None

        # CLIP Label-Embeddings für Laufzeit-Erkennung (aus B21-Checkpoint)
        self._label_clip_embs = None

        # Strategie-System (B22+B23)
        self.strategy_gen  = None
        self.strategy_exec = None

        # Letztes Gemini-Hochreis-Bild (persistiert zwischen Updates)
        self._last_gemini_image = None

        # Letztes Gemini-Event seit dem letzten Dashboard-Update (damit kein Event verloren geht
        # wenn Gemini auf einem Nicht-Display-Step feuert)
        self._pending_gemini_event = None

        # Setup-Status: verhindert Checkpoint-Schreiben bei vorzeitigem Schließen
        self._setup_complete = False

        # Gemini Ausweich-Override: wenn Gemini "ausweichen_links/rechts" sagt,
        # wird die Aktion für N Steps überschrieben.
        self._gemini_override_action = None   # np.ndarray oder None
        self._gemini_override_steps  = 0      # verbleibende Steps
        self._gemini_override_queue  = []     # Sequenz [(action, steps), ...]

    def setup(self):
        """Erstellt alle Komponenten."""
        print("B19 – Orchestrator Setup")
        print(f"  Modus: {self.cfg['mode'].upper()}")
        print()

        # ── Gemini ────────────────────────────────────
        print("Gemini:")
        self.gemini = GeminiClients(api_key=self.api_key)
        print()

        # ── ObservationSource ──────────────────────────
        print("ObservationSource:")
        mode = self.cfg["mode"]
        if mode == "miniworld":
            self.obs_source = MiniWorldObsSource(
                env_name=self.cfg["miniworld_env"],
                low_res=(128,128), high_res=(256,256),
            )
            if not self.obs_source.is_miniworld:
                print("  → Fallback auf Mock")
        elif mode == "ros2":
            print("  ROS2ObsSource (Platzhalter)")
            self.obs_source = ROS2ObsSource()   # ohne Node → Platzhalter
        else:
            self.obs_source = MockObsSource(
                scene_switch_steps=self.cfg["scene_switch"],
                low_res=(128,128), high_res=(256,256),
            )
            print(f"  MockObsSource  ✓")
        print()

        # ── ActionSink ─────────────────────────────────
        print("ActionSink:")
        if mode == "ros2":
            self.action_sink = ROS2ActionSink()
        else:
            self.action_sink = MockActionSink()
            print(f"  MockActionSink ✓")
        print()

        # ── RobotInterface ──────────────────────────────
        self.robot = RobotInterface(self.obs_source, self.action_sink)
        print(f"RobotInterface: ✓")
        print()

        # ── ML-System (B16) ────────────────────────────
        print("ML-System (B16):")
        self.ml_system = IntegratedSystem(
            config={
                "buffer_size":           self.cfg["buffer_size"],
                "batch_size":            self.cfg["batch_size"],
                "lr":                    self.cfg["lr"],
                "beta_max":              self.cfg["beta_max"],
                "beta_warmup":           self.cfg["beta_warmup"],
                "min_gemini_interval":   self.cfg["min_gemini_interval"],
                "max_gemini_interval":   self.cfg["max_gemini_interval"],
            },
            gemini=self.gemini,
        )
        # Initiales Ziel setzen (Zufällig aus allen Objekten in der Szene)
        import random
        random_goal = random.choice(SCENE_TYPES)
        self.ml_system.set_goal(f"find the {random_goal.replace('_',' ')}")
        print(f"  IntegratedSystem ✓  ({sum(p.numel() for p in self.ml_system.encoder.parameters()) + sum(p.numel() for p in self.ml_system.decoder.parameters()):,} Param.)")
        print()

        # ── Dashboard (B18) – früh öffnen, damit Fenster sofort bewegbar sind ──
        hist_len = self.cfg["n_steps"] if self.cfg["n_steps"] > 0 else 5000
        self.dashboard = TrainingDashboard(
            max_history=hist_len,
            title=(f"B19 – Live Training  |  "
                   f"Modus: {self.cfg['mode'].upper()}  |  "
                   f"Gemini: {'✓' if self.gemini.mode=='gemini' else 'Mock'}")
        )
        self.dashboard.setup()
        print("Dashboard (B18): ✓")
        print()

        # ── Overhead Map – ebenfalls früh öffnen ──────
        if OverheadMapView is not None:
            trail_len = min(hist_len, 400)
            self.overhead = OverheadMapView(
                map_size=30.0, trail_length=trail_len,
                title=f"Draufsicht  |  {self.cfg['mode'].upper()}"
            )
            self.overhead.setup()
            if isinstance(self.obs_source, MiniWorldObsSource) \
                    and self.obs_source.is_miniworld:
                self.overhead.set_miniworld_env(self.obs_source._env)
                print("Overhead Map:   ✓  (MiniWorld Wände + Objekte)")
            else:
                print("Overhead Map:   ✓")
        else:
            print("Overhead Map:   ✗ (OverheadMapView.py nicht gefunden)")
        print()

        # Fenster-Events verarbeiten → Fenster sind jetzt verschiebbar
        try:
            import matplotlib.pyplot as _plt
            _plt.pause(0.05)
        except Exception:
            pass

        # ── Checkpoint laden (B20) – nach Fenster-Öffnung ─────────────────────
        ckpt_path = self.cfg.get("checkpoint", None)
        if ckpt_path:
            import glob as _g
            matches = sorted(_g.glob(ckpt_path))
            if matches:
                ckpt_path = matches[-1]  # neuester
            print(f"Checkpoint laden: {ckpt_path}")
            self.ml_system.load_checkpoint(
                ckpt_path,
                load_optimizer=self.cfg.get("load_optimizer", False),
                strict=self.cfg.get("strict_load", False),
            )
            # CLIP Label-Embeddings laden (aus B21-Checkpoint)
            try:
                import torch as _torch
                raw_ckpt = _torch.load(ckpt_path, weights_only=False)
                embs = raw_ckpt.get("label_clip_embeddings", None)
                if embs is not None:
                    device = next(self.ml_system.encoder.parameters()).device
                    self._label_clip_embs = {k: v.to(device) for k, v in embs.items()}
                    print(f"  Label-Embeddings: {list(self._label_clip_embs.keys())}")
                else:
                    print("  Label-Embeddings: nicht im Checkpoint (B21 erneut ausführen)")
            except Exception as _e:
                print(f"  Label-Embeddings: Fehler beim Laden ({_e})")
            print()

        # Fenster-Events erneut verarbeiten (nach Checkpoint-Load)
        try:
            _plt.pause(0.05)
        except Exception:
            pass

        # ── Strategie-System (B22+B23) ────────────────
        if StrategyExecutor is not None:
            self.strategy_exec = StrategyExecutor()
            goal_text = self.ml_system.current_goal or "explore the environment"

            # Immer sofort mit Mock-Strategie starten (kein Blockieren)
            self.strategy_gen = MockStrategyGenerator()
            strategy = self.strategy_gen.generate(goal_text)
            self.strategy_exec.set_strategy(strategy)
            print(f"Strategie:      ✓  (mock, sofort verfügbar)")

            # Gemini-Strategie asynchron upgraden (kein Warten auf API-Call)
            if self.gemini.mode == "gemini" and GeminiStrategyGenerator is not None:
                import threading as _threading
                def _upgrade_strategy():
                    try:
                        gen   = GeminiStrategyGenerator(client=self.gemini.client)
                        strat = gen.generate(goal_text)
                        self.strategy_gen  = gen
                        self.strategy_exec.set_strategy(strat)
                        print(f"  Strategie upgrade: Gemini ({len(strat.rules)} Regeln)")
                    except Exception as _e:
                        print(f"  Gemini-Strategie: Fehler ({_e}), behalte Mock")
                _threading.Thread(target=_upgrade_strategy, daemon=True).start()
        else:
            print("Strategie:      ✗ (B22/B23 nicht gefunden)")
        print()

        self._setup_complete = True
        return self

    def _get_miniworld_action(self, step: int) -> np.ndarray:
        """
        Explorations-Policy für MiniWorld.
        Wechselt zwischen Vorwärts-Fahren und Drehen
        um die Wand-Problem zu vermeiden.
        """
        phase = (step // 8) % 4   # alle 8 Steps Phase wechseln
        if phase == 0:             # vorwärts
            act = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif phase == 1:           # links drehen
            act = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        elif phase == 2:           # vorwärts
            act = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:                      # rechts drehen
            act = [0.0, -0.8, 0.0, 0.0, 0.0, 0.0]
        noise = 0.05 * np.random.randn(6).astype(np.float32)
        return np.clip(np.array(act, dtype=np.float32) + noise, -1, 1)

    def run(self):
        """Haupt-Loop."""
        n = self.cfg["n_steps"]
        mode = self.cfg["mode"]
        unlimited = (n <= 0)
        if unlimited:
            print("Starte Training-Loop: unbegrenzt (Fenster-X oder Ctrl+C zum Beenden)\n")
        else:
            print(f"Starte Training-Loop: {n} Steps\n")

        # Erste Obs holen
        obs = self.robot.obs.get_observation()

        step = 0
        while unlimited or step < n:
            self._step = step

            # ── Fenster geschlossen? → sauber beenden ──
            if self.dashboard.window_closed:
                print("\n[Fenster geschlossen] Beende Training...")
                break
            if self.overhead is not None and self.overhead._window_closed:
                print("\n[Karten-Fenster geschlossen] Beende Training...")
                break

            # ── Szene / Ziel (Mock-Modus) ──────────────
            if mode == "mock":
                # MockObsSource wechselt intern, wir lesen mit
                self._scene = self.obs_source.current_scene
                self._goal  = SCENE_GOALS[self._scene]
                if (step > 0 and
                        step % self.cfg["scene_switch"] == 0):
                    result = self.ml_system.set_goal(
                        f"find the {self._scene.replace('_',' ')}"
                    )
                    print(f"  [Step {step:4d}] Ziel → "
                          f"'{result['primary_goal']}'")
                    # Neue Strategie für neues Ziel
                    if self.strategy_gen is not None:
                        strategy = self.strategy_gen.generate(result['primary_goal'])
                        self.strategy_exec.set_strategy(strategy)
                    # Trail beim Szenenwechsel löschen für übersichtliche Karte
                    if self.overhead is not None:
                        self.overhead.clear_trail()
            elif mode == "miniworld":
                self._scene = "miniworld"
                self._goal  = self.ml_system.current_goal
                # Episode-Reset: Trail löschen + RSSM Hidden-State zurücksetzen
                if (isinstance(self.obs_source, MiniWorldObsSource)
                        and getattr(self.obs_source, "episode_reset", False)):
                    self.ml_system.reset_hidden_state()
                    if self.overhead is not None:
                        self.overhead.clear_trail()
                    
                    import random
                    random_goal = random.choice(SCENE_TYPES)
                    result = self.ml_system.set_goal(f"find the {random_goal.replace('_',' ')}")
                    self._goal = self.ml_system.current_goal
                    
                    # Strategie aktualisieren für das neue Ziel
                    if self.strategy_gen is not None:
                        strategy = self.strategy_gen.generate(result['primary_goal'])
                        self.strategy_exec.set_strategy(strategy)

                    print(f"  [Step {step:4d}] Episode-Reset → Neues Ziel: '{self._goal}'")

            # ── Aktion aus ML-System + Strategie ──────
            ml_result_pre = self.ml_system.step(
                obs_np=obs.image,
                action_np=np.zeros(6, dtype=np.float32),  # Dummy für Vorhersage
                next_obs_np=obs.image,
                scene=self._scene,
                train=False,  # Nur Vorhersage, kein Training
            ) if False else None  # Vorhersage kommt nach dem Step

            if mode == "miniworld" and self._gemini_override_steps > 0:
                # Gemini-Ausweich-Override hat Priorität über alles
                act_arr = self._gemini_override_action.copy()
                self._gemini_override_steps -= 1
                if self._gemini_override_steps == 0:
                    # Nächste Phase aus Queue laden, falls vorhanden
                    if self._gemini_override_queue:
                        action, steps = self._gemini_override_queue.pop(0)
                        self._gemini_override_action = action
                        self._gemini_override_steps = steps
                    else:
                        self._gemini_override_action = None
                        print(f"  [Step {step:4d}] Gemini Override beendet")

            elif mode == "miniworld" and self.strategy_exec is not None:
                # Strategie-Aktion berechnen
                obs_info = {
                    "image_nn": obs.image,
                    "reward":      self.ml_system.metrics["r_total"][-1]
                                   if self.ml_system.metrics["r_total"] else 0.0,
                    "r_intr":      self.ml_system.metrics["r_intrinsic"][-1]
                                   if self.ml_system.metrics.get("r_intrinsic") else 0.05,
                    "sigma":       0.5,  # wird nach ml_result aktualisiert
                    "cam_pan":     self.obs_source._cam_pan
                                   if isinstance(self.obs_source, MiniWorldObsSource)
                                   else 0.0,
                    "step":        step,
                }
                strategy_action = self.strategy_exec.get_action(obs_info)

                if strategy_action is not None:
                    nn_action = self._get_miniworld_action(step)
                    # Sigma aus letztem Step (wenn verfügbar)
                    last_sigma = (self.ml_system.metrics.get("sigma", [0.5])[-1]
                                  if self.ml_system.metrics.get("sigma") else 0.5)
                    act_arr = self.strategy_exec.blend(
                        strategy_action, nn_action, last_sigma
                    )
                else:
                    # T19: Actor-Critic Action (Action Head liefert O(1) geplante Aktion)
                    act_arr = self._get_miniworld_action(step)
            elif mode == "miniworld":
                act_arr = self._get_miniworld_action(step)
            else:
                act_arr = np.clip(
                    np.array(SCENE_ACTIONS.get(
                        self._scene, SCENE_ACTIONS["red_box"]
                    ), dtype=np.float32) +
                    0.08*np.random.randn(6).astype(np.float32), -1, 1
                )

            # ── Robot Step ─────────────────────────────
            action_obj = Action.from_array(act_arr, source="policy")
            if isinstance(self.obs_source, MiniWorldObsSource):
                self.obs_source.apply_action(action_obj)
            self.action_sink.send(action_obj)

            next_obs = self.robot.step(act_arr, source="policy")

            # ── ML-System Step (B16) ───────────────────
            ml_result = self.ml_system.step(
                obs_np=obs.image,
                action_np=act_arr,
                next_obs_np=next_obs.image,
                scene=self._scene,
                terminal_reward=getattr(self.obs_source, '_terminal_reward', None)
            )

            # ── High-res für Gemini ─────────────────────
            # Nutze get_high_res() für alle Quellen (MiniWorld hochskaliert auf 256x256, Mock nativ 256x256).
            # Das löst das Problem des falschen Seitenverhältnisses und der zu kleinen Darstellung im Dashboard.
            gemini_image = None
            if ml_result.get("gem_called"):
                gemini_image = self.robot.get_high_res().image

                # Wand-Filter: einheitliche graue Fläche nicht an Gemini senden
                # (Gemini sagt "unscharf", aber es ist eine Wand → nutzloser API-Call)
                if gemini_image is not None:
                    _img_var = float(np.var(gemini_image.astype(np.float32)))
                    if _img_var < 200.0:
                        gemini_image = None
                        print(f"  [Step {step:4d}] Gemini übersprungen (Wand, var={_img_var:.0f})")

                if gemini_image is None:
                    gemini_event = None
                else:
                    # Persistiert bis zum nächsten Gemini-Call
                    self._last_gemini_image = gemini_image

                    ass = self.gemini.assess_image(
                        gemini_image,
                        self.ml_system.current_goal,
                        {"linear_x":   float(act_arr[0]),
                         "angular_z":  float(act_arr[1]),
                         "camera_pan": float((act_arr[2]+1)/2*180-90),
                         "camera_tilt":float((act_arr[3]+1)/2*90-45)},
                    )
                    print(f"  [Step {step:4d}] Gemini ER: r={ass['reward']:.3f}")
                    print(f"             Situation:  {ass.get('situation','')}")
                    print(f"             Recommendation: {ass.get('recommendation','')}")
                    print(f"             Hint:       {ass.get('next_action_hint','')}")
                    if "raw_response" in ass:
                        print(f"             Raw:        {ass['raw_response']}")
                    gemini_event = ass
                    self._pending_gemini_event = ass

                    # Gemini Ausweich-Override prüfen
                    hint = ass.get("next_action_hint", "").lower()
                    if "avoid_left" in hint:
                        # Hindernis ist links -> Weiche nach RECHTS aus
                        self._gemini_override_action = np.array(
                            [0.3, -1.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 5
                        print(f"             → Avoid-Override: AVOID LEFT (turn right) ({self._gemini_override_steps} Steps)")
                    elif "avoid_right" in hint:
                        # Hindernis ist rechts -> Weiche nach LINKS aus
                        self._gemini_override_action = np.array(
                            [0.3, 1.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 5
                        print(f"             → Avoid-Override: AVOID RIGHT (turn left) ({self._gemini_override_steps} Steps)")
                    elif "backward" in hint:
                        self._gemini_override_action = np.array(
                            [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 3
                        print(f"             → Backward-Override: BACK ({self._gemini_override_steps} Steps)")
                    elif "camera_down" in hint:
                        self._gemini_override_action = np.array(
                            [0.0, 0.0, 0.0, -0.35, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: TILT DOWN ({self._gemini_override_steps} Steps)")
                    elif "camera_up" in hint:
                        self._gemini_override_action = np.array(
                            [0.0, 0.0, 0.0, 0.35, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: TILT UP ({self._gemini_override_steps} Steps)")
                    elif "camera_left" in hint:
                        self._gemini_override_action = np.array(
                            [0.0, 0.0, -0.5, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: PAN LEFT ({self._gemini_override_steps} Steps)")
                    elif "camera_right" in hint:
                        self._gemini_override_action = np.array(
                            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: PAN RIGHT ({self._gemini_override_steps} Steps)")
                    elif "free_drive" in hint:
                        # Mehrstufig: 4 Steps rückwärts, dann 4 Steps scharf drehen
                        turn_dir = 0.9 if np.random.rand() > 0.5 else -0.9
                        self._gemini_override_action = np.array(
                            [-0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 4
                        self._gemini_override_queue = [
                            (np.array([0.2, turn_dir, 0.0, 0.0, 0.0, 0.0],
                                      dtype=np.float32), 4),
                        ]
                        side = "L" if turn_dir > 0 else "R"
                        print(f"             → FREE DRIVE: 4× back, 4× turn {side}")
                    elif "forward" in hint:
                        self._gemini_override_action = np.array(
                            [0.7, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 3
                        print(f"             → Override: FORWARD ({self._gemini_override_steps} Steps)")
                    elif "left" in hint and "camera" not in hint and "avoid" not in hint:
                        self._gemini_override_action = np.array(
                            [0.3, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: TURN LEFT ({self._gemini_override_steps} Steps)")
                    elif "right" in hint and "camera" not in hint and "avoid" not in hint:
                        self._gemini_override_action = np.array(
                            [0.3, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        self._gemini_override_steps = 2
                        print(f"             → Override: TURN RIGHT ({self._gemini_override_steps} Steps)")
            else:
                gemini_event = None

            # ── Live-Update: Kamera-Bilder jeden Step ──────
            # NN-Auflösung (128×128) – das was das NN sieht
            live_obs = obs.image  # bereits 128×128 vom ML-System
            self.dashboard.update_live(live_obs, ml_result["pred_obs"])

            # ── Dashboard Update (B18) – volle Metriken ────
            if (step % self.cfg["update_display"] == 0 or
                    step == n-1):

                # Original-Auflösung für Dashboard
                if isinstance(self.obs_source, MiniWorldObsSource) \
                        and self.obs_source.is_miniworld:
                    display_obs = self.obs_source._obs   # 60×80 Original
                    # MiniWorld Top-Down Karte (Wände + Objekte sichtbar)
                    try:
                        topdown = self.obs_source._env.render_top_view()
                    except Exception:
                        try:
                            topdown = self.obs_source._env.render(
                                mode='top_down')
                        except Exception:
                            topdown = None
                else:
                    display_obs = obs.image
                    topdown     = None

                m = self.ml_system.metrics

                # ── NN-Erkennung: cos_sim(z, goal_proj(label_emb)) ──────────
                recognition_scores = None
                z_np = ml_result.get("latent_z")
                if self._label_clip_embs is not None and z_np is not None:
                    try:
                        import torch as _torch
                        import torch.nn.functional as _F
                        device = next(self.ml_system.goal_proj.parameters()).device
                        z_t = _torch.from_numpy(z_np).float().unsqueeze(0).to(device)
                        recognition_scores = {}
                        with _torch.no_grad():
                            for lbl, emb_512 in self._label_clip_embs.items():
                                proj = self.ml_system.goal_proj(
                                    emb_512.unsqueeze(0))          # (1, 64)
                                sim  = _F.cosine_similarity(z_t, proj).item()
                                recognition_scores[lbl] = (sim + 1.0) / 2.0  # [-1,1] → [0,1]
                    except Exception as _e:
                        recognition_scores = None

                # Pending-Event nutzen (enthält auch Events von Nicht-Display-Steps)
                dashboard_gemini = self._pending_gemini_event
                self._pending_gemini_event = None

                self.dashboard.update(
                    obs=display_obs,
                    pred=ml_result["pred_obs"],
                    metrics={
                        "fe":              m["fe"][-1]      if m["fe"]      else 0,
                        "recon":           m["recon"][-1]   if m["recon"]   else 0,
                        "kl":              m["kl"][-1]      if m["kl"]      else 0,
                        "r_intrinsic":     m["r_intrinsic"][-1] if m["r_intrinsic"] else 0,
                        "r_gemini":        m["r_gemini"][-1]    if m["r_gemini"]    else 0,
                        "r_total":         m["r_total"][-1]     if m["r_total"]     else 0,
                        "goal_progress":   m["goal_progress"][-1] if m["goal_progress"] else 0,
                        "beta":            self.ml_system.beta,
                        "lr":              self.ml_system.optimizer.param_groups[0]["lr"],
                        "gemini_interval": m["gemini_interval"][-1] if m["gemini_interval"] else 0,
                    },
                    gemini_event=dashboard_gemini,
                    latent_z=ml_result.get("latent_z"),
                    scene=self._scene,
                    goal=self.ml_system.current_goal,
                    action_norm=act_arr,
                    sigma=ml_result.get("sigma"),
                    topdown=topdown,
                    gemini_hires=self._last_gemini_image,
                    recognition_scores=recognition_scores,
                )

            # ── Overhead Map Update (jeden Step) ────────
            if self.overhead is not None:
                self.overhead.update(
                    action_ros2=self.action_sink.last_ros2 or {},
                    scene=self._scene,
                    gemini_event=gemini_event,
                )

            obs = next_obs
            step += 1

        self._print_summary()

    def _print_summary(self):
        m    = self.ml_system.metrics
        rate = self.ml_system.adaptive.call_rate
        calls_per_hour  = int(3600 * rate)
        cost_per_hour   = calls_per_hour * 0.001
        savings         = 1.0 - rate

        print()
        print("=" * 55)
        print("B19 – Training abgeschlossen")
        print("=" * 55)
        print(f"  Steps:           {self._step+1}")
        print(f"  Training Steps:  {self.ml_system.train_steps}")
        print(f"  Gemini ER Calls: {self.ml_system.adaptive.calls}")
        print(f"  Call-Rate:       {rate*100:.1f}%")
        print(f"  Gemini Labels:   {self.ml_system.replay.gemini_count}")
        if m["fe"]:
            print(f"  FE final:        {m['fe'][-1]:.5f}")
            print(f"  Recon final:     {m['recon'][-1]:.5f}")
            print(f"  Reward Ø:        {np.mean(m['r_total'][-20:]):.4f}")
        print()
        print("Kosten-Schätzung (Gemini 2.5 Flash):")
        print(f"  Calls/Stunde:    {calls_per_hour:,}")
        print(f"  $/Stunde:        ${cost_per_hour:.3f}")
        print(f"  Ersparnis:       {savings*100:.0f}%")
        print()
        print("Latenz:")
        print(f"  Ø {self.robot.avg_latency_ms:.2f}ms/Step")

        if self.strategy_exec is not None:
            ss = self.strategy_exec.summary()
            print()
            print("Strategie (B22/B23):")
            print(f"  Strategie-Steps: {ss['strategy_pct']:.0f}%")
            print(f"  NN-Steps:        {ss['nn_pct']:.0f}%")
            print(f"  Blended:         {ss['blended_pct']:.0f}%")
            if ss['top_rules']:
                print(f"  Top-Regeln:")
                for rule, count in ss['top_rules'][:3]:
                    print(f"    {rule}: {count}x")
        print()
        print("ROS2-Umstieg:")
        print("  obs  = ROS2ObsSource(node, '/camera/image_raw')")
        print("  act  = ROS2ActionSink(node)")
        print("  robot = RobotInterface(obs, act)")
        print("  → Rest bleibt identisch")
        print()
        print("Nächste Schritte:")
        print("  B21 – Pre-Training VAE")
        print("  B22 – Pre-Training CLIP")

        # ── Checkpoint speichern (B20) ─────────────────
        # Nur speichern wenn Setup vollständig abgeschlossen war
        # (verhindert leeren Checkpoint bei zu frühem Fenster-Schließen)
        if self.cfg.get("save_checkpoint", True) and self._setup_complete:
            self.ml_system.save_checkpoint(tag="checkpoint")

    def close(self):
        self.robot.close()
        if self.overhead is not None:
            self.overhead.close()
        self.dashboard.close()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="B19 – Orchestrator"
    )
    parser.add_argument(
        "--mode", choices=["mock","miniworld","ros2"],
        default="mock",
        help="Ausführungs-Modus (default: mock)"
    )
    parser.add_argument(
        "--env", default="PredictionWorld-OneRoom-v0",
        help="MiniWorld Gym Environment"
    )
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Anzahl Training-Steps (0 = unbegrenzt)"
    )
    parser.add_argument(
        "--display", type=int, default=8,
        help="Dashboard Update alle N Steps"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint-Datei laden (z.B. checkpoints/pwn_pretrain_vae_*.pt)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "mode":            args.mode,
        "n_steps":         args.steps,
        "miniworld_env":   args.env,
        "update_display":  args.display,
        "checkpoint":      args.checkpoint,
        "scene_switch":    40,
        "buffer_size":     1000,
        "batch_size":      16,
        "lr":              1e-3,
        "beta_max":        0.05,
        "beta_warmup":     200,
        "min_gemini_interval": 8,
        "max_gemini_interval": 80,
    }

    orch = Orchestrator(config)
    try:
        orch.setup()
        orch.run()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Abgebrochen.")
        orch._print_summary()
    finally:
        orch.close()
