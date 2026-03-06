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
        self._cam_pan   = 0.0    # Kamera-Pan in rad (-1.57 .. +1.57)
        self._cam_tilt  = 0.0    # Kamera-Tilt (für spätere Nutzung)

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
                                 view="agent")
            obs, _    = self._env.reset()
            self._obs = obs
            self._available = True
            print(f"  MiniWorld: {env_name}  ✓")
            print(f"    Obs shape: {obs.shape}")
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

    def _render_with_pan(self):
        """Rendert Bild mit simuliertem Kamera-Pan.

        Dreht agent.dir temporär um den Pan-Winkel, rendert,
        und setzt die Richtung zurück. So sieht die Kamera zur Seite
        während der Roboter geradeaus fährt.
        """
        agent = self._env.unwrapped.agent
        original_dir = agent.dir
        # Positive pan = nach rechts, normalisiert auf [0, 2π]
        agent.dir = (original_dir - self._cam_pan) % (2 * np.pi)
        obs = self._env.unwrapped.render_obs()
        agent.dir = original_dir
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

        raw = self._render_with_pan()
        img_high = self._resize(raw, self._high_res)
        return _b17.Observation(
            image=img_high.astype(np.uint8),
            timestamp=time.time(),
            frame_id=self._frame,
            source="miniworld_highres",
            metadata={"env": self._env_name,
                      "cam_pan": self._cam_pan},
        )

    def apply_action(self, action: _b17.Action):
        """
        Überträgt eine Aktion an MiniWorld.
        MiniWorld verwendet diskrete Aktionen:
            0: turn_left
            1: turn_right
            2: move_forward
        Kamera-Pan wird intern gespeichert und beim Rendern angewendet.
        """
        if not self._available:
            return

        ros2 = action.to_ros2()
        lx   = ros2["twist"]["linear"]["x"]
        az   = ros2["twist"]["angular"]["z"]

        # Kamera-Pan/Tilt aus Aktion übernehmen (rad)
        self._cam_pan  = float(ros2["camera"]["pan"])
        self._cam_tilt = float(ros2["camera"]["tilt"])

        # Kontinuierlich → diskret (Bewegung)
        if abs(az) > abs(lx):
            gym_action = 0 if az > 0 else 1   # links / rechts
        elif lx > 0.05:
            gym_action = 2                     # vorwärts
        else:
            gym_action = 2                     # default: vorwärts

        obs, reward, terminated, truncated, info = self._env.step(gym_action)
        # Nicht self._obs speichern – get_observation rendert mit Pan
        self._obs = obs

        if terminated or truncated:
            obs, _ = self._env.reset()
            self._obs = obs
            self._cam_pan = 0.0    # Pan zurücksetzen bei Reset

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
        # Initiales Ziel setzen
        self.ml_system.set_goal(f"find the {self._scene.replace('_',' ')}")
        print(f"  IntegratedSystem ✓  ({sum(p.numel() for p in self.ml_system.encoder.parameters()) + sum(p.numel() for p in self.ml_system.decoder.parameters()):,} Param.)")

        # ── Checkpoint laden (B20) ─────────────────────
        ckpt_path = self.cfg.get("checkpoint", None)
        if ckpt_path:
            import glob as _g
            matches = sorted(_g.glob(ckpt_path))
            if matches:
                ckpt_path = matches[-1]  # neuester
            print(f"\n  Lade Checkpoint: {ckpt_path}")
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

        # ── Dashboard (B18) ────────────────────────────
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

        # ── Overhead Map ───────────────────────────────
        if OverheadMapView is not None:
            self.overhead = OverheadMapView(
                map_size=30.0, trail_length=hist_len,
                title=f"Draufsicht  |  {self.cfg['mode'].upper()}"
            )
            self.overhead.setup()
            # MiniWorld-Env übergeben für echte Wände/Objekte/Position
            if isinstance(self.obs_source, MiniWorldObsSource) \
                    and self.obs_source.is_miniworld:
                self.overhead.set_miniworld_env(self.obs_source._env)
                print("Overhead Map:   ✓  (MiniWorld Wände + Objekte)")
            else:
                print("Overhead Map:   ✓")
        else:
            print("Overhead Map:   ✗ (OverheadMapView.py nicht gefunden)")
        print()

        # ── Strategie-System (B22+B23) ────────────────
        if StrategyExecutor is not None:
            self.strategy_exec = StrategyExecutor()
            # Gemini oder Mock Strategy Generator
            if self.gemini.mode == "gemini" and GeminiStrategyGenerator is not None:
                self.strategy_gen = GeminiStrategyGenerator(
                    client=self.gemini.client,
                )
            else:
                self.strategy_gen = MockStrategyGenerator()
            # Initiale Strategie generieren
            goal_text = self.ml_system.current_goal or "explore the environment"
            strategy = self.strategy_gen.generate(goal_text)
            self.strategy_exec.set_strategy(strategy)
            print(f"Strategie:      ✓  ({strategy.source})")
        else:
            print("Strategie:      ✗ (B22/B23 nicht gefunden)")
        print()

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
            elif mode == "miniworld":
                self._scene = "miniworld"
                self._goal  = self.ml_system.current_goal

            # ── Aktion aus ML-System + Strategie ──────
            ml_result_pre = self.ml_system.step(
                obs_np=obs.image,
                action_np=np.zeros(6, dtype=np.float32),  # Dummy für Vorhersage
                next_obs_np=obs.image,
                scene=self._scene,
                train=False,  # Nur Vorhersage, kein Training
            ) if False else None  # Vorhersage kommt nach dem Step

            if mode == "miniworld" and self.strategy_exec is not None:
                # Strategie-Aktion berechnen
                obs_info = {
                    "image_nn": obs.image,
                    "reward":      self.ml_system.metrics["r_total"][-1]
                                   if self.ml_system.metrics["r_total"] else 0.0,
                    "sigma":       0.5,  # wird nach ml_result aktualisiert
                    "cam_pan":     self.obs_source._cam_pan
                                   if isinstance(self.obs_source, MiniWorldObsSource)
                                   else 0.0,
                    "step":        step,
                }
                strategy_action = self.strategy_exec.get_action(obs_info)

                if strategy_action is not None:
                    # NN-Aktion als Fallback/Blending-Partner
                    nn_action = self._get_miniworld_action(step)
                    # Sigma aus letztem Step (wenn verfügbar)
                    last_sigma = (self.ml_system.metrics.get("sigma", [0.5])[-1]
                                  if self.ml_system.metrics.get("sigma") else 0.5)
                    act_arr = self.strategy_exec.blend(
                        strategy_action, nn_action, last_sigma
                    )
                else:
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
            )

            # ── High-res für Gemini ─────────────────────
            # WICHTIG: Original-Auflösung senden, nicht NN-Auflösung upscaled
            gemini_image = None
            if ml_result.get("gem_called"):
                if isinstance(self.obs_source, MiniWorldObsSource) \
                        and self.obs_source.is_miniworld:
                    # MiniWorld: direkt Original-Frame (60×80) verwenden
                    gemini_image = self.obs_source._obs
                else:
                    gemini_image = self.robot.get_high_res().image

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
                # Vollständige Ausgabe
                print(f"  [Step {step:4d}] Gemini ER: r={ass['reward']:.3f}")
                print(f"             Situation:  {ass.get('situation','')}")
                print(f"             Empfehlung: {ass.get('recommendation','')}")
                gemini_event = ass
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
                    gemini_event=gemini_event,
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
        if self.cfg.get("save_checkpoint", True):
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
