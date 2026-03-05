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
    spec   = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
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

print()

import matplotlib
matplotlib.use('TkAgg')

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
        MiniWorld-Hallway-v0
        MiniWorld-OneRoom-v0
        MiniWorld-FourRooms-v0
        MiniWorld-TMaze-v0
        MiniWorld-Maze-v0
    """

    def __init__(self, env_name: str = "MiniWorld-OneRoom-v0",
                 low_res=(16,16), high_res=(128,128),
                 render_mode: str = "rgb_array"):
        self._env_name  = env_name
        self._low_res   = low_res
        self._high_res  = high_res
        self._env       = None
        self._obs       = None
        self._frame     = 0
        self._available = False

        try:
            import gymnasium as gym
            import miniworld
            self._gym = gym

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

    def get_observation(self) -> _b17.Observation:
        if not self._available:
            return self._mock.get_observation()

        self._frame += 1
        img_low = self._resize(self._obs, self._low_res)
        return _b17.Observation(
            image=img_low.astype(np.uint8),
            timestamp=time.time(),
            frame_id=self._frame,
            source="miniworld",
            metadata={"env": self._env_name},
        )

    def get_high_res(self) -> _b17.Observation:
        if not self._available:
            return self._mock.get_high_res()

        img_high = self._resize(self._obs, self._high_res)
        return _b17.Observation(
            image=img_high.astype(np.uint8),
            timestamp=time.time(),
            frame_id=self._frame,
            source="miniworld_highres",
            metadata={"env": self._env_name},
        )

    def apply_action(self, action: _b17.Action):
        """
        Überträgt eine Aktion an MiniWorld.
        MiniWorld verwendet diskrete Aktionen:
            0: turn_left
            1: turn_right
            2: move_forward
            (3: move_back bei manchen Envs)
        Wir leiten aus dem kontinuierlichen Aktions-Vektor ab.
        """
        if not self._available:
            return

        ros2 = action.to_ros2()
        lx   = ros2["twist"]["linear"]["x"]
        az   = ros2["twist"]["angular"]["z"]

        # Kontinuierlich → diskret
        if abs(az) > abs(lx):
            gym_action = 0 if az > 0 else 1   # links / rechts
        elif lx > 0.05:
            gym_action = 2                     # vorwärts
        else:
            gym_action = 2                     # default: vorwärts

        obs, reward, terminated, truncated, info = self._env.step(gym_action)
        self._obs = obs

        if terminated or truncated:
            obs, _ = self._env.reset()
            self._obs = obs

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
        "miniworld_env":     "MiniWorld-OneRoom-v0",
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
                low_res=(16,16), high_res=(128,128),
            )
            if not self.obs_source.is_miniworld:
                print("  → Fallback auf Mock")
        elif mode == "ros2":
            print("  ROS2ObsSource (Platzhalter)")
            self.obs_source = ROS2ObsSource()   # ohne Node → Platzhalter
        else:
            self.obs_source = MockObsSource(
                scene_switch_steps=self.cfg["scene_switch"],
                low_res=(16,16), high_res=(128,128),
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
        print()

        # ── Dashboard (B18) ────────────────────────────
        self.dashboard = TrainingDashboard(
            max_history=self.cfg["n_steps"],
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
                map_size=30.0, trail_length=self.cfg["n_steps"],
                title=f"Draufsicht  |  {self.cfg['mode'].upper()}"
            )
            self.overhead.setup()
            print("Overhead Map:   ✓")
        else:
            print("Overhead Map:   ✗ (OverheadMapView.py nicht gefunden)")
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
        print(f"Starte Training-Loop: {n} Steps\n")

        # Erste Obs holen
        obs = self.robot.obs.get_observation()

        for step in range(n):
            self._step = step

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
            elif mode == "miniworld":
                self._scene = "miniworld"
                self._goal  = self.ml_system.current_goal

            # ── Aktion aus ML-System ───────────────────
            if mode == "miniworld":
                act_arr = self._get_miniworld_action(step)
            else:
                act_arr = np.clip(
                    np.array(SCENE_ACTIONS.get(
                        self._scene, SCENE_ACTIONS["red_box"]
                    ), dtype=np.float32) +
                    0.08*np.random.randn(6).astype(np.float32), -1, 1
                )

            # ── Robot Step ─────────────────────────────
            if isinstance(self.obs_source, MiniWorldObsSource):
                action_obj = Action.from_array(act_arr, source="policy")
                self.obs_source.apply_action(action_obj)

            next_obs = self.robot.step(act_arr, source="policy")

            # ── ML-System Step (B16) ───────────────────
            ml_result = self.ml_system.step(
                obs_np=obs.image,
                action_np=act_arr,
                next_obs_np=next_obs.image,
                scene=self._scene,
            )

            # ── High-res für Gemini ─────────────────────
            # WICHTIG: Original-Auflösung senden, nicht 16×16 upscaled
            if ml_result.get("gem_called"):
                if isinstance(self.obs_source, MiniWorldObsSource) \
                        and self.obs_source.is_miniworld:
                    # MiniWorld: direkt Original-Frame (60×80) verwenden
                    gemini_image = self.obs_source._obs
                else:
                    gemini_image = self.robot.get_high_res().image

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

            # ── Dashboard Update (B18) ──────────────────
            if (step % self.cfg["update_display"] == 0 or
                    step == n-1):

                # Original-Auflösung für Dashboard (nicht 16×16)
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
                    scene=self._scene,
                    goal=self.ml_system.current_goal,
                    action_norm=act_arr,
                    sigma=ml_result.get("sigma"),
                    topdown=topdown,
                )

            # ── Overhead Map Update (jeden Step) ────────
            if self.overhead is not None:
                self.overhead.update(
                    action_ros2=self.action_sink.last_ros2 or {},
                    scene=self._scene,
                    gemini_event=gemini_event,
                )

            obs = next_obs

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
        print()
        print("ROS2-Umstieg:")
        print("  obs  = ROS2ObsSource(node, '/camera/image_raw')")
        print("  act  = ROS2ActionSink(node)")
        print("  robot = RobotInterface(obs, act)")
        print("  → Rest bleibt identisch")
        print()
        print("Nächste Schritte:")
        print("  B20 – Systemtest / Evaluation")
        print("  B21 – ROS2 Live-Anbindung")

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
        "--env", default="MiniWorld-OneRoom-v0",
        help="MiniWorld Gym Environment"
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Anzahl Training-Steps"
    )
    parser.add_argument(
        "--display", type=int, default=8,
        help="Dashboard Update alle N Steps"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {
        "mode":            args.mode,
        "n_steps":         args.steps,
        "miniworld_env":   args.env,
        "update_display":  args.display,
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
