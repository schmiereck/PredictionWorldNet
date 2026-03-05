"""
B17 – Saubere Roboter-Schnittstellen
======================================
Definiert abstrakte Interfaces für visuelle Eingabe und Aktions-Ausgabe.
Heute: Mock-Implementierungen.
Später: ROS2-Implementierungen – NUR die Klasse tauschen, Rest bleibt.

Interfaces:
    ObservationSource   – Liefert Kamerabilder
        MockObsSource       – Synthetische Szenen (jetzt)
        ROS2ObsSource       – ROS2 sensor_msgs/Image Topic (später)

    ActionSink          – Empfängt Aktionen
        MockActionSink      – Loggt und simuliert (jetzt)
        ROS2ActionSink      – ROS2 geometry_msgs/Twist + custom (später)

    RobotInterface      – Verbindet ObsSource + ActionSink mit B16
        step(action) → observation
        get_goal()   → str

ROS2 Topics (Ausblick):
    Kamera:   /camera/image_raw          (sensor_msgs/Image)
              /camera/image_raw/compressed
    Aktion:   /cmd_vel                   (geometry_msgs/Twist)
              /camera_head/pan_tilt      (custom CameraPanTilt.msg)
              /cmd_arc                   (custom ArcMovement.msg)

Custom Messages (Ausblick):
    CameraPanTilt.msg:
        float32 pan    # rad, -1.57 bis +1.57
        float32 tilt   # rad, -0.79 bis +0.79

    ArcMovement.msg:
        float32 linear_x    # m/s
        float32 arc_radius  # m  (0 = geradeaus)
        float32 duration    # s
"""

import matplotlib
matplotlib.use('TkAgg')

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import deque


# ─────────────────────────────────────────────
# DATEN-TYPEN
# ─────────────────────────────────────────────

@dataclass
class Observation:
    """
    Standardisiertes Beobachtungs-Objekt.
    Unabhängig von der Quelle (Mock oder ROS2).
    """
    image:      np.ndarray          # (H, W, 3) uint8
    timestamp:  float               # Unix-Timestamp
    frame_id:   int = 0             # Frame-Zähler
    source:     str = "unknown"     # "mock" | "ros2" | "file"
    metadata:   dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.image.shape

    @property
    def as_float(self) -> np.ndarray:
        """uint8 [0,255] → float32 [0,1]"""
        return self.image.astype(np.float32) / 255.0


@dataclass
class Action:
    """
    Standardisiertes Aktions-Objekt.
    Enthält alle 6 Dimensionen + abgeleitete ROS2-Felder.
    """
    # Normiert [-1, 1]
    linear_x:    float = 0.0
    angular_z:   float = 0.0
    camera_pan:  float = 0.0   # normiert
    camera_tilt: float = 0.0   # normiert
    arc_radius:  float = 0.0   # normiert
    duration:    float = 0.0   # normiert

    timestamp:   float = field(default_factory=time.time)
    source:      str   = "policy"   # "policy" | "manual" | "gemini"

    # ROS2-Felder (physikalisch, werden in to_ros2() befüllt)
    _ros2: dict = field(default_factory=dict, repr=False)

    # Physikalische Grenzen
    BOUNDS = {
        "linear_x":    (-0.5,  0.5),
        "angular_z":   (-1.0,  1.0),
        "camera_pan":  (-1.57, 1.57),
        "camera_tilt": (-0.79, 0.79),
        "arc_radius":  (-2.0,  2.0),
        "duration":    (0.1,   2.0),
    }

    @classmethod
    def from_array(cls, arr: np.ndarray, source: str = "policy") -> "Action":
        """numpy (6,) → Action"""
        assert len(arr) == 6, f"Erwartet 6 Werte, bekam {len(arr)}"
        return cls(
            linear_x=float(arr[0]),
            angular_z=float(arr[1]),
            camera_pan=float(arr[2]),
            camera_tilt=float(arr[3]),
            arc_radius=float(arr[4]),
            duration=float(arr[5]),
            source=source,
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.linear_x, self.angular_z,
                         self.camera_pan, self.camera_tilt,
                         self.arc_radius, self.duration],
                        dtype=np.float32)

    def _denorm(self, key: str, val: float) -> float:
        lo, hi = self.BOUNDS[key]
        return (val + 1.0) / 2.0 * (hi - lo) + lo

    def to_ros2(self) -> dict:
        """
        Normierte Werte → physikalische ROS2-Kommandos.

        Returns dict mit:
            twist:       geometry_msgs/Twist Felder
            camera:      CameraPanTilt.msg Felder
            arc:         ArcMovement.msg Felder
            description: Lesbare Beschreibung
        """
        lx  = self._denorm("linear_x",   self.linear_x)
        az  = self._denorm("angular_z",  self.angular_z)
        pan = self._denorm("camera_pan", self.camera_pan)
        tlt = self._denorm("camera_tilt",self.camera_tilt)
        arc = self._denorm("arc_radius", self.arc_radius)
        dur = self._denorm("duration",   self.duration)

        # Arc-Override: angular_z aus Kreisbogen-Kinematik
        if abs(arc) > 0.1:
            az_actual = lx / arc
        else:
            az_actual = az
            arc       = 0.0

        # Beschreibung
        dist_cm   = abs(lx) * dur * 100
        angle_deg = abs(az_actual) * dur * 180 / np.pi
        dir_lin   = "vor"  if lx > 0.05 else ("zurück" if lx < -0.05 else "")
        dir_rot   = "links" if az_actual > 0.05 else \
            ("rechts" if az_actual < -0.05 else "")
        desc_parts = []
        if dir_lin:
            desc_parts.append(f"{dist_cm:.0f}cm {dir_lin}wärts")
        if dir_rot and abs(arc) < 0.1:
            desc_parts.append(f"{angle_deg:.0f}° {dir_rot}")
        if abs(arc) > 0.1:
            desc_parts.append(f"Kurve {'L' if arc>0 else 'R'} R={abs(arc):.1f}m")
        if not desc_parts:
            desc_parts.append("Stopp")
        pan_deg = pan * 180 / np.pi
        tlt_deg = tlt * 180 / np.pi
        if abs(pan_deg) > 5:
            desc_parts.append(f"Kamera Pan {pan_deg:+.0f}°")
        if abs(tlt_deg) > 5:
            desc_parts.append(f"Kamera Tilt {tlt_deg:+.0f}°")

        return {
            # geometry_msgs/Twist
            "twist": {
                "linear":  {"x": lx,       "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": az_actual},
            },
            # CameraPanTilt.msg (custom)
            "camera": {
                "pan":  pan,    # rad
                "tilt": tlt,    # rad
            },
            # ArcMovement.msg (custom)
            "arc": {
                "linear_x":   lx,
                "arc_radius": arc,
                "duration":   dur,
            },
            # Meta
            "duration":    dur,
            "description": " | ".join(desc_parts),
            "source":      self.source,
        }

    def __repr__(self):
        ros = self.to_ros2()
        return (f"Action({ros['description']} | "
                f"Pan={ros['camera']['pan']*180/np.pi:+.0f}° "
                f"Tilt={ros['camera']['tilt']*180/np.pi:+.0f}°)")


# ─────────────────────────────────────────────
# OBSERVATION SOURCE (Abstrakt)
# ─────────────────────────────────────────────

class ObservationSource(ABC):
    """
    Abstrakte Basis für alle Kamera-Quellen.

    Implementiere diese Klasse für:
        - Mock (Szenen-Generator)
        - ROS2 Topic (/camera/image_raw)
        - Datei / Video
        - MiniWorld Gym
    """

    @abstractmethod
    def get_observation(self) -> Observation:
        """Gibt das aktuelle Kamerabild zurück."""
        ...

    @abstractmethod
    def get_high_res(self) -> Observation:
        """
        Gibt hochauflösendes Bild für Gemini ER zurück.
        Bei ROS2: anderes Topic oder up-scaling.
        """
        ...

    @property
    @abstractmethod
    def obs_shape(self) -> Tuple[int, int, int]:
        """(H, W, C) des low-res Bildes."""
        ...

    def close(self):
        """Ressourcen freigeben (optional)."""
        pass


class MockObsSource(ObservationSource):
    """
    Mock-Implementierung: synthetische Szenen.
    Wird in B17 Demo verwendet.
    Später ersetzt durch ROS2ObsSource.
    """

    SCENE_TYPES = ["red_box", "blue_ball", "green_door", "corridor", "corner"]

    def __init__(self, scene_switch_steps: int = 30,
                 low_res: Tuple = (16, 16),
                 high_res: Tuple = (128, 128)):
        self._step            = 0
        self._scene_idx       = 0
        self._scene_switch    = scene_switch_steps
        self._low_res         = low_res
        self._high_res        = high_res
        self._frame_counter   = 0

    @property
    def current_scene(self) -> str:
        return self.SCENE_TYPES[self._scene_idx]

    def _draw(self, scene: str, size: Tuple,
              noise: float = 0.0) -> np.ndarray:
        """Zeichnet eine Szene in gewünschter Auflösung."""
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for y in range(10, 16):
            img[y, :] = [int(60+(y-10)*15)]*3
        img[0:2,:] = [40,40,60]
        img[2:10,1] = img[2:10,14] = [70,70,90]
        for y in range(2,8):
            img[y,2:14] = [100,100,120]
        if scene == "red_box":
            img[8:12,5:9]=[200,40,40]; img[6:9,6:10]=[160,30,30]
        elif scene == "blue_ball":
            for y in range(16):
                for x in range(16):
                    d = np.sqrt((x-8)**2+(y-10)**2)
                    if d < 3.2:
                        b = int(255*max(0,1-d/3.2))
                        img[y,x] = [0,b//3,min(255,b)]
        elif scene == "green_door":
            img[3:8,6:10]=[30,140,50]; img[5,9]=[200,180,0]
        elif scene == "corridor":
            img[2:10,2:14]=[90,90,110]; img[4:6,7:9]=[220,220,180]
        elif scene == "corner":
            img[2:14,2:8]=[95,90,115]; img[2:14,8:14]=[110,105,130]
        img[10,2:14] = [50,50,50]

        # Auf Zielgröße skalieren
        if size != (16, 16):
            try:
                from PIL import Image as PILImage
                img = np.array(
                    PILImage.fromarray(img).resize(
                        (size[1], size[0]), PILImage.NEAREST)
                )
            except ImportError:
                img = np.repeat(np.repeat(img,
                                          size[0]//16, axis=0), size[1]//16, axis=1)

        if noise > 0:
            img = np.clip(
                img.astype(int) +
                (np.random.randn(*img.shape)*noise*255).astype(int),
                0, 255
            ).astype(np.uint8)
        return img

    def get_observation(self) -> Observation:
        self._step += 1
        self._frame_counter += 1
        if self._step % self._scene_switch == 0:
            self._scene_idx = (self._scene_idx+1) % len(self.SCENE_TYPES)

        img = self._draw(self.current_scene, self._low_res, noise=0.02)
        return Observation(
            image=img,
            timestamp=time.time(),
            frame_id=self._frame_counter,
            source="mock",
            metadata={"scene": self.current_scene},
        )

    def get_high_res(self) -> Observation:
        img = self._draw(self.current_scene, self._high_res, noise=0.0)
        return Observation(
            image=img,
            timestamp=time.time(),
            frame_id=self._frame_counter,
            source="mock_highres",
            metadata={"scene": self.current_scene},
        )

    @property
    def obs_shape(self):
        return (*self._low_res, 3)


class ROS2ObsSource(ObservationSource):
    """
    ROS2-Implementierung (Platzhalter).
    Aktiviere durch:
        import rclpy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

    Topics:
        /camera/image_raw            → low-res (oder direkt 16×16 pub)
        /camera/image_raw/compressed → compressed JPEG für Gemini

    Nutzung:
        source = ROS2ObsSource(
            node=rclpy_node,
            topic_low="/camera/image_raw",
            topic_high="/camera/image_raw/compressed",
            low_res=(16,16),
        )
        obs = source.get_observation()
    """

    def __init__(self, node=None, topic_low: str = "/camera/image_raw",
                 topic_high: str = "/camera/image_raw",
                 low_res=(16,16), high_res=(640,480)):
        self._node       = node
        self._topic_low  = topic_low
        self._topic_high = topic_high
        self._low_res    = low_res
        self._high_res   = high_res
        self._latest_low  = None
        self._latest_high = None
        self._lock        = threading.Lock()
        self._frame_count = 0

        if node is not None:
            self._setup_subscribers()

    def _setup_subscribers(self):
        """
        ROS2 Subscriber Setup.
        Wird aktiv wenn node übergeben wird.
        """
        # import rclpy
        # from sensor_msgs.msg import Image, CompressedImage
        # from cv_bridge import CvBridge
        # self._bridge = CvBridge()
        # self._sub_low = self._node.create_subscription(
        #     Image, self._topic_low, self._cb_low, 10)
        # self._sub_high = self._node.create_subscription(
        #     CompressedImage, self._topic_high, self._cb_high, 10)
        print(f"[ROS2ObsSource] Subscriber bereit:")
        print(f"  Low-res:  {self._topic_low}")
        print(f"  High-res: {self._topic_high}")

    def _cb_low(self, msg):
        """ROS2 Callback für low-res Bild."""
        # img = self._bridge.imgmsg_to_cv2(msg, "rgb8")
        # img = cv2.resize(img, self._low_res[::-1])
        # with self._lock:
        #     self._latest_low = img
        pass

    def _cb_high(self, msg):
        """ROS2 Callback für high-res Bild."""
        pass

    def get_observation(self) -> Observation:
        with self._lock:
            img = self._latest_low
        if img is None:
            # Fallback: schwarz
            img = np.zeros((*self._low_res, 3), dtype=np.uint8)
        self._frame_count += 1
        return Observation(
            image=img,
            timestamp=time.time(),
            frame_id=self._frame_count,
            source="ros2",
        )

    def get_high_res(self) -> Observation:
        with self._lock:
            img = self._latest_high
        if img is None:
            img = np.zeros((*self._high_res, 3), dtype=np.uint8)
        return Observation(
            image=img,
            timestamp=time.time(),
            frame_id=self._frame_count,
            source="ros2_highres",
        )

    @property
    def obs_shape(self):
        return (*self._low_res, 3)


# ─────────────────────────────────────────────
# ACTION SINK (Abstrakt)
# ─────────────────────────────────────────────

class ActionSink(ABC):
    """
    Abstrakte Basis für alle Aktions-Ausgaben.

    Implementiere für:
        - Mock (Logging/Simulation)
        - ROS2 Publisher (/cmd_vel, /camera_head/pan_tilt, /cmd_arc)
    """

    @abstractmethod
    def send(self, action: Action) -> bool:
        """
        Sendet eine Aktion.
        Returns True wenn erfolgreich.
        """
        ...

    @abstractmethod
    def stop(self) -> bool:
        """Sendet Stopp-Befehl (Sicherheit)."""
        ...

    def close(self):
        """Ressourcen freigeben."""
        pass


class MockActionSink(ActionSink):
    """
    Mock-Implementierung: Loggt Aktionen, simuliert Ausführung.
    Später ersetzt durch ROS2ActionSink.
    """

    def __init__(self, history_size: int = 200):
        self._history   = deque(maxlen=history_size)
        self._last      = None
        self._send_count = 0

    def send(self, action: Action) -> bool:
        ros2 = action.to_ros2()
        self._history.append({
            "action": action,
            "ros2":   ros2,
            "t":      time.time(),
        })
        self._last      = action
        self._send_count += 1
        return True

    def stop(self) -> bool:
        stop_action = Action(source="safety")
        return self.send(stop_action)

    @property
    def last_ros2(self) -> Optional[dict]:
        return self._last.to_ros2() if self._last else None

    @property
    def history(self):
        return list(self._history)


class ROS2ActionSink(ActionSink):
    """
    ROS2-Implementierung (Platzhalter).

    Publiziert auf:
        /cmd_vel              geometry_msgs/Twist   (Fahrt)
        /camera_head/pan_tilt custom CameraPanTilt  (Kamera)
        /cmd_arc              custom ArcMovement    (Kurvenfahrt)

    Nutzung:
        sink = ROS2ActionSink(
            node=rclpy_node,
            topic_twist="/cmd_vel",
            topic_camera="/camera_head/pan_tilt",
            topic_arc="/cmd_arc",
        )
        sink.send(action)

    Custom Messages installieren:
        ros2 pkg create --build-type ament_cmake robot_interfaces
        # Dann CameraPanTilt.msg und ArcMovement.msg definieren
    """

    def __init__(self, node=None,
                 topic_twist:  str = "/cmd_vel",
                 topic_camera: str = "/camera_head/pan_tilt",
                 topic_arc:    str = "/cmd_arc"):
        self._node         = node
        self._topic_twist  = topic_twist
        self._topic_camera = topic_camera
        self._topic_arc    = topic_arc
        self._pub_twist    = None
        self._pub_camera   = None
        self._pub_arc      = None

        if node is not None:
            self._setup_publishers()

    def _setup_publishers(self):
        """
        ROS2 Publisher Setup.
        Wird aktiv wenn node übergeben wird.
        """
        # from geometry_msgs.msg import Twist
        # from robot_interfaces.msg import CameraPanTilt, ArcMovement
        # self._pub_twist  = self._node.create_publisher(
        #     Twist, self._topic_twist, 10)
        # self._pub_camera = self._node.create_publisher(
        #     CameraPanTilt, self._topic_camera, 10)
        # self._pub_arc    = self._node.create_publisher(
        #     ArcMovement, self._topic_arc, 10)
        print(f"[ROS2ActionSink] Publisher bereit:")
        print(f"  Twist:  {self._topic_twist}")
        print(f"  Camera: {self._topic_camera}")
        print(f"  Arc:    {self._topic_arc}")

    def send(self, action: Action) -> bool:
        ros2 = action.to_ros2()
        try:
            # ── Twist Publisher ────────────────────────
            # twist_msg = Twist()
            # twist_msg.linear.x  = ros2["twist"]["linear"]["x"]
            # twist_msg.angular.z = ros2["twist"]["angular"]["z"]
            # self._pub_twist.publish(twist_msg)

            # ── Camera Publisher ───────────────────────
            # cam_msg = CameraPanTilt()
            # cam_msg.pan  = ros2["camera"]["pan"]
            # cam_msg.tilt = ros2["camera"]["tilt"]
            # self._pub_camera.publish(cam_msg)

            # ── Arc Publisher ──────────────────────────
            # arc_msg = ArcMovement()
            # arc_msg.linear_x   = ros2["arc"]["linear_x"]
            # arc_msg.arc_radius = ros2["arc"]["arc_radius"]
            # arc_msg.duration   = ros2["arc"]["duration"]
            # self._pub_arc.publish(arc_msg)
            return True
        except Exception as e:
            print(f"[ROS2ActionSink] Fehler: {e}")
            return False

    def stop(self) -> bool:
        """Sendet Null-Twist (Sicherheits-Stopp)."""
        return self.send(Action(source="safety_stop"))


# ─────────────────────────────────────────────
# ROBOT INTERFACE (verbindet alles)
# ─────────────────────────────────────────────

class RobotInterface:
    """
    Verbindet ObservationSource und ActionSink.
    Ist der einzige Punkt der B16 kennt.

    Austausch auf ROS2:
        obs_source = ROS2ObsSource(node, ...)   # statt MockObsSource
        action_sink = ROS2ActionSink(node, ...)  # statt MockActionSink
        robot = RobotInterface(obs_source, action_sink)
        # Rest des Codes bleibt identisch!
    """

    def __init__(self, obs_source: ObservationSource,
                 action_sink: ActionSink):
        self.obs    = obs_source
        self.act    = action_sink
        self._steps = 0

        # Statistiken
        self.stats = {
            "steps":          0,
            "actions_sent":   0,
            "obs_received":   0,
            "latencies_ms":   deque(maxlen=100),
        }

    def step(self, action_array: np.ndarray,
             source: str = "policy") -> Observation:
        """
        Vollständiger Schritt:
            1. Aktion senden
            2. Neue Observation holen
            3. Statistiken aktualisieren

        Args:
            action_array: np.ndarray (6,) normiert [-1,1]
            source:       "policy" | "manual" | "gemini"

        Returns:
            Observation (nächster Frame)
        """
        t0 = time.time()

        # Aktion konvertieren und senden
        action = Action.from_array(action_array, source=source)
        self.act.send(action)
        self.stats["actions_sent"] += 1

        # Nächste Observation
        obs = self.obs.get_observation()
        self.stats["obs_received"] += 1
        self.stats["steps"]        += 1

        # Latenz
        latency_ms = (time.time() - t0) * 1000
        self.stats["latencies_ms"].append(latency_ms)

        return obs

    def get_high_res(self) -> Observation:
        """High-res für Gemini ER."""
        return self.obs.get_high_res()

    def stop(self):
        """Sicherheits-Stopp."""
        self.act.stop()

    def close(self):
        self.stop()
        self.obs.close()
        self.act.close()

    @property
    def avg_latency_ms(self) -> float:
        lats = list(self.stats["latencies_ms"])
        return float(np.mean(lats)) if lats else 0.0

    def summary(self) -> str:
        return (f"RobotInterface: {self.stats['steps']} Steps | "
                f"Latenz Ø {self.avg_latency_ms:.1f}ms | "
                f"Obs-Source: {type(self.obs).__name__} | "
                f"Action-Sink: {type(self.act).__name__}")


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    N_STEPS = 200

    print("B17 – Saubere Roboter-Schnittstellen")
    print()
    print("Schnittstellen:")
    print("  ObservationSource  ← heute: Mock | später: ROS2")
    print("  ActionSink         ← heute: Mock | später: ROS2")
    print("  RobotInterface     ← verbindet beide")
    print()
    print("ROS2-Umstieg (eine Zeile ändern):")
    print("  obs_source  = ROS2ObsSource(node, '/camera/image_raw')")
    print("  action_sink = ROS2ActionSink(node, '/cmd_vel', ...)")
    print()

    # ── Interfaces erstellen ──────────────────────────────
    obs_source  = MockObsSource(scene_switch_steps=30,
                                low_res=(16,16), high_res=(128,128))
    action_sink = MockActionSink(history_size=200)
    robot       = RobotInterface(obs_source, action_sink)

    print(f"Aktive Implementierungen:")
    print(f"  Obs:    {type(obs_source).__name__}")
    print(f"  Action: {type(action_sink).__name__}")
    print()

    # Mock-Policy (später: B16 IntegratedSystem)
    ACTION_BOUNDS = {
        "linear_x":   (-0.5, 0.5), "angular_z":  (-1.0, 1.0),
        "camera_pan": (-1.57,1.57),"camera_tilt":(-0.79,0.79),
        "arc_radius": (-2.0, 2.0), "duration":   (0.1,  2.0),
    }
    SCENE_ACTIONS = {
        "red_box":    [ 0.6,  0.0,  0.0,  0.1,  0.0, -0.5],
        "blue_ball":  [ 0.4,  0.6, -0.3,  0.2,  0.0, -0.5],
        "green_door": [ 0.8,  0.0,  0.0,  0.0,  0.0, -0.3],
        "corridor":   [ 1.0,  0.0,  0.0,  0.0,  0.4, -0.4],
        "corner":     [ 0.3,  0.8,  0.5,  0.0,  0.0, -0.6],
    }

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B17 – Roboter-Schnittstellen: Mock → ROS2 ready',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.38)

    ax_lowres  = fig.add_subplot(gs[0, 0])
    ax_highres = fig.add_subplot(gs[0, 1])
    ax_action  = fig.add_subplot(gs[0, 2])
    ax_ros2    = fig.add_subplot(gs[0, 3]); ax_ros2.axis('off')
    ax_arch    = fig.add_subplot(gs[0, 4]); ax_arch.axis('off')

    ax_lat     = fig.add_subplot(gs[1, :2])
    ax_pan     = fig.add_subplot(gs[1, 2:4])
    ax_info    = fig.add_subplot(gs[1, 4]); ax_info.axis('off')

    ax_lx      = fig.add_subplot(gs[2, :2])
    ax_arc     = fig.add_subplot(gs[2, 2:4])
    ax_msgs    = fig.add_subplot(gs[2, 4]); ax_msgs.axis('off')

    # Tracking
    track = {k: [] for k in [
        "latency", "linear_x", "angular_z", "pan_deg", "tilt_deg",
        "arc_radius", "duration",
    ]}

    print(f"Starte Demo-Loop: {N_STEPS} Steps\n")

    for step in range(N_STEPS):
        scene = obs_source.current_scene

        # Policy: typische Szenen-Aktion
        base = np.array(SCENE_ACTIONS[scene], dtype=np.float32)
        act_arr = np.clip(base + 0.08*np.random.randn(6).astype(np.float32),
                          -1, 1)

        # ── Robot Step ────────────────────────────────
        t0  = time.time()
        obs = robot.step(act_arr, source="policy")
        lat = (time.time() - t0) * 1000

        # Letztes ROS2-Kommando
        ros2 = action_sink.last_ros2

        # Tracking
        track["latency"].append(lat)
        track["linear_x"].append(ros2["twist"]["linear"]["x"])
        track["angular_z"].append(ros2["twist"]["angular"]["z"])
        track["pan_deg"].append(ros2["camera"]["pan"] * 180/np.pi)
        track["tilt_deg"].append(ros2["camera"]["tilt"] * 180/np.pi)
        track["arc_radius"].append(ros2["arc"]["arc_radius"])
        track["duration"].append(ros2["arc"]["duration"])

        if step % 15 == 0 or step == N_STEPS-1:
            steps_x = list(range(len(track["latency"])))

            # ── Low-res Bild ───────────────────────────
            ax_lowres.clear()
            ax_lowres.imshow(obs.image, interpolation='nearest')
            ax_lowres.set_title(
                f'Low-res (16×16)\n{obs.source} | f#{obs.frame_id}',
                fontsize=7)
            ax_lowres.axis('off')

            # ── High-res Bild ──────────────────────────
            ax_highres.clear()
            hi = robot.get_high_res()
            ax_highres.imshow(hi.image, interpolation='bilinear')
            ax_highres.set_title(
                f'High-res (128×128)\n→ Gemini ER Input',
                fontsize=7)
            ax_highres.axis('off')

            # ── Aktion Balken ──────────────────────────
            ax_action.clear()
            anames = ["lx","az","pan","tilt","arc","dur"]
            colors = ['steelblue' if v>=0 else 'tomato' for v in act_arr]
            ax_action.bar(anames, act_arr, color=colors, alpha=0.85)
            ax_action.axhline(0, color='white', linewidth=0.5)
            ax_action.set_ylim(-1.3, 1.3)
            ax_action.set_title('Normierte Aktion (6D)', fontsize=8)
            ax_action.tick_params(labelsize=7)
            ax_action.set_facecolor('#0d0d0d')
            ax_action.tick_params(colors='white')

            # ── ROS2 Kommando ──────────────────────────
            ax_ros2.clear(); ax_ros2.axis('off')
            r2 = ros2
            ros_lines = [
                "── ROS2 Output ──────────",
                "",
                "geometry_msgs/Twist:",
                f"  /cmd_vel",
                f"  linear.x  = {r2['twist']['linear']['x']:+.3f} m/s",
                f"  angular.z = {r2['twist']['angular']['z']:+.3f} rad/s",
                "",
                "CameraPanTilt.msg:",
                f"  /camera_head/pan_tilt",
                f"  pan  = {r2['camera']['pan']*180/np.pi:+.0f}°",
                f"  tilt = {r2['camera']['tilt']*180/np.pi:+.0f}°",
                "",
                "ArcMovement.msg:",
                f"  /cmd_arc",
                f"  linear_x   = {r2['arc']['linear_x']:+.3f}",
                f"  arc_radius = {r2['arc']['arc_radius']:+.2f}m",
                f"  duration   = {r2['arc']['duration']:.2f}s",
                "",
                f"→ {r2['description'][:28]}",
            ]
            ax_ros2.text(0.03, 0.98, "\n".join(ros_lines),
                         transform=ax_ros2.transAxes,
                         fontsize=7.5, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round',
                                   facecolor='#0d1b2a', alpha=0.9),
                         color='lightcyan')

            # ── Architektur ────────────────────────────
            ax_arch.clear(); ax_arch.axis('off')
            arch_lines = [
                "── Schnittstellen ───────",
                "",
                "ObservationSource",
                "  abstract",
                "    ↓",
                "  MockObsSource  ✓",
                "  ROS2ObsSource  □",
                "    topic: /camera",
                "           /image_raw",
                "",
                "ActionSink",
                "  abstract",
                "    ↓",
                "  MockActionSink ✓",
                "  ROS2ActionSink □",
                "    /cmd_vel",
                "    /camera_head",
                "      /pan_tilt",
                "    /cmd_arc",
                "",
                "RobotInterface",
                "  verbindet beide",
                "  → B16 System",
                "",
                f"Step:  {step+1}/{N_STEPS}",
                f"Szene: {scene}",
                f"Latenz:{robot.avg_latency_ms:.1f}ms Ø",
            ]
            ax_arch.text(0.03, 0.98, "\n".join(arch_lines),
                         transform=ax_arch.transAxes,
                         fontsize=7, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round',
                                   facecolor='lightyellow', alpha=0.8))

            # ── Latenz ────────────────────────────────
            ax_lat.clear()
            ax_lat.plot(steps_x, track["latency"],
                        color='steelblue', linewidth=1.2, alpha=0.7,
                        label='Latenz')
            if len(track["latency"]) >= 10:
                ma = np.convolve(track["latency"],
                                 np.ones(10)/10, mode='valid')
                ax_lat.plot(range(9, len(track["latency"])), ma,
                            color='white', linewidth=2, label='MA-10')
            ax_lat.set_title('Step-Latenz (ms)', fontsize=9)
            ax_lat.set_ylabel('ms')
            ax_lat.legend(fontsize=7)
            ax_lat.set_facecolor('#0d0d0d')
            ax_lat.tick_params(colors='white')

            # ── Pan/Tilt ──────────────────────────────
            ax_pan.clear()
            ax_pan.plot(steps_x, track["pan_deg"],
                        color='cyan', linewidth=1.3, label='Pan°')
            ax_pan.plot(steps_x, track["tilt_deg"],
                        color='lightblue', linewidth=1.3,
                        linestyle='--', label='Tilt°')
            ax_pan.axhline(0, color='gray', linewidth=0.5)
            ax_pan.fill_between(steps_x, track["pan_deg"], 0,
                                alpha=0.2, color='cyan')
            ax_pan.set_title('Kamera-Kopf: Pan / Tilt (°)', fontsize=9)
            ax_pan.set_ylim(-100, 100)
            ax_pan.legend(fontsize=7)
            ax_pan.set_facecolor('#0d0d0d')
            ax_pan.tick_params(colors='white')

            # ── Linear_x ──────────────────────────────
            ax_lx.clear()
            ax_lx.plot(steps_x, track["linear_x"],
                       color='seagreen', linewidth=1.3,
                       label='linear_x (m/s)')
            ax_lx.plot(steps_x, track["angular_z"],
                       color='orange', linewidth=1.3,
                       label='angular_z (rad/s)')
            ax_lx.axhline(0, color='gray', linewidth=0.5)
            ax_lx.set_title('Fahrt: linear_x + angular_z', fontsize=9)
            ax_lx.legend(fontsize=7)
            ax_lx.set_facecolor('#0d0d0d')
            ax_lx.tick_params(colors='white')

            # ── Arc Radius ─────────────────────────────
            ax_arc.clear()
            ax_arc.plot(steps_x, track["arc_radius"],
                        color='orange', linewidth=1.3,
                        label='Arc Radius (m)')
            ax_arc.fill_between(steps_x, track["arc_radius"], 0,
                                alpha=0.3, color='orange')
            ax_arc.axhline(0, color='white', linewidth=0.8,
                           linestyle='--')
            ax_arc.set_title('Arc-Movement Radius (m)\n'
                             '+= Linkskurve  −= Rechtskurve  0= gerade',
                             fontsize=9)
            ax_arc.legend(fontsize=7)
            ax_arc.set_facecolor('#0d0d0d')
            ax_arc.tick_params(colors='white')

            # ── Letzte Messages ────────────────────────
            ax_msgs.clear(); ax_msgs.axis('off')
            hist  = action_sink.history[-5:]
            mlines = ["── Letzte Aktionen ──────", ""]
            for h in reversed(hist):
                ros = h["ros2"]
                mlines += [
                    f"  {ros['description'][:30]}",
                    f"  src={h['action'].source}",
                    "",
                ]
            ax_msgs.text(0.03, 0.98, "\n".join(mlines),
                         transform=ax_msgs.transAxes,
                         fontsize=7, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round',
                                   facecolor='lightcyan', alpha=0.8))

            plt.pause(0.02)

    # ── Finale Ausgabe ────────────────────────────────────
    print(robot.summary())
    print()
    print("ROS2-Umstieg:")
    print("  1. pip install rclpy cv_bridge")
    print("  2. ros2 pkg create robot_interfaces")
    print("     → CameraPanTilt.msg, ArcMovement.msg")
    print("  3. Code:")
    print("     obs_source  = ROS2ObsSource(node, '/camera/image_raw')")
    print("     action_sink = ROS2ActionSink(node)")
    print("     robot = RobotInterface(obs_source, action_sink)")
    print("     # Alle anderen Klassen bleiben unverändert!")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()


if __name__ == "__main__":
    run_demo()
