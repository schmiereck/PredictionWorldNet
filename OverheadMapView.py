"""
OverheadMapView – Draufsicht-Karte
=====================================
Zeigt den Roboter von oben/außen ohne Umbauten an B16-B19.

Funktioniert durch Dead Reckoning:
    Position und Heading werden aus den Aktionen berechnet:
        x     += linear_x * duration * cos(heading)
        y     += linear_x * duration * sin(heading)
        heading += angular_z * duration

Keine Änderungen an B16-B19 nötig.
Einfach in B19Orchestrator.py einbinden:

    from OverheadMapView import OverheadMapView
    overhead = OverheadMapView()
    overhead.setup()

    # Im Loop nach robot.step():
    overhead.update(action_ros2, scene, gemini_event)

Zeigt:
    - Roboter-Position als Pfeil (Richtung = Heading)
    - Fahrt-Spur (Trail)
    - Kamera-Sichtfeld (Pan/Tilt als Kegel)
    - Szenen-Objekte als farbige Marker
    - Gemini-Call Punkte (Cyan-Diamanten)
    - Arc-Bewegung als geschwungene Linie
    - Statistiken (Distanz, Drehung, Calls)
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Wedge
from collections import deque
import time


# ─────────────────────────────────────────────
# OVERHEAD MAP
# ─────────────────────────────────────────────

# Farben pro Szene
SCENE_COLORS = {
    "red_box":    "#e05050",
    "blue_ball":  "#5080e0",
    "green_door": "#50c050",
    "corridor":   "#c0c050",
    "corner":     "#c050c0",
    "miniworld":  "#80c0c0",
}

SCENE_MARKERS = {
    "red_box":    "s",    # Quadrat
    "blue_ball":  "o",    # Kreis
    "green_door": "D",    # Diamant
    "corridor":   "^",    # Dreieck
    "corner":     "p",    # Pentagon
    "miniworld":  "*",
}


class RobotPose:
    """Einfache 2D-Pose mit Dead Reckoning."""

    def __init__(self, x=0.0, y=0.0, heading=np.pi/2):
        self.x       = x
        self.y       = y
        self.heading = heading   # rad, 0=rechts, pi/2=oben

    def apply(self, linear_x: float, angular_z: float,
              arc_radius: float, duration: float):
        """
        Aktualisiert Pose aus physikalischen ROS2-Werten.
        arc_radius=0 → Differentialantrieb
        arc_radius≠0 → Kreisbogen
        """
        if abs(arc_radius) > 0.1:
            # Kreisbogen
            d_theta  = linear_x / arc_radius * duration
            r        = arc_radius
            dx = r * (np.sin(self.heading + d_theta) - np.sin(self.heading))
            dy = r * (-np.cos(self.heading + d_theta) + np.cos(self.heading))
            self.heading += d_theta
        else:
            # Differentialantrieb
            dist     = linear_x * duration
            d_theta  = angular_z * duration
            dx = dist * np.cos(self.heading + d_theta/2)
            dy = dist * np.sin(self.heading + d_theta/2)
            self.heading += d_theta

        self.x += dx
        self.y += dy
        self.heading = self.heading % (2*np.pi)

    @property
    def pos(self):
        return np.array([self.x, self.y])


class OverheadMapView:
    """
    Separates Fenster mit Draufsicht-Karte.
    Kein Eingriff in B16-B19 nötig.
    """

    def __init__(self, map_size: float = 30.0,
                 trail_length: int = 300,
                 title: str = "Draufsicht – Roboter-Karte"):
        self.map_size     = map_size
        self.initial_size = map_size
        self.title        = title
        self.pose         = RobotPose(x=0.0, y=0.0, heading=np.pi/2)
        self.trail        = deque(maxlen=trail_length)
        self.gemini_pts   = deque(maxlen=15)   # Nur letzte 15 Calls anzeigen
        self.scene_visits = deque(maxlen=500)  # (x,y,scene)
        self.fig          = None
        self.ax           = None
        self.ax_info      = None

        # Statistiken
        self.total_dist   = 0.0
        self.total_rot    = 0.0
        self.step_count   = 0
        self.last_scene   = ""
        self.last_reward  = 0.0

        # Kamera-Winkel (für Sichtfeld-Kegel)
        self.cam_pan_rad  = 0.0
        self.cam_tilt_rad = 0.0

        # Kamera-Verlauf (für Pan/Tilt-Plot)
        self.cam_pan_hist  = deque(maxlen=trail_length)
        self.cam_tilt_hist = deque(maxlen=trail_length)

        # MiniWorld Environment (optional, für echte Wände/Objekte)
        self._mw_env = None

        # Fenster-geschlossen Flag
        self._window_closed = False

    def setup(self):
        """Öffnet das Karten-Fenster."""
        plt.ion()   # Interactive Mode
        self.fig = plt.figure(
            figsize=(9, 8),
            num="Draufsicht"
        )
        try:
            tk_window = self.fig.canvas.manager.window
            tk_window.attributes('-topmost', False)
            self._tk_window = tk_window
        except Exception:
            self._tk_window = None
        self.fig.canvas.mpl_connect('close_event', lambda evt: setattr(self, '_window_closed', True))
        self.fig.patch.set_facecolor('#0d0d0d')
        self.fig.suptitle(self.title, fontsize=11,
                          fontweight='bold', color='white')

        gs = self.fig.add_gridspec(
            3, 3, height_ratios=[5, 1.5, 1.5],
            hspace=0.4, wspace=0.35
        )
        self.ax      = self.fig.add_subplot(gs[0, :])
        self.ax_dist = self.fig.add_subplot(gs[1, 0])
        self.ax_rot  = self.fig.add_subplot(gs[1, 1])
        self.ax_cam  = self.fig.add_subplot(gs[1, 2])
        self.ax_info = self.fig.add_subplot(gs[2, :])

        for ax in [self.ax, self.ax_dist, self.ax_rot, self.ax_cam, self.ax_info]:
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')

        self.ax_info.axis('off')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        return self

    def set_miniworld_env(self, env):
        """
        Setzt das MiniWorld Gym Environment für echte Wand-/Objekt-Anzeige.
        Erwartet das gym-wrapped env (env.unwrapped wird intern genutzt).
        """
        self._mw_env = env

    def _draw_miniworld(self):
        """
        Zeichnet Wände und Objekte aus MiniWorld.
        Koordinaten-Mapping: MiniWorld (x, z) → Overhead (x, y).
        MiniWorld y-Achse ist vertikal (Höhe), z ist Tiefe.
        """
        if self._mw_env is None:
            return False

        try:
            mw = getattr(self._mw_env, 'unwrapped', self._mw_env)
        except Exception:
            return False

        # Räume (Wände) zeichnen
        try:
            for room in mw.rooms:
                outline = room.outline  # shape (N, 3) mit [x, y, z]
                if outline is not None and len(outline) >= 3:
                    # (x, -z) extrahieren → 2D Polygon (Z negiert für korrekte Links/Rechts-Zuordnung)
                    poly_x = [p[0] for p in outline]
                    poly_y = [-p[2] for p in outline]
                    # Polygon schließen
                    poly_x.append(poly_x[0])
                    poly_y.append(poly_y[0])
                    # Boden als gefülltes Polygon
                    self.ax.fill(poly_x, poly_y,
                                 color='#1a2a3a', alpha=0.4, zorder=1)
                    # Wände als dicke Linien
                    self.ax.plot(poly_x, poly_y,
                                 color='#60a0c0', linewidth=2.5,
                                 solid_capstyle='round', zorder=2)
        except Exception:
            pass

        # Entities (Objekte) zeichnen
        try:
            for ent in mw.entities:
                if ent is mw.agent:
                    continue
                ex, _, ez = ent.pos
                # Farbe aus Entity ableiten
                ent_color = '#ffffff'
                ent_marker = 'o'
                ent_label = type(ent).__name__
                _mw_colors = {
                    "red": "#ff0000", "green": "#00ff00",
                    "blue": "#0000ff", "yellow": "#ffff00",
                    "grey": "#666666", "purple": "#7025c2",
                    "orange": "#ff8000", "white": "#ffffff",
                }
                from MiniWorldRegistry import get_entity_color_name, get_entity_type_name
                try:
                    color_name = get_entity_color_name(ent)
                    if color_name:
                        ent_color = _mw_colors.get(color_name, ent_color)
                    elif hasattr(ent, 'color') and hasattr(ent.color, '__len__') and len(ent.color) >= 3 and not isinstance(ent.color, str):
                        c = ent.color
                        ent_color = '#{:02x}{:02x}{:02x}'.format(
                            int(c[0] * 255) if c[0] <= 1 else int(c[0]),
                            int(c[1] * 255) if c[1] <= 1 else int(c[1]),
                            int(c[2] * 255) if c[2] <= 1 else int(c[2]),
                        )
                except Exception:
                    pass
                # Marker nach Typ
                name = get_entity_type_name(ent)
                if 'box' in name:
                    ent_marker = 's'
                elif 'ball' in name:
                    ent_marker = 'o'
                elif 'key' in name:
                    ent_marker = 'P'
                self.ax.scatter(
                    float(ex), float(-ez),
                    c=ent_color, marker=ent_marker,
                    s=120, zorder=5, edgecolors='white',
                    linewidths=0.8, label=ent_label
                )
        except Exception:
            pass

        return True

    def _draw_miniworld_pose(self):
        """Übernimmt die echte Agenten-Position aus MiniWorld."""
        if self._mw_env is None:
            return False
        try:
            mw = getattr(self._mw_env, 'unwrapped', self._mw_env)
            ax, _, az = mw.agent.pos
            agent_dir = mw.agent.dir
            self.pose.x = float(ax)
            self.pose.y = float(-az)
            self.pose.heading = float(agent_dir) % (2 * np.pi)
            return True
        except Exception:
            return False

    def update(self, action_ros2: dict, scene: str = "",
               gemini_event: dict = None):
        """
        Aktualisiert die Karte.

        Args:
            action_ros2:   dict aus Action.to_ros2()
            scene:         aktuelle Szene
            gemini_event:  dict mit reward etc. (optional)
        """
        if self.fig is None:
            return

        self.step_count += 1
        self.last_scene  = scene

        # ── Dead Reckoning ─────────────────────────────
        twist = action_ros2.get("twist", {})
        arc   = action_ros2.get("arc",   {})
        lx    = twist.get("linear",  {}).get("x", 0.0)
        az    = twist.get("angular", {}).get("z", 0.0)
        ar    = arc.get("arc_radius", 0.0)
        dur   = arc.get("duration",   0.5)

        old_pos     = self.pose.pos.copy()
        old_heading = self.pose.heading

        # MiniWorld: echte Position/Richtung übernehmen (kein Dead Reckoning)
        has_miniworld = self._draw_miniworld_pose()
        if not has_miniworld:
            self.pose.apply(lx, az, ar, dur)

        dist = float(np.linalg.norm(self.pose.pos - old_pos))
        self.total_dist += dist
        self.total_rot  += abs(az * dur)

        # Kamera-Winkel
        cam = action_ros2.get("camera", {})
        self.cam_pan_rad  = cam.get("pan",  0.0)
        self.cam_tilt_rad = cam.get("tilt", 0.0)
        self.cam_pan_hist.append(self.cam_pan_rad  * 180.0 / np.pi)
        self.cam_tilt_hist.append(self.cam_tilt_rad * 180.0 / np.pi)

        # Trail + Szenen-Visits
        self.trail.append((self.pose.x, self.pose.y, self.pose.heading, scene))
        self.scene_visits.append((self.pose.x, self.pose.y, scene))

        # Gemini-Event
        if gemini_event is not None:
            self.last_reward = gemini_event.get("reward", 0)
            self.gemini_pts.append({
                "pos":    self.pose.pos.copy(),
                "reward": self.last_reward,
                "scene":  scene,
                "situation": gemini_event.get("situation", ""),
            })

        # ── Map zeichnen ───────────────────────────────
        self.ax.clear()
        self.ax.set_facecolor('#0a0f1a')

        # ── MiniWorld: Wände und Objekte ───────────────
        has_mw = self._draw_miniworld()
        # ── Auto-Zoom: Bereich an Roboter anpassen ────
        # Alle bisherigen Positionen berücksichtigen
        if len(self.trail) >= 2:
            xs = [t[0] for t in self.trail]
            ys = [t[1] for t in self.trail]
            margin = max(3.0, self.map_size * 0.2)
            x_min = min(xs) - margin
            x_max = max(xs) + margin
            y_min = min(ys) - margin
            y_max = max(ys) + margin
            # Quadratisch halten
            size  = max(x_max - x_min, y_max - y_min)
            cx    = (x_min + x_max) / 2
            cy    = (y_min + y_max) / 2
        else:
            size = self.map_size
            cx, cy = self.pose.x, self.pose.y

        self.ax.set_xlim(cx - size/2, cx + size/2)
        self.ax.set_ylim(cy - size/2, cy + size/2)
        self.ax.set_aspect('equal')
        title_mode = 'MiniWorld (echte Position)' if has_mw else 'Dead Reckoning'
        self.ax.set_title(f'Draufsicht ({title_mode})',
                          fontsize=9, color='white')
        self.ax.tick_params(colors='gray', labelsize=7)
        self.ax.grid(True, color='#1a2a1a', linewidth=0.5)

        # ── Gitter-Linien (Meter) ──────────────────────
        for v in np.arange(-self.map_size/2, self.map_size/2+1, 1.0):
            self.ax.axhline(v, color='#1a2a3a', linewidth=0.4)
            self.ax.axvline(v, color='#1a2a3a', linewidth=0.4)

        # ── Start-Marker ───────────────────────────────
        self.ax.scatter([0], [0], marker='*', s=150,
                        color='gold', zorder=5, label='Start')

        # ── Trail ──────────────────────────────────────
        if len(self.trail) >= 2:
            trail_arr = np.array([(t[0],t[1]) for t in self.trail])
            # Färbung nach Szene
            for i in range(1, len(self.trail)):
                s   = self.trail[i][3]
                col = SCENE_COLORS.get(s, '#888888')
                self.ax.plot(
                    [trail_arr[i-1,0], trail_arr[i,0]],
                    [trail_arr[i-1,1], trail_arr[i,1]],
                    color=col, linewidth=1.5, alpha=0.6
                )

        # ── Szenen-Visit Marker (alle 10 Steps) ────────
        visits = list(self.scene_visits)
        for j in range(0, len(visits), 10):
            x, y, s = visits[j]
            col = SCENE_COLORS.get(s, '#888888')
            mrk = SCENE_MARKERS.get(s, 'o')
            self.ax.scatter(x, y, c=col, marker=mrk,
                            s=20, alpha=0.4, zorder=3)

        # ── Gemini-Call Punkte ─────────────────────────
        for gp in self.gemini_pts:
            r   = gp["reward"]
            col = ('seagreen' if r > 0.6 else
                   'gold'     if r > 0.3 else 'tomato')
            self.ax.scatter(
                gp["pos"][0], gp["pos"][1],
                marker='D', s=40, color=col,
                zorder=6, edgecolors='none'
            )
            self.ax.annotate(
                f'{r:.2f}',
                xy=gp["pos"],
                xytext=(gp["pos"][0]+0.15, gp["pos"][1]+0.15),
                fontsize=5.5, color=col
            )

        # ── Kamera-Sichtfeld (Kegel) ───────────────────
        fov_angle = np.radians(60)   # 60° Kamera-FOV
        # positive pan = rechts = CW, normalisiert auf [0, 2π]
        cam_dir   = (self.pose.heading - self.cam_pan_rad) % (2 * np.pi)
        cone_len  = 1.2
        left_ang  = cam_dir + fov_angle/2
        right_ang = cam_dir - fov_angle/2
        px, py    = self.pose.x, self.pose.y

        # Kamera sitzt vorne am Körper
        cam_offset = 0.25
        cam_x = px + cam_offset * np.cos(self.pose.heading)
        cam_y = py + cam_offset * np.sin(self.pose.heading)

        cone = plt.Polygon([
            [cam_x, cam_y],
            [cam_x + cone_len*np.cos(left_ang),
             cam_y + cone_len*np.sin(left_ang)],
            [cam_x + cone_len*np.cos(right_ang),
             cam_y + cone_len*np.sin(right_ang)],
        ], closed=True, color='cyan', alpha=0.12, zorder=4)
        self.ax.add_patch(cone)

        # Kamera-Richtungs-Linie
        self.ax.plot(
            [cam_x, cam_x + cone_len*0.8*np.cos(cam_dir)],
            [cam_y, cam_y + cone_len*0.8*np.sin(cam_dir)],
            color='cyan', linewidth=1.2, alpha=0.7, zorder=5
        )

        # Kamera-Punkt (vorne am Körper)
        self.ax.plot(cam_x, cam_y, 'o', color='cyan',
                     markersize=4, zorder=8)

        # ── Roboter-Körper als Rechteck + Richtungspfeil ──
        arrow_len = 0.5
        dx = arrow_len * np.cos(self.pose.heading)
        dy = arrow_len * np.sin(self.pose.heading)
        scene_col = SCENE_COLORS.get(scene, '#ffffff')

        # Rechteckiger Körper (gedreht in Fahrtrichtung)
        body_w2, body_h2 = 0.45, 0.30
        corners = np.array([
            [-body_w2/2, -body_h2/2],
            [ body_w2/2, -body_h2/2],
            [ body_w2/2,  body_h2/2],
            [-body_w2/2,  body_h2/2],
        ])
        cos_h = np.cos(self.pose.heading)
        sin_h = np.sin(self.pose.heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = corners @ rot.T + np.array([px, py])
        body_rect = plt.Polygon(
            corners, closed=True,
            facecolor='#2a3a4a', edgecolor='white',
            linewidth=1.5, alpha=0.9, zorder=6
        )
        self.ax.add_patch(body_rect)

        # Richtungspfeil (Fahrtrichtung)
        self.ax.annotate(
            "", xy=(px+dx, py+dy), xytext=(px-dx*0.3, py-dy*0.3),
            arrowprops=dict(
                arrowstyle='->', color='yellow',
                lw=2.5, mutation_scale=20
            ), zorder=7
        )

        # ── Legende ───────────────────────────────────
        legend_handles = []
        seen_scenes = set(t[3] for t in self.trail)
        for s in seen_scenes:
            col = SCENE_COLORS.get(s, '#888888')
            legend_handles.append(
                mpatches.Patch(color=col,
                               label=s.replace('_',' '))
            )
        legend_handles.append(
            mpatches.Patch(color='cyan', alpha=0.4, label='Kamera-FOV')
        )
        if self.gemini_pts:
            legend_handles.append(
                plt.Line2D([0],[0], marker='D', color='w',
                           markerfacecolor='gold', markersize=5,
                           label='Gemini-Call')
            )
        self.ax.legend(handles=legend_handles,
                       fontsize=6, loc='upper right',
                       facecolor='#0d1b2a', labelcolor='white')

        # ── Distanz-Plot ──────────────────────────────
        self.ax_dist.clear()
        self.ax_dist.set_facecolor('#111111')
        if len(self.trail) >= 2:
            trail_arr = np.array([(t[0],t[1]) for t in self.trail])
            dists = np.cumsum(np.linalg.norm(
                np.diff(trail_arr, axis=0), axis=1))
            self.ax_dist.plot(dists, color='seagreen',
                              linewidth=1.5)
            self.ax_dist.fill_between(range(len(dists)), dists,
                                      alpha=0.3, color='seagreen')
        self.ax_dist.set_title('Distanz (m)', fontsize=8, color='white')
        self.ax_dist.tick_params(colors='white', labelsize=6)

        # ── Rotation-Plot ──────────────────────────────
        self.ax_rot.clear()
        self.ax_rot.set_facecolor('#111111')
        if len(self.trail) >= 2:
            # unwrap verhindert Sprünge bei 0°/360° Übergang
            raw_rad = np.array([t[2] for t in self.trail])
            unwrapped = np.unwrap(raw_rad) * 180 / np.pi
            self.ax_rot.plot(unwrapped, color='mediumpurple', linewidth=1.5)
        heading_deg = (self.pose.heading * 180 / np.pi) % 360
        self.ax_rot.set_title(f'Heading: {heading_deg:.0f}°',
                              fontsize=8, color='white')
        self.ax_rot.tick_params(colors='white', labelsize=6)

        # ── Kamera Pan / Tilt Verlauf ───────────────────
        self.ax_cam.clear()
        self.ax_cam.set_facecolor('#111111')
        pan_deg  = self.cam_pan_rad  * 180.0 / np.pi
        tilt_deg = self.cam_tilt_rad * 180.0 / np.pi
        if len(self.cam_pan_hist) >= 2:
            xs = range(len(self.cam_pan_hist))
            self.ax_cam.plot(xs, list(self.cam_pan_hist),
                             color='cyan', linewidth=1.3, label='Pan')
            self.ax_cam.fill_between(xs, list(self.cam_pan_hist), 0,
                                     alpha=0.15, color='cyan')
            self.ax_cam.plot(xs, list(self.cam_tilt_hist),
                             color='#ff9944', linewidth=1.3,
                             linestyle='--', label='Tilt')
            self.ax_cam.fill_between(xs, list(self.cam_tilt_hist), 0,
                                     alpha=0.1, color='#ff9944')
            self.ax_cam.legend(fontsize=6, loc='upper left',
                               facecolor='#0d1b2a', labelcolor='white',
                               framealpha=0.7)
        self.ax_cam.axhline(0,   color='gray',  linewidth=0.6, alpha=0.5)
        self.ax_cam.axhline( 45, color='white', linewidth=0.4,
                             linestyle=':', alpha=0.3)
        self.ax_cam.axhline(-45, color='white', linewidth=0.4,
                             linestyle=':', alpha=0.3)
        self.ax_cam.set_ylim(-95, 95)
        self.ax_cam.set_title(
            f'Kamera  Pan={pan_deg:+.0f}°  Tilt={tilt_deg:+.0f}°',
            fontsize=8, color='white')
        self.ax_cam.tick_params(colors='white', labelsize=6)
        self.ax_cam.set_ylabel('°', fontsize=7, color='white')

        # ── Info-Zeile ─────────────────────────────────
        self.ax_info.clear(); self.ax_info.axis('off')
        pan_deg  = self.cam_pan_rad  * 180/np.pi
        tilt_deg = self.cam_tilt_rad * 180/np.pi
        info_txt = (
            f"Step: {self.step_count:4d}  |  "
            f"Pos: ({self.pose.x:+.2f}, {self.pose.y:+.2f})  |  "
            f"Heading: {(self.pose.heading*180/np.pi) % 360:.0f}°  |  "
            f"Distanz: {self.total_dist:.2f}m  |  "
            f"Kamera Pan={pan_deg:+.0f}° Tilt={tilt_deg:+.0f}°  |  "
            f"Gemini-Calls: {len(self.gemini_pts)}  |  "
            f"Szene: {scene}  |  "
            f"r_gem: {self.last_reward:.2f}"
        )
        self.ax_info.text(
            0.01, 0.5, info_txt,
            transform=self.ax_info.transAxes,
            fontsize=7.5, verticalalignment='center',
            fontfamily='monospace', color='lightcyan',
            bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9)
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def clear_trail(self):
        """Löscht den Trail, Gemini-Markierungen und Kamera-Verlauf (z.B. beim Szenenwechsel)."""
        self.trail.clear()
        self.scene_visits.clear()
        self.gemini_pts.clear()
        self.cam_pan_hist.clear()
        self.cam_tilt_hist.clear()
        print("[OverheadMap] Trail + Gemini-Calls + Kamera-Verlauf gelöscht")

    def close(self):
        try:
            plt.figure("Draufsicht")
            plt.close("Draufsicht")
        except Exception:
            pass


# ─────────────────────────────────────────────
# DEMO (standalone)
# ─────────────────────────────────────────────

SCENE_TYPES   = ["red_box","blue_ball","green_door","corridor","corner"]
SCENE_ACTIONS = {
    "red_box":    {"linear_x": 0.30, "angular_z":  0.00, "arc_radius": 0.0, "duration": 0.6},
    "blue_ball":  {"linear_x": 0.20, "angular_z":  0.60, "arc_radius": 0.0, "duration": 0.5},
    "green_door": {"linear_x": 0.40, "angular_z":  0.00, "arc_radius": 0.0, "duration": 0.8},
    "corridor":   {"linear_x": 0.50, "angular_z":  0.00, "arc_radius": 0.8, "duration": 0.7},
    "corner":     {"linear_x": 0.15, "angular_z":  0.80, "arc_radius": 0.0, "duration": 0.5},
}

def run_demo():
    N_STEPS = 300

    print("OverheadMapView – Demo")
    print(f"  {N_STEPS} Steps, Dead Reckoning aus Mock-Aktionen")
    print()

    overhead = OverheadMapView(map_size=30.0, trail_length=N_STEPS)
    overhead.setup()

    for step in range(N_STEPS):
        scene    = SCENE_TYPES[(step // 30) % len(SCENE_TYPES)]
        base_act = SCENE_ACTIONS[scene]

        # Rauschen
        lx  = float(np.clip(base_act["linear_x"]  + 0.05*np.random.randn(), -0.5, 0.5))
        az  = float(np.clip(base_act["angular_z"]  + 0.1 *np.random.randn(), -1.0, 1.0))
        arc = base_act["arc_radius"]
        dur = base_act["duration"]

        # Physikalische ROS2-Werte (wie Action.to_ros2() sie liefert)
        if abs(arc) > 0.1:
            az_actual = lx / arc
        else:
            az_actual = az

        action_ros2 = {
            "twist":  {"linear":  {"x": lx,         "y": 0, "z": 0},
                       "angular": {"x": 0, "y": 0, "z": az_actual}},
            "camera": {"pan":  float(0.3*np.random.randn()),
                       "tilt": float(0.1*np.random.randn())},
            "arc":    {"linear_x": lx, "arc_radius": arc, "duration": dur},
        }

        # Gemini-Event alle 30 Steps
        gemini_event = None
        if step % 30 == 0:
            gemini_event = {
                "reward":      float(np.clip(0.7 + 0.2*np.random.randn(), 0, 1)),
                "situation":   f"Szene: {scene}",
                "goal_progress": float(np.clip(step/N_STEPS, 0, 1)),
            }
            print(f"  [Step {step:4d}] Gemini: r={gemini_event['reward']:.2f}  {scene}")

        overhead.update(action_ros2, scene=scene, gemini_event=gemini_event)

    print(f"\nDemo abgeschlossen!")
    print(f"  Gesamtstrecke: {overhead.total_dist:.2f}m")
    print(f"  Drehung total: {overhead.total_rot*180/np.pi:.0f}°")
    print(f"  Gemini-Calls:  {len(overhead.gemini_pts)}")
    print()
    print("Einbinden in B19:")
    print("  from OverheadMapView import OverheadMapView")
    print("  overhead = OverheadMapView()")
    print("  overhead.setup()")
    print("  # Im Loop:")
    print("  overhead.update(action_sink.last_ros2, scene, gemini_event)")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        overhead.close()


if __name__ == "__main__":
    run_demo()
