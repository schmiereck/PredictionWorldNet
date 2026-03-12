"""
B18 – Visualisierungs-Dashboard
=================================
Das ursprüngliche B16+B17+B18 aus dem Plan:

    B16 (original): Live-Anzeige
        → aktuelles Bild vs. vorhergesagtes Bild
        → Differenz-Bild (Prediction Error)

    B17 (original): Training-Metriken
        → Loss-Kurven (Free Energy, Recon, KL)
        → Reward-Verlauf (Intrinsic, Gemini, Total)
        → Latent-Space Visualisierung (t-SNE / PCA)

    B18 (original): Gemini-Feedback-Anzeige
        → Wann wurde Gemini gefragt?
        → Was hat er geantwortet?
        → Goal-Progress über Zeit

Alles in einem Dashboard-Fenster.
Kann live während des Trainings (B19 Vollintegration) laufen.
"""

import matplotlib
matplotlib.use('TkAgg')

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.widgets import TextBox
from collections import deque


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

class TrainingDashboard:
    """
    Live-Dashboard für das Training.

    Nutzung:
        dash = TrainingDashboard()
        dash.setup()

        # Im Training-Loop:
        dash.update(
            obs=obs_np,
            pred=pred_np,
            metrics={"fe": 0.1, "recon": 0.02, ...},
            gemini_event={"reward": 0.8, "situation": "..."},  # optional
        )

        dash.close()
    """

    def __init__(self, max_history: int = 500, title: str = "B18 Dashboard",
                 initial_display_every: int = 8, on_display_every_changed=None,
                 initial_display_every_live: int = 1, on_display_every_live_changed=None):
        self.max_history = max_history
        self.title       = title
        self.fig         = None
        self._step       = 0
        self._initial_display_every = initial_display_every
        self._on_display_every_changed = on_display_every_changed
        self._initial_display_every_live = initial_display_every_live
        self._on_display_every_live_changed = on_display_every_live_changed

        # Ringpuffer für alle Metriken
        self.hist = {k: deque(maxlen=max_history) for k in [
            "fe", "recon", "kl", "action",
            "r_intrinsic", "r_gemini", "r_total", "r_reward_pred",
            "goal_progress", "pred_error",
            "gemini_interval", "beta", "lr",
            "sigma_mean", "strategy_blend",
            # T18: neue Metriken aus T10/T13/T14
            "l_pred_img", "l_reward", "l_scene",
            "complexity",   # T18: beta * KL  (Complexity-Term der FE)
            "inaccuracy",   # T18: recon + l_pred_img  (Inaccuracy-Term der FE)
        ]}

        # Gemini-Events (Step + Inhalt)
        self.gemini_events = deque(maxlen=50)

        # Latent-Space Punkte für PCA-Visualisierung
        self.latent_points  = deque(maxlen=200)
        self.latent_labels  = deque(maxlen=200)

        # Letzte Bilder
        self._last_obs  = None
        self._last_pred = None
        self._last_goal = ""
        self._last_scene = ""
        self._last_scene_pred = "?"   # T13: Szenen-Klasse aus scene_head

        # Strategie-Info
        self._last_strategy_rule = None

        # Imshow-Objekte für schnelles set_data() (kein clear() nötig)
        self._im_obs  = None
        self._im_pred = None
        self._im_diff = None

        # Letztes hochaufgelöstes Bild das an Gemini ging
        self._last_gemini_img  = None
        self._im_gemini        = None

        # Fenster-geschlossen Flag
        self._window_closed = False

    @property
    def window_closed(self):
        return self._window_closed

    def setup(self):
        """Erstellt das Dashboard-Fenster."""
        plt.ion()   # Interactive Mode – kein Blockieren
        self.fig = plt.figure(figsize=(12, 9),
                              num="Dashboard"
        )
        try:
            tk_window = self.fig.canvas.manager.window
            tk_window.attributes('-topmost', False)
            self._tk_window = tk_window
        except Exception:
            self._tk_window = None
        self.fig.canvas.mpl_connect('close_event', lambda evt: setattr(self, '_window_closed', True))
        self.fig.patch.set_facecolor('#0d0d0d')
        self.fig.suptitle(
            self.title,
            fontsize=14, fontweight='bold', color='white'
        )

        # Ränder und Abstände optimieren
        self.fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)
        gs = gridspec.GridSpec(3, 6, figure=self.fig,
                               hspace=0.45, wspace=0.25)

        # ── Zeile 0: Bilder ───────────────────────────
        self.ax_obs   = self.fig.add_subplot(gs[0, 0])
        self.ax_pred  = self.fig.add_subplot(gs[0, 1])
        self.ax_diff  = self.fig.add_subplot(gs[0, 2])
        self.ax_goal  = self.fig.add_subplot(gs[0, 3])
        self.ax_cam   = self.fig.add_subplot(gs[0, 4])
        self.ax_arc   = self.fig.add_subplot(gs[0, 5])

        for ax in [self.ax_obs, self.ax_pred, self.ax_diff,
                   self.ax_goal, self.ax_cam, self.ax_arc]:
            ax.set_facecolor('#111111')

        # ── Zeile 1: Loss-Kurven ──────────────────────
        self.ax_fe      = self.fig.add_subplot(gs[1, :2])
        self.ax_rewards = self.fig.add_subplot(gs[1, 2:4])
        self.ax_latent  = self.fig.add_subplot(gs[1, 4:6])

        # ── Zeile 2: Gemini + Progress + Stats + Erkennung ─
        self.ax_gemini  = self.fig.add_subplot(gs[2, :2])
        self.ax_prog    = self.fig.add_subplot(gs[2, 2:4])
        self.ax_stats   = self.fig.add_subplot(gs[2, 4])
        self.ax_recog   = self.fig.add_subplot(gs[2, 5])

        for ax in [self.ax_fe, self.ax_rewards, self.ax_latent,
                   self.ax_gemini, self.ax_prog, self.ax_stats, self.ax_recog]:
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')

        self.ax_stats.axis('off')
        self.ax_goal.axis('off')
        self.ax_cam.axis('off')
        self.ax_arc.axis('off')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Textbox für DISPLAY_EVERY
        self._textbox_ax = self.fig.add_axes([0.84, 0.015, 0.06, 0.03])
        self._textbox_ax.set_facecolor('#222222')
        self._textbox = TextBox(self._textbox_ax, 'Update-Rate: ', initial=str(self._initial_display_every), color='#222222', hovercolor='#333333')
        self._textbox.label.set_color('white')
        self._textbox.text_disp.set_color('white')
        if hasattr(self._textbox, 'cursor'):
            self._textbox.cursor.set_color('white')
        self._textbox.on_submit(self._handle_display_every_submit)

        # Textbox für DISPLAY_EVERY_LIVE (Video Stream)
        self._textbox_live_ax = self.fig.add_axes([0.70, 0.015, 0.06, 0.03])
        self._textbox_live_ax.set_facecolor('#222222')
        self._textbox_live = TextBox(self._textbox_live_ax, 'Live-Rate: ', initial=str(self._initial_display_every_live), color='#222222', hovercolor='#333333')
        self._textbox_live.label.set_color('white')
        self._textbox_live.text_disp.set_color('white')
        if hasattr(self._textbox_live, 'cursor'):
            self._textbox_live.cursor.set_color('white')
        self._textbox_live.on_submit(self._handle_display_every_live_submit)

        return self

    def _handle_display_every_submit(self, text):
        try:
            val = int(text)
            if val > 0:
                if self._on_display_every_changed:
                    self._on_display_every_changed(val)
        except ValueError:
            pass

    def _handle_display_every_live_submit(self, text):
        try:
            val = int(text)
            if val > 0:
                if self._on_display_every_live_changed:
                    self._on_display_every_live_changed(val)
        except ValueError:
            pass

    def process_events(self):
        """Verarbeitet UI-Events (wie Texteingaben), ohne neu zu zeichnen."""
        if self.fig is not None:
            self.fig.canvas.flush_events()

    def update(
            self,
            obs:                 np.ndarray,           # (H,W,3) uint8
            pred:                np.ndarray,           # (H,W,3) float [0,1]
            metrics:             dict,
            gemini_event:        dict       = None,
            latent_z:            np.ndarray = None,
            scene:               str        = "",
            goal:                str        = "",
            action_norm:         np.ndarray = None,
            sigma:               np.ndarray = None,
            topdown:             np.ndarray = None,    # (H,W,3) Top-Down Karte optional
            gemini_hires:        np.ndarray = None,    # (H,W,3) letztes Bild an Gemini
            recognition_scores:  dict       = None,    # {label: float [0,1]}
            step:                int        = None,
    ):
        """
        Aktualisiert das Dashboard (volle Metriken).

        Für live Kamera-Bilder jeden Step: update_live() aufrufen.
        Zwei Geschwindigkeiten:
            Schnell (jeden Aufruf):  Kamerabild, Vorhersage, Differenz, Auftrag
            Langsam (alle 10 Aufrufe): Loss-Kurven, Rewards, Latent-Space, Gemini-Timeline
        """
        if step is not None:
            self._step = step
        else:
            self._step += 1

        self._last_obs  = obs
        self._last_pred = pred
        self._last_goal = goal
        self._last_scene = scene
        slow_update = (self._step % 10 == 0) or (self._step == 1)

        # Gemini-Hochreis-Bild speichern
        if gemini_hires is not None:
            self._last_gemini_img = gemini_hires

        # Metriken speichern
        for k, v in metrics.items():
            if k in self.hist:
                self.hist[k].append(float(v))

        # Sigma-Mittelwert berechnen und speichern
        if sigma is not None:
            sigma_mean = float(np.mean(sigma))
            self.hist["sigma_mean"].append(sigma_mean)

        # T18: FE-Zerlegung — Complexity (β·KL) vs Inaccuracy (Recon+Pred)
        beta_v  = float(metrics.get("beta",       0.0))
        kl_v    = float(metrics.get("kl",         0.0))
        recon_v = float(metrics.get("recon",      0.0))
        pred_v  = float(metrics.get("l_pred_img", 0.0))
        self.hist["complexity"].append(beta_v * kl_v)
        self.hist["inaccuracy"].append(recon_v + pred_v)

        # T13: Scene Prediction aus scene_head
        if "scene_pred" in metrics:
            self._last_scene_pred = str(metrics["scene_pred"])

        # Strategie-Info speichern
        if "strategy_rule" in metrics:
            self._last_strategy_rule = metrics["strategy_rule"]
        if "strategy_blend" in metrics:
            self.hist["strategy_blend"].append(float(metrics["strategy_blend"]))

        # pred auf obs-Größe skalieren falls nötig (z.B. 16×16 → 60×80)
        obs_h, obs_w = obs.shape[:2]
        pred_f = np.clip(pred, 0, 1)
        if pred_f.shape[:2] != (obs_h, obs_w):
            try:
                from PIL import Image as _PILImage
                pred_pil = _PILImage.fromarray(
                    (pred_f * 255).astype(np.uint8)
                ).resize((obs_w, obs_h), _PILImage.NEAREST)
                pred_f = np.array(pred_pil).astype(float) / 255.0
            except ImportError:
                ry = max(1, obs_h // pred_f.shape[0])
                rx = max(1, obs_w // pred_f.shape[1])
                pred_f = np.repeat(np.repeat(pred_f, ry, axis=0),
                                   rx, axis=1)
                pred_f = pred_f[:obs_h, :obs_w]

        # Prediction Error
        obs_f = obs.astype(float) / 255.0
        self.hist["pred_error"].append(
            float(np.mean((obs_f - pred_f)**2))
        )

        # Gemini-Event
        if gemini_event is not None:
            self.gemini_events.append({
                "step":   self._step,
                "event":  gemini_event,
            })

        # Latent-Space: goal als Label (wechselt, aussagekräftig);
        # falls kein goal, scene als Fallback
        if latent_z is not None:
            self.latent_points.append(latent_z.copy())
            lbl = goal.strip() if goal and goal.strip() else scene
            self.latent_labels.append(lbl)

        steps_x = list(range(len(self.hist["fe"])))

        # ── Panel: Aktuelles Bild (16×16 NN-Input) ────
        # obs wird hier NICHT angezeigt – update_live() macht das
        # ax_obs wird nur noch von update_live() beschrieben
        self.ax_obs.set_title(
            f'Kamera NN (live)\n{scene}', fontsize=8, color='lime'
        )

        # ── Panel: Vorhergesagtes Bild ─────────────────
        pred_disp = (pred_f * 255).astype(np.uint8)
        pe = self.hist["pred_error"][-1] if self.hist["pred_error"] else 0
        if self._im_pred is None:
            self.ax_pred.clear()
            self._im_pred = self.ax_pred.imshow(pred_disp, interpolation='nearest')
            self.ax_pred.axis('off')
        else:
            self._im_pred.set_data(pred_disp)
        self.ax_pred.set_title(
            f'Vorhersage\nMSE={pe:.4f}', fontsize=8, color='white'
        )

        # ── Panel: Differenz-Bild ──────────────────────
        diff = np.abs(obs_f - pred_f)
        diff_amp = np.clip(diff * 5, 0, 1)
        if self._im_diff is None:
            self.ax_diff.clear()
            self._im_diff = self.ax_diff.imshow(
                diff_amp, cmap='hot', interpolation='nearest', vmin=0, vmax=1
            )
            self.ax_diff.axis('off')
        else:
            self._im_diff.set_data(diff_amp)
        self.ax_diff.set_title(
            f'Prediction Error\n(×5 Verstärkt)', fontsize=8, color='white'
        )

        # ── Panel: Ziel + Action ──────────────────────
        self.ax_goal.clear(); self.ax_goal.axis('off')
        g_lines = [
            f"── Ziel ─────────────────",
            f'"{goal[:35]}"',
            f"Szene: {scene}",
            f"Step:  {self._step}",
        ]
        if action_norm is not None:
            anames = ["lin_x","ang_z","pan","tilt","arc","dur"]
            g_lines += ["", "── Aktion (normiert) ────"]
            for i, (n, v) in enumerate(zip(anames, action_norm)):
                sig_str = f" ±{sigma[i]:.2f}" if sigma is not None else ""
                bar = "█" * int(abs(v)*8)
                sign = "+" if v >= 0 else "-"
                g_lines.append(f"  {n:5s}: {sign}{abs(v):.2f}{sig_str}")
        self.ax_goal.text(
            0.03, 0.98, "\n".join(g_lines),
            transform=self.ax_goal.transAxes,
            fontsize=7.5, verticalalignment='top',
            fontfamily='monospace', color='lightcyan',
            bbox=dict(boxstyle='round', facecolor='#0d1b2a', alpha=0.9)
        )

        # ── Panel: AUFTRAG (prominent, große Schrift) ──
        self.ax_cam.clear(); self.ax_cam.set_facecolor('#0d1b2a')
        self.ax_cam.axis('off')
        # Farbbalken Reward
        if self.gemini_events:
            r = self.gemini_events[-1]["event"].get("reward", 0)
            bar_color = ('seagreen' if r > 0.6 else
                         'gold'     if r > 0.3 else 'tomato')
        else:
            r, bar_color = 0.0, 'gray'
        # Großes Ziel-Label
        self.ax_cam.text(
            0.5, 0.72, "AUFTRAG",
            transform=self.ax_cam.transAxes,
            fontsize=9, fontweight='bold',
            ha='center', color='gold'
        )
        self.ax_cam.text(
            0.5, 0.52, f'"{goal}"',
            transform=self.ax_cam.transAxes,
            fontsize=8.5, ha='center', color='white',
            wrap=True
        )
        prog = self.hist["goal_progress"][-1] \
            if self.hist["goal_progress"] else 0
        # Fortschrittsbalken
        bar_bg = plt.Rectangle((0.05, 0.28), 0.9, 0.1,
                               transform=self.ax_cam.transAxes,
                               color='#333333', clip_on=False)
        bar_fg = plt.Rectangle((0.05, 0.28), 0.9*prog, 0.1,
                               transform=self.ax_cam.transAxes,
                               color=bar_color, clip_on=False)
        self.ax_cam.add_patch(bar_bg)
        self.ax_cam.add_patch(bar_fg)
        self.ax_cam.text(
            0.5, 0.22, f"Progress: {prog*100:.0f}%  |  r={r:.2f}",
            transform=self.ax_cam.transAxes,
            fontsize=7.5, ha='center', color=bar_color
        )
        self.ax_cam.text(
            0.5, 0.10, f"Szene: {scene}  |  Step: {self._step}",
            transform=self.ax_cam.transAxes,
            fontsize=7, ha='center', color='lightgray'
        )
        self.ax_cam.set_title('Current goal',
                              fontsize=9, color='white')

        # Panel: Letztes Gemini-Bild (hochaufgeloest)
        if self._last_gemini_img is not None:
            if self._im_gemini is None:
                self.ax_arc.clear()
                self._im_gemini = self.ax_arc.imshow(
                    self._last_gemini_img, interpolation='nearest'
                )
                self.ax_arc.axis('off')
            else:
                self._im_gemini.set_data(self._last_gemini_img)
                h, w = self._last_gemini_img.shape[:2]
                self.ax_arc.set_xlim(-0.5, w - 0.5)
                self.ax_arc.set_ylim(h - 0.5, -0.5)
            n_calls = len(self.gemini_events)
            last_r  = (self.gemini_events[-1]['event'].get('reward', 0)
                       if self.gemini_events else 0)
            self.ax_arc.set_title(
                f'Letztes Gemini-Bild  (#{n_calls}  r={last_r:.2f})',
                fontsize=8, color='cyan'
            )
        elif topdown is not None:
            self.ax_arc.clear(); self._im_gemini = None
            self.ax_arc.imshow(topdown, interpolation='nearest')
            self.ax_arc.set_title('Top-Down (MiniWorld)',
                                  fontsize=8, color='white')
            self.ax_arc.axis('off')
        else:
            self.ax_arc.clear(); self._im_gemini = None
            self.ax_arc.set_facecolor('#0d1b2a')
            self.ax_arc.axis('off')
            if action_norm is not None:
                lx_norm  = float(action_norm[0])
                arc_norm = float(action_norm[4])
                arc_phys = (arc_norm+1)/2*4-2
                robot_c  = plt.Circle((0.5, 0.35), 0.07,
                                       color='steelblue', zorder=4)
                self.ax_arc.add_patch(robot_c)
                if abs(arc_phys) > 0.1:
                    r_n = np.clip(arc_phys/2, -1, 1)
                    cx2 = 0.5 + r_n*0.25
                    th  = np.linspace(-np.pi/2, np.pi/6, 40)
                    self.ax_arc.plot(
                        cx2 + abs(r_n*0.25)*np.cos(th),
                        0.35 + abs(r_n*0.25)*np.sin(th),
                        color='orange', linewidth=2.5
                    )
                    desc = f'Kurve R={arc_phys:+.1f}m'
                else:
                    self.ax_arc.annotate(
                        '', xy=(0.5, 0.35+lx_norm*0.35),
                        xytext=(0.5, 0.35),
                        arrowprops=dict(arrowstyle='->',
                                        color='lime', lw=2.5,
                                        mutation_scale=15)
                    )
                    desc = 'geradeaus'
                self.ax_arc.set_title(
                    f'Movement: {desc} / waiting for Gemini...',
                    fontsize=8, color='gray'
                )
            else:
                self.ax_arc.set_title(
                    'Last Gemini picture (no call yet)',
                    fontsize=8, color='gray'
                )
            self.ax_arc.set_xlim(0, 1); self.ax_arc.set_ylim(0, 1)


        # ── Panels: Kurven + Latent + Gemini + Statistiken (langsam) ──
        if slow_update:
            # ── T18: Free Energy Zerlegung ─────────────
            # Complexity (β·KL) = Prior-Kosten des latenten Raums
            # Inaccuracy (Recon+Pred) = Rekonstruktions- + Vorhersagefehler
            # Residual = Rest (Action, Sigma, Reward, Scene Loss)
            self.ax_fe.clear()
            if self.hist["fe"]:
                fe_list   = list(self.hist["fe"])
                comp_list = list(self.hist["complexity"])
                inacc_list= list(self.hist["inaccuracy"])
                n = min(len(fe_list), len(comp_list), len(inacc_list))
                sx = steps_x[:n]
                fe_n    = fe_list[:n]
                comp_n  = comp_list[:n]
                inacc_n = inacc_list[:n]
                resid_n = [max(0.0, f - c - i)
                           for f, c, i in zip(fe_n, comp_n, inacc_n)]

                # Gestapelte Flächen: Complexity → Inaccuracy → Residual
                comp_arr  = np.array(comp_n)
                inacc_arr = np.array(inacc_n)
                resid_arr = np.array(resid_n)
                self.ax_fe.fill_between(
                    sx, 0, comp_arr,
                    color='darkorange', alpha=0.55, label='Complexity (β·KL)')
                self.ax_fe.fill_between(
                    sx, comp_arr, comp_arr + inacc_arr,
                    color='steelblue', alpha=0.55, label='Inaccuracy (Recon+Pred)')
                self.ax_fe.fill_between(
                    sx, comp_arr + inacc_arr, comp_arr + inacc_arr + resid_arr,
                    color='mediumpurple', alpha=0.4, label='Residual (Action+σ+...)')

                # Gesamt-FE als weiße Linie
                self.ax_fe.plot(sx, fe_n, color='white',
                                linewidth=1.5, alpha=0.8, label='FE total')
                if n >= 20:
                    ma = np.convolve(fe_n, np.ones(20)/20, mode='valid')
                    self.ax_fe.plot(
                        sx[19:], ma,
                        color='red', linewidth=2, linestyle='--', label='MA-20')

                # Gemini-Calls als vertikale Linien
                for ev in self.gemini_events:
                    si = ev["step"] - (self._step - n)
                    if 0 <= si < n:
                        self.ax_fe.axvline(si, color='cyan',
                                           linewidth=1, alpha=0.4)

            self.ax_fe.set_title(
                'FE Decomposition: Complexity (β·KL) + Inaccuracy (Recon+Pred)  |  Cyan=Gemini',
                fontsize=8, color='white')
            if self.ax_fe.get_legend_handles_labels()[1]:
                self.ax_fe.legend(fontsize=5.5, ncol=3)
            self.ax_fe.set_facecolor('#111111')
            self.ax_fe.tick_params(colors='white')

        # ── Panel: Rewards ─────────────────────────────
        self.ax_rewards.clear()
        if self.hist["r_total"]:
            # ── Rewards ────────────────────────────────
            self.ax_rewards.clear()
            if self.hist["r_total"]:
                self.ax_rewards.plot(steps_x, list(self.hist["r_intrinsic"]),
                                     color='steelblue', linewidth=1.2,
                                     alpha=0.7, label='Intrinsic')
                self.ax_rewards.plot(steps_x, list(self.hist["r_gemini"]),
                                     color='gold', linewidth=1.5,
                                     label='Gemini ER')
                self.ax_rewards.plot(steps_x, list(self.hist["r_total"]),
                                     color='white', linewidth=2,
                                     label='Total')
                if self.hist["r_reward_pred"]:
                    self.ax_rewards.plot(
                        steps_x[-len(self.hist["r_reward_pred"]):],
                        list(self.hist["r_reward_pred"]),
                        color='violet', linewidth=1.2, linestyle=':',
                        alpha=0.85, label='r_pred (T14)')
                if len(self.hist["r_total"]) >= 20:
                    ma = np.convolve(list(self.hist["r_total"]),
                                     np.ones(20)/20, mode='valid')
                    self.ax_rewards.plot(range(19, len(self.hist["r_total"])),
                                         ma, color='orange', linewidth=2.5,
                                         linestyle='--', label='Total MA-20')
            self.ax_rewards.set_title('Rewards', fontsize=9, color='white')
            self.ax_rewards.legend(fontsize=6, ncol=2)
            self.ax_rewards.set_facecolor('#111111')
            self.ax_rewards.tick_params(colors='white')

            # ── Latent-Space (PCA) ─────────────────────
            self.ax_latent.clear()
            if len(self.latent_points) >= 5:
                pts    = np.stack(list(self.latent_points))
                labels = list(self.latent_labels)
                pts_c  = pts - pts.mean(axis=0)
                cov    = pts_c.T @ pts_c / len(pts_c)
                vals, vecs = np.linalg.eigh(cov)
                idx    = np.argsort(vals)[::-1]
                pc     = pts_c @ vecs[:, idx[:2]]
                # Dynamische Farb-Map: nur Labels die tatsächlich vorkommen
                unique_labels = list(dict.fromkeys(labels))  # Reihenfolge erhalten
                label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
                n_labels      = max(len(unique_labels), 1)
                colors_pca = plt.cm.tab10(
                    [label_to_idx[l] / n_labels for l in labels]
                )
                self.ax_latent.scatter(
                    pc[:,0], pc[:,1], c=colors_pca, s=15, alpha=0.7)
                for l, i in label_to_idx.items():
                    disp = l.replace("_"," ")
                    # Lange Goal-Strings kürzen
                    if len(disp) > 20:
                        disp = disp[:18] + "…"
                    self.ax_latent.scatter(
                        [], [], c=[plt.cm.tab10(i / n_labels)],
                        label=disp, s=30)
                self.ax_latent.legend(fontsize=5, loc='upper right')
            self.ax_latent.set_title('Latent-Space (PCA)',
                                     fontsize=9, color='white')
            self.ax_latent.set_facecolor('#111111')
            self.ax_latent.tick_params(colors='white')

            # ── Gemini-Feedback ────────────────────────
            self.ax_gemini.clear()
            self.ax_gemini.set_facecolor('#111111')
            if self.gemini_events:
                gem_steps = [ev["step"] for ev in self.gemini_events]
                gem_rew   = [ev["event"].get("reward", 0)
                             for ev in self.gemini_events]
                self.ax_gemini.scatter(
                    gem_steps, gem_rew,
                    c=['seagreen' if r>0.6 else 'gold' if r>0.3 else 'tomato'
                       for r in gem_rew],
                    s=80, zorder=4, marker='D', label='Gemini ER')
                self.ax_gemini.vlines(gem_steps, 0, gem_rew,
                                      colors='gray', linewidth=0.8, alpha=0.5)
                self.ax_gemini.axhline(0.6, color='seagreen',
                                       linestyle='--', linewidth=1, alpha=0.6)
                self.ax_gemini.axhline(0.3, color='gold',
                                       linestyle='--', linewidth=1, alpha=0.6)
                last_ev = self.gemini_events[-1]["event"]
                self.ax_gemini.text(
                    0.01, 0.02,
                    f"Step {self.gemini_events[-1]['step']}:\n"
                    f"r={last_ev.get('reward',0):.3f}  "
                    f"prog={last_ev.get('goal_progress',0)*100:.0f}%\n"
                    f"\"{last_ev.get('situation','')}\"",
                    transform=self.ax_gemini.transAxes,
                    fontsize=6, verticalalignment='bottom',
                    fontfamily='monospace', color='lightcyan',
                    bbox=dict(boxstyle='round',
                              facecolor='#0d1b2a', alpha=0.8)
                )
                self.ax_gemini.legend(fontsize=6)
            self.ax_gemini.set_ylim(-0.05, 1.1)
            self.ax_gemini.set_title(
                f'Gemini ER Calls ({len(self.gemini_events)} total)',
                fontsize=9, color='white')
            self.ax_gemini.tick_params(colors='white')

            # ── Goal Progress + Sigma + Strategy Blend ─
            self.ax_prog.clear()
            if self.hist["goal_progress"]:
                prog = list(self.hist["goal_progress"])
                self.ax_prog.fill_between(steps_x, 0, prog,
                                          color='seagreen', alpha=0.4)
                self.ax_prog.plot(steps_x, prog, color='seagreen',
                                  linewidth=1.5, label='Goal Progress')
                self.ax_prog.plot(steps_x, list(self.hist["pred_error"]),
                                  color='tomato', linewidth=1.2,
                                  alpha=0.7, label='Pred. Error')

                # Sigma (NN-Unsicherheit)
                if self.hist["sigma_mean"]:
                    self.ax_prog.plot(steps_x[-len(self.hist["sigma_mean"]):],
                                      list(self.hist["sigma_mean"]),
                                      color='cyan', linewidth=1.3,
                                      alpha=0.8, linestyle='--',
                                      label='Sigma (NN-Unsicherheit)')

                # Strategy Blend Factor (0 = nur NN, 1 = nur Strategie)
                if self.hist["strategy_blend"]:
                    self.ax_prog.plot(steps_x[-len(self.hist["strategy_blend"]):],
                                      list(self.hist["strategy_blend"]),
                                      color='magenta', linewidth=1.5,
                                      alpha=0.9, linestyle=':',
                                      label='Strategy Dominance')

                self.ax_prog.axhline(1.0, color='gold', linestyle='--',
                                     linewidth=1, label='Ziel erreicht')
            self.ax_prog.set_title('Progress, Error, Sigma & Strategy',
                                   fontsize=9, color='white')
            self.ax_prog.set_ylim(0, 1.15)
            self.ax_prog.legend(fontsize=6, ncol=2)
            self.ax_prog.set_facecolor('#111111')
            self.ax_prog.tick_params(colors='white')

            # ── Statistiken + T18 EFE-Proxy ────────────
            self.ax_stats.clear(); self.ax_stats.axis('off')
            fe_now  = list(self.hist["fe"])[-1]       if self.hist["fe"]      else 0
            r_now   = list(self.hist["r_total"])[-1]  if self.hist["r_total"] else 0
            rec_now = list(self.hist["recon"])[-1]    if self.hist["recon"]   else 0
            kl_now  = list(self.hist["kl"])[-1]       if self.hist["kl"]      else 0
            pe_now  = list(self.hist["pred_error"])[-1] if self.hist["pred_error"] else 0
            lr_now  = list(self.hist["lr"])[-1]       if self.hist["lr"]      else 0
            beta_now= list(self.hist["beta"])[-1]     if self.hist["beta"]    else 0
            gem_int = list(self.hist["gemini_interval"])[-1] \
                if self.hist["gemini_interval"] else 0
            sigma_now = list(self.hist["sigma_mean"])[-1] \
                if self.hist["sigma_mean"] else 0
            blend_now = list(self.hist["strategy_blend"])[-1] \
                if self.hist["strategy_blend"] else 0
            # T14 / T13 Metriken
            r_pred_now  = list(self.hist["r_reward_pred"])[-1] \
                if self.hist["r_reward_pred"] else 0
            l_rew_now   = list(self.hist["l_reward"])[-1] \
                if self.hist["l_reward"] else 0
            l_scn_now   = list(self.hist["l_scene"])[-1] \
                if self.hist["l_scene"] else 0
            comp_now    = list(self.hist["complexity"])[-1] \
                if self.hist["complexity"] else 0
            inacc_now   = list(self.hist["inaccuracy"])[-1] \
                if self.hist["inaccuracy"] else 0

            # EFE-Proxy: Epistemisch (Unsicherheit) vs Pragmatisch (Reward)
            # Epistemic  ≈ sigma_mean  (hohe Unsicherheit = mehr explorieren)
            # Pragmatic  ≈ r_reward_pred (hoher Reward = Ziel nah)
            efe_epist = sigma_now
            efe_prag  = r_pred_now
            efe_label = ("EXPLORE" if efe_epist > efe_prag + 0.1
                         else "CLOSER TO THE GOAL" if efe_prag > efe_epist + 0.1
                         else "BALANCED")
            efe_color = ('cyan' if efe_epist > efe_prag + 0.1
                         else 'gold' if efe_prag > efe_epist + 0.1
                         else 'lightgreen')

            # Strategie-Info aufbereiten
            strategy_lines = []
            if self._last_strategy_rule:
                strategy_lines = [
                    "", "── Strategy ────────────",
                    f"Rule:  {self._last_strategy_rule[:28]}",
                    f"Blend: {blend_now:.2f}"
                    f"{'  (Strat)' if blend_now > 0.7 else '  (Mix)' if blend_now > 0.3 else '  (NN)'}",
                ]

            self.ax_stats.text(
                0.03, 0.98,
                "\n".join([
                    "── Training ─────────────",
                    f"Step:      {self._step}",
                    f"FE:        {fe_now:.5f}",
                    f"  Compl:   {comp_now:.5f}  (β·KL)",
                    f"  Inacc:   {inacc_now:.5f}  (R+P)",
                    f"Beta:      {beta_now:.4f}",
                    f"LR:        {lr_now:.2e}",
                    "── T13/T14 ──────────────",
                    f"Scene:     {self._last_scene_pred}",
                    f"l_scene:   {l_scn_now:.5f}",
                    f"r_pred:    {r_pred_now:.4f}",
                    f"l_reward:  {l_rew_now:.5f}",
                    "── EFE Proxy ────────────",
                    f"Epistemic: {efe_epist:.3f}  (σ)",
                    f"Pragmatic: {efe_prag:.3f}  (r_pred)",
                    f"→ {efe_label}",
                    *strategy_lines,
                    "── Gemini ───────────────",
                    f"Calls:     {len(self.gemini_events)}",
                    f"Interval:  {gem_int:.0f} Steps",
                    f"Target:    {goal[:22]}",
                ]),
                transform=self.ax_stats.transAxes,
                fontsize=6.2, verticalalignment='top',
                fontfamily='monospace', color='white',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9)
            )
            # EFE-Label farbig hervorheben
            self.ax_stats.text(
                0.5, 0.01, f"EFE: {efe_label}",
                transform=self.ax_stats.transAxes,
                fontsize=7.5, va='top', ha='center', fontweight='bold',
                color=efe_color,
                bbox=dict(boxstyle='round', facecolor='#0d0d0d', alpha=0.7)
            )

        # ── NN-Erkennung (Balkendiagramm, jeden Aufruf) ──────────
        self.ax_recog.clear()
        self.ax_recog.set_facecolor('#111111')
        if recognition_scores:
            sorted_items = sorted(
                recognition_scores.items(), key=lambda x: x[1], reverse=True
            )
            lbls   = [item[0] for item in sorted_items]
            scores = [item[1] for item in sorted_items]
            bar_colors = [
                '#44ee88' if i == 0 else '#3a3a5e'
                for i in range(len(lbls))
            ]
            bars = self.ax_recog.barh(
                lbls, scores, color=bar_colors, height=0.65
            )
            self.ax_recog.set_xlim(0, 1.0)
            for bar, sc in zip(bars, scores):
                self.ax_recog.text(
                    min(sc + 0.03, 0.97),
                    bar.get_y() + bar.get_height() / 2,
                    f'{sc:.2f}',
                    va='center', fontsize=6.5, color='white'
                )
            self.ax_recog.invert_yaxis()
            self.ax_recog.set_title(
                f'NN Recognition →  {lbls[0]}\n'
                f'Scene-Head (T13): {self._last_scene_pred}',
                fontsize=7.5, color='#44ee88', fontweight='bold'
            )
        else:
            self.ax_recog.text(
                0.5, 0.5,
                'No Label-Embeddings\n(B21 execute)',
                ha='center', va='center',
                color='gray', fontsize=8,
                transform=self.ax_recog.transAxes
            )
            self.ax_recog.set_title(
                'NN Recognition', fontsize=8, color='gray'
            )
        self.ax_recog.tick_params(colors='white', labelsize=7)
        self.ax_recog.set_facecolor('#111111')

        # ── Immer: figure rendern ──────────────────
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    # ─────────────────────────────────────────────
    # SCHNELLES LIVE-UPDATE (jeden Step aufrufen)
    # ─────────────────────────────────────────────

    def update_live(self, obs: np.ndarray, pred: np.ndarray):
        """
        Schnelles Update nur für die 3 Kamera-Panels (jeden Step).
        Nutzt set_data() statt clear()+imshow() → deutlich schneller.

        Aufruf im Loop:
            dashboard.update_live(obs.image, ml_result["pred_obs"])
        """
        if self.fig is None:
            return

        self._last_obs  = obs
        self._last_pred = pred

        # Prediction skalieren falls nötig
        obs_h, obs_w = obs.shape[:2]
        pred_f = np.clip(pred, 0, 1)
        if pred_f.shape[:2] != (obs_h, obs_w):
            try:
                from PIL import Image as _PILImage
                pred_pil = _PILImage.fromarray(
                    (pred_f * 255).astype(np.uint8)
                ).resize((obs_w, obs_h), _PILImage.NEAREST)
                pred_f = np.array(pred_pil).astype(float) / 255.0
            except ImportError:
                ry = max(1, obs_h // pred_f.shape[0])
                rx = max(1, obs_w // pred_f.shape[1])
                pred_f = np.repeat(np.repeat(pred_f, ry, axis=0), rx, axis=1)
                pred_f = pred_f[:obs_h, :obs_w]

        obs_f     = obs.astype(float) / 255.0
        pred_disp = (pred_f * 255).astype(np.uint8)
        diff_amp  = np.clip(np.abs(obs_f - pred_f) * 5, 0, 1)
        pe        = float(np.mean((obs_f - pred_f) ** 2))

        if self._im_obs is None:
            # Erste Initialisierung
            self.ax_obs.clear()
            self._im_obs = self.ax_obs.imshow(obs, interpolation='nearest')
            self.ax_obs.set_title('Camera NN (live)', fontsize=8, color='lime')
            self.ax_obs.axis('off')

            self.ax_pred.clear()
            self._im_pred = self.ax_pred.imshow(pred_disp, interpolation='nearest')
            self.ax_pred.set_title(f'Prediction\nMSE={pe:.4f}',
                                   fontsize=8, color='white')
            self.ax_pred.axis('off')

            self.ax_diff.clear()
            self._im_diff = self.ax_diff.imshow(
                diff_amp, cmap='hot', interpolation='nearest', vmin=0, vmax=1
            )
            self.ax_diff.set_title('Pred. Error (×5)', fontsize=8, color='white')
            self.ax_diff.axis('off')
        else:
            self._im_obs.set_data(obs)
            self._im_pred.set_data(pred_disp)
            self._im_diff.set_data(diff_amp)
            self.ax_obs.set_title('Camera NN (live)', fontsize=8, color='lime')
            self.ax_pred.set_title(f'Prediction\nMSE={pe:.4f}',
                                   fontsize=8, color='white')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ─────────────────────────────────────────────
# MOCK-DATEN (simuliert B16 Training-Loop)
# ─────────────────────────────────────────────

def draw_scene(scene_type: str, noise: float = 0.0) -> np.ndarray:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    for y in range(10, 16):
        img[y, :] = [int(60+(y-10)*15)]*3
    img[0:2,:] = [40,40,60]
    img[2:10,1] = img[2:10,14] = [70,70,90]
    for y in range(2,8):
        img[y,2:14] = [100,100,120]
    if scene_type == "red_box":
        img[8:12,5:9]=[200,40,40]; img[6:9,6:10]=[160,30,30]
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
                    g=int(255*max(0,1-d/3.2))
                    img[y,x]=[0,min(255,g),0]
    elif scene_type == "blue_ball":
        for y in range(16):
            for x in range(16):
                d=np.sqrt((x-8)**2+(y-10)**2)
                if d<3.2:
                    b=int(255*max(0,1-d/3.2))
                    img[y,x]=[0,b//3,min(255,b)]
    img[10,2:14]=[50,50,50]
    if noise > 0:
        img = np.clip(img.astype(int) +
                      (np.random.randn(*img.shape)*noise*255).astype(int),
                      0, 255).astype(np.uint8)
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
SCENE_ACTIONS = {
    "red_box":    [ 0.6,  0.0,  0.0,  0.1,  0.0, -0.5],
    "green_ball": [ 0.5,  0.3,  0.0,  0.0,  0.0, -0.4],
    "blue_ball":  [ 0.4,  0.6, -0.3,  0.2,  0.0, -0.5],
    "orange_box": [ 0.6, -0.2,  0.1,  0.0,  0.0, -0.5],
    "yellow_box": [ 0.5,  0.0,  0.2,  0.1,  0.0, -0.5],
    "white_box":  [ 0.5,  0.1, -0.1,  0.0,  0.0, -0.5],
}

GEMINI_MOCK_RESPONSES = {
    "red_box":    {"reward":0.9, "goal_progress":0.75,
                   "situation":"red box visible",
                   "recommendation":"continue driving forward"},
    "green_ball": {"reward":0.85,"goal_progress":0.65,
                   "situation":"green ball visible",
                   "recommendation":"continue driving forward"},
    "blue_ball":  {"reward":0.85,"goal_progress":0.65,
                   "situation":"blue ball visible",
                   "recommendation":"continue driving forward"},
    "orange_box": {"reward":0.8, "goal_progress":0.6,
                   "situation":"orange box visible",
                   "recommendation":"continue driving forward"},
    "yellow_box": {"reward":0.8, "goal_progress":0.6,
                   "situation":"yellow box visible",
                   "recommendation":"continue driving forward"},
    "white_box":  {"reward":0.7, "goal_progress":0.5,
                   "situation":"white box visible",
                   "recommendation":"continue driving forward"},
}


def run_demo():
    N_STEPS          = 400
    SCENE_SWITCH     = 40
    GEMINI_INTERVAL  = 30
    UPDATE_EVERY     = 10

    print("B18 – Visualisierungs-Dashboard")
    print(f"  Panels:     3×6 = 18 Ansichten")
    print(f"  Bausteine:  B16 (original): Live-Bilder")
    print(f"              B17 (original): Loss-Kurven")
    print(f"              B18 (original): Gemini-Feedback")
    print(f"  Steps:      {N_STEPS}")
    print()

    dash = TrainingDashboard(
        max_history=N_STEPS,
        title="B18 – Live Training Dashboard  |  B16+B17+B18"
    )
    dash.setup()

    scene_idx = 0
    scene     = SCENE_TYPES[scene_idx]

    # Simulierter Trainings-Zustand
    fe_val   = 0.4
    recon    = 0.3
    kl_val   = 0.8
    r_intr   = 0.3
    r_gem    = 0.3
    r_total  = 0.3
    goal_prog = 0.0
    beta     = 0.0
    lr_val   = 1e-3
    gem_int  = 30.0

    print(f"Starte Dashboard: {N_STEPS} Steps\n")

    for step in range(N_STEPS):

        # Szene wechseln
        if step > 0 and step % SCENE_SWITCH == 0:
            scene_idx = (scene_idx+1) % len(SCENE_TYPES)
            scene     = SCENE_TYPES[scene_idx]
            goal_prog = 0.0   # reset bei neuer Szene
            print(f"  [Step {step:4d}] Szene → {scene}")

        # Simulierte Metriken (konvergieren über Zeit)
        t = step / N_STEPS
        fe_val  = max(0.005, 0.4 * np.exp(-t*3) + 0.01*np.random.randn())
        recon   = max(0.001, 0.3 * np.exp(-t*4) + 0.005*np.random.randn())
        kl_val  = max(0.05,  0.8 * np.exp(-t*2) + 0.02*np.random.randn())
        r_intr  = max(0.0,   0.3 * np.exp(-t*5) + 0.02*np.random.rand())
        beta    = min(0.05,  0.05 * t * 3)
        goal_prog = min(1.0, goal_prog + 0.003 + 0.01*np.random.rand())
        gem_int = min(80, 10 + 70*t)   # Interval wächst

        # Bilder
        obs_np  = draw_scene(scene, noise=0.02)
        # Vorhersage: wird über Zeit besser
        noise_l = max(0.02, 0.3*(1-t))
        pred_np = draw_scene(scene, noise=noise_l).astype(float)/255.0

        # Aktion
        base_act = np.array(SCENE_ACTIONS[scene], dtype=np.float32)
        act      = np.clip(base_act + 0.1*np.random.randn(6).astype(np.float32), -1, 1)
        sigma    = np.abs(0.3 * np.random.randn(6).astype(np.float32))

        # Latent-Vektor (mock: szenen-spezifisch + Rauschen)
        rng   = np.random.default_rng(scene_idx*100)
        z_base = rng.standard_normal(256).astype(np.float32)
        z_now  = z_base + 0.3*np.random.randn(256).astype(np.float32)

        # Gemini-Event
        gemini_event = None
        if step % GEMINI_INTERVAL == 0:
            r_gem     = GEMINI_MOCK_RESPONSES[scene]["reward"]
            r_gem    += 0.1 * np.random.randn()
            r_gem     = float(np.clip(r_gem, 0, 1))
            goal_prog = GEMINI_MOCK_RESPONSES[scene]["goal_progress"]
            gemini_event = {**GEMINI_MOCK_RESPONSES[scene], "reward": r_gem}
            print(f"  [Step {step:4d}] Gemini ER: r={r_gem:.3f}  "
                  f"'{gemini_event['situation']}'")
        else:
            r_gem = r_gem   # letzter Wert bleibt

        r_total = float(np.clip(
            0.3*r_intr + 0.4*r_gem +
            0.2*goal_prog + 0.1*(1-np.mean(sigma)),
            0, 1
        ))

        # Dashboard nur alle UPDATE_EVERY Steps updaten
        if step % UPDATE_EVERY == 0 or step == N_STEPS-1:
            # Simuliere Strategy Blend (oszilliert mit Sigma)
            blend_factor = np.clip(
                1.0 / (1.0 + np.exp(-(np.mean(sigma) - 0.4) * 8.0)),
                0.1, 0.9
            )
            strategy_rule = "no_target → turn_left" if blend_factor > 0.5 else "target_centered → move_forward"

            # T18 Mock: neue Metriken simulieren
            l_pred_img   = max(0, recon * 0.6 + 0.002*np.random.randn())
            l_reward_val = max(0, 0.05 * np.exp(-t*2) + 0.005*np.random.rand())
            l_scene_val  = max(0, 2.0  * np.exp(-t*3) + 0.05*np.random.rand())
            r_reward_pred= float(np.clip(r_gem + 0.05*np.random.randn(), 0, 1))
            # T13: Scene Prediction (konvergiert über Zeit zur richtigen Klasse)
            scene_vocab = ["red_box","yellow_box","orange_box","white_box",
                           "green_ball","blue_ball","exploring","unknown"]
            scene_map   = {"red_box": "red_box", "green_ball": "green_ball",
                           "blue_ball": "blue_ball", "orange_box": "orange_box",
                           "yellow_box": "yellow_box", "white_box": "white_box"}
            scene_pred  = scene_map.get(scene, "unknown") if t > 0.3 else "unknown"

            dash.update(
                obs=obs_np,
                pred=pred_np,
                metrics={
                    "fe":              fe_val,
                    "recon":           recon,
                    "kl":              kl_val,
                    "r_intrinsic":     r_intr,
                    "r_gemini":        r_gem,
                    "r_total":         r_total,
                    "r_reward_pred":   r_reward_pred,
                    "goal_progress":   goal_prog,
                    "beta":            beta,
                    "lr":              lr_val,
                    "gemini_interval": gem_int,
                    "strategy_rule":   strategy_rule,
                    "strategy_blend":  blend_factor,
                    "l_pred_img":      l_pred_img,
                    "l_reward":        l_reward_val,
                    "l_scene":         l_scene_val,
                    "scene_pred":      scene_pred,
                },
                gemini_event=gemini_event,
                latent_z=z_now,
                scene=scene,
                goal=SCENE_GOALS[scene],
                action_norm=act,
                sigma=sigma,
                step=step,
            )

    print("\nDashboard Demo abgeschlossen!")
    print(f"  Steps:          {N_STEPS}")
    print(f"  Gemini-Events:  {len(dash.gemini_events)}")
    print()
    print("Das Dashboard kann live im Training-Loop verwendet werden:")
    print("  dash = TrainingDashboard()")
    print("  dash.setup()")
    print("  # Im B19 Training-Loop:")
    print("  dash.update(obs, pred, metrics, gemini_event, ...)")

    dash.close()


if __name__ == "__main__":
    run_demo()
