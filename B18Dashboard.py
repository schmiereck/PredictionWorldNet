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

    def __init__(self, max_history: int = 500, title: str = "B18 Dashboard"):
        self.max_history = max_history
        self.title       = title
        self.fig         = None
        self._step       = 0

        # Ringpuffer für alle Metriken
        self.hist = {k: deque(maxlen=max_history) for k in [
            "fe", "recon", "kl", "action",
            "r_intrinsic", "r_gemini", "r_total",
            "goal_progress", "pred_error",
            "gemini_interval", "beta", "lr",
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

        # Imshow-Objekte für schnelles set_data() (kein clear() nötig)
        self._im_obs  = None
        self._im_pred = None
        self._im_diff = None

        # Letztes hochaufgelöstes Bild das an Gemini ging
        self._last_gemini_img  = None
        self._im_gemini        = None

    def setup(self):
        """Erstellt das Dashboard-Fenster."""
        plt.ion()   # Interactive Mode – kein Blockieren
        self.fig = plt.figure(figsize=(18, 11))
        self.fig.patch.set_facecolor('#0d0d0d')
        self.fig.suptitle(
            self.title,
            fontsize=14, fontweight='bold', color='white'
        )

        gs = gridspec.GridSpec(3, 6, figure=self.fig,
                               hspace=0.55, wspace=0.38)

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

        # ── Zeile 2: Gemini + Progress + Stats ────────
        self.ax_gemini  = self.fig.add_subplot(gs[2, :2])
        self.ax_prog    = self.fig.add_subplot(gs[2, 2:4])
        self.ax_stats   = self.fig.add_subplot(gs[2, 4:6])

        for ax in [self.ax_fe, self.ax_rewards, self.ax_latent,
                   self.ax_gemini, self.ax_prog, self.ax_stats]:
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')

        self.ax_stats.axis('off')
        self.ax_goal.axis('off')
        self.ax_cam.axis('off')
        self.ax_arc.axis('off')

        plt.pause(0.01)
        return self

    def update(
            self,
            obs:           np.ndarray,           # (H,W,3) uint8
            pred:          np.ndarray,           # (H,W,3) float [0,1]
            metrics:       dict,
            gemini_event:  dict       = None,
            latent_z:      np.ndarray = None,
            scene:         str        = "",
            goal:          str        = "",
            action_norm:   np.ndarray = None,
            sigma:         np.ndarray = None,
            topdown:       np.ndarray = None,    # (H,W,3) Top-Down Karte optional
            gemini_hires:  np.ndarray = None,    # (H,W,3) letztes Bild an Gemini
    ):
        """
        Aktualisiert das Dashboard (volle Metriken).

        Für live Kamera-Bilder jeden Step: update_live() aufrufen.
        Zwei Geschwindigkeiten:
            Schnell (jeden Aufruf):  Kamerabild, Vorhersage, Differenz, Auftrag
            Langsam (alle 10 Aufrufe): Loss-Kurven, Rewards, Latent-Space, Gemini-Timeline
        """
        self._step  += 1
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

        # Latent-Space
        if latent_z is not None:
            self.latent_points.append(latent_z.copy())
            self.latent_labels.append(scene)

        steps_x = list(range(len(self.hist["fe"])))

        # ── Panel: Aktuelles Bild ──────────────────────
        if self._im_obs is None:
            self.ax_obs.clear()
            self._im_obs = self.ax_obs.imshow(obs, interpolation='nearest')
            self.ax_obs.axis('off')
        else:
            self._im_obs.set_data(obs)
        self.ax_obs.set_title(
            f'Kamera (aktuell)\n{scene}', fontsize=8, color='white'
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
            0.5, 0.22, f"Fortschritt: {prog*100:.0f}%  |  r={r:.2f}",
            transform=self.ax_cam.transAxes,
            fontsize=7.5, ha='center', color=bar_color
        )
        self.ax_cam.text(
            0.5, 0.10, f"Szene: {scene}  |  Step: {self._step}",
            transform=self.ax_cam.transAxes,
            fontsize=7, ha='center', color='lightgray'
        )
        self.ax_cam.set_title('Aktueller Auftrag',
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
                    f'Bewegung: {desc} / warte auf Gemini...',
                    fontsize=8, color='gray'
                )
            else:
                self.ax_arc.set_title(
                    'Letztes Gemini-Bild (noch kein Call)',
                    fontsize=8, color='gray'
                )
            self.ax_arc.set_xlim(0, 1); self.ax_arc.set_ylim(0, 1)


        # ── Panels: Kurven + Latent + Gemini + Statistiken (langsam) ──
        if slow_update:
            # ── Free Energy ────────────────────────────
            self.ax_fe.clear()
            if self.hist["fe"]:
                self.ax_fe.plot(steps_x, list(self.hist["fe"]),
                                color='white', linewidth=1.2,
                                alpha=0.5, label='Free Energy')
                self.ax_fe.plot(steps_x, list(self.hist["recon"]),
                                color='steelblue', linewidth=1.3,
                                label='Recon')
                self.ax_fe.plot(steps_x, list(self.hist["kl"]),
                                color='darkorange', linewidth=1.3,
                                label='KL')
                if len(self.hist["fe"]) >= 20:
                    ma = np.convolve(list(self.hist["fe"]),
                                     np.ones(20)/20, mode='valid')
                    self.ax_fe.plot(range(19, len(self.hist["fe"])), ma,
                                    color='red', linewidth=2, label='FE MA-20')
                for ev in self.gemini_events:
                    si = ev["step"] - (self._step - len(self.hist["fe"]))
                    if 0 <= si < len(self.hist["fe"]):
                        self.ax_fe.axvline(si, color='cyan',
                                           linewidth=1, alpha=0.5)
            self.ax_fe.set_title('Loss-Kurven  |  Cyan = Gemini-Call',
                                 fontsize=9, color='white')
            self.ax_fe.legend(fontsize=6, ncol=2)
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
                scene_map = {s: i for i, s in enumerate([
                    "red_box","blue_ball","green_door","corridor","corner"
                ])}
                colors_pca = plt.cm.tab10(
                    [scene_map.get(l, 0)/5 for l in labels]
                )
                self.ax_latent.scatter(
                    pc[:,0], pc[:,1], c=colors_pca, s=15, alpha=0.7)
                for s, i in scene_map.items():
                    self.ax_latent.scatter(
                        [], [], c=[plt.cm.tab10(i/5)],
                        label=s.replace("_"," "), s=30)
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
            self.ax_gemini.set_ylim(-0.05, 1.1)
            self.ax_gemini.set_title(
                f'Gemini ER Calls ({len(self.gemini_events)} total)',
                fontsize=9, color='white')
            self.ax_gemini.legend(fontsize=6)
            self.ax_gemini.tick_params(colors='white')

            # ── Goal Progress ──────────────────────────
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
                self.ax_prog.axhline(1.0, color='gold', linestyle='--',
                                     linewidth=1, label='Ziel erreicht')
            self.ax_prog.set_title('Goal Progress + Prediction Error',
                                   fontsize=9, color='white')
            self.ax_prog.set_ylim(0, 1.15)
            self.ax_prog.legend(fontsize=6)
            self.ax_prog.set_facecolor('#111111')
            self.ax_prog.tick_params(colors='white')

            # ── Statistiken ────────────────────────────
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
            self.ax_stats.text(
                0.03, 0.98,
                "\n".join([
                    "── Training ─────────────",
                    f"Step:      {self._step}",
                    f"FE:        {fe_now:.5f}",
                    f"Recon:     {rec_now:.5f}",
                    f"KL:        {kl_now:.5f}",
                    f"Pred.Err:  {pe_now:.5f}",
                    f"Beta:      {beta_now:.4f}",
                    f"LR:        {lr_now:.2e}",
                    "", "── Rewards ──────────────",
                    f"Total:     {r_now:.4f}",
                    f"Gemini:    {list(self.hist['r_gemini'])[-1] if self.hist['r_gemini'] else 0:.4f}",
                    f"Intrinsic: {list(self.hist['r_intrinsic'])[-1] if self.hist['r_intrinsic'] else 0:.4f}",
                    "", "── Gemini ───────────────",
                    f"Calls:     {len(self.gemini_events)}",
                    f"Interval:  {gem_int:.0f} Steps",
                    f"Ziel:      {goal[:25]}",
                    "", "── Szene ────────────────",
                    f"{scene}",
                    "", "── Sync ─────────────────",
                    "Dashboard: synchron",
                    "→ was NN gerade sieht",
                    "Gemini ER: adaptiv",
                    f"→ letzter Call: Step",
                    f"  {self.gemini_events[-1]['step'] if self.gemini_events else 0}",
                ]),
                transform=self.ax_stats.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace', color='white',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9)
            )

        # ── Immer: figure rendern ──────────────────
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

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
            self.ax_obs.set_title('Kamera NN (live)', fontsize=8, color='lime')
            self.ax_obs.axis('off')

            self.ax_pred.clear()
            self._im_pred = self.ax_pred.imshow(pred_disp, interpolation='nearest')
            self.ax_pred.set_title(f'Vorhersage\nMSE={pe:.4f}',
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
            self.ax_obs.set_title('Kamera NN (live)', fontsize=8, color='lime')
            self.ax_pred.set_title(f'Vorhersage\nMSE={pe:.4f}',
                                   fontsize=8, color='white')

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


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
    elif scene_type == "blue_ball":
        for y in range(16):
            for x in range(16):
                d=np.sqrt((x-8)**2+(y-10)**2)
                if d<3.2:
                    b=int(255*max(0,1-d/3.2))
                    img[y,x]=[0,b//3,min(255,b)]
    elif scene_type == "green_door":
        img[3:8,6:10]=[30,140,50]; img[5,9]=[200,180,0]
    elif scene_type == "corridor":
        img[2:10,2:14]=[90,90,110]; img[4:6,7:9]=[220,220,180]
    elif scene_type == "corner":
        img[2:14,2:8]=[95,90,115]; img[2:14,8:14]=[110,105,130]
    img[10,2:14]=[50,50,50]
    if noise > 0:
        img = np.clip(img.astype(int) +
                      (np.random.randn(*img.shape)*noise*255).astype(int),
                      0, 255).astype(np.uint8)
    return img


SCENE_TYPES = ["red_box","blue_ball","green_door","corridor","corner"]
SCENE_GOALS = {
    "red_box":    "find the red box",
    "blue_ball":  "find the blue ball",
    "green_door": "navigate to the exit door",
    "corridor":   "explore the corridor",
    "corner":     "navigate to the corner",
}
SCENE_ACTIONS = {
    "red_box":    [ 0.6,  0.0,  0.0,  0.1,  0.0, -0.5],
    "blue_ball":  [ 0.4,  0.6, -0.3,  0.2,  0.0, -0.5],
    "green_door": [ 0.8,  0.0,  0.0,  0.0,  0.0, -0.3],
    "corridor":   [ 1.0,  0.0,  0.0,  0.0,  0.4, -0.4],
    "corner":     [ 0.3,  0.8,  0.5,  0.0,  0.0, -0.6],
}

GEMINI_MOCK_RESPONSES = {
    "red_box":    {"reward":0.9, "goal_progress":0.75,
                   "situation":"Rote Box klar sichtbar",
                   "recommendation":"Weiter vorwärts"},
    "blue_ball":  {"reward":0.85,"goal_progress":0.65,
                   "situation":"Blauer Ball zentriert",
                   "recommendation":"Langsam nähern"},
    "green_door": {"reward":0.7, "goal_progress":0.5,
                   "situation":"Tür teilweise sichtbar",
                   "recommendation":"Leicht links"},
    "corridor":   {"reward":0.5, "goal_progress":0.4,
                   "situation":"Korridor erkennbar",
                   "recommendation":"Geradeaus"},
    "corner":     {"reward":0.3, "goal_progress":0.2,
                   "situation":"Ecke schwer erkennbar",
                   "recommendation":"Mehr explorieren"},
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
        z_base = rng.standard_normal(64).astype(np.float32)
        z_now  = z_base + 0.3*np.random.randn(64).astype(np.float32)

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
                    "goal_progress":   goal_prog,
                    "beta":            beta,
                    "lr":              lr_val,
                    "gemini_interval": gem_int,
                },
                gemini_event=gemini_event,
                latent_z=z_now,
                scene=scene,
                goal=SCENE_GOALS[scene],
                action_norm=act,
                sigma=sigma,
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
