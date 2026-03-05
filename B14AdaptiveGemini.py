"""
B14 – Adaptive Gemini-Frequenz
================================
Steuert wie oft Gemini befragt wird basierend auf dem aktuellen
Zustand des Agenten – spart API-Kosten ohne Qualität zu verlieren.

Kernidee:
    Gemini ist teuer (Latenz + API-Kosten).
    Wir brauchen Gemini NUR wenn:
        1. Neuer Befehl vom User
        2. Agent ist verwirrt (hohe Free Energy)
        3. Ziel wurde erreicht (neues Ziel nötig)
        4. Lange kein Update (Timeout)
        5. Hohe Novelty → unbekannte Situation

    Wir brauchen Gemini NICHT wenn:
        - Agent weiß was er tut (niedrige FE)
        - Gleiches Ziel wie letztes Mal
        - Gerade erst befragt

Frequenz-Steuerung:
    call_interval = base_interval * (1 / urgency)
    urgency = f(FE, novelty, goal_reached, timeout)

    Hohe Urgency → kurzes Interval → oft fragen
    Niedrige Urgency → langes Interval → selten fragen

Typische Kosten-Ersparnis:
    Ohne Adaptive: 1 Call/Step → 10.000 Calls/Stunde
    Mit Adaptive:  1 Call/~50 Steps → 200 Calls/Stunde  (95% Ersparnis)
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import time


# ─────────────────────────────────────────────
# ADAPTIVE FREQUENCY CONTROLLER
# ─────────────────────────────────────────────

class AdaptiveGeminiController:
    """
    Entscheidet wann Gemini befragt werden soll.

    Urgency-Berechnung (0 = entspannt, 1 = dringend):
        u_fe      : Free Energy Urgency  (hohe FE → dringend)
        u_novelty : Novelty Urgency      (unbekannte Situation → dringend)
        u_timeout : Timeout Urgency      (zu lange kein Update → dringend)
        u_goal    : Goal-Reached Flag    (Ziel erreicht → sofort neu fragen)

        urgency = clip(w_fe*u_fe + w_nov*u_novelty + w_to*u_timeout + u_goal, 0, 1)

    Call-Interval:
        min_interval ≤ call_interval ≤ max_interval
        call_interval = max_interval * (1 - urgency) + min_interval * urgency
    """

    def __init__(
            self,
            min_interval:   int   = 5,      # Mindestabstand zwischen Calls (Steps)
            max_interval:   int   = 100,    # Max. Abstand (bei niedrigem FE)
            fe_threshold:   float = 0.15,   # FE über diesem Wert → dringend
            fe_low:         float = 0.05,   # FE unter diesem Wert → entspannt
            novelty_thresh: float = 0.6,    # Novelty über diesem Wert → dringend
            timeout_steps:  int   = 80,     # Nach N Steps ohne Call → dringend
            w_fe:           float = 0.5,
            w_novelty:      float = 0.3,
            w_timeout:      float = 0.2,
    ):
        self.min_interval   = min_interval
        self.max_interval   = max_interval
        self.fe_threshold   = fe_threshold
        self.fe_low         = fe_low
        self.novelty_thresh = novelty_thresh
        self.timeout_steps  = timeout_steps
        self.w_fe           = w_fe
        self.w_novelty      = w_novelty
        self.w_timeout      = w_timeout

        # Zustand
        self.last_call_step  = -max_interval   # Beim ersten Step sofort fragen
        self.call_count      = 0
        self.total_steps     = 0

        # Gleitender Durchschnitt FE (EMA)
        self.fe_ema          = 0.2
        self.ema_alpha       = 0.1

        # Historie für Visualisierung
        self.history = {
            "urgency":       [],
            "interval":      [],
            "fe_ema":        [],
            "novelty":       [],
            "called":        [],    # 1 wenn Gemini aufgerufen, sonst 0
            "u_fe":          [],
            "u_novelty":     [],
            "u_timeout":     [],
        }

    def update(
            self,
            fe:           float,
            novelty:      float,
            goal_reached: bool  = False,
            force_call:   bool  = False,
    ) -> bool:
        """
        Entscheidet ob Gemini jetzt aufgerufen werden soll.

        Args:
            fe:           Aktuelle Free Energy
            novelty:      Aktuelle Novelty (0-1)
            goal_reached: True wenn Agent Ziel erreicht hat
            force_call:   True für neuen User-Befehl

        Returns:
            True  → Gemini jetzt aufrufen
            False → Kein Call nötig
        """
        self.total_steps += 1

        # EMA der Free Energy
        self.fe_ema = (1 - self.ema_alpha) * self.fe_ema + self.ema_alpha * fe

        steps_since_call = self.total_steps - self.last_call_step

        # ── Urgency-Komponenten ────────────────────────────
        # FE Urgency: linear zwischen fe_low (0) und fe_threshold (1)
        u_fe = np.clip(
            (self.fe_ema - self.fe_low) / (self.fe_threshold - self.fe_low + 1e-8),
            0.0, 1.0
        )

        # Novelty Urgency
        u_novelty = np.clip(novelty / self.novelty_thresh, 0.0, 1.0)

        # Timeout Urgency: linear nach timeout_steps
        u_timeout = np.clip(steps_since_call / self.timeout_steps, 0.0, 1.0)

        # Gesamt-Urgency
        urgency = np.clip(
            self.w_fe * u_fe +
            self.w_novelty * u_novelty +
            self.w_timeout * u_timeout,
            0.0, 1.0
        )

        # Adaptives Interval
        call_interval = int(
            self.max_interval * (1 - urgency) +
            self.min_interval * urgency
        )

        # ── Call-Entscheidung ──────────────────────────────
        should_call = (
                force_call or
                goal_reached or
                steps_since_call >= call_interval
        )

        if should_call:
            self.last_call_step = self.total_steps
            self.call_count    += 1

        # Historie
        self.history["urgency"].append(float(urgency))
        self.history["interval"].append(float(call_interval))
        self.history["fe_ema"].append(float(self.fe_ema))
        self.history["novelty"].append(float(novelty))
        self.history["called"].append(1.0 if should_call else 0.0)
        self.history["u_fe"].append(float(u_fe))
        self.history["u_novelty"].append(float(u_novelty))
        self.history["u_timeout"].append(float(u_timeout))

        return should_call

    @property
    def call_rate(self) -> float:
        """Anteil der Steps mit Gemini-Call (0-1)."""
        if self.total_steps == 0:
            return 0.0
        return self.call_count / self.total_steps

    @property
    def estimated_hourly_cost(self) -> dict:
        """
        Schätzt API-Kosten bei typischer Step-Rate.
        Gemini 2.5 Flash: ~$0.001 pro Call (grobe Schätzung).
        """
        steps_per_hour  = 3600    # ~1 Step/Sek
        calls_per_hour  = steps_per_hour * self.call_rate
        cost_per_call   = 0.001   # USD
        return {
            "calls_per_hour":    int(calls_per_hour),
            "cost_per_hour_usd": calls_per_hour * cost_per_call,
            "savings_vs_always": 1.0 - self.call_rate,
        }

    def summary(self) -> dict:
        return {
            "total_steps":    self.total_steps,
            "total_calls":    self.call_count,
            "call_rate":      f"{self.call_rate*100:.1f}%",
            "min_interval":   self.min_interval,
            "max_interval":   self.max_interval,
        }


# ─────────────────────────────────────────────
# MOCK AGENT STATE (simuliert B11 Outputs)
# ─────────────────────────────────────────────

class MockAgentState:
    """
    Simuliert den Agenten-Zustand über Zeit:
        - Phase 1 (0-100):   Hohe FE, hohe Novelty (Agent lernt)
        - Phase 2 (100-200): Niedrige FE (Agent kennt Umgebung)
        - Phase 3 (200-250): Spike! Neue Szene (unbekannt)
        - Phase 4 (250-350): Wieder stabil
        - Phase 5 (350-400): User gibt neuen Befehl
    """

    def __init__(self):
        self.step = 0

    def get_state(self) -> dict:
        s = self.step
        self.step += 1

        if s < 100:
            # Phase 1: Lernen – hohe FE + Novelty
            fe      = 0.25 - 0.15 * (s/100) + 0.05 * np.random.randn()
            novelty = 0.8  - 0.5  * (s/100) + 0.1  * np.random.rand()
            goal_reached = False
        elif s < 200:
            # Phase 2: Stabil – niedrige FE
            fe      = 0.04 + 0.02 * np.random.randn()
            novelty = 0.2  + 0.1  * np.random.rand()
            goal_reached = (s == 180)   # Ziel einmal erreicht
        elif s < 250:
            # Phase 3: Neue unbekannte Szene!
            fe      = 0.20 + 0.08 * np.random.rand()
            novelty = 0.75 + 0.15 * np.random.rand()
            goal_reached = False
        elif s < 350:
            # Phase 4: Wieder stabil
            fe      = 0.05 + 0.02 * np.random.randn()
            novelty = 0.15 + 0.1  * np.random.rand()
            goal_reached = False
        else:
            # Phase 5: Stabil, neuer User-Befehl kommt bei s=370
            fe      = 0.04 + 0.01 * np.random.randn()
            novelty = 0.1  + 0.05 * np.random.rand()
            goal_reached = False

        return {
            "fe":           float(np.clip(fe, 0.01, 0.5)),
            "novelty":      float(np.clip(novelty, 0, 1)),
            "goal_reached": goal_reached,
            "force_call":   (s == 370),   # Neuer User-Befehl
        }


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

def run_demo():
    N_STEPS = 400

    controller = AdaptiveGeminiController(
        min_interval=5,
        max_interval=100,
        fe_threshold=0.15,
        fe_low=0.05,
        novelty_thresh=0.6,
        timeout_steps=80,
        w_fe=0.5,
        w_novelty=0.3,
        w_timeout=0.2,
    )
    agent = MockAgentState()

    print("B14 – Adaptive Gemini-Frequenz")
    print(f"  min_interval:   {controller.min_interval} Steps")
    print(f"  max_interval:   {controller.max_interval} Steps")
    print(f"  fe_threshold:   {controller.fe_threshold}")
    print(f"  novelty_thresh: {controller.novelty_thresh}")
    print(f"  Gewichte:       FE={controller.w_fe}  "
          f"Nov={controller.w_novelty}  Timeout={controller.w_timeout}")
    print()

    # Phasen-Labels für Plot
    phase_labels = [
        (0,   100, "Phase 1\nLernen",   '#2a1a0a'),
        (100, 200, "Phase 2\nStabil",   '#0a1a0a'),
        (200, 250, "Phase 3\nNeu!",     '#1a0a0a'),
        (250, 350, "Phase 4\nStabil",   '#0a0a1a'),
        (350, 400, "Phase 5\nBefehl",   '#1a1a0a'),
    ]

    # ── Matplotlib Setup ──────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle('B14 – Adaptive Gemini-Frequenz: Wann fragen?',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38)

    ax_fe       = fig.add_subplot(gs[0, :3])   # FE + Novelty
    ax_costs    = fig.add_subplot(gs[0, 3])    # Kosten-Uhr
    ax_urgency  = fig.add_subplot(gs[1, :3])   # Urgency + Interval
    ax_comps    = fig.add_subplot(gs[1, 3])    # Urgency-Komponenten
    ax_calls    = fig.add_subplot(gs[2, :3])   # Call-Zeitpunkte
    ax_stats    = fig.add_subplot(gs[2, 3])
    ax_stats.axis('off')
    ax_costs.axis('off')

    print(f"Simuliere {N_STEPS} Steps...\n")

    for step in range(N_STEPS):
        state = agent.get_state()
        called = controller.update(
            fe=state["fe"],
            novelty=state["novelty"],
            goal_reached=state["goal_reached"],
            force_call=state["force_call"],
        )

        if called:
            reason = "force" if state["force_call"] else \
                "goal"  if state["goal_reached"] else "adaptive"
            # print nur wichtige Events
            if reason in ("force", "goal") or step % 50 == 0:
                print(f"  Step {step:4d}: Gemini aufgerufen "
                      f"[{reason}]  "
                      f"FE={state['fe']:.3f}  "
                      f"Nov={state['novelty']:.2f}")

        # Visualisierung alle 20 Steps
        if step % 20 == 0 or step == N_STEPS - 1:
            h       = controller.history
            steps_x = list(range(len(h["urgency"])))

            # Hintergrund-Phasen
            for ax in [ax_fe, ax_urgency, ax_calls]:
                ax.clear()
                for p_start, p_end, p_label, p_color in phase_labels:
                    if p_start <= step:
                        end = min(p_end, step+1)
                        ax.axvspan(p_start, end, alpha=0.15,
                                   color=p_color, zorder=0)

            # ── FE + Novelty ───────────────────────────
            ax_fe.plot(steps_x, h["fe_ema"],
                       color='tomato', linewidth=2,
                       label='Free Energy (EMA)', zorder=3)
            ax_fe.plot(steps_x, h["novelty"],
                       color='mediumpurple', linewidth=1.3,
                       alpha=0.7, label='Novelty', zorder=2)
            ax_fe.axhline(controller.fe_threshold, color='tomato',
                          linestyle='--', linewidth=1,
                          alpha=0.6, label=f'FE thresh={controller.fe_threshold}')
            ax_fe.axhline(controller.fe_low, color='seagreen',
                          linestyle='--', linewidth=1,
                          alpha=0.6, label=f'FE low={controller.fe_low}')

            # Phasen-Labels
            for p_start, p_end, p_label, _ in phase_labels:
                mid = (p_start + min(p_end, step+1)) / 2
                if mid <= step:
                    ax_fe.text(mid, 0.48, p_label, ha='center',
                               fontsize=6.5, color='white', alpha=0.7)

            # Call-Zeitpunkte als vertikale Linien
            call_steps = [i for i, c in enumerate(h["called"]) if c > 0]
            for cs in call_steps:
                ax_fe.axvline(cs, color='gold', linewidth=0.8,
                              alpha=0.5, zorder=1)

            ax_fe.set_title('Free Energy + Novelty  |  Gold = Gemini aufgerufen',
                            fontsize=9)
            ax_fe.set_ylabel('Wert')
            ax_fe.set_ylim(-0.02, 0.55)
            ax_fe.legend(fontsize=7, loc='upper right', ncol=2)
            ax_fe.set_facecolor('#0d0d0d')
            ax_fe.tick_params(colors='white')

            # ── Urgency + Interval ─────────────────────
            ax_urgency.plot(steps_x, h["urgency"],
                            color='orange', linewidth=2,
                            label='Urgency', zorder=3)
            ax2 = ax_urgency.twinx()
            ax2.plot(steps_x, h["interval"],
                     color='steelblue', linewidth=1.5,
                     linestyle='--', alpha=0.8,
                     label=f'Interval (Steps)')
            ax2.set_ylabel('Call-Interval (Steps)', color='steelblue',
                           fontsize=8)
            ax2.tick_params(axis='y', colors='steelblue')
            ax2.set_ylim(0, controller.max_interval * 1.1)

            for cs in call_steps:
                ax_urgency.axvline(cs, color='gold', linewidth=0.8,
                                   alpha=0.5, zorder=1)

            ax_urgency.set_title('Urgency (orange) + adaptives Call-Interval (blau)',
                                 fontsize=9)
            ax_urgency.set_ylabel('Urgency (0-1)')
            ax_urgency.set_ylim(-0.05, 1.1)
            ax_urgency.legend(fontsize=7, loc='upper left')
            ax_urgency.set_facecolor('#0d0d0d')
            ax_urgency.tick_params(colors='white')

            # ── Call-Zeitpunkte ─────────────────────────
            # Zeigt als Eventplot wann Gemini aufgerufen wurde
            called_arr = np.array(h["called"])
            ax_calls.fill_between(steps_x, 0, called_arr,
                                  color='gold', alpha=0.8, step='mid')
            ax_calls.set_ylim(-0.1, 1.5)
            ax_calls.set_title('Gemini Call-Zeitpunkte  (Gold = Call)',
                               fontsize=9)
            ax_calls.set_xlabel('Step')
            ax_calls.set_yticks([])
            ax_calls.set_facecolor('#0d0d0d')
            ax_calls.tick_params(colors='white')

            # Kumulativer Call-Zähler als Linie
            ax_call2 = ax_calls.twinx()
            cumcalls = np.cumsum(called_arr)
            ax_call2.plot(steps_x, cumcalls, color='white',
                          linewidth=1.5, label='Kumulativ')
            ax_call2.set_ylabel('Kumulierte Calls', color='white', fontsize=8)
            ax_call2.tick_params(axis='y', colors='white')
            ax_call2.legend(fontsize=7, loc='upper left')

            # ── Urgency-Komponenten ────────────────────
            ax_comps.clear()
            if h["u_fe"]:
                n_show = min(50, len(h["u_fe"]))
                xs     = steps_x[-n_show:]
                ax_comps.stackplot(
                    xs,
                    [h["u_fe"][-n_show:],
                     h["u_novelty"][-n_show:],
                     h["u_timeout"][-n_show:]],
                    labels=['u_FE', 'u_Novelty', 'u_Timeout'],
                    colors=['tomato', 'mediumpurple', 'steelblue'],
                    alpha=0.75
                )
            ax_comps.set_title('Urgency-Komponenten\n(letzte 50 Steps)', fontsize=8)
            ax_comps.set_ylim(0, 2)
            ax_comps.legend(fontsize=6, loc='upper left')
            ax_comps.set_facecolor('#0d0d0d')
            ax_comps.tick_params(colors='white')

            # ── Kosten-Uhr ─────────────────────────────
            ax_costs.clear()
            ax_costs.axis('off')
            cost = controller.estimated_hourly_cost
            rate = controller.call_rate

            # Gauge (Halbkreis)
            theta  = np.linspace(np.pi, 0, 100)
            r_out, r_in = 0.9, 0.6
            # Hintergrund
            ax_costs.fill_between(
                np.cos(theta)*r_out, np.sin(theta)*r_out,
                np.cos(theta)*r_in, np.sin(theta)*r_in,
                color='#1a1a1a', alpha=0.8
            )
            # Füllung bis aktueller Rate
            fill_end = max(0.01, 1 - rate)
            theta_f  = np.linspace(np.pi, np.pi * fill_end, 100)
            color_g  = 'seagreen' if rate < 0.1 else \
                'gold'     if rate < 0.3 else 'tomato'
            ax_costs.fill_between(
                np.cos(theta_f)*r_out, np.sin(theta_f)*r_out,
                np.cos(theta_f)*r_in,  np.sin(theta_f)*r_in,
                color=color_g, alpha=0.9
            )
            ax_costs.set_xlim(-1.1, 1.1)
            ax_costs.set_ylim(-0.3, 1.1)
            ax_costs.set_aspect('equal')
            ax_costs.text(0, 0.15,  f"{rate*100:.1f}%",
                          ha='center', fontsize=18, fontweight='bold',
                          color=color_g)
            ax_costs.text(0, -0.05, "Call-Rate",
                          ha='center', fontsize=9, color='white')
            ax_costs.text(0, -0.2,
                          f"~{cost['calls_per_hour']:,}/h\n"
                          f"${cost['cost_per_hour_usd']:.2f}/h\n"
                          f"{cost['savings_vs_always']*100:.0f}% Ersparnis",
                          ha='center', fontsize=8, color='lightgray')
            ax_costs.set_facecolor('#0d0d0d')
            ax_costs.set_title('API-Kosten Schätzung\n(Gemini 2.5 Flash)',
                               fontsize=8, color='white')

            # ── Statistiken ────────────────────────────
            ax_stats.clear()
            ax_stats.axis('off')
            summ = controller.summary()
            urg_now  = h["urgency"][-1]  if h["urgency"]  else 0
            int_now  = h["interval"][-1] if h["interval"] else 0

            lines = [
                "── Adaptive Controller ──",
                f"Steps:    {summ['total_steps']}",
                f"Calls:    {summ['total_calls']}",
                f"Rate:     {summ['call_rate']}",
                "",
                "── Aktuell ──────────────",
                f"FE EMA:   {controller.fe_ema:.4f}",
                f"Urgency:  {urg_now:.4f}",
                f"Interval: {int_now:.0f} Steps",
                "",
                "── Schwellwerte ─────────",
                f"FE thresh:{controller.fe_threshold}",
                f"FE low:   {controller.fe_low}",
                f"Nov.thr.: {controller.novelty_thresh}",
                f"Timeout:  {controller.timeout_steps}",
                "",
                "── Kosten (geschätzt) ───",
                f"Calls/h:  {controller.estimated_hourly_cost['calls_per_hour']:,}",
                f"$/h:      {controller.estimated_hourly_cost['cost_per_hour_usd']:.3f}",
                f"Ersparnis:{controller.estimated_hourly_cost['savings_vs_always']*100:.0f}%",
                "",
                "── Phasen ───────────────",
                "1: Lernen    (FE hoch)",
                "2: Stabil    (FE niedrig)",
                "3: Neu!      (Novelty↑)",
                "4: Stabil",
                "5: Befehl    (force)",
            ]
            ax_stats.text(
                0.03, 0.98, "\n".join(lines),
                transform=ax_stats.transAxes,
                fontsize=7, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            )

            plt.pause(0.02)

    # ── Finale Ausgabe ────────────────────────────────────
    summ = controller.summary()
    cost = controller.estimated_hourly_cost
    print("\nSimulation abgeschlossen!")
    print(f"  Steps gesamt:    {summ['total_steps']}")
    print(f"  Gemini Calls:    {summ['total_calls']}")
    print(f"  Call-Rate:       {summ['call_rate']}")
    print()
    print("Kosten-Abschätzung (Gemini 2.5 Flash):")
    print(f"  Calls/Stunde:    {cost['calls_per_hour']:,}")
    print(f"  Kosten/Stunde:   ${cost['cost_per_hour_usd']:.3f} USD")
    print(f"  Ersparnis ggü.   immer fragen: "
          f"{cost['savings_vs_always']*100:.0f}%")
    print()
    print("Wann wurde Gemini aufgerufen:")
    call_steps = [i for i, c in enumerate(controller.history["called"]) if c > 0]
    print(f"  {call_steps[:20]}{'...' if len(call_steps)>20 else ''}")
    print()
    print("Naechste Schritte:")
    print("  B15 – Reward-Kombination: Intrinsic + Gemini")
    print("  B16 – Vollintegration aller Bausteine")

    plt.show()


if __name__ == "__main__":
    run_demo()
