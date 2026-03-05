"""
B19OrchestratorModeMiniworld.py
================================
Startet B19 im MiniWorld-Modus.

Installation (einmalig):
    pip install gymnasium miniworld pyopengl

Konfiguration: ENV_NAME, N_STEPS, DISPLAY_EVERY unten anpassen.
"""

import sys
import os

# ─────────────────────────────────────────────
# KONFIGURATION – hier anpassen
# ─────────────────────────────────────────────

ENV_NAME      = "PredictionWorld-OneRoom-v0"
N_STEPS       = 0           # 0 = unbegrenzt (bis Fenster-X oder Ctrl+C)
DISPLAY_EVERY = 8
CHECKPOINT    = "checkpoints/pwn_checkpoint_*.pt"  # neuester wird automatisch gewählt


# ─────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────

def check_dependencies() -> bool:
    ok = True
    for pkg, import_name in [
        ("gymnasium", "gymnasium"),
        ("miniworld",  "miniworld"),
        ("pyglet",     "pyglet"),
    ]:
        try:
            m   = __import__(import_name)
            ver = getattr(m, "__version__", "?")
            print(f"  {pkg:12s} ✓  ({ver})")
        except ImportError:
            print(f"  {pkg:12s} ✗  → pip install {pkg}")
            ok = False
    return ok


def check_env(env_name: str) -> bool:
    try:
        import gymnasium as gym
        import miniworld  # noqa: F401
        # Custom Env registrieren (wird ggf. in B19Orchestrator nochmal aufgerufen)
        _register_pw_env(gym)
        env    = gym.make(env_name, render_mode="rgb_array", view="agent")
        obs, _ = env.reset()
        env.close()
        print(f"  {env_name}")
        print(f"    Obs-Shape: {obs.shape}")
        return True
    except Exception as e:
        print(f"  {env_name}  ✗  {e}")
        return False


def _register_pw_env(gym):
    """Registriert PredictionWorld-OneRoom-v0."""
    env_id = "PredictionWorld-OneRoom-v0"
    if env_id in gym.envs.registry:
        return
    import numpy as _np
    from miniworld.envs.oneroom import OneRoom
    from miniworld.entity import Box, Ball, COLORS, COLOR_NAMES
    if "orange" not in COLORS:
        COLORS["orange"] = _np.array([1.0, 0.5, 0.0])
    if "white" not in COLORS:
        COLORS["white"] = _np.array([1.0, 1.0, 1.0])
    for c in ("orange", "white"):
        if c not in COLOR_NAMES:
            COLOR_NAMES.append(c)

    class PredictionWorldRoom(OneRoom):
        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            self.box = self.place_entity(Box(color="red"))
            self.place_entity(Box(color="yellow"))
            self.place_entity(Box(color="white"))
            self.place_entity(Box(color="orange"))
            self.place_entity(Ball(color="green"))
            self.place_entity(Ball(color="blue"))
            self.place_agent()

    gym.register(id=env_id,
                 entry_point=lambda **kw: PredictionWorldRoom(**kw),
                 max_episode_steps=300)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("B19 – MiniWorld Modus")
    print("=" * 45)
    print()

    print("Abhängigkeiten:")
    deps_ok = check_dependencies()
    print()

    if not deps_ok:
        print("  pip install gymnasium miniworld pyopengl")
        print()
        print("Starte trotzdem mit Mock-Fallback? (j/n): ", end="", flush=True)
        try:
            if input().strip().lower() not in ("j", "ja", "y", "yes"):
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        print()

    if deps_ok:
        print("Environment-Test:")
        if not check_env(ENV_NAME):
            print("\nVerfügbare Environments:")
            for e in ["MiniWorld-OneRoom-v0", "MiniWorld-Hallway-v0",
                      "MiniWorld-FourRooms-v0", "MiniWorld-TMaze-v0"]:
                print(f"  {e}")
            print("\nENV_NAME in diesem Script anpassen.")
            sys.exit(1)
        print()

    # ── B19 laden und direkt aufrufen ─────────
    here     = os.path.dirname(os.path.abspath(__file__))
    b19_path = os.path.join(here, "B19Orchestrator.py")
    if not os.path.exists(b19_path):
        print(f"Fehler: B19Orchestrator.py nicht gefunden in:\n  {here}")
        sys.exit(1)

    print("Konfiguration:")
    print(f"  Environment:  {ENV_NAME}")
    print(f"  Steps:        {N_STEPS}")
    print(f"  Display alle: {DISPLAY_EVERY} Steps")
    print()
    print("Starte B19Orchestrator...")
    print("=" * 45)
    print()

    import importlib.util
    spec   = importlib.util.spec_from_file_location("B19Orchestrator", b19_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # __main__ läuft bei importlib nicht → Orchestrator direkt aufrufen
    config = {
        "mode":                "miniworld",
        "n_steps":             N_STEPS,
        "miniworld_env":       ENV_NAME,
        "update_display":      DISPLAY_EVERY,
        "checkpoint":          CHECKPOINT,
        "scene_switch":        40,
        "buffer_size":         1000,
        "batch_size":          16,
        "lr":                  1e-3,
        "beta_max":            0.05,
        "beta_warmup":         200,
        "min_gemini_interval": 8,
        "max_gemini_interval": 80,
    }

    orch = module.Orchestrator(config)
    try:
        orch.setup()
        orch.run()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Abgebrochen.")
        orch._print_summary()
    finally:
        orch.close()


if __name__ == "__main__":
    main()
