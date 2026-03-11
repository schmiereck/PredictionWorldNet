"""
MiniWorldRegistry.py
================================
Zentrale Registrierung der MiniWorld-Umgebungen für das PredictionWorldNet.
Diese Datei beseitigt die doppelte Definition von _gen_world in mehreren Modulen.
"""

import numpy as np

def register_prediction_world_environments():
    """Registriert alle Varianten der PredictionWorld-Umgebungen in Gymnasium."""
    import gymnasium as gym
    
    env_id_room = "PredictionWorld-OneRoom-v0"
    env_id_empty = "PredictionWorld-Empty-v0"
    env_id_single = "PredictionWorld-Single-v0"
    
    if env_id_room in gym.envs.registry:
        return # Bereits registriert
        
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

    # 1. Haupt-Umgebung mit mehreren Objekten
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
        id=env_id_room,
        entry_point=lambda **kw: PredictionWorldRoom(**kw),
        max_episode_steps=300,
    )

    # 2. Leere Umgebung (für freie Exploration)
    class PredictionWorldEmpty(OneRoom):
        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            # Die Original-Umgebung braucht self.box für das step-Belohnungs-System,
            # aber wir überschreiben step() weiter unten in der Verwendung oft.
            self.box = self.place_entity(Box(color="red"))
            self.place_agent()
            from B16FullIntegration import CAM_HEIGHT
            self.agent.cam_height = CAM_HEIGHT
            
        def step(self, action):
            # Kein Zielobjekt → nur Basis-Step ohne near(self.box)-Check
            from miniworld.miniworld import MiniWorldEnv
            return MiniWorldEnv.step(self, action)

    gym.register(
        id=env_id_empty,
        entry_point=lambda **kw: PredictionWorldEmpty(**kw),
        max_episode_steps=300,
    )

    # 3. Umgebung mit einzelnem variablen Objekt (wird im Konstruktor gesetzt)
    class PredictionWorldSingle(OneRoom):
        def __init__(self, target_entity_cls=Box, target_color="red", **kwargs):
            self.target_cls = target_entity_cls
            self.target_color = target_color
            super().__init__(**kwargs)

        def _gen_world(self):
            self.add_rect_room(min_x=0, max_x=self.size,
                               min_z=0, max_z=self.size)
            ent = self.target_cls(color=self.target_color)
            self.box = self.place_entity(ent)   # OneRoom.step() erwartet self.box
            self.place_agent()
            from B16FullIntegration import CAM_HEIGHT
            self.agent.cam_height = CAM_HEIGHT

    gym.register(
        id=env_id_single,
        entry_point=lambda **kw: PredictionWorldSingle(**kw),
        max_episode_steps=300,
    )

def get_entity_color_name(ent) -> str | None:
    """
    Gibt den Farbnamen einer MiniWorld-Entity als String (z.B. 'red') zurück.
    Wenn keine eindeutige Farbe erkannt wird, wird None zurückgegeben.
    - Box: ent.color ist in der Regel ein String.
    - Ball: Besitzt oft kein .color Attribut; extrahiert die Farbe stattdessen aus dem ObjMesh-Namen.
    """
    KNOWN_COLORS = ("red", "green", "blue", "yellow", "orange", "white", "grey", "purple", "black")
    
    # 1. Normale Entities (.color Attribut vorhanden und ist ein String)
    if hasattr(ent, 'color') and isinstance(ent.color, str):
        col = ent.color.lower()
        if col in KNOWN_COLORS:
            return col
        return col

    # 2. Besondere Entities wie Ball, bei denen die Farbe im Dateinamen des Meshes steckt (z.B. "ball_green.obj")
    if hasattr(ent, 'mesh') and ent.mesh is not None:
        try:
            import os, re
            from miniworld.objmesh import ObjMesh
            for k, v in ObjMesh.cache.items():
                if v is ent.mesh:
                    base = os.path.basename(k).lower()
                    m = re.match(r'.*_([a-z]+)\.obj$', base)
                    if m:
                        return m.group(1)
                    break
        except Exception:
            pass
            
    return None

def get_entity_type_name(ent) -> str:
    """Gibt den Basis-Namen (Typ) einer Entity in Kleinbuchstaben zurück (z.B. 'box', 'ball')."""
    return type(ent).__name__.lower()
