import gymnasium as gym
from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, Ball

class MyCustomWorld(MiniWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault('obs_width', 16)   # Nur setzen wenn nicht bereits in kwargs
        kwargs.setdefault('obs_height', 16)
        kwargs.setdefault('max_episode_steps', 100)
        super().__init__(**kwargs)

    def _gen_world(self):
        # Einen einfachen Raum erstellen
        self.add_rect_room(min_x=0, max_x=10, min_z=0, max_z=10)

        # Ein statisches Objekt (Würfel)
        self.place_entity(Box(color='red'), pos=(2, 0, 2))

        # Ein dynamisches Objekt (Kugel)
        self.dynamic_ball = self.place_entity(Ball(color='blue'), pos=(5, 0, 5))

        # Den Roboter (die Kamera) platzieren
        self.place_agent()

# Registriere und erstelle die Welt mit 16x16 Pixeln
env = MyCustomWorld(obs_width=16, obs_height=16, render_mode='rgb_array')
obs, info = env.reset()

print("Setup erfolgreich! 16x16 Bild generiert.")
