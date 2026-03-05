# B01 – MiniWorld Custom Environment

from miniworld.miniworld import MiniWorldEnv
from miniworld.entity import Box, Ball

import matplotlib
matplotlib.use('TkAgg')  # Erzwingt interaktives Fenster

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MyCustomWorld(MiniWorldEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault('obs_width', 16)   # Nur setzen wenn nicht bereits in kwargs
        kwargs.setdefault('obs_height', 16)
        kwargs.setdefault('max_episode_steps', 100)
        super().__init__(**kwargs)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=10, min_z=0, max_z=10)
        self.place_entity(Box(color='red'), pos=(2, 0, 2))
        self.dynamic_ball = self.place_entity(Ball(color='blue'), pos=(5, 0, 5))
        self.place_agent()

env = MyCustomWorld(obs_width=16, obs_height=16, render_mode='rgb_array')
obs, info = env.reset()

print("Setup erfolgreich! 16x16 Bild generiert.")
print(f"Beobachtungs-Shape: {obs.shape}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('MiniWorld – Live View', fontsize=14, fontweight='bold')

axes[0].set_title('Agenten-Kamera (16×16 px)', fontsize=10)
img_obs = axes[0].imshow(obs, interpolation='nearest')
axes[0].axis('off')

render_frame = env.render()
axes[1].set_title('3D-Umgebung (gerendert)', fontsize=10)
img_render = axes[1].imshow(render_frame, interpolation='bilinear')
axes[1].axis('off')

plt.tight_layout()

def update(frame):
    # 1. Eine zufällige Aktion ausführen (z.B. "dreh links", "geh vor")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # 2. Falls Episode beendet → Umgebung zurücksetzen
    if terminated or truncated:
        obs, info = env.reset()

    # 3. Beide Bilder im Fenster aktualisieren:
    # Linkes Bild: 16x16 Agenten-Kamera
    img_obs.set_data(obs)
    render_frame = env.render()
    # Rechtes Bild: 3D-Ansicht
    img_render.set_data(render_frame)
    # 4. Titel mit aktuellem Frame und Reward aktualisieren
    fig.suptitle(f'MiniWorld – Frame {frame} | Reward: {reward:.2f}', fontsize=14)
    return img_obs, img_render

ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=False, repeat=False)

plt.show()
env.close()
print("Fenster geschlossen.")
