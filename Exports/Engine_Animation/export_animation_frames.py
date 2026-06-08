import base64
import io
import os
from PIL import Image

from physics_v9_4 import PROTOTYPE, to_si, build_geometry
from animation_v9_4 import build_engine_animation

params = dict(PROTOTYPE)
params_si = to_si(params)
geom = build_geometry(params_si)

print("Generating animation... this may take a minute")

gif_b64 = build_engine_animation(geom, params)
gif_bytes = base64.b64decode(gif_b64)
gif = Image.open(io.BytesIO(gif_bytes))

out_dir = "engine_phase_frames"
os.makedirs(out_dir, exist_ok=True)

frame_indices = [
    0,
    gif.n_frames // 4,
    gif.n_frames // 2,
    (3 * gif.n_frames) // 4,
]

names = [
    "phase_1_0deg.png",
    "phase_2_90deg.png",
    "phase_3_180deg.png",
    "phase_4_270deg.png",
]

for idx, name in zip(frame_indices, names):
    gif.seek(idx)
    frame = gif.convert("RGBA")
    path = os.path.join(out_dir, name)
    frame.save(path)
    print(f"Saved {path}")

print("Done.")
print("Frames exported to:", os.path.abspath(out_dir))
