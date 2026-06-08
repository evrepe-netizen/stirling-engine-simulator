import base64
import io
import os
import subprocess
from PIL import Image

from physics_v9_4 import PROTOTYPE, to_si, build_geometry
from animation_v9_4 import build_engine_animation

params = dict(PROTOTYPE)
params_si = to_si(params)
geom = build_geometry(params_si)

print("Generating animation GIF... this may take a minute")

gif_b64 = build_engine_animation(geom, params)
gif_bytes = base64.b64decode(gif_b64)

out_dir = "engine_video"
os.makedirs(out_dir, exist_ok=True)

gif_path = os.path.join(out_dir, "stirling_engine_animation.gif")
mp4_path = os.path.join(out_dir, "stirling_engine_animation.mp4")

with open(gif_path, "wb") as f:
    f.write(gif_bytes)

print(f"Saved GIF: {gif_path}")

# Convert GIF to MP4 using ffmpeg
# yuv420p makes it compatible with Windows / PowerPoint
cmd = [
    "ffmpeg",
    "-y",
    "-i", gif_path,
    "-movflags", "+faststart",
    "-pix_fmt", "yuv420p",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    mp4_path
]

print("Converting to MP4...")
try:
    subprocess.run(cmd, check=True)
    print(f"Saved MP4: {mp4_path}")
    print("Done.")
    print("Open folder:")
    print(os.path.abspath(out_dir))
except FileNotFoundError:
    print("ERROR: ffmpeg is not installed.")
    print("Install it with:")
    print("brew install ffmpeg")
except subprocess.CalledProcessError as e:
    print("ERROR: ffmpeg conversion failed.")
    print(e)
