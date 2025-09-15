from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from eelbrain import plot
import mne


def GlassBrainVideo(tmin, tmax, dt, nd, vname):
    mne.set_log_level("warning")
    
    out_dir = Path("videos/cache/")
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tmin = max(tmin, float(nd.time.tmin))      
    tmax = min(float(nd.time.tmax),tmax)     
    dt   = min(dt, 0.1)                               
    times = np.arange(tmin, tmax + 1e-12, dt)

    dpi = 100           # image quality
    fps = 5             # video frame rate
    video_path = f"videos/{vname}.mp4"

    stitched_paths = []

    for k, t in enumerate(times, 1):
        p = plot.Butterfly(nd.norm('space'), color='k', title=vname)
        p.add_vline(t, color='tab:blue', lw=2)
        f1 = out_dir / f"butter_{k:04d}.png"
        p.figure.savefig(f1, dpi=dpi, bbox_inches="tight")
        plt.close(p.figure) 

        gb = plot.GlassBrain(nd.sub(time=t), title=f" {t:.2f}s")
        f2 = out_dir / f"glass_{k:04d}.png"
        gb.figure.savefig(f2, dpi=dpi, bbox_inches="tight")
        plt.close(gb.figure)

        a = Image.open(f1); b = Image.open(f2)
        H = max(a.height, b.height)
        def resize_keep_h(img, H):
            W = int(round(img.width * (H / img.height)))
            return img.resize((W, H), Image.LANCZOS)
        a = resize_keep_h(a, H); b = resize_keep_h(b, H)
        W = a.width + b.width
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))

        frame_path = frames_dir / f"frame_{k:04d}.png"
        canvas.save(frame_path)
        stitched_paths.append(frame_path)

    frames = [imageio.imread(p) for p in stitched_paths]
    imageio.mimsave(video_path, frames, fps=fps, quality=8)
    print(f"Saved video to: {video_path}")


    #-------------
    """
    from Beyond import GlassBrainVideo
    GlassBrainVideo(0.1, 0.8, 0.1, anim, "v4")

    from IPython.display import Video
    video_path="videos/v4.mp4"
    Video(video_path, embed=True, width=1200)
    """
