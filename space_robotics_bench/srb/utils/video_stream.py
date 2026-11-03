"""Streaming video recorder for long-running envs (`srb agent ros` missions).

`gymnasium.wrappers.RecordVideo` keeps every frame of an episode in memory and writes on episode
end; a lunar base mission is ~20 min of sim time at 25 Hz and never ends an episode, so it would
exhaust memory and write nothing. This wrapper renders one frame every `SRB_VIDEO_STRIDE` steps
and pipes it straight into ffmpeg as fragmented MP4, which stays playable even if the process is
killed during a teardown.

Environment:
    SRB_VIDEO_PATH     output .mp4 (default: <video_folder>/<name_prefix>.mp4)
    SRB_VIDEO_STRIDE   env steps per frame (default 5: 10 fps of sim time at the 50 Hz step)
    SRB_VIDEO_SPEEDUP  playback speed relative to sim time (default 4)
    SRB_VIDEO_FOLLOW   asset whose root the camera follows (e.g. `robot`); unset = the env's viewer as configured
    SRB_VIDEO_EYE      camera offset from the followed root, world axes (default "-4,-4,3")
    SRB_VIDEO_LOOKAT   look-at offset from the followed root (default "0,0,0.4")
    SRB_VIDEO_RES      "W,H" (default "960,540")

Next to the video, `<video>.times` holds one line per written frame: the sim time (the `/clock` the ROS interface
publishes, common_step_counter x step_dt) of that frame, so logs stamped on `/clock` can be overlaid frame-exactly.

The camera is IsaacLab's viewport camera controller, switched to `asset_root` tracking here rather than
through Hydra `env.viewer.*` overrides (which did not reach the controller in a live check).
"""

import os
import shutil
import subprocess
from pathlib import Path

import gymnasium


class StreamVideo(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, video_folder: str, name_prefix: str):
        super().__init__(env)
        self._path = Path(os.environ.get("SRB_VIDEO_PATH") or Path(video_folder) / f"{name_prefix}.mp4")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stride = int(os.environ.get("SRB_VIDEO_STRIDE", "5"))
        step_dt = float(env.unwrapped.step_dt)  # type: ignore
        self._fps = round(float(os.environ.get("SRB_VIDEO_SPEEDUP", "4")) / (self._stride * step_dt), 3)
        self._steps = 0
        self._step_dt = step_dt
        self._ffmpeg: subprocess.Popen | None = None
        self._times = None
        base = env.unwrapped
        base.cfg.viewer.resolution = tuple(int(v) for v in os.environ.get("SRB_VIDEO_RES", "960,540").split(","))  # type: ignore
        follow = os.environ.get("SRB_VIDEO_FOLLOW")
        if follow:
            eye = [float(v) for v in os.environ.get("SRB_VIDEO_EYE", "-4,-4,3").split(",")]
            lookat = [float(v) for v in os.environ.get("SRB_VIDEO_LOOKAT", "0,0,0.4").split(",")]
            vcc = base.viewport_camera_controller  # type: ignore
            vcc.update_view_to_asset_root(follow)
            vcc.update_view_location(eye=eye, lookat=lookat)

    def _write(self) -> None:
        frame = self.env.render()
        if frame is None or not frame.any():  # renderer warming up
            return
        if self._ffmpeg is None:
            h, w = frame.shape[:2]
            self._ffmpeg = subprocess.Popen(
                [shutil.which("ffmpeg") or "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(self._fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
                 str(self._path)],
                stdin=subprocess.PIPE,
            )
            print(f"[srb.video] streaming {w}x{h} @ {self._fps} fps (every {self._stride} steps) -> {self._path}", flush=True)
            self._times = open(f"{self._path}.times", "w")
        self._ffmpeg.stdin.write(frame.tobytes())  # type: ignore
        self._times.write(f"{self.env.unwrapped.common_step_counter * self._step_dt:.3f}\n")  # type: ignore
        self._times.flush()

    def step(self, action):
        out = self.env.step(action)
        if self._steps % self._stride == 0:
            self._write()
        self._steps += 1
        return out

    def close(self):
        if self._ffmpeg is not None:
            self._ffmpeg.stdin.close()  # type: ignore
            self._ffmpeg.wait()
            self._ffmpeg = None
        return super().close()
