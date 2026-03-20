"""
video_render.py
===============
Handles offscreen rendering and MP4 file writing.

Optimisations applied:
  1. Low FPS (VIDEO_FPS=20) — fewer renderer.render() calls.
  2. Low resolution (320×240) — 4× fewer pixels per frame.
  3. libx264 ultrafast preset — ~10× faster encoding, slightly larger files.
  4. Background encoder thread — writer.append_data() runs in a separate
     thread so the sim loop is never blocked by encoding.

Usage
-----
    recorder = VideoRecorder(n=4, physics_model=...)
    recorder.capture(step, robot_states)   # call every sim step
    recorder.close()                        # call after the loop
"""

import queue
import threading
from typing import List

import imageio
import mujoco

from sim_config import VIDEO_FPS, RENDER_WIDTH, RENDER_HEIGHT, RENDER_DIR


class VideoRecorder:
    """
    Renders each robot to a separate MP4 file in RENDER_DIR.

    Capturing (renderer.render) happens in the sim loop.
    Encoding  (writer.append_data) happens in a background thread.
    """

    def __init__(self, n: int, physics_model: mujoco.MjModel):
        self.n = n

        # Steps to skip between frames so the video plays at VIDEO_FPS
        self.render_interval = max(1, round(1.0 / (VIDEO_FPS * physics_model.opt.timestep)))
        self.actual_fps      = 1.0 / (self.render_interval * physics_model.opt.timestep)

        # One renderer, reused for every robot each frame
        self.renderer = mujoco.Renderer(physics_model, height=RENDER_HEIGHT, width=RENDER_WIDTH)

        self.camera = mujoco.MjvCamera()
        self.camera.azimuth   = 45
        self.camera.elevation = -25
        self.camera.distance  = 1.8
        self.camera.lookat[:] = [0.0, 0.0, 0.3]

        RENDER_DIR.mkdir(exist_ok=True)

        # ultrafast preset: ~10× faster encoding vs default "medium"
        # crf=28: slightly lower quality than default (23) to keep file size small
        self.writers = [
            imageio.get_writer(
                str(RENDER_DIR / f"r{i}.mp4"),
                fps=self.actual_fps,
                codec="libx264",
                output_params=["-preset", "ultrafast", "-crf", "28"],
            )
            for i in range(n)
        ]

        # Background thread: pulls (robot_index, frame) from the queue and encodes
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._encoder_worker, daemon=True)
        self._thread.start()

        print(f"VideoRecorder : {n} files → {RENDER_DIR}/")
        print(f"  {self.actual_fps:.0f} fps | {RENDER_WIDTH}×{RENDER_HEIGHT} | ultrafast | async encoding")

    def _encoder_worker(self):
        """Runs in background — encodes frames without blocking the sim loop."""
        while True:
            item = self._queue.get()
            if item is None:        # stop signal sent by close()
                break
            robot_index, frame = item
            self.writers[robot_index].append_data(frame)

    def capture(self, step: int, robot_states: List[mujoco.MjData]):
        """
        Render one frame per robot (every render_interval steps) and push it
        to the background encoder queue. Returns immediately — never blocks.
        """
        if step % self.render_interval != 0:
            return

        for i, state in enumerate(robot_states):
            self.renderer.update_scene(state, camera=self.camera)
            frame = self.renderer.render().copy()   # copy so the buffer is safe to hand off
            self._queue.put((i, frame))

    def close(self):
        """Wait for all frames to finish encoding, then close files."""
        self._queue.put(None)       # tell the worker to stop
        self._thread.join()         # wait until the queue is fully drained

        for writer in self.writers:
            writer.close()
        self.renderer.close()
        print(f"Videos saved to {RENDER_DIR}/")
