# Taken from https://github.com/MishaLaskin/curl/blob/019a229eb049b9400e97f142f32dd47b4567ba8a/video.py#L6

import imageio
import os
import numpy as np
import cv2


class VideoRecorder(object):
    def __init__(self, log_dir, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = log_dir
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, camera_id=0):
        if self.enabled:
            """
            try:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except:
                frame = env.render(
                    mode='rgb_array',
                )
            """
            frame = env.render(mode='rgb_array')
            shape = (self.height, self.width)
            if frame.shape[0] == 3: #channels first rendering case
                frame = frame.transpose((1, 2, 0))
            frame = cv2.resize(frame, shape, interpolation=cv2.INTER_AREA)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)