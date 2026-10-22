"""Browser-compatible H.264 encoding with bounded frame memory."""
from pathlib import Path


class StreamingMovie:
    def __init__(self, path, fps):
        self.path, self.fps = Path(path), fps
        self.working_path = self.path.with_name(self.path.stem + '.recording.mp4')
        self.writer = None
        self.frame_count = 0
        self.closed = False

    def write(self, rgb):
        import imageio.v2 as imageio
        if self.closed:
            raise RuntimeError('Cannot append frames to a closed movie')
        if self.writer is None:
            self.writer = imageio.get_writer(
                self.working_path, format='FFMPEG', mode='I', fps=self.fps,
                codec='libx264', pixelformat='yuv420p', macro_block_size=2,
                ffmpeg_params=['-movflags', '+faststart'],
            )
        self.writer.append_data(rgb)
        self.frame_count += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.working_path.replace(self.path)
