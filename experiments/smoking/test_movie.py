import tempfile
import unittest
from pathlib import Path
import av
import cv2
import numpy as np
from movie import StreamingMovie


class MovieTests(unittest.TestCase):
    def test_streamed_frames_are_readable_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'movie.mp4'
            movie=StreamingMovie(path,30)
            frame=np.zeros((48,64,3),dtype=np.uint8);frame[:,:,0]=255
            for _ in range(60):movie.write(frame)
            self.assertFalse(path.exists())
            self.assertTrue(movie.working_path.exists())
            movie.close();movie.close()
            self.assertFalse(movie.working_path.exists())
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                self.assertEqual(stream.codec_context.name,'h264')
                self.assertEqual(stream.codec_context.pix_fmt,'yuv420p')
            content=path.read_bytes()
            self.assertLess(content.index(b'moov'),content.index(b'mdat'))
            cap=cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),60)
            ok,decoded=cap.read();self.assertTrue(ok)
            self.assertGreater(decoded[:,:,2].mean(),200)
            cap.release()
