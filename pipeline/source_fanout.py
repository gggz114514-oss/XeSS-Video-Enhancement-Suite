"""Bounded source RGB fan-out for existing SR guide consumers.

One decoder stream feeds the preparer and existing guard/MFSR consumers. Each
guide receives the exact original bytes before optional source fusion.
"""
import os
import threading

from shm_ring import RingWriter


class SourceFanout:
    def __init__(self, source, frame_bytes, frames, rings):
        self.source = source
        self.frame_bytes = frame_bytes
        self.frames = frames
        self.rings = rings
        read_fd, write_fd = os.pipe()
        self.reader = os.fdopen(read_fd, "rb", buffering=0)
        self.writer = os.fdopen(write_fd, "wb", buffering=0)
        self.errors = []
        self.frames_read = 0
        self.thread = threading.Thread(target=self._run, name="source-rgb-fanout", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        writers = []
        try:
            writers = [RingWriter(ring.name, ring.slots, ring.slot_size) for ring in self.rings]
            for index in range(self.frames):
                payload = bytearray(self.frame_bytes)
                view = memoryview(payload)
                position = 0
                while position < self.frame_bytes:
                    count = self.source.readinto(view[position:])
                    if not count:
                        raise EOFError(f"source guide decoder ended at frame {index}")
                    position += count
                for writer in writers:
                    writer.write_parts(b"", [payload])
                position = 0
                while position < len(payload):
                    count = self.writer.write(view[position:])
                    if not count:
                        raise BrokenPipeError("source preparer pipe closed")
                    position += count
                self.frames_read += 1
        except BaseException as exc:
            self.errors.append(exc)
        finally:
            self.source.close()
            self.writer.close()
            for writer in writers:
                writer.close()

    def join(self):
        self.thread.join(timeout=35)
        if self.thread.is_alive():
            raise RuntimeError("source guide fan-out did not drain within its bounded timeout")
        if self.errors:
            raise RuntimeError(f"source guide fan-out failed: {self.errors[0]}") from self.errors[0]
