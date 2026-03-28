import os
import threading
import time

try:
    import pygame
except Exception:
    pygame = None


class Alarm:
    def __init__(self, sound_path: str = "assets/alarm.wav"):
        self.sound_path = sound_path
        self.active = False
        self._use_pygame = False

        if pygame is not None and os.path.exists(self.sound_path):
            try:
                pygame.mixer.init()
                self._use_pygame = True
            except Exception:
                self._use_pygame = False

    def start(self):
        if self.active:
            return

        self.active = True

        if self._use_pygame:
            try:
                pygame.mixer.music.load(self.sound_path)
                pygame.mixer.music.play(-1)
                return
            except Exception:
                self._use_pygame = False

        # Fallback beep in a background thread
        threading.Thread(target=self._beep_loop, daemon=True).start()

    def _beep_loop(self):
        try:
            import winsound
            for _ in range(6):
                if not self.active:
                    break
                winsound.Beep(1200, 180)
                time.sleep(0.10)
        except Exception:
            print("\a", end="", flush=True)

    def stop(self):
        self.active = False
        if self._use_pygame:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass