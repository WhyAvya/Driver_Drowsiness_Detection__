# src/alarm.py
# This module handles audio alerts when drowsiness is detected.
# It is intentionally kept independent of ML and vision logic.

import pygame
import os


class Alarm:
    """
    Alarm class responsible for playing and stopping
    a warning sound when the driver is drowsy.
    """

    def __init__(self, sound_path):
        """
        Initialize the alarm system.

        Args:
            sound_path (str): Path to the alarm .wav file
        """

        # Initialize pygame's audio mixer once
        pygame.mixer.init()

        # Validate that the alarm sound file exists
        if not os.path.exists(sound_path):
            raise FileNotFoundError(f"Alarm sound not found: {sound_path}")

        # Load the alarm sound into memory
        self.sound = pygame.mixer.Sound(sound_path)

        # Track whether the alarm is currently playing
        self.is_playing = False

    def start(self):
        """
        Start playing the alarm sound.
        This will loop continuously until stopped.
        """
        if not self.is_playing:
            # Play sound in infinite loop (-1)
            self.sound.play(-1)
            self.is_playing = True

    def stop(self):
        """
        Stop the alarm sound if it is currently playing.
        """
        if self.is_playing:
            self.sound.stop()
            self.is_playing = False
