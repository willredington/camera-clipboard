"""LED and button, real (gpiozero) and fake. gpiozero is imported lazily so laptops don't need it."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Led:
    """MOSFET gate on a GPIO. gpiozero uses lgpio on Bookworm (not RPi.GPIO)."""

    def __init__(self, pin: int, active_high: bool = True):
        from gpiozero import LED
        self._led = LED(pin, active_high=active_high, initial_value=False)

    def on(self):
        self._led.on()

    def off(self):
        self._led.off()

    def close(self):
        self._led.close()


class FakeLed:
    def __init__(self):
        self.is_on = False
        self.history: list[bool] = []

    def on(self):
        self.is_on = True
        self.history.append(True)
        log.info("[fake LED] on")

    def off(self):
        self.is_on = False
        self.history.append(False)
        log.info("[fake LED] off")

    def close(self):
        pass


class Button:
    """Momentary push button between the pin and GND (internal pull-up)."""

    def __init__(self, pin: int, on_press):
        from gpiozero import Button as GzButton
        self._btn = GzButton(pin, pull_up=True, bounce_time=0.05)
        self._btn.when_pressed = on_press

    def close(self):
        self._btn.close()
