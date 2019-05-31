import board
import digitalio
import time
import sys

led = digitalio.DigitalInOut(board.D13)
led.direction = digitalio.Direction.OUTPUT

DigIn1 = digitalio.DigitalInOut(board.D1)
DigIn2 = digitalio.DigitalInOut(board.D2)
DigIn3 = digitalio.DigitalInOut(board.D3)
DigIn4 = digitalio.DigitalInOut(board.D4)

DigIn1.switch_to_input(pull=digitalio.Pull.DOWN)
DigIn2.switch_to_input(pull=digitalio.Pull.DOWN)
DigIn3.switch_to_input(pull=digitalio.Pull.DOWN)
DigIn4.switch_to_input(pull=digitalio.Pull.DOWN)

while True:
    a = sys.stdin.read(1)
    print(("{:1d}"*4).format(DigIn1.value, \
                             DigIn2.value, \
                             DigIn3.value, \
                             DigIn4.value), end='')
    led.value = not led.value

