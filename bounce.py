import sys
import time
import math
from pyglet.gl import *
from pyglet.window import key
from euclid3 import *
import random

window = pyglet.window.Window(512, 512)

batch = pyglet.graphics.Batch()
cleaned = [[False] * 512 for _ in range(512)]

RADIUS = 20
quadric = gluNewQuadric()
ballPos = Vector2(256, 256)
ballVel = Vector2(200, 145)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT)
    batch.draw()
    glPushMatrix()
    glTranslatef(ballPos[0], ballPos[1], 0)
    glColor3f(1, 0, 0)
    gluDisk(quadric, 0, RADIUS, 32, 1)
    glPopMatrix()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        ballVel[0], ballVel[1] = (random.randint(-512, 512),
                                  random.randint(-512, 512))


def checkForBounce():
    if ballPos[0] + RADIUS > 512.0:
        ballVel[0] = -ballVel[0]
        ballPos[0] = 512.0 - (ballPos[0] - 512.0)
    elif ballPos[0] - RADIUS < 0.0:
        ballVel[0] = -ballVel[0]
        ballPos[0] = -ballPos[0]
    if ballPos[1] + RADIUS > 512.0:
        ballVel[1] = -ballVel[1]
        ballPos[1] = 512.0 - (ballPos[1] - 512.0)
    elif ballPos[1] - RADIUS < 0.0:
        ballVel[1] = -ballVel[1]
        ballPos[1] = -ballPos[1]


def clean(pos):
    roomba_x, roomba_y = map(int, pos)
    reward = 0
    points = []
    for x in range(roomba_x - RADIUS, roomba_x + RADIUS):
        for y in range(roomba_y - RADIUS, roomba_y + RADIUS):
            if (0 <= x < 512 and 0 <= y < 512 and
                    (x - roomba_x) ** 2 + (y - roomba_y) ** 2 <= RADIUS ** 2):
                is_new = not cleaned[x][y]
                reward += is_new - (1 - is_new)
                if is_new:
                    points.extend([x, y])
                    cleaned[x][y] = True
    return reward, points


def update(dt):
    global ballPos, ballVel
    reward, points = clean(ballPos)
    nb = len(points) // 2
    colors = [255] * (3 * nb)
    batch.add(nb, pyglet.gl.GL_POINTS, None,
              ('v2i', points),
              ('c3B', colors))
    ballPos += ballVel * dt
    checkForBounce()


pyglet.clock.schedule_interval(update, 1/24.0)
pyglet.app.run()
