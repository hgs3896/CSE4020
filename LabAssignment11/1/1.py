import glfw
from OpenGL.GL import *
import numpy as np

RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

P = [
    np.array([100., 200.]),
    np.array([200., 300.]),
    np.array([300., 300.]),
    np.array([400., 200.]),
]
gEditingPoint = ''


def Lerp(t, q0, q1):
    return (1. - t) * q0 + t * q1


def draw_curve(t, plist):
    P0, P1, P2, P3 = plist
    Q0, Q1, Q2 = Lerp(t, P0, P1), Lerp(t, P1, P2), Lerp(t, P2, P3)
    R0, R1 = Lerp(t, Q0, Q1), Lerp(t, Q1, Q2)
    return Lerp(t, R0, R1)


def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, RESOLUTION[0], 0, RESOLUTION[1], -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glColor3ub(255, 255, 255)
    glBegin(GL_LINE_STRIP)
    for t in np.arange(0, 1, .01):
        p = draw_curve(t, [P[0], P[1], P[2], P[3]])
        glVertex2fv(p)
    glEnd()

    glColor3ub(0, 255, 0)
    glBegin(GL_LINE_LOOP)
    for p in P:
        glVertex2fv(p)
    glEnd()

    glPointSize(20.)
    glBegin(GL_POINTS)
    for p in P:
        glVertex2fv(p)
    glEnd()


def button_callback(window, button, action, mod):
    global P, gEditingPoint
    if button == glfw.MOUSE_BUTTON_LEFT:
        x, y = glfw.get_cursor_pos(window)
        y = RESOLUTION[1] - y
        c = np.array([x, y])
        if action == glfw.PRESS:
            for idx, p in enumerate(P):
                if np.all(np.abs(c-p) < 10):
                    gEditingPoint = f'p{idx}'
                    break
        elif action == glfw.RELEASE:
            gEditingPoint = ''


def cursor_callback(window, xpos, ypos):
    global P, gEditingPoint
    ypos = RESOLUTION[1] - ypos
    if gEditingPoint.startswith('p'):
        idx = int(gEditingPoint[1:])
        P[idx][0] = xpos
        P[idx][1] = ypos


def main():
    if not glfw.init():
        return
    window = glfw.create_window(*RESOLUTION, STUDENT_ID, None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()