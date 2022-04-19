import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

RESOLUTION = (800, 800)
STUDENT_ID = "2017030473"
WINDOW_TITLE = "Basic OpenGL Viewer"

TARGET = np.array([0., 0., 0.])
UP = np.array([0., 1., 0.])

gTime = 0

gProjectionMode = 0
gCursorPos = np.zeros(2)
gScrollOffset = np.zeros(2)

orbit, orbit_s, orbit_e = np.zeros(2), np.zeros(2), np.zeros(2)
pan, pan_s, pan_e, pan_delta = np.zeros(
    2), np.zeros(2), np.zeros(2), np.zeros(2)
zoom = 50

MOUSE_BUTTON_STATE = {
    "NONE": 0,
    "TRIGGER_START": 1,
    "TRIGGERING": 2,
    "TRIGGER_END": 3,
}
MOUSE_BUTTON_LEFT = MOUSE_BUTTON_STATE["NONE"]
MOUSE_BUTTON_RIGHT = MOUSE_BUTTON_STATE["NONE"]


def get_next_state(s):
    if s == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        return MOUSE_BUTTON_STATE["TRIGGERING"]
    elif s == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        return MOUSE_BUTTON_STATE["NONE"]
    return s


def get_next_state_from_button_callback(s, action_type):
    if action_type == glfw.PRESS:
        if s == MOUSE_BUTTON_STATE["NONE"]:
            return MOUSE_BUTTON_STATE["TRIGGER_START"]
    elif action_type == glfw.RELEASE:
        if s == MOUSE_BUTTON_STATE["TRIGGERING"]:
            return MOUSE_BUTTON_STATE["TRIGGER_END"]
    return s


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_V:
        if action == glfw.PRESS:
            global gProjectionMode
            gProjectionMode = 1 - gProjectionMode


def cursor_callback(window, xpos, ypos):
    global gCursorPos, old_orbit, orbit, old_pan, pan, pan_delta
    global MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT

    gCursorPos = np.array([xpos, ypos]) / RESOLUTION

    global orbit_s, orbit_e
    global pan_s, pan_e
    if MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        old_orbit = orbit
        orbit_s = gCursorPos
    elif MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGERING"]:
        orbit_e = gCursorPos
        orbit = old_orbit + (orbit_e - orbit_s)
    elif MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        orbit_e = gCursorPos

    if MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        old_pan = pan
        pan_s = gCursorPos
        pan_delta = np.zeros(2)
    elif MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGERING"]:
        pan_e = gCursorPos
        pan_delta = np.array([1, -1]) * (pan_e - pan_s)
        pan = old_pan + pan_delta
    elif MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        pan_e = gCursorPos
        pan_delta = np.zeros(2)

    MOUSE_BUTTON_LEFT = get_next_state(MOUSE_BUTTON_LEFT)
    MOUSE_BUTTON_RIGHT = get_next_state(MOUSE_BUTTON_RIGHT)


def button_callback(window, button, action, mod):
    if button == glfw.MOUSE_BUTTON_LEFT:
        global MOUSE_BUTTON_LEFT
        MOUSE_BUTTON_LEFT = get_next_state_from_button_callback(
            MOUSE_BUTTON_LEFT, action)

    if button == glfw.MOUSE_BUTTON_RIGHT:
        global MOUSE_BUTTON_RIGHT
        MOUSE_BUTTON_RIGHT = get_next_state_from_button_callback(
            MOUSE_BUTTON_RIGHT, action)


def scroll_callback(window, xoffset, yoffset):
    global gScrollOffset
    gScrollOffset = np.array([xoffset, yoffset]) / RESOLUTION

    global zoom
    if np.abs(gScrollOffset[1]) > 0.001:
        zoom -= gScrollOffset[1] * 100
        zoom = np.clip(zoom, 1, 100)


def hsv2rgb(hsvs):
    hsvs_ = hsvs.reshape(-1, 3)
    K = np.array([1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0])
    p = np.abs(np.modf(hsvs_[:, [0]] + K[:3].reshape(-1, 3))[0] * 6.0 - K[3])
    Kxxx = K[[0, 0, 0]]
    return hsvs_[:, [2]] * (Kxxx * (1 - hsvs_[:, [1]]) + np.clip(p - Kxxx, 0.0, 1.0) * (hsvs_[:, [1]]))

# Draw functions
def drawAxis():
    # x, y, z coordinates
    glBegin(GL_LINE_STRIP)
    glColor3ub(255, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(100, 0, 0)
    glEnd()

    glBegin(GL_LINE_STRIP)
    glColor3ub(0, 255, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 100, 0)
    glEnd()

    glBegin(GL_LINE_STRIP)
    glColor3ub(0, 0, 255)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 100)
    glEnd()


def drawGrids():
    max_length = 100
    for line in range(-max_length, max_length):
        glColor3f(.5, .5, .5)
        # x line
        glBegin(GL_LINE_STRIP)
        glVertex3iv(np.array([line, 0, -max_length]))
        glVertex3iv(np.array([line, 0, max_length]))
        glEnd()

        # z line
        glBegin(GL_LINE_STRIP)
        glVertex3iv(np.array([-max_length, 0, line]))
        glVertex3iv(np.array([max_length, 0, line]))
        glEnd()


def drawUnitCube():
    glBegin(GL_QUADS)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)

    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)

    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)

    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)

    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)

    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glEnd()


def drawCubeArray(initialPosX=0, initialPosY=0, initialPosZ=0):
    glColor3ub(255, 255, 255)
    for i in range(5):
        for j in range(5):
            for k in range(5):
                glPushMatrix()
                glColor3fv(hsv2rgb(
                    np.array([
                        1 - ((i + 1) * 5 ** 2 + (j + 1) *
                             5 ** 1 + (k + 1)) / (5 ** 3),
                        0.6,
                        1,
                    ])
                ))
                glTranslatef(initialPosX, initialPosY, initialPosZ)
                glTranslatef(2*(i-2), 2*(j-2), 3*(k-2))
                glScalef(.5, .5, .5)
                drawUnitCube()
                glPopMatrix()


def drawOrbitingPlannet(d=3):
    glPushMatrix()
    glTranslatef(d * np.sin(gTime), 0, d * np.cos(gTime))
    drawPlannet(0.5)
    glPopMatrix()


def drawPlannet(R=1, M=30, N=30):
    deltaM = np.pi / M
    deltaN = (2 * np.pi) / N

    def getCoord(i, j):
        r = R * np.sin(i*deltaM)
        return (r * np.cos(j*deltaN), r * np.sin(j*deltaN), R * np.cos(i*deltaM))
    planet = []
    for m in range(M):
        for n in range(N):
            planet.append(getCoord(m, n))
            planet.append(getCoord(m+1, n))
            planet.append(getCoord(m+1, n+1))
            planet.append(getCoord(m, n+1))
            planet.append(getCoord(m, n))
            planet.append(getCoord(m+1, n+1))
    planet = np.array(planet, dtype=np.float32)

    glEnableClientState(GL_VERTEX_ARRAY)
    glColor3fv(hsv2rgb(np.array([gTime * 0.1 % 1, 1, 1])))
    glVertexPointer(3, GL_FLOAT, 3 * planet.itemsize, planet)
    glDrawArrays(GL_TRIANGLES, 0, planet.size // 3)


def getRotation(axis, rad):
    c, s = np.cos(rad), np.sin(rad)
    M = np.eye(4)
    if axis == 'x':
        M[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == 'y':
        M[0:3:2, 0:3:2] = [[c, s], [-s, c]]
    elif axis == 'z':
        M[0:2, 0:2] = [[c, -s], [s, c]]
    return M


def getCameraSettings():
    azimuth, elevation = np.radians(orbit * 180)
    return azimuth, elevation, zoom


def getCameraVectors(cam_to_target, up):
    cam_dir = cam_to_target
    dist = np.linalg.norm(cam_dir)
    cam_dir = cam_dir / dist
    up_normalized = up / np.linalg.norm(up)

    w = -cam_dir
    u = np.cross(up_normalized, w)
    v = np.cross(w, u)

    return u, v, w


def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    glLoadIdentity()  # Reset previous state

    # Projection Mode
    aspect_ratio = RESOLUTION[0] / RESOLUTION[1]
    if gProjectionMode == 0:
        gluPerspective(45, aspect_ratio, 0.1, 100)
    else:
        glOrtho(-10, 10, -10, 10, -10*aspect_ratio, 10)

    # Viewing Transform
    azimuth, elevation, zoom = getCameraSettings()

    up = UP

    # When the camera goes into the opposite area due to its elevation in a sudden
    if (elevation + np.pi/2) // np.pi % 2 != 0:
        up = -up  # just change it up vector

    if gProjectionMode == 1:
        zoom *= 0.1  # For similar scales

    # Orbit
    cam_to_target = \
        getRotation('y', -azimuth) @ \
        getRotation('x', -elevation) @ \
        np.array([0, 0, -1, 0])

    # Pan
    global TARGET
    u, v, w = getCameraVectors(cam_to_target[:3], up)
    if np.linalg.norm(pan_delta) > 0.01:
        TARGET += (-pan_delta[0] * u + pan_delta[1] * v) * 3
    target = np.array([*TARGET, 1])

    # Zoom
    cam = target + cam_to_target * zoom

    # Convert homogeneous coordinates into 3d cartesian coordinates
    cam = cam[:3]
    target = target[:3]

    # Using gluLookAt to transform the current space into view(camera) space
    gluLookAt(*cam, *target, *up)

    # World Transform
    # Models to draw

    # 1. Colored Axis
    drawAxis()

    # 2. Grid
    drawGrids()

    # 3. 3D Boxes
    drawCubeArray(initialPosX=10, initialPosY=5, initialPosZ=10)

    # 4. Sphere
    drawPlannet()

    # 5. Orbiting Sphere
    drawOrbitingPlannet()


def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(*RESOLUTION, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Make the window's context current
    glfw.make_context_current(window)

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Update Global Time
        global gTime
        gTime = glfw.get_time()

        # Render here, e.g. using pyOpenGL
        render()

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
