import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# D. Find code for drawFrame(), drawCubeArray() from 6-Viewing,Projection slides.
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

def drawCubeArray():
    for i in range(5):
        for j in range(5):
            for k in range(5):
                glPushMatrix()
                glTranslatef(i, j, -k-1)
                glScalef(.5, .5, .5)
                drawUnitCube()
                glPopMatrix()

def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([1., 0., 0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0., 0., 0.]))
    glVertex3fv(np.array([0., 1., 0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0., 0., 0]))
    glVertex3fv(np.array([0., 0., 1.1]))
    glEnd()

# C. Code Skeleton
def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    myOrtho(-5, 5, -5, 5, -8, 8)
    myLookAt(np.array([5, 3, 5]), np.array([1, 1, -1]), np.array([0, 1, 0]))
    # Above two lines must behaves exactly same as the below two lines
    # glOrtho(-5,5, -5,5, -8,8)
    # gluLookAt(5,3,5, 1,1,-1, 0,1,0)
    drawFrame()
    glColor3ub(255, 255, 255)
    drawCubeArray()
    glLoadIdentity()

# A. Write your own myLookAt() and myOrtho() functions (of the following form) that behaves exactly same as gluLookAt() and glOrtho().
def myOrtho(left, right, bottom, top, near, far):
    # implement here
    glMultMatrixf(np.array([
        [2 / (right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2 / (top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1],
    ]).T)

def myLookAt(eye, at, up):
    # implement here
    cam_dir = at - eye
    dist = np.linalg.norm(cam_dir)
    cam_dir = cam_dir / dist
    up_normalized = up / np.linalg.norm(up)

    w = -cam_dir
    u = np.cross(up_normalized, w)
    v = np.cross(w, u)

    M = np.eye(4)
    M[:3, 0] = u
    M[:3, 1] = v
    M[:3, 2] = w
    M[:3, 3] = eye
    M = np.matrix(M)

    glMultMatrixf(M.I.T)

# B. Set the window title to your student ID and the window size to (480,480).
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# E. DO NOT use gluLookAt() inside myLookAt() and glOrtho() inside myOrtho()!
def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(*RESOLUTION, STUDENT_ID, None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Render here, e.g. using pyOpenGL
        render()

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()