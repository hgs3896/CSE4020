import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

gCamAng = 0
gCamHeight = 0

# A. Write down a Python program to draw following triangular pyramid (삼각뿔) 
# by using separate triangles representation and glDrawArrays().
def render():
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    
    glLoadIdentity()
    gluPerspective(45, 1, 1, 10)
    gluLookAt(5*np.sin(gCamAng),gCamHeight,5*np.cos(gCamAng), 0,0,0, 0,1,0)

    drawFrame()
    glColor3ub(255, 255, 255)
    drawPyramid()

def key_callback(window, key, scancode, action, mods):
    global gCamAng, gCamHeight
    if action==glfw.PRESS or action==glfw.REPEAT:
        if key==glfw.KEY_1:
            gCamAng += np.radians(-10)
        elif key==glfw.KEY_3:
            gCamAng += np.radians(10)
        elif key==glfw.KEY_2:
            gCamHeight += .1
        elif key==glfw.KEY_W:
            gCamHeight += -.1

# B. Start from the code in the lecture slides. Make sure camera manipulation shortcuts ‘1’, ‘3’, ‘2’, ‘w’ work.
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

def drawPyramid():
    pyramid = np.array([
        [0., 0., 0.], # origin
        [1., 0., 0.], # x
        [0., 1., 0.], # y

        [0., 0., 0.], # origin
        [0., 1., 0.], # y
        [0., 0., 1.], # z

        [0., 0., 0.], # origin
        [0., 0., 1.], # z
        [1., 0., 0.], # x
    ], dtype=np.float32) * 1.5
    
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 3 * pyramid.itemsize, pyramid)
    glDrawArrays(GL_TRIANGLES, 0, pyramid.size // 3)

# C. Set the window title to your student ID and the window size to (480,480).
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

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
    glfw.set_key_callback(window, key_callback)

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