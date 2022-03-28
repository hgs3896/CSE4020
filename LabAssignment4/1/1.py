import numpy as np
import glfw
from OpenGL.GL import *

# Write down a Python program to draw a transformed triangle in a 2D space.
# A. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# B. Complete the render() function below to draw a triangle in the manner described in C.
# You have to use OpenGL transformation functions. Do not use numpy matrix multiplication for composing transformations.
def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    # draw cooridnates
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()

    glColor3ub(255, 255, 255)

    ###########################
    for key in reversed(keyPressed):
        KeyFuncs[key]()
    ###########################

    drawTriangle()

def drawTriangle():
    glBegin(GL_TRIANGLES)
    glVertex2fv(np.array([0.,.5]))
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([.5,0.]))
    glEnd()

# C. If you press or repeat a key, the triangle should be transformed as shown in the Table
def resetKeys():
    global keyPressed
    keyPressed = []

KeyFuncs = {
    glfw.KEY_Q: lambda : glTranslatef(-0.1, 0., 0.),
    glfw.KEY_E: lambda : glTranslatef(+0.1, 0., 0.),
    glfw.KEY_A: lambda : glRotatef(10, 0, 0, 1),
    glfw.KEY_D: lambda : glRotatef(-10, 0, 0, 1),
    glfw.KEY_1: lambda : resetKeys(),
}

# D. Transformations should be accumulated (composed with previous one) unless you press ‘1’.
# You may need a global variable (like a python list object) to store key inputs.
keyPressed = []

def key_callback(window, key, scancode, action, mods):
    global keyPressed
    if action==glfw.PRESS or action==glfw.REPEAT:
        if key in KeyFuncs:
            keyPressed.append(key)

def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(*RESOLUTION, STUDENT_ID, None, None)
    if not window:
        glfw.terminate()
        return

    # Set a key callback function to handle the keyboard events
    glfw.set_key_callback(window, key_callback)

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