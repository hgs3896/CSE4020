import numpy as np
import glfw
from OpenGL.GL import *

# Write down a python program to draw a transformed triangle
# A. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# B. Draw a triangle using render() function of prob 1 (Do not modify it)
def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    # draw coordinate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()

    # draw triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv( (T @ np.array([.0,.5,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.0,.0,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.5,.0,1.]))[:-1] )
    glEnd()

# C. If you press or repeat a key, the triangle should be transformed as shown in the Table.
# D. Transformations should be accumulated (composed with previous one) unless you press '1'.
gComposedM = np.identity(3)

ten_degree_rad = np.deg2rad(10)
def scale(s):
    return np.diag([s,s,1])

def rotate(rad):
    c, s = np.cos(rad), np.sin(rad)
    r = np.identity(3)
    r[0:2,0:2] = [[c, -s], [s, c]]
    return r

def shear(offsetX=0, offsetY=0):
    sh = np.identity(3)
    sh[0, 1] = offsetX
    sh[1, 0] = offsetY
    return sh

def reflectX():
    M = np.identity(3)
    M[1,1] = -1
    return M

KeyFuncs = {
    glfw.KEY_W: lambda M: scale(0.9) @ M,
    glfw.KEY_E: lambda M: scale(1.1) @ M,
    glfw.KEY_S: lambda M: rotate(ten_degree_rad) @ M,
    glfw.KEY_D: lambda M: rotate(-ten_degree_rad) @ M,
    glfw.KEY_X: lambda M: shear(-0.1) @ M,
    glfw.KEY_C: lambda M: shear(0.1) @ M,
    glfw.KEY_R: lambda M: reflectX() @ M,
    glfw.KEY_1: lambda M: np.identity(3),
}

def key_callback(window, key, scancode, action, mods):
    if action==glfw.PRESS:
        if key in KeyFuncs:
            global gComposedM
            gComposedM = KeyFuncs[key](gComposedM)

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
        render(gComposedM)

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()