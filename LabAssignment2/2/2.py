import numpy as np
import glfw
from OpenGL.GL import *

# A. Set the window title to your student ID and the window size to (480,480).
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# B. The width and height of the rectangle are 1.0.
SIZE = 1.0

# C. The 4 vertices should be specified counterclockwise.
vertices = np.array([
    [SIZE/2.0, SIZE/2.0], [-SIZE/2.0, SIZE/2.0], [-SIZE/2.0, -SIZE/2.0], [SIZE/2.0, -SIZE/2.0]
])

# D. When the program starts, the vertices are connected with GL_LINE_LOOP.
PrimitiveType = GL_LINE_LOOP

# E. If the keys 1, 2, 3, ... 9, 0 are entered, the primitive type should be changed.
# i. Hint: Use a global variable to store the primitive type
KeyEnums = {
    glfw.KEY_1: GL_POINTS,
    glfw.KEY_2: GL_LINES,
    glfw.KEY_3: GL_LINE_STRIP,
    glfw.KEY_4: GL_LINE_LOOP,
    glfw.KEY_5: GL_TRIANGLES,
    glfw.KEY_6: GL_TRIANGLE_STRIP,
    glfw.KEY_7: GL_TRIANGLE_FAN,
    glfw.KEY_8: GL_QUADS,
    glfw.KEY_9: GL_QUAD_STRIP,
    glfw.KEY_0: GL_POLYGON,
}

def key_callback(window, key, scancode, action, mods):
    if action==glfw.PRESS:
        if key in KeyEnums:
            global PrimitiveType
            PrimitiveType = KeyEnums[key]

def render():
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glBegin(PrimitiveType)
    for vertex in vertices:
        glVertex2fv(vertex)
    glEnd()

def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(*RESOLUTION, STUDENT_ID, None, None)
    if not window:
        glfw.terminate()
        return

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