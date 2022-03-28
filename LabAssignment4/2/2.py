import numpy as np
import glfw
from OpenGL.GL import *

# Write down a Python program to draw rotating point p=(0.5, 0) and vector v=(0.5, 0) in a 2D space.
# A. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# B. Use the following render() and fill "# your implementation" parts to render p and v.
# Hint: Render the vector v as a line segment starting from the origin (0,0).
def render(M):
    glClear(GL_COLOR_BUFFER_BIT)

    glLoadIdentity()

    # draw cooridnate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()

    glColor3ub(255, 255, 255)

    # draw point p
    glBegin(GL_POINTS)
    # your implementation
    glVertex2fv((M @ np.array([0.5, 0, 1]))[:-1])
    glEnd()
    
    # draw vector v
    glBegin(GL_LINES)
    # your implementation
    glVertex2fv((M @ np.array([0, 0, 0]))[:-1])
    glVertex2fv((M @ np.array([0.5, 0, 0]))[:-1])
    glEnd()

# C. Expected result: Uploaded LabAssignment4-2.mp4
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

    global start_time
    start_time = glfw.get_time()

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # D. p and v should be t rad rotated when t seconds have elapsed since the program was executed.
        t_elapsed = glfw.get_time() - start_time
        c, s = np.cos(t_elapsed), np.sin(t_elapsed)
        
        # E. You need to somehow combine a rotation matrix and a translation matrix to produce the expected result.
        M = np.array([c, -s, 0, s, c, 0, 0, 0, 1]).reshape(3, 3) @ \
            np.array([1, 0, 0.5, 0, 1, 0, 0, 0, 1]).reshape(3, 3)

        # Render here, e.g. using pyOpenGL
        render(M)

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()