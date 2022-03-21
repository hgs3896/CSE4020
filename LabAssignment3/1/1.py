import numpy as np
import glfw
from OpenGL.GL import *

# Write down a python program to draw a rotating triangle
# A. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

# B. Draw a triangle using render() function below (Do not modify it)
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

# C. Expected result: Uploaded LabAssignment3-1.mp4
# D. The triangle should be t rad rotated when t seconds have elapsed since the program was executed.
t_elpased = 0

# E. You need to somehow combine a rotation matrix and a translation matrix to produce the expected result.
def getMatrix(t = 0, dx = 0, dy = 0):
    return np.array([ # Rotation
        [np.cos(t), -np.sin(t), 0.],
        [np.sin(t), np.cos(t), 0.],
        [0, 0., 1]
    ]) @ np.array([ # Translation
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1],
    ])

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

    start_time = glfw.get_time()

    # Loop until the user closes the window
    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()

        # Render here, e.g. using pyOpenGL
        t_elapsed = glfw.get_time() - start_time
        
        M = getMatrix(t_elapsed, 0.5, 0)
        render(M)

        # Swap front and back buffers
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()