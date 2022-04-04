import numpy as np
import glfw
from OpenGL.GL import *

# Write down a Python program to draw three objects transformed with different modeling transformations and their local frames in a 3D space.
# A. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glEnd()

    glBegin(GL_LINES)
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()

def drawTriangle(): 
    glBegin(GL_TRIANGLES)
    glVertex2fv(np.array([0.,.5]))
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([.5,0.]))
    glEnd()
    
def drawBox():
    glBegin(GL_QUADS)
    glVertex2fv(np.array([0.,.5]))
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([.5,0.]))
    glVertex2fv(np.array([.5,.5]))
    glEnd()

def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)


    # B. Object 1 is a white triangle. Its modeling transform is the identity.
    glLoadIdentity()
    drawFrame()
    glColor3ub(255, 255, 255)
    drawTriangle()

    # C. Object 2 is a blue box. Its modeling transform is first rotation by 30 degrees about z and then translation by (0.6, 0, 0) w.r.t. the global frame.
    glLoadIdentity()
    glTranslatef(0.6, 0, 0)
    glRotatef(30, 0, 0, 1)

    drawFrame()
    glColor3ub(0, 0, 255)
    drawBox()

    # D. Object 3 is a red triangle. Its modeling transform is first translation by (0.3, 0, 0) and then rotation by -90 degrees about z w.r.t. the global frame.
    glLoadIdentity()
    glRotatef(-90, 0, 0, 1)
    glTranslatef(0.3, 0, 0)

    drawFrame()
    glColor3ub(255, 0, 0)
    drawTriangle()

    # E. All objects should be rendered using the drawTriangle() and drawBox()
    # F. Render the local frame of each object using the drawFrame().
    # G. Do not use gluLookAt() or any other viewing & projection manipulation functions.

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