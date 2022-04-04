import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

# As mentioned in the lecture, “moving camera” and “moving world” are two equivalent operations. Based on the following figure, replace the gluLookAt call() in the following code with two glRotatef() calls and one glTranslatef() call and complete the program.
# C. Set the window title to your student ID and the window size to (480, 480)
RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

def drawUnitCube() :
    glBegin(GL_QUADS)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)

    glVertex3f( 0.5,-0.5,0.5)
    glVertex3f(-0.5,-0.5,0.5)
    glVertex3f(-0.5,-0.5,-0.5)
    glVertex3f( 0.5,-0.5,-0.5)

    glVertex3f( 0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5,-0.5, 0.5)
    glVertex3f( 0.5,-0.5, 0.5)

    glVertex3f( 0.5,-0.5,-0.5)
    glVertex3f(-0.5,-0.5,-0.5)
    glVertex3f(-0.5, 0.5,-0.5)
    glVertex3f( 0.5, 0.5,-0.5)

    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5,-0.5)
    glVertex3f(-0.5,-0.5,-0.5)
    glVertex3f(-0.5,-0.5, 0.5)

    glVertex3f( 0.5, 0.5,-0.5)
    glVertex3f( 0.5, 0.5, 0.5)
    glVertex3f( 0.5,-0.5, 0.5)
    glVertex3f( 0.5,-0.5,-0.5)
    glEnd()

def drawCubeArray():
    for i in range (5):
        for j in range (5):
            for k in range (5):
                glPushMatrix()
                glTranslatef(i,j,-k-1)
                glScalef(.5,.5,.5)
                drawUnitCube()
                glPopMatrix()

def drawFrame() :
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array([1.,0.,0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array ([0.,1.,0.]) )
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0.,0.,0]))
    glVertex3fv(np.array([0.,0.,1.1]))
    glEnd()

def f(a, b):
    return np.arcsin(np.linalg.norm(np.cross(a, b))/np.linalg.norm(a)/np.linalg.norm(b))

def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
    glLoadIdentity()

    gluPerspective(45, 1, 1, 10)

    # Replace this call with two glRotatef() calls and one glTranslatef() call
    eye = np.array([3, 3, 3])
    w = eye / np.linalg.norm(eye)
    up = np.array([0, 1, 0])
    u = np.cross(up, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    
    alpha = np.degrees(f(eye * np.array([0, 1, 1]), eye))
    beta = np.degrees(f(np.array([0, 0, 1]), eye * np.array([1, 0, 1])))
    
    glRotatef(alpha, 1, 0, 0)
    glRotatef(-beta, 0, 1, 0)
    glTranslatef(*(-eye))
    
    drawFrame()

    glColor3ub(255, 255, 255)
    drawCubeArray()

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