import glfw
from OpenGL.GL import *
import numpy as np
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import ctypes

RESOLUTION = (480, 480)
STUDENT_ID = "2017030473"

gCamAng = 0.
gCamHeight = 1.

def createVertexArraySeparate():
    varr = np.array([
            [0,1,0],            # v0 normal
            [ 0.5, 0.5,-0.5],   # v0 position
            [0,1,0],            # v1 normal
            [-0.5, 0.5,-0.5],   # v1 position
            [0,1,0],            # v2 normal
            [-0.5, 0.5, 0.5],   # v2 position

            [0,1,0],            # v3 normal
            [ 0.5, 0.5,-0.5],   # v3 position
            [0,1,0],            # v4 normal
            [-0.5, 0.5, 0.5],   # v4 position
            [0,1,0],            # v5 normal
            [ 0.5, 0.5, 0.5],   # v5 position

            [0,-1,0],           # v6 normal
            [ 0.5,-0.5, 0.5],   # v6 position
            [0,-1,0],           # v7 normal
            [-0.5,-0.5, 0.5],   # v7 position
            [0,-1,0],           # v8 normal
            [-0.5,-0.5,-0.5],   # v8 position

            [0,-1,0],
            [ 0.5,-0.5, 0.5],
            [0,-1,0],
            [-0.5,-0.5,-0.5],
            [0,-1,0],
            [ 0.5,-0.5,-0.5],

            [0,0,1],
            [ 0.5, 0.5, 0.5],
            [0,0,1],
            [-0.5, 0.5, 0.5],
            [0,0,1],
            [-0.5,-0.5, 0.5],

            [0,0,1],
            [ 0.5, 0.5, 0.5],
            [0,0,1],
            [-0.5,-0.5, 0.5],
            [0,0,1],
            [ 0.5,-0.5, 0.5],

            [0,0,-1],
            [ 0.5,-0.5,-0.5],
            [0,0,-1],
            [-0.5,-0.5,-0.5],
            [0,0,-1],
            [-0.5, 0.5,-0.5],

            [0,0,-1],
            [ 0.5,-0.5,-0.5],
            [0,0,-1],
            [-0.5, 0.5,-0.5],
            [0,0,-1],
            [ 0.5, 0.5,-0.5],

            [-1,0,0],
            [-0.5, 0.5, 0.5],
            [-1,0,0],
            [-0.5, 0.5,-0.5],
            [-1,0,0],
            [-0.5,-0.5,-0.5],

            [-1,0,0],
            [-0.5, 0.5, 0.5],
            [-1,0,0],
            [-0.5,-0.5,-0.5],
            [-1,0,0],
            [-0.5,-0.5, 0.5],

            [1,0,0],
            [ 0.5, 0.5,-0.5],
            [1,0,0],
            [ 0.5, 0.5, 0.5],
            [1,0,0],
            [ 0.5,-0.5, 0.5],

            [1,0,0],
            [ 0.5, 0.5,-0.5],
            [1,0,0],
            [ 0.5,-0.5, 0.5],
            [1,0,0],
            [ 0.5,-0.5,-0.5],
            # ...
            ], 'float32')
    return varr

def drawUnitCube_glDrawArray():
    global gVertexArraySeparate
    varr = gVertexArraySeparate
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 6*varr.itemsize, varr)
    glVertexPointer(3, GL_FLOAT, 6*varr.itemsize, ctypes.c_void_p(varr.ctypes.data + 3*varr.itemsize))
    glDrawArrays(GL_TRIANGLES, 0, int(varr.size/6))

def drawFrame():
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array([1.,0.,0.]))
    glColor3ub(0, 255, 0)
    glVertex3fv(np.array([0.,0.,0.]))
    glVertex3fv(np.array([0.,1.,0.]))
    glColor3ub(0, 0, 255)
    glVertex3fv(np.array([0.,0.,0]))
    glVertex3fv(np.array([0.,0.,1.]))
    glEnd()

# euler[0]: zang
# euler[1]: yang
# euler[2]: zang_2
def ZXZEulerToRotMat(euler):
    zang, xang, zang_2 = euler
    Rz = np.array([[np.cos(zang), -np.sin(zang), 0],
                   [np.sin(zang), np.cos(zang), 0],
                   [0,0,1]])
    Rx = np.array([[1,0,0],
                   [0,np.cos(xang),-np.sin(xang)],
                   [0,np.sin(xang),np.cos(xang)]])
    Rz_2 = np.array([[np.cos(zang_2), -np.sin(zang_2), 0],
                   [np.sin(zang_2), np.cos(zang_2), 0],
                   [0,0,1]])
    return Rz @ Rx @ Rz_2

def drawCubes(brightness):
    glPushMatrix()
    glScalef(.5,.5,.5)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (.5*brightness,.5*brightness,.5*brightness,1.))
    drawUnitCube_glDrawArray()

    glTranslatef(1.5,0,0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (1.*brightness,0.,0.,1.))
    drawUnitCube_glDrawArray()

    glTranslatef(-1.5,1.5,0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.,1.*brightness,0.,1.))
    drawUnitCube_glDrawArray()

    glTranslatef(0,-1.5,1.5)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.,0.,1.*brightness,1.))
    drawUnitCube_glDrawArray()
    glPopMatrix()

#################################################
gEulerParams = [0, 0, 0]
def render():
    global gCamAng, gCamHeight
    global gEulerParams
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1, 1,10)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(5*np.sin(gCamAng), gCamHeight, 5*np.cos(gCamAng), 0,0,0, 0,1,0)

    drawFrame() # draw global frame

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_RESCALE_NORMAL) # rescale normal vectors after transformation and before lighting to have unit length

    glLightfv(GL_LIGHT0, GL_POSITION, (1.,2.,3.,1.))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (.1,.1,.1,1.))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.,1.,1.,1.))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.,1.,1.,1.))

    # end orientation
    euler = np.array(gEulerParams)*np.radians(1)   # in ZYX Euler angles
    R = ZXZEulerToRotMat(euler)  # in rotation matrix
    M = np.identity(4)

    # slerp
    glPushMatrix()
    M[:3,:3] = R
    glMultMatrixf(M.T)
    drawCubes(1.)
    glPopMatrix()

    glDisable(GL_LIGHTING)

def key_callback(window, key, scancode, action, mods):
    global gCamAng, gCamHeight
    global gEulerParams
    # rotate the camera when 1 or 3 key is pressed or repeated
    if action==glfw.PRESS or action==glfw.REPEAT:
        if key==glfw.KEY_1:
            gCamAng += np.radians(-10)
        elif key==glfw.KEY_3:
            gCamAng += np.radians(10)
        elif key==glfw.KEY_2:
            gCamHeight += .1
        elif key==glfw.KEY_W:
            gCamHeight += -.1
        elif key==glfw.KEY_A:
            gEulerParams[0] += 10
        elif key==glfw.KEY_Z:
            gEulerParams[0] -= 10
        elif key==glfw.KEY_S:
            gEulerParams[1] += 10
        elif key==glfw.KEY_X:
            gEulerParams[1] -= 10
        elif key==glfw.KEY_D:
            gEulerParams[2] += 10
        elif key==glfw.KEY_C:
            gEulerParams[2] -= 10
        elif key==glfw.KEY_V:
            gEulerParams = [0, 0, 0]

gVertexArraySeparate = None
def main():
    if not glfw.init():
        return
    window = glfw.create_window(*RESOLUTION, STUDENT_ID, None,None)
    if not window:
        glfw.terminate()
        return
    glfw.set_key_callback(window, key_callback)

    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    global gVertexArraySeparate
    gVertexArraySeparate = createVertexArraySeparate()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()