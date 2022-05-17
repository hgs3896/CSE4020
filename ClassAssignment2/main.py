import re
from os.path import exists as file_exists
from os.path import splitext as split_ext
from os.path import basename
from os.path import join as path_join
import concurrent.futures as futures

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

RESOLUTION = (800, 800)
STUDENT_ID = "2017030473"
WINDOW_TITLE = "Basic OpenGL Viewer"

RENDERING_MODE = 0
single_model = None
multi_models = {
    "Building": path_join("..", "objs", "building.obj"),
    "Bugatti": path_join("..", "objs", "bugatti.obj"),
    "Male": path_join("..", "objs", "male.obj"),
    "Skeletons": path_join("..", "objs", "skeletons.obj"),
    "Plane": path_join("..", "objs", "airplane.obj"),
    "Cow": path_join("..", "objs", "cow.obj"),
    "Cat": path_join("..", "objs", "cat.obj"),
}

STOP_THE_WORLD = 0
SHOW_WIREFRAME = 1
FORCE_SMOOTH_SHADING = 0

TARGET = np.array([0., 0., 0.])
UP = np.array([0., 1., 0.])

gTime = 0

gProjectionMode = 0
gCursorPos = np.zeros(2)
gScrollOffset = np.zeros(2)

orbit, orbit_s, orbit_e = np.zeros(2), np.zeros(2), np.zeros(2)
pan, pan_s, pan_e, pan_delta = np.zeros(
    2), np.zeros(2), np.zeros(2), np.zeros(2)
zoom = 50

MOUSE_BUTTON_STATE = {
    "NONE": 0,
    "TRIGGER_START": 1,
    "TRIGGERING": 2,
    "TRIGGER_END": 3,
}
MOUSE_BUTTON_LEFT = MOUSE_BUTTON_STATE["NONE"]
MOUSE_BUTTON_RIGHT = MOUSE_BUTTON_STATE["NONE"]


def get_next_state(s):
    if s == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        return MOUSE_BUTTON_STATE["TRIGGERING"]
    elif s == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        return MOUSE_BUTTON_STATE["NONE"]
    return s


def get_next_state_from_button_callback(s, action_type):
    if action_type == glfw.PRESS:
        if s == MOUSE_BUTTON_STATE["NONE"]:
            return MOUSE_BUTTON_STATE["TRIGGER_START"]
    elif action_type == glfw.RELEASE:
        if s == MOUSE_BUTTON_STATE["TRIGGERING"]:
            return MOUSE_BUTTON_STATE["TRIGGER_END"]
    return s


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_V:
        if action == glfw.PRESS:
            global gProjectionMode
            gProjectionMode = 1 - gProjectionMode
    elif key == glfw.KEY_H:
        if action == glfw.PRESS:
            global RENDERING_MODE
            RENDERING_MODE = 1
    elif key == glfw.KEY_Z:
        if action == glfw.PRESS:
            global SHOW_WIREFRAME
            SHOW_WIREFRAME = 1 - SHOW_WIREFRAME
    elif key == glfw.KEY_S:
        if action == glfw.PRESS:
            global FORCE_SMOOTH_SHADING
            FORCE_SMOOTH_SHADING = 1 - FORCE_SMOOTH_SHADING
    elif key == glfw.KEY_SPACE:
        if action == glfw.PRESS:
            global STOP_THE_WORLD
            STOP_THE_WORLD = 1 - STOP_THE_WORLD


def cursor_callback(window, xpos, ypos):
    global gCursorPos, old_orbit, orbit, old_pan, pan, pan_delta
    global MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT

    gCursorPos = np.array([xpos, ypos]) / RESOLUTION

    global orbit_s, orbit_e
    global pan_s, pan_e
    if MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        old_orbit = orbit
        orbit_s = gCursorPos
    elif MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGERING"]:
        orbit_e = gCursorPos
        orbit = old_orbit + np.array([1, -1]) * (orbit_e - orbit_s)
    elif MOUSE_BUTTON_LEFT == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        orbit_e = gCursorPos

    if MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGER_START"]:
        old_pan = pan
        pan_s = gCursorPos
        pan_delta = np.zeros(2)
    elif MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGERING"]:
        pan_e = gCursorPos
        pan_delta = np.array([1, -1]) * (pan_e - pan_s)
        pan = old_pan + pan_delta
    elif MOUSE_BUTTON_RIGHT == MOUSE_BUTTON_STATE["TRIGGER_END"]:
        pan_e = gCursorPos
        pan_delta = np.zeros(2)

    MOUSE_BUTTON_LEFT = get_next_state(MOUSE_BUTTON_LEFT)
    MOUSE_BUTTON_RIGHT = get_next_state(MOUSE_BUTTON_RIGHT)


def button_callback(window, button, action, mod):
    if button == glfw.MOUSE_BUTTON_LEFT:
        global MOUSE_BUTTON_LEFT
        MOUSE_BUTTON_LEFT = get_next_state_from_button_callback(
            MOUSE_BUTTON_LEFT, action)

    if button == glfw.MOUSE_BUTTON_RIGHT:
        global MOUSE_BUTTON_RIGHT
        MOUSE_BUTTON_RIGHT = get_next_state_from_button_callback(
            MOUSE_BUTTON_RIGHT, action)


def scroll_callback(window, xoffset, yoffset):
    global gScrollOffset
    gScrollOffset = np.array([xoffset, yoffset]) / RESOLUTION

    global zoom
    if np.abs(gScrollOffset[1]) > 0.001:
        zoom -= gScrollOffset[1] * 100
        zoom = np.clip(zoom, 1, 1000)

def parse_obj_format(file):
    f_type = re.compile("[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
    index_type = re.compile("[-/\d]+")

    vertex_type = re.compile("v\s+([-+eE\d\. ]+)")
    normal_type = re.compile("vn\s+([-+eE\d\. ]+)")
    face_type = re.compile("f\s+([-/\d ]+)")

    def convert_indices(indices, n):
        indices = np.array(indices, dtype=np.int32)
        indices = (indices >= 1) * (indices - 1) + \
            (indices < 1) * (n + indices)
        return indices.astype(np.uint32)

    def create_new_model():
        return {
            "vertices": [],
            "normals": [],
            "vertex_indices": {3: [], 4: []},
            "normal_indices": {3: [], 4: []},
        }

    current = create_new_model()

    # Parse the obj format
    while True:
        line = file.readline()
        if not line:
            break
        if line[:2] in ("o ", "g "):
            continue

        vertex_type_match = vertex_type.match(line)
        if vertex_type_match:
            numbers = [float(number[0]) for number in f_type.finditer(
                vertex_type_match.group(0))]

            current["vertices"].append(numbers)
            continue

        normal_type_match = normal_type.match(line)
        if normal_type_match:
            numbers = [float(number[0]) for number in f_type.finditer(
                normal_type_match.group(0))]

            current["normals"].append(numbers)
            continue

        face_type_match = face_type.match(line)
        if face_type_match:
            indices = [index[0].split(
                "/") for index in index_type.finditer(face_type_match.group(1))]

            vindex = []
            nindex = []
            for index in indices:
                if len(index) > 0 and index[0]:
                    vindex.append(index[0])
                if len(index) > 2 and index[2]:
                    nindex.append(index[2])

            if len(vindex) != 0:
                if len(vindex) not in current["vertex_indices"]:
                    current["vertex_indices"][len(vindex)] = []
                current["vertex_indices"][len(vindex)].append(vindex)
            
            if len(nindex) != 0:
                if len(nindex) not in current["normal_indices"]:
                    current["normal_indices"][len(nindex)] = []
                current["normal_indices"][len(nindex)].append(nindex)
            continue

    # 2. Convert the parsed obj data into numpy array
    current["vertices"] = np.array(current["vertices"], dtype=np.float32)
    current["normals"] = np.array(current["normals"], dtype=np.float32)
    NO_NORMAL = len(current["normals"]) == 0

    n = current["vertices"].shape[0]

    current["triangular_vertices_indices"] = []
    current["triangular_normals_indices"] = []

    num_all_vertices = sum(len(current["vertex_indices"][num_vertices])
                           for num_vertices in current["vertex_indices"])
    
    # Triangularization
    for num_vertices in current["vertex_indices"]:
        current["vertex_indices"][num_vertices] = convert_indices(current["vertex_indices"][num_vertices], n)
        if not NO_NORMAL:
            current["normal_indices"][num_vertices] = convert_indices(current["normal_indices"][num_vertices], n)

        if len(current["vertex_indices"][num_vertices]) == 0:
            continue

        indices_to_fetch = ((0, i - 1, i) for i in range(2, num_vertices))
        triangular_vertices_indices = []
        triangular_normals_indices = []
        
        for index_to_fetch in indices_to_fetch:
            triangular_vertices_indices.extend(current["vertex_indices"][num_vertices].reshape(-1, num_vertices)[:, index_to_fetch])
            if not NO_NORMAL:
                triangular_normals_indices.extend(current["normal_indices"][num_vertices].reshape(-1, num_vertices)[:, index_to_fetch])

        triangular_vertices_indices = np.concatenate(triangular_vertices_indices, dtype=np.uint32)
        current["triangular_vertices_indices"].append(np.array(triangular_vertices_indices))
        if not NO_NORMAL:
            triangular_normals_indices = np.concatenate(triangular_normals_indices, dtype=np.uint32)
            current["triangular_normals_indices"].append(np.array(triangular_normals_indices))
        
    current["triangular_vertices_indices"] = np.concatenate(current["triangular_vertices_indices"], dtype=np.uint32).reshape(-1, 3)
    current["triangular_vertices"] = current["vertices"][current["triangular_vertices_indices"]].reshape(-1, 3)

    current["triangular_normals_indices"] = np.concatenate(current["triangular_normals_indices"], dtype=np.uint32).reshape(-1, 3)
    current["triangular_normals"] = current["normals"][current["triangular_normals_indices"]].reshape(-1, 3)

    v_a = current["vertices"][current["triangular_vertices_indices"][:, 0]]
    v_b = current["vertices"][current["triangular_vertices_indices"][:, 1]]
    v_c = current["vertices"][current["triangular_vertices_indices"][:, 2]]

    v1 = v_b - v_a
    v2 = v_c - v_a

    computed_normals = np.cross(v1, v2)
    computed_normals = computed_normals / np.linalg.norm(computed_normals)

    current["computed_normals"] = np.zeros_like(current["vertices"])
    for (i, j, k), computed_normal in zip(current["triangular_vertices_indices"], computed_normals):
        current["computed_normals"][i] += computed_normal
        current["computed_normals"][j] += computed_normal
        current["computed_normals"][k] += computed_normal

    current["computed_normals"] = current["computed_normals"] / np.linalg.norm(current["computed_normals"], axis=0)
    current["computed_triangular_normals"] = current["computed_normals"][current["triangular_vertices_indices"]]

    if not NO_NORMAL:
        current["triangular"] = np.hstack((
            current["triangular_normals"].reshape(-1, 3),
            current["triangular_vertices"].reshape(-1, 3)
        )).reshape(-1, 3)
    else:
        current["triangular"] = current["triangular_vertices"].reshape(-1, 3)

    current["computed_triangular"] = np.hstack((
        current["computed_triangular_normals"].reshape(-1, 3),
        current["triangular_vertices"].reshape(-1, 3)
    )).reshape(-1, 3)

    return {"_filename": file.name, "base": {
        "has_normals": not NO_NORMAL,
        "num_vertices": {
            "3": len(current["vertex_indices"][3]),
            "4": len(current["vertex_indices"][4]),
            "all": num_all_vertices,
        },
        "data": current["triangular"].astype(np.float32),
        "computed_data": current["computed_triangular"].astype(np.float32),
    }}

def read_model(model_path):
    with open(model_path) as obj_file:
        return parse_obj_format(obj_file)

def load_models(executer):
    # Single Model 처리
    global single_model
    model = single_model

    if type(model) == str:
        # 비동기로 로딩
        print("File is loading")
        path = model
        single_model = executer.submit(read_model, path)
        print("Loading an obj file is on progress.")
    elif type(model) == futures.Future:
        if model.done():
            print("Finished to load obj file")
            single_model = model.result()

            filename = basename(single_model["_filename"])
            
            all = single_model['base']['num_vertices']['all']
            three_sided = single_model['base']['num_vertices']['3']
            four_sided = single_model['base']['num_vertices']['4']
            print(f"Filename: {filename}")
            print(f"Total number of faces: {all}")
            print(f"Number of faces with 3 vertices: {three_sided}")
            print(f"Number of faces with 4 vertices: {four_sided}")
            print(f"Number of faces with more than 4 vertices: {all-three_sided-four_sided}")

    # Hirachical Animated Models 처리
    global multi_models
    for k in multi_models:
        model = multi_models[k]
        if type(model) == str:
            # 비동기로 로딩
            path = model
            multi_models[k] = executer.submit(read_model, path)
        elif type(model) == futures.Future:
            if model.done():
                multi_models[k] = model.result()
            elif model.cancelled():
                del multi_models[k]

def drop_callback(window, paths):
    global single_model
    global RENDERING_MODE

    if len(paths) != 1:
        return

    path = paths[0]
    if not file_exists(path) or not (split_ext(path)[1] in (".obj",)):
        print("파일이 존재하지 않거나 obj 파일이 아닙니다.")
        return
    
    if type(single_model) == futures.Future:
        single_model.cancel()
    
    single_model = path
    RENDERING_MODE = 0
    

def getRotation(axis, rad):
    c, s = np.cos(rad), np.sin(rad)
    M = np.eye(4)
    if axis == 'x':
        M[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == 'y':
        M[0:3:2, 0:3:2] = [[c, s], [-s, c]]
    elif axis == 'z':
        M[0:2, 0:2] = [[c, -s], [s, c]]
    return M

def getCameraSettings():
    azimuth, elevation = np.radians(orbit * 180)
    return azimuth, elevation, zoom

def getCameraVectors(cam_to_target, up):
    cam_dir = cam_to_target
    dist = np.linalg.norm(cam_dir)
    cam_dir = cam_dir / dist
    up_normalized = up / np.linalg.norm(up)

    w = -cam_dir
    u = np.cross(up_normalized, w)
    v = np.cross(w, u)

    return u, v, w

# Draw functions
def drawAxis():
    # x, y, z coordinates
    glBegin(GL_LINE_STRIP)
    glColor3ub(255, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(100, 0, 0)
    glEnd()

    glBegin(GL_LINE_STRIP)
    glColor3ub(0, 255, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 100, 0)
    glEnd()

    glBegin(GL_LINE_STRIP)
    glColor3ub(0, 0, 255)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 100)
    glEnd()


def drawGrids():
    max_length = 100
    for line in range(-max_length, max_length):
        glColor3f(.5, .5, .5)
        # x line
        glBegin(GL_LINE_STRIP)
        glVertex3iv(np.array([line, 0, -max_length]))
        glVertex3iv(np.array([line, 0, max_length]))
        glEnd()

        # z line
        glBegin(GL_LINE_STRIP)
        glVertex3iv(np.array([-max_length, 0, line]))
        glVertex3iv(np.array([max_length, 0, line]))
        glEnd()

def drawParsedModel(model):
    if type(model) != dict:
        return

    submodel = model["base"]

    if FORCE_SMOOTH_SHADING:
        data = submodel["computed_data"]
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)
        glNormalPointer(GL_FLOAT, 6 * data.itemsize, data)
        glVertexPointer(3, GL_FLOAT, 6 * data.itemsize, ctypes.c_void_p(data.ctypes.data + 3 * data.itemsize))
        glDrawArrays(GL_TRIANGLES, 0, data.size // 6)
    else:
        has_normals = submodel["has_normals"]
        data = submodel["data"]
        if has_normals:
            glEnableClientState(GL_NORMAL_ARRAY)
            glEnableClientState(GL_VERTEX_ARRAY)
            glNormalPointer(GL_FLOAT, 6 * data.itemsize, data)
            glVertexPointer(3, GL_FLOAT, 6 * data.itemsize, ctypes.c_void_p(data.ctypes.data + 3 * data.itemsize))
            glDrawArrays(GL_TRIANGLES, 0, data.size // 6)
        else:
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 3 * data.itemsize, data)
            glDrawArrays(GL_TRIANGLES, 0, data.size // 3)

def drawSingleModel():
    global single_model
    model = single_model

    glPushMatrix()
    drawParsedModel(model)
    glPopMatrix()

def smoothstep(x, min, max):
    t = np.clip(x, min, max)
    return t * t * (3.0 - 2.0 * t)

def drawMultiModels():
    global multi_models

    duration = 10
    progress = gTime % (2 * duration) / duration
    inverse_direction = progress > 1
    progress = min(progress, 2 - progress)
    
    total_length = np.sqrt(2) + 1
    d = progress * total_length
    l = 40
    h = 50
    partA_end = 1
    partB_end = np.sqrt(2)
    partC_end = total_length

    x, y, z = 0, h, 0
    elevation = 0
    thetaDelta = 0
    if d < partA_end:
        p = d / partA_end
        z = l * p
        pp = smoothstep(p, 0.0, 1)
        y = h * pp
        elevation = np.degrees(np.arctan(2*pp*(1-pp)))
    elif d < partB_end:
        p = (d - partA_end) / (partB_end - partA_end)
        theta = 1.25 * np.pi * p
        r = (np.sqrt(2)-1) * l
        x = r * (1 + np.cos(np.pi-theta))
        z = r * ((np.sqrt(2) + 1) + np.sin(np.pi-theta))
        thetaDelta = np.degrees(theta)
    else:
        p = (d - partB_end) / (partC_end - partB_end)
        x = l * (1-p) * np.sqrt(0.5)
        z = l * (1-p) * np.sqrt(0.5)
        thetaDelta = 225
    
    if inverse_direction:
        x, z = -x, -z
        elevation = -elevation

    r = 15
    theta = 2 * gTime
    c, s = np.cos(theta), np.sin(theta)

    # Hierachical View
    # Building
    # - Bugatti
    # - Plane
    #    - Cat
    #    - Skeletons
    #       - Male
    #       - Cow

    glPushMatrix() # Building - start
    applyColor((0.1, 0.1, 0.1, 0.3), shininess=10)
    glScalef(0.3,0.3,0.3)
    drawParsedModel(multi_models["Building"])

    glPushMatrix() # Bugatti - start
    applyColor((1.0, 0.0, 0.0, 1.0), shininess=10)
    glRotatef(np.degrees(theta), 0, 1, 0)
    glTranslatef(0, 0, 50)
    drawParsedModel(multi_models["Bugatti"])

    glRotatef(90, 0, 1, 0)

    glPushMatrix() # Cow - start
    applyColor((1.0, 0.9, 0.05, 1.0))
    glTranslatef(0, 3, -30)
    glRotatef(-90 + 20 * np.sin(gTime), 0, 1, 0)
    drawParsedModel(multi_models["Cow"])
    glPopMatrix()  # Cow - end

    glPushMatrix() # Male - start
    applyColor((251/255, 206/255, 177/255, 1.0))
    glTranslatef(0, 5 * np.abs(s) + 3, 0)
    glRotatef(np.degrees(3 * theta), 0, 1, 0)
    drawParsedModel(multi_models["Male"])
    glPopMatrix() # Male - end

    glPopMatrix() # Bugatti - end

    glPushMatrix() # Plane - start
    applyColor((0.4, 0.4, 0.4, 1.0), shininess=10)
    glTranslatef(-70, 0, 0)
    glTranslatef(x, y, z)
    glRotatef(thetaDelta-90, 0, 1, 0)
    glRotatef(elevation, 0, 0, 1)
    drawParsedModel(multi_models["Plane"])

    glPushMatrix() # Cat - start
    applyColor((1.0, 1.0, 0.5, 1.0))
    glTranslatef(5 * s, 1.8, 0)
    glScalef(0.1, 0.1, 0.1)
    glRotatef(90 + np.degrees(theta), 0, 1, 0)
    glRotatef(-90, 1, 0, 0)
    drawParsedModel(multi_models["Cat"])
    glPopMatrix()  # Cat - end

    glPushMatrix() # Skeletons - start
    applyColor((0.9, 0.9, 0.9, 1.0))
    glRotatef(90, 0, 1, 0)
    glTranslatef(0, 0, 2.5 * s)
    glScalef(0.8, 0.8, 0.8)
    drawParsedModel(multi_models["Skeletons"])

    glPopMatrix() # Skeletons - end

    glPopMatrix() # Plane - end

    glPopMatrix() # Building - end

_light_constants_ = (GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7)
def applyLights(lightInfo):
    if len(lightInfo) <= 0:
        return

    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)

    for light_idx, light_info in zip(_light_constants_, lightInfo):
        if type(light_info) != dict:
            continue

        glEnable(light_idx)

        lightPos = (0., 1., 0., 1.)
        if "lightPos" in light_info:
            lightPos = light_info["lightPos"]
        
        lightColor = (1.,1.,1.,1.)
        if "lightColor" in light_info:
            lightColor = light_info["lightColor"]
        
        ambientLightColor = (.1,.1,.1,1.)
        if "ambientLightColor" in light_info:
            ambientLightColor = light_info["ambientLightColor"]
        
        glPushMatrix()
        glLightfv(light_idx, GL_POSITION, lightPos)
        glPopMatrix()

        # light intensity for each color channel
        glLightfv(light_idx, GL_DIFFUSE, lightColor)
        glLightfv(light_idx, GL_SPECULAR, lightColor)
        glLightfv(light_idx, GL_AMBIENT, ambientLightColor)

def applyColor(objectColor = (1., 1., 1., 1.), specularObjectColor = (1., 1., 1., 1.), shininess = 10):
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, objectColor)
    glMaterialfv(GL_FRONT, GL_SHININESS, shininess)
    glMaterialfv(GL_FRONT, GL_SPECULAR, specularObjectColor)

def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    # glEnable(GL_BLEND)
    # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # glEnable(GL_POINT_SMOOTH)
    # glEnable(GL_LINE_SMOOTH)
    # glEnable(GL_POLYGON_SMOOTH)

    # hint = GL_NICEST # GL_FASTEST # 
    # glHint(GL_POINT_SMOOTH_HINT, hint)
    # glHint(GL_LINE_SMOOTH_HINT, hint)
    # glHint(GL_POLYGON_SMOOTH_HINT, hint)

    glLoadIdentity()  # Reset previous state
    if SHOW_WIREFRAME:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # Projection Mode
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect_ratio = RESOLUTION[0] / RESOLUTION[1]
    if gProjectionMode == 0:
        gluPerspective(45, aspect_ratio, 0.0001, 10000)
    else:
        glOrtho(-30, 30, -30, 30, -30*aspect_ratio, 30)

    # Viewing Transform
    azimuth, elevation, zoom = getCameraSettings()

    up = UP

    # When the camera goes into the opposite area due to its elevation in a sudden
    if (elevation + np.pi/2) // np.pi % 2 != 0:
        up = -up  # just change it up vector

    if gProjectionMode == 1:
        zoom *= 0.1  # For similar scales

    # Orbit
    cam_to_target = \
        getRotation('y', -azimuth) @ \
        getRotation('x', -elevation) @ \
        np.array([0, 0, -1, 0])

    # Pan
    global TARGET
    u, v, w = getCameraVectors(cam_to_target[:3], up)
    if np.linalg.norm(pan_delta) > 0.01:
        TARGET += (-pan_delta[0] * u + pan_delta[1] * v) * 3
    target = np.array([*TARGET, 1])

    # Zoom
    cam = target + cam_to_target * zoom

    # Convert homogeneous coordinates into 3d cartesian coordinates
    cam = cam[:3]
    target = target[:3]

    # Using gluLookAt to transform the current space into view(camera) space
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(*cam, *target, *up)

    # World Transform
    # Models to draw

    # # 1. Colored Axis
    drawAxis()

    # # 2. Grid
    drawGrids()

    if RENDERING_MODE == 0:
        # 3-1. single mesh rendering mode

        applyLights([
            {
                "lightPos": (3, 4, 5, 1),
                "lightColor": (1.0, 1.0, 1.0, 1.0),
                "ambientLightColor": (0.1, 0.1, 0.1, 1.0)
            },
            {
                "lightPos": (-5, 4, -3, 1),
                "lightColor": (1.0, 1.0, 1.0, 1.0),
                "ambientLightColor": (0.1, 0.1, 0.1, 1.0)
            },
            {
                "lightPos": (0, 5, 10, 0),
                "lightColor": (1.0, 1.0, 0.8, 1.0),
                "ambientLightColor": (0.1, 0.1, 0.1, 1.0)
            },
        ])

        applyColor(
            objectColor=(1.0, 1.0, 1.0, 1.0),
            specularObjectColor=(1.0, 1.0, 1.0, 1.0),
            shininess=10
        )

        drawSingleModel()
    else:
        # 3-2. predefined hierachical animation mode
        applyLights([
            {
                "lightPos": (30, 30, 30, 0.0),
                "lightColor": (1.0, 1.0, 1.0, 1.0),
                "ambientLightColor": (0.1, 0.1, 0.1, 1.0)
            },
            {
                "lightPos": (-30, 70, -30, 1.0),
                "lightColor": (1.0, 1.0, 1.0, 1.0),
                "ambientLightColor": (0.1, 0.1, 0.1, 1.0)
            }
        ])

        drawMultiModels()

    glDisable(GL_LIGHTING)

def main():
    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(*RESOLUTION, WINDOW_TITLE, None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_mouse_button_callback(window, button_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_drop_callback(window, drop_callback)

    # Make the window's context current
    glfw.make_context_current(window)

    # For file IO and parsing asynchronously
    with futures.ProcessPoolExecutor() as executer:
        prevTime = 0
        # Loop until the user closes the window
        while not glfw.window_should_close(window):
            # Poll events
            glfw.poll_events()

            # Update Global Time
            curTime = glfw.get_time()
            delta = curTime - prevTime
            prevTime = curTime
            if STOP_THE_WORLD:
                delta = 0
            
            global gTime
            gTime += delta

            # Load models automatically
            load_models(executer)

            # Render here, e.g. using pyOpenGL
            render()

            # Swap front and back buffers
            glfw.swap_buffers(window)
    
    glfw.terminate()


if __name__ == "__main__":
    main()