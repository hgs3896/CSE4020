import concurrent.futures as futures
from importlib.resources import path
import re
from io import FileIO
from os.path import basename
from os.path import exists as file_exists
from os.path import join as path_join
from os.path import splitext as split_ext
from tkinter import N

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
    "male_head": None,             # path_join("..", "hidden_objs", "male_head.obj"),
    "male_neck": None,             # path_join("..", "hidden_objs", "male_neck.obj"),
    "male_spine": None,            # path_join("..", "hidden_objs", "male_spine.obj"),
    "male_left_shoulder": None,    # path_join("..", "hidden_objs", "male_left_shoulder.obj"),
    "male_left_forearm": None,     # path_join("..", "hidden_objs", "male_left_forearm.obj"),
    "male_left_hand": None,        # path_join("..", "hidden_objs", "male_left_hand.obj"),
    "male_left_leg": None,         # path_join("..", "hidden_objs", "male_left_leg.obj"),
    "male_left_up_leg": None,      # path_join("..", "hidden_objs", "male_left_up_leg.obj"),
    "male_right_shoulder": None,   # path_join("..", "hidden_objs", "male_right_shoulder.obj"),
    "male_right_forearm": None,    # path_join("..", "hidden_objs", "male_right_forearm.obj"),
    "male_right_hand": None,       # path_join("..", "hidden_objs", "male_right_hand.obj"),
    "male_right_leg": None,        # path_join("..", "hidden_objs", "male_right_leg.obj"),
    "male_right_up_leg": None,     # path_join("..", "hidden_objs", "male_right_up_leg.obj"),
}

RENDERING_TYPE = 0
HAVE_BVH_BEEN_PLAYED = 0
single_bvh = None

STOP_THE_WORLD = 1
SHOW_WIREFRAME = 0
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
    global RENDERING_TYPE
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
        global STOP_THE_WORLD
        if action == glfw.PRESS:
            if type(single_bvh) == dict:
                global HAVE_BVH_BEEN_PLAYED
                HAVE_BVH_BEEN_PLAYED = 1
            STOP_THE_WORLD = 0
        if action == glfw.RELEASE:
            STOP_THE_WORLD = 1
    elif key == glfw.KEY_1:
        if action == glfw.PRESS:
            RENDERING_TYPE = 0
    elif key == glfw.KEY_2:
        if action == glfw.PRESS:
            RENDERING_TYPE = 1
    elif key == glfw.KEY_3:
        if action == glfw.PRESS:
            RENDERING_TYPE = 2
    elif key == glfw.KEY_4:
        if action == glfw.PRESS:
            RENDERING_TYPE = 3


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

def parse_bvh_format(file: FileIO):
    SECTION_TYPE = {"NONE": 0, "HIERARCHY": 1, "MOTION": 2}
    HIERARCHICAL_ITEM_TYPE = {"NONE": 0, "ROOT": 1, "JOINT": 2, "End Site": 3}

    section_type = SECTION_TYPE["NONE"]
    hierarchical_item_type = HIERARCHICAL_ITEM_TYPE["NONE"]

    item_idx = 0
    cur_node_name = ""
    cur_node = None
    bvh_data = {"name": file.name}

    while True:
        line = file.readline()

        if not line:
            break

        # Remove trailing newline character
        line = line[:-1]

        if line == "HIERARCHY":
            section_type = SECTION_TYPE["HIERARCHY"]
            continue
        elif line == "MOTION":
            section_type = SECTION_TYPE["MOTION"]
            continue

        # Remove leading & trailing whitespaces
        line = line.strip()

        if line == '{':
            if hierarchical_item_type == HIERARCHICAL_ITEM_TYPE["ROOT"]:
                bvh_data["children"] = {cur_node_name: {
                    "name": cur_node_name, "parent": bvh_data}}
                cur_node = bvh_data["children"][cur_node_name]
            else:
                if "children" not in cur_node:
                    cur_node["children"] = {}
                cur_node["children"][cur_node_name] = {
                    "name": cur_node_name, "parent": cur_node}
                cur_node = cur_node["children"][cur_node_name]

            continue
        elif line == '}':
            cur_node = cur_node["parent"]
            cur_node_name = cur_node["name"]

            continue

        if line.startswith("ROOT"):
            assert section_type == SECTION_TYPE["HIERARCHY"]
            hierarchical_item_type = HIERARCHICAL_ITEM_TYPE["ROOT"]
            cur_node_name = line[5:]
            continue
        elif line.startswith("JOINT"):
            assert section_type == SECTION_TYPE["HIERARCHY"]
            hierarchical_item_type = HIERARCHICAL_ITEM_TYPE["JOINT"]
            cur_node_name = line[6:]
            continue
        elif line == "End Site":
            assert section_type == SECTION_TYPE["HIERARCHY"]
            hierarchical_item_type = HIERARCHICAL_ITEM_TYPE["End Site"]
            cur_node_name = "End Site"
            continue

        if line.startswith("OFFSET"):
            assert section_type == SECTION_TYPE["HIERARCHY"]
            offsets = line[7:].split()
            assert len(offsets) == 3
            offsets = list(map(np.float32, offsets))
            cur_node["OFFSET"] = np.array(offsets)
            continue

        if line.startswith("CHANNELS"):
            assert section_type == SECTION_TYPE["HIERARCHY"]
            assert hierarchical_item_type != HIERARCHICAL_ITEM_TYPE["End Site"]
            channel_info = line[9:].split()
            assert len(channel_info) > 1
            channel_cnt = int(channel_info[0])
            assert len(channel_info) == channel_cnt + 1
            cur_node["CHANNELS"] = list(
                zip(map(str.upper, channel_info[1:]), range(item_idx, item_idx + channel_cnt)))
            item_idx = item_idx + channel_cnt
            continue

        if line.startswith("Frames:"):
            assert section_type == SECTION_TYPE["MOTION"]
            bvh_data["Frames"] = int(line[8:])
            continue

        if line.startswith("Frame Time:"):
            assert section_type == SECTION_TYPE["MOTION"]
            bvh_data["Frame Time"] = np.float32(line[11:])
            continue

        if section_type == SECTION_TYPE["MOTION"]:
            values = np.array(list(map(np.float32, line.split())))
            if "MOTION" not in bvh_data:
                bvh_data["MOTION"] = []
            bvh_data["MOTION"].append(values)

    return bvh_data

def read_model(model_path):
    with open(model_path) as obj_file:
        return parse_obj_format(obj_file)

def read_bvh(bvh_path):
    with open(bvh_path) as bvh_file:
        return parse_bvh_format(bvh_file)

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

def read_joint_from_bvh(bvh_node):
    if not bvh_node:
        return None

    if "name" not in bvh_node:
        return None
    result = [bvh_node["name"]]

    if "children" not in bvh_node:
        return result

    for child_node_name in bvh_node["children"]:
        if child_node_name == "End Site":
            continue
        child_node = bvh_node["children"][child_node_name]
        result.append(child_node_name)
        child_result = read_joint_from_bvh(child_node)
        if child_result:
            result += child_result
    return result

def load_bvh_animation(executer):
    # Single BVH 처리
    global single_bvh
    model = single_bvh

    if type(model) == str:
        # 비동기로 로딩
        print("BVH File is loading")
        path = model
        single_bvh = executer.submit(read_bvh, path)
        print("Loading a bvh file is on progress.")
    elif type(model) == futures.Future:
        if model.done():
            print("Finished to load bvh file")
            single_bvh = model.result()

            bvh_joint_info = list(set(read_joint_from_bvh(single_bvh)[1:]))

            # 1. File name
            print(f"1. File name: {basename(single_bvh['name'])}")
            # 2. Number of frames
            print(f"2. Number of frames: {single_bvh['Frames']}")
            # 3. FPS (which is 1/FrameTime)
            print(f"3. FPS (which is 1/FrameTime): {1 / single_bvh['Frame Time']}")
            # 4. Number of joints (including root)
            print(f"4. Number of joints (including root): {len(bvh_joint_info)}")
            # 5. List of all joint names
            print(f"5. List of all joint names: {bvh_joint_info}")
            
            global HAVE_BVH_BEEN_PLAYED
            HAVE_BVH_BEEN_PLAYED = 0

def drop_callback(window, paths):
    global single_bvh
    global RENDERING_MODE

    if len(paths) != 1:
        return

    path = paths[0]
    if not file_exists(path) or not (split_ext(path)[1] in (".bvh",)):
        print("파일이 존재하지 않거나 bvh 파일이 아닙니다.")
        return
    
    # if type(single_model) == futures.Future:
    #     single_model.cancel()
    
    # single_model = path
    # RENDERING_MODE = 0

    if type(single_bvh) == futures.Future:
        single_bvh.cancel()

    single_bvh = path
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

def getTranslatation(axis, v):
    M = np.eye(4)
    if axis == 'x':
        M[0, 3] = v
    elif axis == 'y':
        M[1, 3] = v
    elif axis == 'z':
        M[2, 3] = v
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

def drawCube(__varr = createVertexArraySeparate()):
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glNormalPointer(GL_FLOAT, 6*__varr.itemsize, __varr)
    glVertexPointer(3, GL_FLOAT, 6*__varr.itemsize, ctypes.c_void_p(__varr.ctypes.data + 3*__varr.itemsize))
    glDrawArrays(GL_TRIANGLES, 0, int(__varr.size/6))

def drawAnimationModel(t):
    global single_bvh
    bvh = single_bvh

    if type(bvh) != dict:
        return

    total_frames = bvh["Frames"]
    spf = bvh["Frame Time"]
    fps = 1 / spf
    total_secs = total_frames * spf
    t = t % total_secs
    frame_idx = int(t * fps)

    motion_data = bvh["MOTION"][frame_idx]

    channel_funcs = {
        'XPOSITION': lambda x: glTranslatef(x, 0, 0),
        'YPOSITION': lambda y: glTranslatef(0, y, 0),
        'ZPOSITION': lambda z: glTranslatef(0, 0, z),
        'XROTATION': lambda deg: glRotatef(deg, 1, 0, 0),
        'YROTATION': lambda deg: glRotatef(deg, 0, 1, 0),
        'ZROTATION': lambda deg: glRotatef(deg, 0, 0, 1),
    }

    mat_funcs = {
        'XPOSITION': lambda x: getTranslatation('x', x),
        'YPOSITION': lambda y: getTranslatation('y', y),
        'ZPOSITION': lambda z: getTranslatation('z', z),
        'XROTATION': lambda deg: getRotation('x', np.radians(deg)),
        'YROTATION': lambda deg: getRotation('y', np.radians(deg)),
        'ZROTATION': lambda deg: getRotation('z', np.radians(deg)),
    }

    root_key = list(bvh["children"])[0]

    def drawNode(node):
        glPushMatrix()

        if 'name' in node and node["name"] != root_key and 'OFFSET' in node:
            if RENDERING_TYPE == 0:
                glLineWidth(30.)
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(*(node['OFFSET']))
                glEnd()
                glLineWidth(1.)
            elif RENDERING_TYPE == 1:
                glPushMatrix()
                offset_v = node['OFFSET']
                
                s = np.linalg.norm(offset_v)
                ss = np.sqrt(1 + s**2)
                d = offset_v / s
                rotAxis = np.cross(np.array([0, 0, 1]), d)
                deg = np.degrees(np.arcsin(np.linalg.norm(rotAxis)))

                glTranslatef(*(offset_v / 2))
                glRotatef(deg, *rotAxis)
                glScalef(0.25 * s / ss, 0.25 * s / ss, 0.8 * s)

                # Draw a cube
                drawCube()
                glPopMatrix()
            elif RENDERING_TYPE == 2:
                glPointSize(20.)
                glBegin(GL_POINTS)
                glVertex3f(*node['OFFSET'])
                glEnd()
                glPointSize(1.)
            elif RENDERING_TYPE == 3:
                if "name" in node:
                    glPushMatrix()
                    offset_v = node['OFFSET']
                
                    s = np.linalg.norm(offset_v)
                    ss = np.sqrt(1 + s**2)
                    d = offset_v / s
                    rotAxis = np.cross(np.array([0, 0, 1]), d)
                    deg = np.degrees(np.arcsin(np.linalg.norm(rotAxis)))

                    glRotatef(deg, *rotAxis)
                    glRotatef(90, 1, 0, 0)

                    if node["name"] == "Head":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_head"])
                        glTranslatef(0, -0.1, 0)
                        drawParsedModel(multi_models["male_neck"])
                        glPopMatrix()
                    elif node["name"] == "Spine1":
                        glPushMatrix()
                        glScalef(5,5,5)
                        glTranslatef(0, -2, 0)
                        drawParsedModel(multi_models["male_spine"])
                        glPopMatrix()
                    elif node["name"] == "LeftArm":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_left_shoulder"])
                        glPopMatrix()
                    elif node["name"] == "RightArm":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_right_shoulder"])
                        glPopMatrix()
                    elif node["name"] == "LeftForeArm":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_left_forearm"])
                        glPopMatrix()
                    elif node["name"] == "RightForeArm":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_right_forearm"])
                        glPopMatrix()
                    elif node["name"] == "LeftHand":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_left_hand"])
                        glPopMatrix()
                    elif node["name"] == "RightHand":
                        glPushMatrix()
                        glScalef(5,5,5)
                        drawParsedModel(multi_models["male_right_hand"])
                        glPopMatrix()
                    elif node["name"] == "LeftLeg":
                        glPushMatrix()
                        glScalef(5,5,5)
                        glRotatef(180, 0, 1, 0)
                        glTranslatef(0, 1, 0)
                        glScalef(1, -1, 1)
                        glTranslatef(0, -1, 0)
                        drawParsedModel(multi_models["male_left_up_leg"])
                        glPopMatrix()
                    elif node["name"] == "LeftFoot":
                        glPushMatrix()
                        glScalef(5,5,5)
                        glRotatef(180, 0, 1, 0)
                        glTranslatef(0, 1, 0)
                        glScalef(1, -1, 1)
                        glTranslatef(0, -1, 0)
                        drawParsedModel(multi_models["male_left_leg"])
                        glPopMatrix()
                    elif node["name"] == "RightLeg":
                        glPushMatrix()
                        glScalef(5,5,5)
                        glRotatef(180, 0, 1, 0)
                        glTranslatef(0, 1, 0)
                        glScalef(1, -1, 1)
                        glTranslatef(0, -1, 0)
                        drawParsedModel(multi_models["male_right_up_leg"])
                        glPopMatrix()
                    elif node["name"] == "RightFoot":
                        glPushMatrix()
                        glScalef(5,5,5)
                        glRotatef(180, 0, 1, 0)
                        glTranslatef(0, 1, 0)
                        glScalef(1, -1, 1)
                        glTranslatef(0, -1, 0)
                        drawParsedModel(multi_models["male_right_leg"])
                        glPopMatrix()
                    glPopMatrix()
        
        if 'OFFSET' in node:
            glTranslatef(*node['OFFSET'])

        if HAVE_BVH_BEEN_PLAYED and 'CHANNELS' in node and len(node['CHANNELS']) > 0:
            for channel_name, channel_id in node['CHANNELS']:
                channel_funcs[channel_name](motion_data[channel_id])

        if 'children' in node and len(node['children']) > 0:
            for child_node_id in node['children']:
                drawNode(node['children'][child_node_id])
        glPopMatrix()

    glPushMatrix()
    drawSingleModel()
    drawNode(bvh["children"][root_key])
    glPopMatrix()

def smoothstep(x, min, max):
    t = np.clip(x, min, max)
    return t * t * (3.0 - 2.0 * t)

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
        glOrtho(-30, 30, -30, 30, -30 * aspect_ratio, 30)

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

        drawAnimationModel(gTime)
    else:
        # 3-2. predefined hierachical animation mode
        pass

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

            # Load bvh animation automatically
            load_bvh_animation(executer)

            # Render here, e.g. using pyOpenGL
            render()

            # Swap front and back buffers
            glfw.swap_buffers(window)
    
    glfw.terminate()


if __name__ == "__main__":
    main()
