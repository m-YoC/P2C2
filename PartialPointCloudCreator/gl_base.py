# Usage
# PyOpenGL
# glfw
# numpy

# https://codelabo.com/posts/20200228182137
# https://medium.com/@shintaroshiba/python-openglでディスプレイなしで3dレンダリング結果を保存-5ce9f56a7890
# https://metamost.com/opengl-with-python/
# https://www.metamost.com/opengl-with-python-pt2/

import contextlib
import sys

# OpenGLとGLFWをインポートします
from OpenGL import GL as gl
import glfw

import numpy as np
import math

from logging import getLogger
logger = getLogger(__name__)


# ----------------------------------------------------------------------


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2 == 0] = 1
    return v/l2


# aspect = width / height
class Perspective:
    def __init__(self, fov, aspect, z_near, z_far):
        self.fov = fov
        self.aspect = aspect
        self.z_near = z_near
        self.z_far = z_far

    def matrix(self):
        zn = self.z_near
        zf = self.z_far
        top = np.tan(math.radians(self.fov) / 2) * self.z_near
        # bottom = -top
        right = top * self.aspect
        # left = -top * aspect
        projection = np.array([[zn / right,        0,                        0,  0],
                               [         0, zn / top,                        0,  0],
                               [         0,        0,   -(zf + zn) / (zf - zn), -1],
                               [         0,        0, -2 * zf * zn / (zf - zn),  0]])
        return projection


class LookAt:
    def __init__(self, eye, target, up):
        self.eye = eye
        self.target = target
        self.up = up

    def matrix(self):
        zax = normalize(self.eye - self.target)
        xax = normalize(np.cross(self.up, zax))
        yax = np.cross(zax, xax)
        x = -xax.dot(self.eye)
        y = -yax.dot(self.eye)
        z = -zax.dot(self.eye)
        view = np.array([[xax[0], yax[0], zax[0], 0],
                         [xax[1], yax[1], zax[1], 0],
                         [xax[2], yax[2], zax[2], 0],
                         [     x,      y,      z, 1]])
        return view


class Transform:
    def __init__(self, scale, angle, axis, trans):
        self.scale = scale
        self.angle = angle
        self.axis = axis
        self.trans = trans

    def __rot(self):
        c = math.cos(math.radians(self.angle))
        s = math.sin(math.radians(self.angle))
        x = self.axis[0]
        y = self.axis[1]
        z = self.axis[2]
        rotmat = np.array([[    c + x * x * (1 - c), x * y * (1 - c) + z * s, x * z * (1 - c) - y * s],
                           [x * y * (1 - c) - z * s,     c + y * y * (1 - c), y * z * (1 - c) + x * s],
                           [x * z * (1 - c) + y * s, y * z * (1 - c) - x * s,     c + z * z * (1 - c)]])
        return rotmat

    def matrix(self):
        mat = np.identity(4)
        mat[0:3, 0:3] = (self.scale * np.identity(3)) @ self.__rot()
        mat[3, 0:3] = self.trans
        return mat


def create_mvp_id(program_id):
    matrix_id = gl.glGetUniformLocation(program_id, 'MVP')
    return matrix_id


def create_mvp(projection, view, model):
    mvp = model @ view @ projection
    return mvp.astype(np.float32)


# ----------------------------------------------------------------------


def check_shader_compilation(shader_id):
    # check if compilation was successful
    result = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
    info_log_len = gl.glGetShaderiv(shader_id, gl.GL_INFO_LOG_LENGTH)
    if info_log_len:
        logger.error(gl.glGetShaderInfoLog(shader_id))
        sys.exit(10)


def check_program_linking(program_id):
    # check if linking was successful
    result = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
    info_log_len = gl.glGetProgramiv(program_id, gl.GL_INFO_LOG_LENGTH)
    if info_log_len:
        logger.error(gl.glGetProgramInfoLog(program_id))
        sys.exit(11)


@contextlib.contextmanager
def load_shaders(vertex_shader_file, fragment_shader_file):
    # シェーダーファイルからソースコードを読み込む
    with open(vertex_shader_file, 'r', encoding='utf-8') as f:
        vertex_shader_src = f.read()

    with open(fragment_shader_file, 'r', encoding='utf-8') as f:
        fragment_shader_src = f.read()

    shader_ids = []
    program_id = gl.glCreateProgram()
    try:
        # 作成したシェーダオブジェクトにソースコードを渡しコンパイルする
        vertex_shader_id = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertex_shader_id, vertex_shader_src)
        gl.glCompileShader(vertex_shader_id)
        # チェック
        check_shader_compilation(vertex_shader_id)
        # programに紐付け
        gl.glAttachShader(program_id, vertex_shader_id)
        shader_ids.append(vertex_shader_id)

        # 作成したシェーダオブジェクトにソースコードを渡しコンパイルする
        fragment_shader_id = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragment_shader_id, fragment_shader_src)
        gl.glCompileShader(fragment_shader_id)
        # チェック
        check_shader_compilation(fragment_shader_id)
        # programに紐付け
        gl.glAttachShader(program_id, fragment_shader_id)
        shader_ids.append(fragment_shader_id)

        # 作成したプログラムオブジェクトをリンク
        gl.glLinkProgram(program_id)

        check_program_linking(program_id)
        gl.glUseProgram(program_id)
        yield program_id
    finally:
        gl.glDetachShader(program_id, shader_ids[0])
        gl.glDeleteShader(shader_ids[0])
        gl.glDetachShader(program_id, shader_ids[1])
        gl.glDeleteShader(shader_ids[1])
        gl.glUseProgram(0)
        gl.glDeleteProgram(program_id)


# vao: vertex_array_object
@contextlib.contextmanager
def create_vao():
    # VAOを作成してバインド
    vao_id = gl.glGenVertexArrays(1)
    try:
        gl.glBindVertexArray(vao_id)
        yield
    finally:
        gl.glDeleteVertexArrays(1, [vao_id])


# vbo: vertex_buffer_object
@contextlib.contextmanager
def create_vbo(attr_id, vertex_data, dim):
    try:
        # 頂点バッファオブジェクトを作成してデータをGPU側に送る
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)
        # 頂点バッファオブジェクトの位置を指定
        # attribute, dim, type, use_normalized, stride, array_buffer_offset
        gl.glVertexAttribPointer(attr_id,  dim, gl.GL_FLOAT, False, 0, None)

        # アトリビュート変数を有効化
        gl.glEnableVertexAttribArray(attr_id)
        yield
    finally:
        gl.glDisableVertexAttribArray(attr_id)
        gl.glDeleteBuffers(1, [vbo])


@contextlib.contextmanager
def create_index_object(index_data):
    try:
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data, gl.GL_STATIC_DRAW)
        yield
    finally:
        gl.glDeleteBuffers(1, [vbo])


# ----------------------------------------------------------------------


@contextlib.contextmanager
def create_window(width, height, title):
    # GLFW初期化
    if not glfw.init():
        sys.exit(1)

    # ウィンドウ不可視状態
    # glfw.window_hint(glfw.VISIBLE, False)

    try:
        # OpenGLのバージョン指定
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        # windowのリサイズの禁止
        glfw.window_hint(glfw.RESIZABLE, False)

        # ウィンドウを作成
        window = glfw.create_window(width, height, title, None, None)
        if not window:
            print('Failed to create window')
            sys.exit(2)

        # コンテキストの作成
        glfw.make_context_current(window)
        # Keyの有効化
        glfw.set_input_mode(window, glfw.STICKY_KEYS, True)

        yield window
    finally:
        glfw.terminate()


def init_gl(background_color):
    # バッファを指定色で初期化
    gl.glClearColor(background_color[0], background_color[1], background_color[2], background_color[3])
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glCullFace(gl.GL_BACK)


def print_gl_version():
    # OpenGLのバージョン等を表示します
    print('Vendor :', gl.glGetString(gl.GL_VENDOR))
    print('GPU :', gl.glGetString(gl.GL_RENDERER))
    print('OpenGL version :', gl.glGetString(gl.GL_VERSION))


def triangle_position():
    return np.array([[-1, -1, 0],
                     [ 1, -1, 0],
                     [ 0,  1, 0]], dtype=np.float32)


def triangle_color():
    return np.array([[1.0, 0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)


def triangle_index():
    return np.array([[0, 1, 2]], dtype=np.uint32)

