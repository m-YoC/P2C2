
import contextlib
import sys
from abc import ABCMeta, abstractmethod

# OpenGLとGLFWをインポートします
from OpenGL import GL as gl
import glfw
import cv2

import numpy as np
import math
import stl_base as stlb
import gl_base as glb
import write_base as wb

from logging import getLogger
logger = getLogger(__name__)


# ----------------------------------------------------------------------


class ControlBase(metaclass=ABCMeta):
    # 戻り値はbool値(main_loopを続けるかどうか)
    @abstractmethod
    def do(self, window, meta):
        pass

    @staticmethod
    def set_angle_degree(base, add):
        return (base + add + 360) % 360

    @staticmethod
    def save_snapshot(meta, save_name):
        print('start save snapshot')
        image_buffer = gl.glReadPixels(0, 0, meta['width'], meta['height'],
                                       gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(meta['height'], meta['width'], 3)
        cv2.imwrite(save_name, cv2.flip(image, 0))
        print('finish save snapshot')

    @staticmethod
    def save_depth_to_xyz(meta, save_name):
        print('start save depth')
        image_buffer = gl.glReadPixels(0, 0, meta['width'], meta['height'],
                                       gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        image = np.frombuffer(image_buffer, dtype=np.float32).reshape(meta['height'], meta['width'])
        xyz = wb.get_xyz(image, meta['width'], meta['height'],
                         meta['projection'].fov, meta['projection'].z_near, meta['projection'].z_far,
                         meta['view'].matrix())
        wb.write_xyz_file(xyz, save_name)
        print('finish save depth')

    @staticmethod
    def save_depth_to_stl(meta, save_name):
        print('start save depth')
        image_buffer = gl.glReadPixels(0, 0, meta['width'], meta['height'],
                                       gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        image = np.frombuffer(image_buffer, dtype=np.float32).reshape(meta['height'], meta['width'])
        v, f = wb.get_vertex_and_face(image, meta['width'], meta['height'],
                                      meta['projection'].fov, meta['projection'].z_near, meta['projection'].z_far,
                                      meta['view'].matrix())
        print('get vertices and faces...')
        wb.write_stl_file(v, f, save_name)
        print('writing...')
        print('finish save depth')


class ControlManual(ControlBase):
    def __init__(self, d_angle):
        self.__lock_key = 0
        self.d_angle = d_angle

    def do(self, window, meta):
        # z keyで右回り, x keyで左回り
        # s keyで保存

        # イベント待機
        glfw.wait_events()

        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS and self.__lock_key == 0:
            self.__lock_key = 's'
            name_base = meta['save_path'] + meta['save_name_func'](meta['model'].angle)
            self.save_depth_to_xyz(meta, name_base + '.xyz') if not meta['save_stl_mode'] else self.save_depth_to_stl(meta, name_base + '.stl')
            if meta['save_snapshot'] : self.save_snapshot(meta, name_base + '_ss.png')

        if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS and self.__lock_key == 0:
            self.__lock_key = 'z'
            meta['model'].angle = self.set_angle_degree(meta['model'].angle, -self.d_angle)
            print(meta['model'].angle)
        if glfw.get_key(window, glfw.KEY_X) == glfw.PRESS and self.__lock_key == 0:
            self.__lock_key = 'x'
            meta['model'].angle = self.set_angle_degree(meta['model'].angle, +self.d_angle)
            print(meta['model'].angle)

        if glfw.get_key(window, glfw.KEY_S) == glfw.RELEASE and self.__lock_key == 's':
            self.__lock_key = 0
        if glfw.get_key(window, glfw.KEY_Z) == glfw.RELEASE and self.__lock_key == 'z':
            self.__lock_key = 0
        if glfw.get_key(window, glfw.KEY_X) == glfw.RELEASE and self.__lock_key == 'x':
            self.__lock_key = 0

        return False


class ControlAuto(ControlBase):
    def __init__(self, d_angle, repeat):
        self.__counter = 0
        self.d_angle = d_angle
        self.repeat = repeat

    def do(self, window, meta):
        # 回転とセーブを自動で繰り返す

        glfw.poll_events()

        name_base = meta['save_path'] + meta['save_name_func'](meta['model'].angle)
        self.save_depth_to_xyz(meta, name_base + '.xyz') if not meta['save_stl_mode'] else self.save_depth_to_stl(meta, name_base + '.stl')
        if meta['save_snapshot'] : self.save_snapshot(meta, name_base + '_ss.png')
        meta['model'].angle = self.set_angle_degree(meta['model'].angle, self.d_angle)
        print(meta['model'].angle)

        self.__counter += 1
        return True if self.__counter == self.repeat else False


def run(meta):

    # Matrixの準備
    projection = meta['projection'].matrix()
    view = meta['view'].matrix()
    # 初期位置の調整用
    model_init = meta['ini_model'].matrix()

    # load stl
    position, normal, face = stlb.load_stl(meta['load_name'])
    color = stlb.false_color_array2(stlb.get_vertex_normal(position, normal, face))

    with glb.create_window(meta['width'], meta['height'], 'P2C2 - Partial PointCloud Creator') as window:
        with glb.create_vao():
            with glb.create_vbo(0, position, 3), glb.create_vbo(1, color, 4), \
                 glb.create_index_object(face), glb.load_shaders('shader.vert', 'shader.frag') as program_id:

                # matrix用領域の準備
                mvp_matrix_id = glb.create_mvp_id(program_id)
                while glfw.get_key(window, glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):

                    # 初期化
                    glb.init_gl([0.1, 0.1, 0.1, 1])
                    # Model変換行列の生成
                    model = meta['model'].matrix()
                    # GPUに行列をロード
                    gl.glUniformMatrix4fv(mvp_matrix_id, 1, False, glb.create_mvp(projection, view, model_init @ model))
                    # バインドしたVAOを用いて描画
                    gl.glDrawElements(gl.GL_TRIANGLES, 3*position.shape[0], gl.GL_UNSIGNED_INT, None)
                    # バッファを入れ替えて画面を更新
                    glfw.swap_buffers(window)

                    if meta['control'].do(window, meta):
                        break
    return


# ----------------------------------------------------------------------

def create_meta_data(load_name, **kwargs):
    angle_degree = kwargs.get('angle_degree', 20)
    repeat = kwargs.get('repeat', 18)

    meta_data = {'load_name': load_name, # 読み込むstlファイルのパス
                 'save_path': './result/', # 保存ディレクトリ
                 'save_name_func': lambda i: 'pc_' + str(i).zfill(3), # 保存ファイル名. 引数に角度(deg)が入る
                 'save_snapshot': False, # スナップショットを保存するか
                 'save_stl_mode': True, # stl形式で保存するか
                 'width': 640, # ウィンドウの横幅. この大きさが最終的な点群の密度にも影響する
                 'height': 480, # ウィンドウの縦幅. この大きさが最終的な点群の密度にも影響する
                 'projection': glb.Perspective(60, 0, 0.1, 100), # projection matrix
                 'view': glb.LookAt(np.array([0, 1, 3]), np.array([0, 0, 0]), np.array([0, 1, 0])), # view matrix
                 'ini_model': glb.Transform(1, -10, np.array([1, 0, 0]), np.array([0, 0, 0])), # モデルの初期状態の調整用
                 'model': glb.Transform(1, 0, np.array([0, 1, 0]), np.array([0, 0, 0])), # model matrix. モデルの状態更新に使用される
                 'control': ControlAuto(angle_degree, repeat)} # 動作を定義するクラス

    meta_data['projection'].aspect = meta_data['width'] / meta_data['height']

    return meta_data

