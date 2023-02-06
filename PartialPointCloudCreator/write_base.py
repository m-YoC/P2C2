

import numpy as np
import math

from logging import getLogger
logger = getLogger(__name__)


# ----------------------------------------------------------------------


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2 == 0] = 1
    return v/l2


# pz:depth-bufferの値, zn: z-near, zf: z-far
def get_vz(pz, zn, zf):
    return zf * zn / (pz * (zf - zn) - zf)


# sx: screen-x, vz: view-z
def get_vx(sx, vz, width, height, fov):
    return vz / height * (width - 2 * sx) * math.tan(math.radians(fov) * 0.5)


# sy: screen-y, vz: view-z
def get_vy(sy, vz, width, height, fov):
    return vz / height * (height - 2 * sy) * math.tan(math.radians(fov) * 0.5)


def get_xyz(image, width, height, fov, zn, zf, view_mat):
    res = []
    view_inv = np.linalg.inv(view_mat)
    for iy in range(height):
        for ix in range(width):
            if image[iy, ix] == 1:
                continue
            vz = get_vz(image[iy, ix], zn, zf)
            vx = get_vx(ix, vz, width, height, fov)
            vy = get_vy(iy, vz, width, height, fov)
            xyz = np.array([vx, vy, vz, 1]) @ view_inv
            res.append([xyz[0], xyz[1], xyz[2]])

    return res


def write_xyz_file(xyz_data, path):
    with open(path, mode='w') as f:
        for i in range(len(xyz_data)):
            f.write(str(xyz_data[i][0]) + ' ' + str(xyz_data[i][1]) + ' ' + str(xyz_data[i][2]) + '\n')


# 速度の為にエラー処理は書かない
# (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1)をチェックするので
# ix == width-1 or iy == height-1となるような値は使わないこと
def add_vertex_and_face(ix, iy, points, is_existence, indices, out_vertices, out_faces):
    def check_use_index(c0, c1, c2, c3):
        if c0 and indices[ix, iy] == -1:
            indices[ix, iy] = len(out_vertices)
            out_vertices.append([points[ix, iy, 0], points[ix, iy, 1], points[ix, iy, 2]])
        if c1 and indices[ix + 1, iy] == -1:
            indices[ix + 1, iy] = len(out_vertices)
            out_vertices.append([points[ix+1, iy, 0], points[ix+1, iy, 1], points[ix+1, iy, 2]])
        if c2 and indices[ix, iy + 1] == -1:
            indices[ix, iy + 1] = len(out_vertices)
            out_vertices.append([points[ix, iy+1, 0], points[ix, iy+1, 1], points[ix, iy+1, 2]])
        if c3 and indices[ix + 1, iy + 1] == -1:
            indices[ix + 1, iy + 1] = len(out_vertices)
            out_vertices.append([points[ix+1, iy+1, 0], points[ix+1, iy+1, 1], points[ix+1, iy+1, 2]])

    def get_indices(ix, iy):
        return indices[ix, iy], indices[ix + 1, iy], indices[ix, iy + 1], indices[ix + 1, iy + 1]

    if is_existence[ix + 1, iy] and is_existence[ix, iy + 1] and is_existence[ix + 1, iy + 1]:
        # 本当は三角形の貼り方が2種類あるので分けた方がいい
        check_use_index(True, True, True, True)
        iv0, iv1, iv2, iv3 = get_indices(ix, iy)
        out_faces.append([iv0, iv1, iv3])
        out_faces.append([iv0, iv3, iv2])
        return

    if is_existence[ix + 1, iy] and is_existence[ix, iy + 1] and not is_existence[ix + 1, iy + 1]:
        check_use_index(True, True, True, False)
        iv0, iv1, iv2, iv3 = get_indices(ix, iy)
        out_faces.append([iv0, iv1, iv2])
        return

    if is_existence[ix + 1, iy] and not is_existence[ix, iy + 1] and is_existence[ix + 1, iy + 1]:
        check_use_index(True, True, False, True)
        iv0, iv1, iv2, iv3 = get_indices(ix, iy)
        out_faces.append([iv0, iv1, iv3])
        return

    if not is_existence[ix + 1, iy] and is_existence[ix, iy + 1] and is_existence[ix + 1, iy + 1]:
        check_use_index(True, False, True, True)
        iv0, iv1, iv2, iv3 = get_indices(ix, iy)
        out_faces.append([iv0, iv3, iv2])
        return

    return


def get_vertex_and_face(image, width, height, fov, zn, zf, view_mat):
    # points[width, height, xyz]
    points = np.zeros((width, height, 3))
    # is_existence[width, height] True, False
    is_existence = np.zeros((width, height), dtype=bool)
    # indices[width, height] 0, 1, 2, ...
    indices = -np.ones((width, height), dtype=np.int32)

    view_inv = np.linalg.inv(view_mat)
    for iy in range(height):
        for ix in range(width):

            if image[iy, ix] == 1:
                points[ix, iy] = np.array([0., 0., 0.])
                is_existence[ix, iy] = False
                continue

            vz = get_vz(image[iy, ix], zn, zf)
            vx = get_vx(ix, vz, width, height, fov)
            vy = get_vy(iy, vz, width, height, fov)
            xyz = np.array([vx, vy, vz, 1]) @ view_inv
            points[ix, iy] = xyz[0:3]
            is_existence[ix, iy] = True

    vertices = []
    faces = []

    for iy in range(height-1):
        for ix in range(width-1):
            if not is_existence[ix, iy]:
                continue
            add_vertex_and_face(ix, iy, points, is_existence, indices, vertices, faces)

    return vertices, faces


def write_stl_file(vertex, face, path):
    with open(path, mode='w') as f:
        f.write('solid ' + 'p2c2data' + '\n')
        for i in range(len(face)):
            # 面の頂点の値を取得
            v0 = np.array(vertex[face[i][0]])
            v1 = np.array(vertex[face[i][1]])
            v2 = np.array(vertex[face[i][2]])
            # 法線
            normal = normalize(np.cross(v1 - v0, v2 - v0))
            f.write('facet normal ' + str(normal[0]) + ' ' + str(normal[1]) + ' ' + str(normal[2]) + '\n')
            f.write('outer loop' + '\n')
            f.write('vertex ' + str(v0[0]) + ' ' + str(v0[1]) + ' ' + str(v0[2]) + '\n')
            f.write('vertex ' + str(v1[0]) + ' ' + str(v1[1]) + ' ' + str(v1[2]) + '\n')
            f.write('vertex ' + str(v2[0]) + ' ' + str(v2[1]) + ' ' + str(v2[2]) + '\n')
            f.write('endloop' + '\n')
            f.write('endfacet' + '\n')
        f.write('endsolid ' + 'p2c2data' + '\n')

