

import numpy as np
import math

from logging import getLogger
logger = getLogger(__name__)


_is_good_triangle_thr = 1.15

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
    def dist(ax, ay, bx, by):
        return math.sqrt(math.fsum( [(points[ax, ay, i] - points[bx, by, i]) ** 2.0 for i in range(0, 3)] ))

    def is_good_triangle(iix, iiy, ijx, ijy, x=ix, y=iy):
        if not is_existence[iix, iiy] or not is_existence[ijx, ijy] or not is_existence[x, y]: return False

        ab = dist(x, y, iix, iiy)
        ac = dist(x, y, ijx, ijy)
        bc = dist(iix, iiy, ijx, ijy)
        max_d = max(ab, ac, bc)
        min_d = min(ab, ac, bc)
        sum_d = math.fsum([ab, ac, bc])
        return (sum_d - max_d) / max_d > _is_good_triangle_thr # and max_d / min_d < 3.0

    def append_point_indices_and_vertices(c0, c1, c2, c3):
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

    def get_indices():
        return indices[ix, iy], indices[ix + 1, iy], indices[ix, iy + 1], indices[ix + 1, iy + 1]

    if is_good_triangle(ix + 1, iy, ix + 1, iy + 1) and is_good_triangle(ix, iy + 1, ix + 1, iy + 1):
        append_point_indices_and_vertices(True, True, True, True)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv0, iv1, iv3])
        out_faces.append([iv0, iv3, iv2])
        return

    if is_good_triangle(ix + 1, iy, ix, iy + 1) and is_good_triangle(ix + 1, iy, ix, iy + 1, ix + 1, iy + 1):
        append_point_indices_and_vertices(True, True, True, True)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv0, iv1, iv2])
        out_faces.append([iv1, iv3, iv2])
        return

    if is_good_triangle(ix + 1, iy, ix, iy + 1):
        append_point_indices_and_vertices(True, True, True, False)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv0, iv1, iv2])
        return

    if is_good_triangle(ix + 1, iy, ix + 1, iy + 1):
        append_point_indices_and_vertices(True, True, False, True)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv0, iv1, iv3])
        return

    if is_good_triangle(ix, iy + 1, ix + 1, iy + 1):
        append_point_indices_and_vertices(True, False, True, True)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv0, iv3, iv2])
        return

    if is_good_triangle(ix + 1, iy, ix, iy + 1, ix + 1, iy + 1):
        append_point_indices_and_vertices(False, True, True, True)
        iv0, iv1, iv2, iv3 = get_indices()
        out_faces.append([iv1, iv3, iv2])
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

