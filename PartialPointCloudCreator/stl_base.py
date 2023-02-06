# Usage
# numpy
# numpy-stl

# https://codelabo.com/posts/20200228182137
# https://medium.com/@shintaroshiba/python-openglでディスプレイなしで3dレンダリング結果を保存-5ce9f56a7890
# https://metamost.com/opengl-with-python/
# https://www.metamost.com/opengl-with-python-pt2/

import numpy as np
import stl


# ----------------------------------------------------------------------


def updates(mesh):
    mesh.update_areas()
    mesh.update_max()
    mesh.update_min()
    mesh.update_normals()
    return mesh


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord=order, axis=axis, keepdims=True)
    l2[l2 == 0] = 1
    return v/l2


def normalize_model(mesh, z_swap):
    mid_pos_rel = (mesh.max_ - mesh.min_) / 2
    mesh.x = mesh.x - (mid_pos_rel[0] + mesh.min_[0])
    mesh.y = mesh.y - (mid_pos_rel[1] + mesh.min_[1])
    mesh.z = mesh.z - (mid_pos_rel[2] + mesh.min_[2])
    updates(mesh)

    max1 = np.amax(mesh.max_)
    max2 = np.amax(-mesh.min_)
    scale = max1 if max1 > max2 else max2
    mesh.x /= scale
    mesh.y /= scale
    mesh.z /= scale
    if z_swap:
        mesh.z *= -1
    updates(mesh)

    return mesh


def get_vertex_normal(points, face_normals, faces):
    vn = np.zeros((points.shape[0], 3), dtype=np.float32)
    for i in range(face_normals.shape[0]):
        vn[faces[i, 0]] += face_normals[i]
        vn[faces[i, 1]] += face_normals[i]
        vn[faces[i, 2]] += face_normals[i]
    normalize(vn, axis=1)
    return vn.astype(np.float32)


def load_stl(filename, z_swap=False):
    m = stl.mesh.Mesh.from_file(filename)
    # 正規化
    normalize_model(m, z_swap)
    points = m.points.reshape(-1, 3)
    faces = np.arange(points.shape[0]).reshape(-1, 3)
    # get normal of face
    normals = m.normals.reshape(-1, 3)
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    normals = normalize(normals, axis=1)
    print(points.shape[0])
    print(faces.shape[0])
    return points.astype(np.float32), normals.astype(np.float32), faces.astype(np.uint32)


def false_color_array(points_num):
    return np.ones((points_num, 4), dtype=np.float32)


def false_color_array2(normals):
    color3 = normals * 0.5 + 0.5
    color4 = np.append(color3, np.ones((color3.shape[0], 1)), axis=1)
    return color4.astype(np.float32)
