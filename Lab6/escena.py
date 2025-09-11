import pygame
import sys
import numpy as np
from gl import Renderer, TRIANGLES, barycentric, sample_texture_nearest
from model import Model
from shaders import vertexShader, radioactive_fragment_shader, copperFragmentShader, bwFragmentShader
from BMP_Writer import GenerateBMP

BACKGROUND_PATH = "background.bmp"
OBJ_PATH1 = "hola.obj"
TEXTURE_PATH1 = "holaBMP.bmp"
OBJ_PATH2 = "among.obj"
TEXTURE_PATH2 = "among_t.bmp"
OBJ_PATH3 = "charlie2.obj"
TEXTURE_PATH3 = "charlie_t.bmp"
OBJ_PATH4 = "pim.obj"
TEXTURE_PATH4 = "pim_t.bmp"
OBJ_PATH5 = "DJ.obj"
TEXTURE_PATH5 = "DJ_t.bmp"
WIDTH, HEIGHT = 800, 600

def normalize(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def compute_face_normal(A, B, C):
    A3 = np.array([A[0], A[1], A[2]], dtype=float)
    B3 = np.array([B[0], B[1], B[2]], dtype=float)
    C3 = np.array([C[0], C[1], C[2]], dtype=float)
    u = B3 - A3
    v = C3 - A3
    n = np.cross(u, v)
    norm = np.linalg.norm(n)
    if norm == 0:
        return np.array([0, 0, 1], dtype=float)
    return n / norm

def parse_obj_with_uv(path):
    verts = []
    texcoords = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append(tuple(map(float, parts[1:4])))
            elif parts[0] == "vt":
                texcoords.append(tuple(map(float, parts[1:3])))
            elif parts[0] == "f":
                face = []
                for p in parts[1:]:
                    comps = p.split("/")
                    v_idx = int(comps[0]) - 1
                    vt_idx = int(comps[1]) - 1 if len(comps) > 1 and comps[1] != "" else None
                    face.append((v_idx, vt_idx))
                if len(face) == 3:
                    faces.append(tuple(face))
                elif len(face) > 3:
                    for i in range(1, len(face)-1):
                        faces.append((face[0], face[i], face[i+1]))
    return verts, texcoords, faces

def transform_with_model_matrix(pos, model_matrix):
    vec = np.array([pos[0], pos[1], pos[2], 1.0], dtype=float).reshape(4,1)
    transformed = model_matrix @ vec
    transformed = np.asarray(transformed).flatten()
    w = transformed[3] if abs(transformed[3]) > 1e-6 else 1.0
    x = transformed[0] / w
    y = transformed[1] / w
    z = transformed[2] / w
    return [x, y, z]

def project(p):
    x, y, z = p
    factor = 1 / (1 + z * 0.001)
    return [x * factor, y * factor, z]

def blend(src_rgb, dst_rgb, alpha):
    src = np.array(src_rgb, dtype=float)
    dst = np.array(dst_rgb, dtype=float)
    return (alpha * src + (1 - alpha) * dst).tolist()

def main():
    # Inicializar pygame y ventana antes de cargar cualquier textura
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smiling Friends")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    # --- Modelo DJ ---
    texture5 = pygame.image.load(TEXTURE_PATH5).convert()
    verts5, texcoords5, faces5 = parse_obj_with_uv(OBJ_PATH5)
    vertex_normals5 = [np.array([0.0,0.0,0.0], dtype=float) for _ in verts5]
    for face in faces5:
        i0, _ = face[0]
        i1, _ = face[1]
        i2, _ = face[2]
        v0 = np.array(verts5[i0], dtype=float)
        v1 = np.array(verts5[i1], dtype=float)
        v2 = np.array(verts5[i2], dtype=float)
        u = v1 - v0
        v_ = v2 - v0
        fn = np.cross(u, v_)
        if np.linalg.norm(fn) != 0:
            fn = fn / np.linalg.norm(fn)
        vertex_normals5[i0] += fn
        vertex_normals5[i1] += fn
        vertex_normals5[i2] += fn
    vertex_normals5 = [normalize(n) for n in vertex_normals5]

    xs_dj = [v[0] for v in verts5]
    ys_dj = [v[1] for v in verts5]
    minX_dj, maxX_dj = min(xs_dj), max(xs_dj)
    minY_dj, maxY_dj = min(ys_dj), max(ys_dj)
    objW_dj = maxX_dj - minX_dj if maxX_dj - minX_dj != 0 else 1
    objH_dj = maxY_dj - minY_dj if maxY_dj - minY_dj != 0 else 1
    scaleFactor_dj = min(WIDTH, HEIGHT) * 0.3 / max(objW_dj, objH_dj)
    centerX_dj = (minX_dj + maxX_dj) / 2
    centerY_dj = (minY_dj + maxY_dj) / 2
    angle_y_dj = np.radians(180)
    cos_ay_dj = np.cos(angle_y_dj)
    sin_ay_dj = np.sin(angle_y_dj)
    rotation_y_dj = np.array([
        [cos_ay_dj, 0, sin_ay_dj, 0],
        [0,      1, 0,      0],
        [-sin_ay_dj,0, cos_ay_dj, 0],
        [0,      0, 0,      1]
    ])
    model_matrix_dj = np.identity(4)
    model_matrix_dj[0][0] = scaleFactor_dj
    model_matrix_dj[1][1] = scaleFactor_dj
    model_matrix_dj[2][2] = scaleFactor_dj
    offset_x_dj = -240  
    offset_y_dj = -180  
    model_matrix_dj[0][3] = WIDTH/2 - centerX_dj * scaleFactor_dj + offset_x_dj
    model_matrix_dj[1][3] = HEIGHT/2 - centerY_dj * scaleFactor_dj + offset_y_dj
    model_matrix_dj = model_matrix_dj @ rotation_y_dj

    # --- Modelo pim ---
    texture4 = pygame.image.load(TEXTURE_PATH4).convert()
    verts4, texcoords4, faces4 = parse_obj_with_uv(OBJ_PATH4)
    vertex_normals4 = [np.array([0.0,0.0,0.0], dtype=float) for _ in verts4]
    for face in faces4:
        i0, _ = face[0]
        i1, _ = face[1]
        i2, _ = face[2]
        v0 = np.array(verts4[i0], dtype=float)
        v1 = np.array(verts4[i1], dtype=float)
        v2 = np.array(verts4[i2], dtype=float)
        u = v1 - v0
        v_ = v2 - v0
        fn = np.cross(u, v_)
        if np.linalg.norm(fn) != 0:
            fn = fn / np.linalg.norm(fn)
        vertex_normals4[i0] += fn
        vertex_normals4[i1] += fn
        vertex_normals4[i2] += fn
    vertex_normals4 = [normalize(n) for n in vertex_normals4]

    xs_pim = [v[0] for v in verts4]
    ys_pim = [v[1] for v in verts4]
    minX_pim, maxX_pim = min(xs_pim), max(xs_pim)
    minY_pim, maxY_pim = min(ys_pim), max(ys_pim)
    objW_pim = maxX_pim - minX_pim if maxX_pim - minX_pim != 0 else 1
    objH_pim = maxY_pim - minY_pim if maxY_pim - minY_pim != 0 else 1
    scaleFactor_pim = min(WIDTH, HEIGHT) * 0.05 / max(objW_pim, objH_pim)
    centerX_pim = (minX_pim + maxX_pim) / 2
    centerY_pim = (minY_pim + maxY_pim) / 2
    angle_y_pim = np.radians(180)
    cos_ay_pim = np.cos(angle_y_pim)
    sin_ay_pim = np.sin(angle_y_pim)
    rotation_y_pim = np.array([
        [cos_ay_pim, 0, sin_ay_pim, 0],
        [0,      1, 0,      0],
        [-sin_ay_pim,0, cos_ay_pim, 0],
        [0,      0, 0,      1]
    ])
    model_matrix_pim = np.identity(4)
    model_matrix_pim[0][0] = scaleFactor_pim
    model_matrix_pim[1][1] = scaleFactor_pim
    model_matrix_pim[2][2] = scaleFactor_pim
    offset_x_pim = -190  
    offset_y_pim = -100   
    model_matrix_pim[0][3] = WIDTH/2 - centerX_pim * scaleFactor_pim + offset_x_pim
    model_matrix_pim[1][3] = HEIGHT/2 - centerY_pim * scaleFactor_pim + offset_y_pim
    model_matrix_pim = model_matrix_pim @ rotation_y_pim

    # --- Modelo among ---
    texture2 = pygame.image.load(TEXTURE_PATH2).convert()
    verts2, texcoords2, faces2 = parse_obj_with_uv(OBJ_PATH2)
    vertex_normals2 = [np.array([0.0,0.0,0.0], dtype=float) for _ in verts2]
    for face in faces2:
        i0, _ = face[0]
        i1, _ = face[1]
        i2, _ = face[2]
        v0 = np.array(verts2[i0], dtype=float)
        v1 = np.array(verts2[i1], dtype=float)
        v2 = np.array(verts2[i2], dtype=float)
        u = v1 - v0
        v_ = v2 - v0
        fn = np.cross(u, v_)
        if np.linalg.norm(fn) != 0:
            fn = fn / np.linalg.norm(fn)
        vertex_normals2[i0] += fn
        vertex_normals2[i1] += fn
        vertex_normals2[i2] += fn
    vertex_normals2 = [normalize(n) for n in vertex_normals2]

    xs_among = [v[0] for v in verts2]
    ys_among = [v[1] for v in verts2]
    minX_among, maxX_among = min(xs_among), max(xs_among)
    minY_among, maxY_among = min(ys_among), max(ys_among)
    objW_among = maxX_among - minX_among if maxX_among - minX_among != 0 else 1
    objH_among = maxY_among - minY_among if maxY_among - minY_among != 0 else 1
    scaleFactor_among = min(WIDTH, HEIGHT) * 0.15 / max(objW_among, objH_among)
    centerX_among = (minX_among + maxX_among) / 2
    centerY_among = (minY_among + maxY_among) / 2
    angle_y_among = np.radians(180)
    cos_ay_among = np.cos(angle_y_among)
    sin_ay_among = np.sin(angle_y_among)
    rotation_y_among = np.array([
        [cos_ay_among, 0, sin_ay_among, 0],
        [0,      1, 0,      0],
        [-sin_ay_among,0, cos_ay_among, 0],
        [0,      0, 0,      1]
    ])
    model_matrix_among = np.identity(4)
    model_matrix_among[0][0] = scaleFactor_among
    model_matrix_among[1][1] = scaleFactor_among
    model_matrix_among[2][2] = scaleFactor_among
    offset_x_among = 20  
    offset_y_among = -50
    model_matrix_among[0][3] = WIDTH/2 - centerX_among * scaleFactor_among + offset_x_among
    model_matrix_among[1][3] = HEIGHT/2 - centerY_among * scaleFactor_among + offset_y_among
    model_matrix_among = model_matrix_among @ rotation_y_among
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smiling Friends")
    clock = pygame.time.Clock()
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)


    # --- Modelo charlie ---
    texture3 = pygame.image.load(TEXTURE_PATH3).convert()
    verts3, texcoords3, faces3 = parse_obj_with_uv(OBJ_PATH3)
    vertex_normals3 = [np.array([0.0,0.0,0.0], dtype=float) for _ in verts3]
    for face in faces3:
        i0, _ = face[0]
        i1, _ = face[1]
        i2, _ = face[2]
        v0 = np.array(verts3[i0], dtype=float)
        v1 = np.array(verts3[i1], dtype=float)
        v2 = np.array(verts3[i2], dtype=float)
        u = v1 - v0
        v_ = v2 - v0
        fn = np.cross(u, v_)
        if np.linalg.norm(fn) != 0:
            fn = fn / np.linalg.norm(fn)
        vertex_normals3[i0] += fn
        vertex_normals3[i1] += fn
        vertex_normals3[i2] += fn
    vertex_normals3 = [normalize(n) for n in vertex_normals3]

    xs_charlie = [v[0] for v in verts3]
    ys_charlie = [v[1] for v in verts3]
    minX_charlie, maxX_charlie = min(xs_charlie), max(xs_charlie)
    minY_charlie, maxY_charlie = min(ys_charlie), max(ys_charlie)
    objW_charlie = maxX_charlie - minX_charlie if maxX_charlie - minX_charlie != 0 else 1
    objH_charlie = maxY_charlie - minY_charlie if maxY_charlie - minY_charlie != 0 else 1
    scaleFactor_charlie = min(WIDTH, HEIGHT) * 0.1 / max(objW_charlie, objH_charlie)
    centerX_charlie = (minX_charlie + maxX_charlie) / 2
    centerY_charlie = (minY_charlie + maxY_charlie) / 2
    angle_y_charlie = np.radians(90)
    cos_ay_charlie = np.cos(angle_y_charlie)
    sin_ay_charlie = np.sin(angle_y_charlie)
    rotation_y_charlie = np.array([
        [cos_ay_charlie, 0, sin_ay_charlie, 0],
        [0,      1, 0,      0],
        [-sin_ay_charlie,0, cos_ay_charlie, 0],
        [0,      0, 0,      1]
    ])
    model_matrix_charlie = np.identity(4)
    model_matrix_charlie[0][0] = scaleFactor_charlie
    model_matrix_charlie[1][1] = scaleFactor_charlie
    model_matrix_charlie[2][2] = scaleFactor_charlie
    offset_x_charlie = -175  
    offset_y_charlie = -80    
    model_matrix_charlie[0][3] = WIDTH/2 - centerX_charlie * scaleFactor_charlie + offset_x_charlie
    model_matrix_charlie[1][3] = HEIGHT/2 - centerY_charlie * scaleFactor_charlie + offset_y_charlie
    model_matrix_charlie = model_matrix_charlie @ rotation_y_charlie

    background = pygame.image.load(BACKGROUND_PATH).convert()
    rend = Renderer(screen)
    texture = pygame.image.load(TEXTURE_PATH1).convert()
    verts, texcoords, faces = parse_obj_with_uv(OBJ_PATH1)

    # Calcular normales de vértices
    vertex_normals = [np.array([0.0,0.0,0.0], dtype=float) for _ in verts]
    for face in faces:
        i0, _ = face[0]
        i1, _ = face[1]
        i2, _ = face[2]
        v0 = np.array(verts[i0], dtype=float)
        v1 = np.array(verts[i1], dtype=float)
        v2 = np.array(verts[i2], dtype=float)
        u = v1 - v0
        v_ = v2 - v0
        fn = np.cross(u, v_)
        if np.linalg.norm(fn) != 0:
            fn = fn / np.linalg.norm(fn)
        vertex_normals[i0] += fn
        vertex_normals[i1] += fn
        vertex_normals[i2] += fn
    vertex_normals = [normalize(n) for n in vertex_normals]

    rend.glBindTexture(texture)

    # Matriz de transformación solo para el modelo de hola.obj
    xs_hola = [v[0] for v in verts]
    ys_hola = [v[1] for v in verts]
    minX_hola, maxX_hola = min(xs_hola), max(xs_hola)
    minY_hola, maxY_hola = min(ys_hola), max(ys_hola)
    objW_hola = maxX_hola - minX_hola if maxX_hola - minX_hola != 0 else 1
    objH_hola = maxY_hola - minY_hola if maxY_hola - minY_hola != 0 else 1
    scaleFactor_hola = min(WIDTH, HEIGHT) * 0.35 / max(objW_hola, objH_hola)  
    centerX_hola = (minX_hola + maxX_hola) / 2
    centerY_hola = (minY_hola + maxY_hola) / 2

    angle_y = np.radians(270)  
    cos_ay = np.cos(angle_y)
    sin_ay = np.sin(angle_y)
    rotation_y = np.array([
        [cos_ay, 0, sin_ay, 0],
        [0,      1, 0,      0],
        [-sin_ay,0, cos_ay, 0],
        [0,      0, 0,      1]
    ])

    model_matrix_hola = np.identity(4)
    model_matrix_hola[0][0] = scaleFactor_hola
    model_matrix_hola[1][1] = scaleFactor_hola
    model_matrix_hola[2][2] = scaleFactor_hola
    offset_x = 100
    offset_y = -175
    model_matrix_hola[0][3] = WIDTH/2 - centerX_hola * scaleFactor_hola + offset_x
    model_matrix_hola[1][3] = HEIGHT/2 - centerY_hola * scaleFactor_hola + offset_y

    model_matrix_hola = model_matrix_hola @ rotation_y

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        rend.glDrawBackground(background)

        for face in faces5:
            idx0, vt0 = face[0]
            idx1, vt1 = face[1]
            idx2, vt2 = face[2]

            n0 = vertex_normals5[idx0]
            n1 = vertex_normals5[idx1]
            n2 = vertex_normals5[idx2]

            A = vertexShader(verts5[idx0], modelMatrix=model_matrix_dj, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            B = vertexShader(verts5[idx1], modelMatrix=model_matrix_dj, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            C = vertexShader(verts5[idx2], modelMatrix=model_matrix_dj, viewMatrix=np.identity(4), projMatrix=np.identity(4))

            def unpack5(vs, vt_idx):
                x, y, z = vs[0], vs[1], vs[2]
                u, v = (0, 0)
                if vt_idx is not None and vt_idx < len(texcoords5):
                    u, v = texcoords5[vt_idx]
                return [x, y, z, u, v]

            A_uv = unpack5(A, vt0)
            B_uv = unpack5(B, vt1)
            C_uv = unpack5(C, vt2)

            xs5 = [A_uv[0], B_uv[0], C_uv[0]]
            ys5 = [A_uv[1], B_uv[1], C_uv[1]]
            x_min = max(0, int(min(xs5)))
            x_max = min(rend.width - 1, int(max(xs5)))
            y_min = max(0, int(min(ys5)))
            y_max = min(rend.height - 1, int(max(ys5)))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    bc = barycentric((A_uv[0], A_uv[1]), (B_uv[0], B_uv[1]), (C_uv[0], C_uv[1]), (x, y))
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    z = w0 * A_uv[2] + w1 * B_uv[2] + w2 * C_uv[2]
                    u = w0 * A_uv[3] + w1 * B_uv[3] + w2 * C_uv[3]
                    v = w0 * A_uv[4] + w1 * B_uv[4] + w2 * C_uv[4]
                    tex_color = sample_texture_nearest(texture5, u, v)
                    base_color = np.array(tex_color, dtype=float) / 255.0
                    rend.glPoint(x, y, base_color.tolist(), z=z)

        for face in faces4:
            idx0, vt0 = face[0]
            idx1, vt1 = face[1]
            idx2, vt2 = face[2]

            n0 = vertex_normals4[idx0]
            n1 = vertex_normals4[idx1]
            n2 = vertex_normals4[idx2]

            A = vertexShader(verts4[idx0], modelMatrix=model_matrix_pim, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            B = vertexShader(verts4[idx1], modelMatrix=model_matrix_pim, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            C = vertexShader(verts4[idx2], modelMatrix=model_matrix_pim, viewMatrix=np.identity(4), projMatrix=np.identity(4))

            def unpack4(vs, vt_idx):
                x, y, z = vs[0], vs[1], vs[2]
                u, v = (0, 0)
                if vt_idx is not None and vt_idx < len(texcoords4):
                    u, v = texcoords4[vt_idx]
                return [x, y, z, u, v]

            A_uv = unpack4(A, vt0)
            B_uv = unpack4(B, vt1)
            C_uv = unpack4(C, vt2)

            xs4 = [A_uv[0], B_uv[0], C_uv[0]]
            ys4 = [A_uv[1], B_uv[1], C_uv[1]]
            x_min = max(0, int(min(xs4)))
            x_max = min(rend.width - 1, int(max(xs4)))
            y_min = max(0, int(min(ys4)))
            y_max = min(rend.height - 1, int(max(ys4)))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    bc = barycentric((A_uv[0], A_uv[1]), (B_uv[0], B_uv[1]), (C_uv[0], C_uv[1]), (x, y))
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    z = w0 * A_uv[2] + w1 * B_uv[2] + w2 * C_uv[2]
                    u = w0 * A_uv[3] + w1 * B_uv[3] + w2 * C_uv[3]
                    v = w0 * A_uv[4] + w1 * B_uv[4] + w2 * C_uv[4]
                    tex_color = sample_texture_nearest(texture4, u, v)
                    base_color = np.array(tex_color, dtype=float) / 255.0
                    rend.glPoint(x, y, base_color.tolist(), z=z)

        for face in faces2:
            idx0, vt0 = face[0]
            idx1, vt1 = face[1]
            idx2, vt2 = face[2]

            n0 = vertex_normals2[idx0]
            n1 = vertex_normals2[idx1]
            n2 = vertex_normals2[idx2]

            A = vertexShader(verts2[idx0], modelMatrix=model_matrix_among, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            B = vertexShader(verts2[idx1], modelMatrix=model_matrix_among, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            C = vertexShader(verts2[idx2], modelMatrix=model_matrix_among, viewMatrix=np.identity(4), projMatrix=np.identity(4))

            def unpack2(vs, vt_idx):
                x, y, z = vs[0], vs[1], vs[2]
                u, v = (0, 0)
                if vt_idx is not None and vt_idx < len(texcoords2):
                    u, v = texcoords2[vt_idx]
                return [x, y, z, u, v]

            A_uv = unpack2(A, vt0)
            B_uv = unpack2(B, vt1)
            C_uv = unpack2(C, vt2)

            xs2 = [A_uv[0], B_uv[0], C_uv[0]]
            ys2 = [A_uv[1], B_uv[1], C_uv[1]]
            x_min = max(0, int(min(xs2)))
            x_max = min(rend.width - 1, int(max(xs2)))
            y_min = max(0, int(min(ys2)))
            y_max = min(rend.height - 1, int(max(ys2)))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    bc = barycentric((A_uv[0], A_uv[1]), (B_uv[0], B_uv[1]), (C_uv[0], C_uv[1]), (x, y))
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    z = w0 * A_uv[2] + w1 * B_uv[2] + w2 * C_uv[2]
                    u = w0 * A_uv[3] + w1 * B_uv[3] + w2 * C_uv[3]
                    v = w0 * A_uv[4] + w1 * B_uv[4] + w2 * C_uv[4]
                    tex_color = sample_texture_nearest(texture2, u, v)
                    base_color = np.array(tex_color, dtype=float) / 255.0
                    rend.glPoint(x, y, base_color.tolist(), z=z)

        for face in faces3:
            idx0, vt0 = face[0]
            idx1, vt1 = face[1]
            idx2, vt2 = face[2]

            n0 = vertex_normals3[idx0]
            n1 = vertex_normals3[idx1]
            n2 = vertex_normals3[idx2]

            A = vertexShader(verts3[idx0], modelMatrix=model_matrix_charlie, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            B = vertexShader(verts3[idx1], modelMatrix=model_matrix_charlie, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            C = vertexShader(verts3[idx2], modelMatrix=model_matrix_charlie, viewMatrix=np.identity(4), projMatrix=np.identity(4))

            def unpack3(vs, vt_idx):
                x, y, z = vs[0], vs[1], vs[2]
                u, v = (0, 0)
                if vt_idx is not None and vt_idx < len(texcoords3):
                    u, v = texcoords3[vt_idx]
                return [x, y, z, u, v]

            A_uv = unpack3(A, vt0)
            B_uv = unpack3(B, vt1)
            C_uv = unpack3(C, vt2)

            xs3 = [A_uv[0], B_uv[0], C_uv[0]]
            ys3 = [A_uv[1], B_uv[1], C_uv[1]]
            x_min = max(0, int(min(xs3)))
            x_max = min(rend.width - 1, int(max(xs3)))
            y_min = max(0, int(min(ys3)))
            y_max = min(rend.height - 1, int(max(ys3)))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    bc = barycentric((A_uv[0], A_uv[1]), (B_uv[0], B_uv[1]), (C_uv[0], C_uv[1]), (x, y))
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    z = w0 * A_uv[2] + w1 * B_uv[2] + w2 * C_uv[2]
                    u = w0 * A_uv[3] + w1 * B_uv[3] + w2 * C_uv[3]
                    v = w0 * A_uv[4] + w1 * B_uv[4] + w2 * C_uv[4]
                    tex_color = sample_texture_nearest(texture3, u, v)
                    base_color = np.array(tex_color, dtype=float) / 255.0
                    rend.glPoint(x, y, base_color.tolist(), z=z)

        for face in faces:
            idx0, vt0 = face[0]
            idx1, vt1 = face[1]
            idx2, vt2 = face[2]

            n0 = vertex_normals[idx0]
            n1 = vertex_normals[idx1]
            n2 = vertex_normals[idx2]

            A = vertexShader(verts[idx0], modelMatrix=model_matrix_hola, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            B = vertexShader(verts[idx1], modelMatrix=model_matrix_hola, viewMatrix=np.identity(4), projMatrix=np.identity(4))
            C = vertexShader(verts[idx2], modelMatrix=model_matrix_hola, viewMatrix=np.identity(4), projMatrix=np.identity(4))

            def unpack(vs, vt_idx):
                x, y, z = vs[0], vs[1], vs[2]
                u, v = (0, 0)
                if vt_idx is not None and vt_idx < len(texcoords):
                    u, v = texcoords[vt_idx]
                return [x, y, z, u, v]

            A_uv = unpack(A, vt0)
            B_uv = unpack(B, vt1)
            C_uv = unpack(C, vt2)

            xs2 = [A_uv[0], B_uv[0], C_uv[0]]
            ys2 = [A_uv[1], B_uv[1], C_uv[1]]
            x_min = max(0, int(min(xs2)))
            x_max = min(rend.width - 1, int(max(xs2)))
            y_min = max(0, int(min(ys2)))
            y_max = min(rend.height - 1, int(max(ys2)))

            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    bc = barycentric((A_uv[0], A_uv[1]), (B_uv[0], B_uv[1]), (C_uv[0], C_uv[1]), (x, y))
                    if bc is None:
                        continue
                    w0, w1, w2 = bc
                    if w0 < 0 or w1 < 0 or w2 < 0:
                        continue

                    z = w0 * A_uv[2] + w1 * B_uv[2] + w2 * C_uv[2]
                    u = w0 * A_uv[3] + w1 * B_uv[3] + w2 * C_uv[3]
                    v = w0 * A_uv[4] + w1 * B_uv[4] + w2 * C_uv[4]
                    tex_color = sample_texture_nearest(rend.activeTexture, u, v)
                    base_color = np.array(tex_color, dtype=float) / 255.0
                    rend.glPoint(x, y, base_color.tolist(), z=z)

        pygame.display.flip()

    pygame.quit()
    GenerateBMP("escena_capturada.bmp", rend.frameBuffer)

if __name__ == "__main__":
    main()
