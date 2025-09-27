import numpy as np

def blend(src_rgb, dst_rgb, alpha):
    src = np.array(src_rgb, dtype=float)
    dst = np.array(dst_rgb, dtype=float)
    return (alpha * src + (1 - alpha) * dst).tolist()

def normalize(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def vertexShader(vertex, **kwargs):
    modelMatrix = kwargs["modelMatrix"]
    viewMatrix = kwargs.get("viewMatrix", np.identity(4))
    projMatrix = kwargs.get("projMatrix", np.identity(4))

    vt = np.array([vertex[0], vertex[1], vertex[2], 1.0], dtype=float).reshape(4,1)
    transformed = projMatrix @ viewMatrix @ modelMatrix @ vt
    transformed = np.asarray(transformed).flatten()
    w = transformed[3] if abs(transformed[3]) > 1e-6 else 1.0
    x = transformed[0] / w
    y = transformed[1] / w
    z = transformed[2] / w
    return [x, y, z]

def radioactive_fragment_shader(base_color, normal, view_dir, time):
    normal = normalize(normal)
    view_dir = normalize(view_dir)

    ndotv = float(max(-1.0, min(1.0, np.dot(normal, view_dir))))
    ndotv = max(0.0, ndotv)

    # rim / glow based on view angle
    rim = (1.0 - ndotv) ** 3.0
    glow_intensity = rim

    # Base de color gris oscuro azulado
    ghost_base = np.array([0.08, 0.08, 0.12], dtype=float)
    min_glow = 0.2
    glow_strength = float(min(1.0, min_glow + glow_intensity * 0.9))

    final_color = np.clip(ghost_base * (0.6 + glow_strength * 1.2), 0, 1)

    alpha = float(max(0.25, min(0.65, 0.25 + glow_strength * 0.4)))

    return final_color.tolist(), alpha



def matrixFragmentShader(base_color, normal, view_dir):
    green = [0.0, 1.0, 0.0]
    alpha = 1.0
    return green, alpha

def copperFragmentShader(base_color, normal, view_dir):
    # Normalizar entradas
    normal = normalize(normal)
    view_dir = normalize(view_dir)

    copper_color = np.array([0.72, 0.45, 0.2], dtype=float)
    light_dir = normalize(np.array([0.5, 0.8, 0.6], dtype=float))

    ambient = 0.12
    ndotl = max(0.0, np.dot(normal, light_dir))
    diffuse = ndotl * 1.0

    # Blinn-Phong specular (simple)
    half_dir = normalize(light_dir + view_dir)
    spec_intensity = max(0.0, np.dot(normal, half_dir))
    specular = (spec_intensity ** 16) * 0.6

    final_color = copper_color * (ambient + diffuse) + np.array([1.0, 0.8, 0.4]) * specular
    final_color = np.clip(final_color, 0, 1)

    return final_color.tolist(), 1.0

def bwFragmentShader(base_color, normal, view_dir):
    normal = normalize(normal)
    view_dir = normalize(view_dir)

    light_dir = normalize(np.array([0.0, 0.0, 1.0], dtype=float))
    ndotl = max(0.0, np.dot(normal, light_dir))
    ndotv = max(0.0, np.dot(normal, view_dir))

    # Thresholds para blanco y negro
    if ndotv < 0.25:
        color = np.array([0.0, 0.0, 0.0], dtype=float)
    elif ndotl > 0.5:
        color = np.array([1.0, 1.0, 1.0], dtype=float)
    else:
        color = np.array([0.15, 0.15, 0.15], dtype=float)

    return color.tolist(), 1.0

def NormalMaping_shader(base_color, normal_map_vec, light_dir=None):
    if light_dir is None:
        light_dir = np.array([1.5, 0, 1.5], dtype=float)
    else:
        light_dir = np.array(light_dir, dtype=float)
    normal = np.array(normal_map_vec, dtype=float)
    normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else normal
    diffuse = max(np.dot(normal, normalize(light_dir)), 0.0)
    color_lit = np.array(base_color, dtype=float) * diffuse * 1.6
    color_lit = np.clip(color_lit, 0, 1)
    return color_lit.tolist(), 1.0
