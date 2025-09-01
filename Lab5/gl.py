
import numpy as np
import pygame
from math import isclose, floor, pi, tan
from camera import Camera
import random

POINTS = 0
LINES = 1
TRIANGLES = 2

def barycentric(A, B, C, P):
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]

    v0 = (B[0] - A[0], B[1] - A[1])
    v1 = (C[0] - A[0], C[1] - A[1])
    v2 = (P[0] - A[0], P[1] - A[1])

    denom = cross(v0, v1)
    if abs(denom) < 1e-6:
        return None
    u = cross(v2, v1) / denom
    v = cross(v0, v2) / denom
    w = 1 - u - v
    return (u, v, w)

def sample_texture_nearest(texture_surface, u, v):
    width, height = texture_surface.get_size()
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    x = min(width - 1, int(u * (width - 1)))
    y = min(height - 1, int((1 - v) * (height - 1)))
    return texture_surface.get_at((x, y))[:3]

def sample_texture_bilinear(texture_surface, u, v):
    width, height = texture_surface.get_size()
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    x = u * (width - 1)
    y = (1 - v) * (height - 1)

    x0 = int(np.floor(x))
    x1 = min(width - 1, x0 + 1)
    y0 = int(np.floor(y))
    y1 = min(height - 1, y0 + 1)

    fx = x - x0
    fy = y - y0

    c00 = np.array(texture_surface.get_at((x0, y0))[:3], dtype=float)
    c10 = np.array(texture_surface.get_at((x1, y0))[:3], dtype=float)
    c01 = np.array(texture_surface.get_at((x0, y1))[:3], dtype=float)
    c11 = np.array(texture_surface.get_at((x1, y1))[:3], dtype=float)

    c0 = c00 * (1 - fx) + c10 * fx
    c1 = c01 * (1 - fx) + c11 * fx
    c = c0 * (1 - fy) + c1 * fy

    return [int(round(v)) for v in c]

class Renderer(object):
    def readNormalMap(self, normal_map_surface, u, v):

        width, height = normal_map_surface.get_size()
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        x = min(width - 1, int(u * (width - 1)))
        y = min(height - 1, int((1 - v) * (height - 1)))
        r, g, b = normal_map_surface.get_at((x, y))[:3]
        # Convertir de [0,255] a [-1,1]
        nx = (r / 255.0) * 2.0 - 1.0
        ny = (g / 255.0) * 2.0 - 1.0
        nz = (b / 255.0) * 2.0 - 1.0
        return np.array([nx, ny, nz], dtype=float)

    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = self.screen.get_rect()

        # Raytracing attributes
        self.camera = Camera()
        self.glViewport(0, 0, self.width, self.height)
        self.glProjection()

        self.glColor(1, 1, 1)
        self.glClearColor(0, 0, 0)

        self.glClear()

        self.scene = []
        self.lights = []

        # Rasterization attributes
        self.primitiveType = TRIANGLES
        self.activeModelMatrix = None
        self.activeVertexShader = None
        self.models = []
        self.activeTexture = None
        self.use_bilinear = False  # toggle: False = nearest, True = bilinear
    # Viewport setup for raytracing
    def glViewport(self, x, y, width, height):
        self.vpX = round(x)
        self.vpY = round(y)
        self.vpWidth = width
        self.vpHeight = height
        self.viewportMatrix = np.matrix([[width/2, 0, 0, x + width/2],
                                         [0, height/2, 0, y + height/2],
                                         [0, 0, 0.5, 0.5],
                                         [0, 0, 0, 1]])

    # Projection setup for raytracing
    def glProjection(self, n=0.1, f=1000, fov=60):
        aspectRatio = self.vpWidth / self.vpHeight
        fov *= pi/180
        self.topEdge = tan(fov/2) * n
        self.rightEdge = self.topEdge * aspectRatio
        self.nearPlane = n
        self.projectionMatrix = np.matrix([[n/self.rightEdge, 0, 0, 0],
                                           [0, n/self.topEdge, 0, 0],
                                           [0, 0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                                           [0, 0, -1, 0]])
    # Raytracing render loop
    def glRenderRaytracing(self):
        import time
        
        for y in range(self.height):
            for x in range(self.width):
                # Convertir coordenadas de pantalla a mundo
                px = (2.0 * x / self.width - 1.0) * (self.width / self.height)
                py = 1.0 - 2.0 * y / self.height
                
                # Origen del rayo (cámara en el origen)
                ray_origin = np.array([0.0, 0.0, 0.0])
                # Dirección del rayo (hacia el plano z=-1)
                ray_direction = np.array([px, py, -1.0])
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                # Buscar intersección
                hit = self.glCastRay(ray_origin, ray_direction)
                
                # Solo dibujar píxeles donde hay intersección con la esfera
                if hit and hasattr(hit, 'obj') and hasattr(hit.obj, 'material'):
                    # Usar modelo de iluminación Phong
                    color = hit.obj.material.GetSurfaceColor(hit, self)
                    self.glPoint(x, y, color)
                    
                    # Actualizar la pantalla después de cada punto para ver el progreso
                    pygame.display.flip()
                    
                    # Pequeña pausa para ver el efecto punto por punto
                    time.sleep(0.001)  # 1 milisegundo de pausa

    # Raytracing intersection
    def glCastRay(self, origin, direction, sceneObj=None):
        depth = float('inf')
        intercept = None
        hit = None
        for obj in self.scene:
            if obj != sceneObj:
                intercept = obj.ray_intersect(origin, direction)
                if intercept is not None:
                    if intercept.distance < depth:
                        hit = intercept
                        depth = intercept.distance
        return hit

    def glClearColor(self, r, g, b):
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))
        self.ClearColor = [r, g, b]

    def glDrawBackground(self, background_surface):
        bg_width, bg_height = background_surface.get_size()
        for x in range(self.width):
            for y in range(self.height):
                bx = int(x * bg_width / self.width)
                by = int(y * bg_height / self.height)
                color = background_surface.get_at((bx, by))[:3]
                self.screen.set_at((x, self.height - y - 1), color)
                self.frameBuffer[x][y] = list(color)

    def glColor(self, r, g, b):
        r = min(1, max(0, r))
        g = min(1, max(0, g))
        b = min(1, max(0, b))
        self.currColor = [r, g, b]

    def glClear(self):
        color = [int(i * 255) for i in self.ClearColor]
        self.screen.fill(color)
        self.frameBuffer = [[color.copy() for y in range(self.height)] for x in range(self.width)]
        self.zbuffer = [[float('inf') for y in range(self.height)] for x in range(self.width)]

    def glPoint(self, x, y, color=None, z=None):
        x = round(x)
        y = round(y)
        if not ((0 <= x < self.width) and (0 <= y < self.height)):
            return

        if z is not None:
            if z >= self.zbuffer[x][y]:
                return
            self.zbuffer[x][y] = z

        color = [int(i * 255) for i in (color or self.currColor)]
        self.screen.set_at((x, self.height - y - 1), color)
        self.frameBuffer[x][y] = color

    def glLine(self, p0, p1, color=None):
        x0, y0 = round(p0[0]), round(p0[1])
        x1, y1 = round(p1[0]), round(p1[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            self.glPoint(x, y, color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def glPolygon(self, puntos, color=None):
        for i in range(len(puntos)):
            punto_actual = puntos[i]
            punto_siguiente = puntos[(i + 1) % len(puntos)]
            self.glLine(punto_actual, punto_siguiente, color or self.currColor)

    def glFillPolygon(self, puntos, color=None):
        self.glPolygon(puntos, color)
        y_min = min(punto[1] for punto in puntos)
        y_max = max(punto[1] for punto in puntos)
        for y in range(y_min, y_max + 1):
            intersecciones = []
            for i in range(len(puntos)):
                p1 = puntos[i]
                p2 = puntos[(i + 1) % len(puntos)]
                x1, y1 = p1
                x2, y2 = p2
                if y1 > y2:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                if y1 <= y < y2:
                    if y2 - y1 != 0:
                        x_interseccion = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                        intersecciones.append(x_interseccion)
            intersecciones.sort()
            for i in range(0, len(intersecciones) - 1, 2):
                x_inicio = int(intersecciones[i])
                x_fin = int(intersecciones[i + 1])
                for x in range(x_inicio, x_fin + 1):
                    self.glPoint(x, y, color or self.currColor)

    def glBindTexture(self, texture_surface):
        self.activeTexture = texture_surface

    def glTexturedTriangle(self, A, B, C):
        # A, B, C: [x, y, z, u, v]
        xs = [A[0], B[0], C[0]]
        ys = [A[1], B[1], C[1]]
        x_min = max(0, int(min(xs)))
        x_max = min(self.width - 1, int(max(xs)))
        y_min = max(0, int(min(ys)))
        y_max = min(self.height - 1, int(max(ys)))

        # Precompute 1/z and u/z, v/z for perspective-correct
        def prep(v):
            x, y, z, u, v_uv = v
            iz = 1.0 / z if abs(z) > 1e-6 else 1.0
            return iz, u * iz, v_uv * iz

        izA, uA_div, vA_div = prep(A)
        izB, uB_div, vB_div = prep(B)
        izC, uC_div, vC_div = prep(C)

        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                bc = barycentric((A[0], A[1]), (B[0], B[1]), (C[0], C[1]), (x, y))
                if bc is None:
                    continue
                w0, w1, w2 = bc
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue

                # Perspective-correct interpolation of UV
                iz = w0 * izA + w1 * izB + w2 * izC
                if iz == 0:
                    continue
                u = (w0 * uA_div + w1 * uB_div + w2 * uC_div) / iz
                v = (w0 * vA_div + w1 * vB_div + w2 * vC_div) / iz

                # Linear z for depth test
                z = w0 * A[2] + w1 * B[2] + w2 * C[2]

                if self.activeTexture:
                    if self.use_bilinear:
                        tex_color = sample_texture_bilinear(self.activeTexture, u, v)
                    else:
                        tex_color = sample_texture_nearest(self.activeTexture, u, v)
                    color_norm = [c / 255 for c in tex_color]
                    self.glPoint(x, y, color_norm, z=z)
                else:
                    self.glPoint(x, y, None, z=z)

    def glRender(self):
        for model in self.models:
            self.activeModelMatrix = model.GetModelMatrix()
            self.activeVertexShader = model.vertexShader

            vertexBuffer = []
            for i in range(0, len(model.vertices), 3):
                x = model.vertices[i]
                y = model.vertices[i + 1]
                z = model.vertices[i + 2]
                if self.activeVertexShader:
                    x, y, z = self.activeVertexShader([x, y, z],
                                                      modelMatrix=self.activeModelMatrix)
                vertexBuffer.extend([x, y, z])

            self.glDrawPrimitives(vertexBuffer, 3)

    def glDrawPrimitives(self, buffer, vertexOffset):
        if self.primitiveType == POINTS:
            for i in range(0, len(buffer), vertexOffset):
                x = buffer[i]
                y = buffer[i + 1]
                self.glPoint(x, y)

        elif self.primitiveType == LINES:
            for i in range(0, len(buffer), vertexOffset * 2):
                x0 = buffer[i + 0]
                y0 = buffer[i + 1]
                x1 = buffer[i + vertexOffset + 0]
                y1 = buffer[i + vertexOffset + 1]
                self.glLine((x0, y0), (x1, y1))

        elif self.primitiveType == TRIANGLES:
            for i in range(0, len(buffer), vertexOffset * 3):
                A = [buffer[i + j + vertexOffset * 0] for j in range(vertexOffset)]
                B = [buffer[i + j + vertexOffset * 1] for j in range(vertexOffset)]
                C = [buffer[i + j + vertexOffset * 2] for j in range(vertexOffset)]
                self.glTriangle(A, B, C)
