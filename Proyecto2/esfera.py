import pygame
import numpy as np
from gl import Renderer
from figures import Sphere
from material import Material, OPAQUE, REFLECTIVE, TRANSPARENT
from lights import DirectionalLight
from camera import Camera

WIDTH, HEIGHT = 800, 800

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing Esfera')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    # Configurar recursión para reflexiones
    renderer.maxRecursionDepth = 5  # Permitir más niveles de reflexión
    
    # Fondo lila/morado
    renderer.glClearColor(0.7, 0.6, 0.9)

    # Cargar textura de fondo (descomenta esta línea para usar textura)
    try:
        from BMPTexture import BMPTexture
        renderer.envMap = BMPTexture("fondo.bmp")
        print("Fondo de textura cargado exitosamente")
    except:
        print("No se pudo cargar la textura de fondo, usando color sólido")

    # Materiales
    # Materiales opacos con texturas
    try:
        dry_texture = BMPTexture("dry.bmp")
        material_azul = Material(diffuse=[0.4, 0.4, 0.8], spec=32.0, ks=0.3, matType=OPAQUE, texture=dry_texture)
        print("Textura dry cargada exitosamente")
    except:
        print("No se pudo cargar dry.bmp, usando material azul sólido")
        material_azul = Material(diffuse=[0.0, 0.0, 1.0], spec=32.0, ks=0.3, matType=OPAQUE)
    
    try:
        mosaico_texture = BMPTexture("mosaico.bmp")
        material_rojo = Material(diffuse=[0.8, 0.4, 0.4], spec=32.0, ks=0.3, matType=OPAQUE, texture=mosaico_texture)
        print("Textura mosaico cargada exitosamente")
    except:
        print("No se pudo cargar mosaico.bmp, usando material rojo sólido")
        material_rojo = Material(diffuse=[1.0, 0.0, 0.0], spec=32.0, ks=0.3, matType=OPAQUE)
    
    # Material reflectivo con textura de cerámica (espejo perfecto)
    try:
        from BMPTexture import BMPTexture
        ceramica_texture = BMPTexture("ceramica.bmp")
        mirror_ceramica = Material(diffuse=[0.9, 0.9, 0.9], spec=512, ks=0.02, matType=REFLECTIVE, texture=ceramica_texture)
        print("Textura de cerámica cargada exitosamente")
    except:
        print("No se pudo cargar ceramica.bmp, usando material blanco brillante")
        mirror_ceramica = Material(diffuse=[0.9, 0.9, 0.9], spec=512, ks=0.02, matType=REFLECTIVE)
    
    # Material reflectivo con textura de metal (menos reflexivo, más metálico)
    try:
        metal_texture = BMPTexture("metal.bmp")
        mirror_metal = Material(diffuse=[0.7, 0.7, 0.7], spec=64, ks=0.4, matType=REFLECTIVE, texture=metal_texture)
        print("Textura de metal cargada exitosamente")
    except:
        print("No se pudo cargar metal.bmp, usando material dorado")
        mirror_metal = Material(diffuse=[0.8, 0.6, 0.3], spec=64, ks=0.4, matType=REFLECTIVE)
    
    # Materiales transparentes con texturas
    try:
        quarzo_texture = BMPTexture("quarzo.bmp")
        glass_claro = Material(diffuse=[1.0, 1.0, 1.0], ior=1.5, ks=0.1, spec=64, matType=TRANSPARENT, texture=quarzo_texture)
        print("Textura quarzo cargada exitosamente")
    except:
        print("No se pudo cargar quarzo.bmp, usando cristal claro sólido")
        glass_claro = Material(diffuse=[1.0, 1.0, 1.0], ior=1.5, ks=0.1, spec=64, matType=TRANSPARENT)
    
    try:
        blanca_texture = BMPTexture("blanca.bmp")
        glass_verde = Material(diffuse=[0.9, 0.9, 0.9], ior=1.3, ks=0.1, spec=64, matType=TRANSPARENT, texture=blanca_texture)
        print("Textura blanca cargada exitosamente")
    except:
        print("No se pudo cargar blanca.bmp, usando cristal verde sólido")
        glass_verde = Material(diffuse=[0.8, 1.0, 0.8], ior=1.3, ks=0.1, spec=64, matType=TRANSPARENT)

    # 6 esferas: 3 arriba y 3 abajo
    # Fila de arriba (y = 0.7)
    esfera1 = Sphere(position=[-1.2, 0.7, -4], radius=0.35, material=material_azul)        # Opaca azul
    renderer.scene.append(esfera1)
    
    esfera2 = Sphere(position=[0, 0.7, -4], radius=0.35, material=mirror_ceramica)           # Reflectiva cerámica
    renderer.scene.append(esfera2)
    
    esfera3 = Sphere(position=[1.2, 0.7, -4], radius=0.35, material=glass_claro)          # Transparente clara
    renderer.scene.append(esfera3)
    
    # Fila de abajo (y = -0.7)
    esfera4 = Sphere(position=[-1.2, -0.7, -4], radius=0.35, material=material_rojo)      # Opaca roja
    renderer.scene.append(esfera4)
    
    esfera5 = Sphere(position=[0, -0.7, -4], radius=0.35, material=mirror_metal)          # Reflectiva con textura metal
    renderer.scene.append(esfera5)
    
    esfera6 = Sphere(position=[1.2, -0.7, -4], radius=0.35, material=glass_verde)         # Transparente verde
    renderer.scene.append(esfera6)

    # Luces
    luz_direccional = DirectionalLight(color=[1,1,1], intensity=0.8, direction=[1, -1, -1])
    renderer.lights.append(luz_direccional)
    from lights import AmbientLight
    luz_ambiental = AmbientLight(color=[1,1,1], intensity=0.2)
    renderer.lights.append(luz_ambiental)

    # Renderizar
    renderer.glRenderRaytracing()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == '__main__':
    main()
