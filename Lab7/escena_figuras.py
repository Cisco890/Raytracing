import pygame
import numpy as np
from gl import Renderer
from figures import Sphere, Cube, Disk, Triangle, Plane
from material import Material, OPAQUE, REFLECTIVE, TRANSPARENT
from lights import DirectionalLight, AmbientLight, PointLight
from camera import Camera

WIDTH, HEIGHT = 800, 800

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing - Disco, Triangulo y Cubo')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    # Configurar recursión para reflexiones
    renderer.maxRecursionDepth = 3
    
    # Sin fondo (negro)
    renderer.glClearColor(0.0, 0.0, 0.0)
    
    # No usar textura de fondo (comentado para crear cuarto cerrado)
    # renderer.envMap = None

    # Materiales
    # Material opaco rojo para el cubo
    try:
        from BMPTexture import BMPTexture
        dry_texture = BMPTexture("dry.bmp")
        material_cubo = Material(diffuse=[0.8, 0.2, 0.2], spec=32.0, ks=0.3, matType=OPAQUE, texture=dry_texture)
        print("Textura dry cargada para el cubo")
    except:
        print("No se pudo cargar dry.bmp, usando material rojo sólido para el cubo")
        material_cubo = Material(diffuse=[0.8, 0.2, 0.2], spec=32.0, ks=0.3, matType=OPAQUE)
    
    # Material dorado (oro) para el disco
    try:
        metal_texture = BMPTexture("metal.bmp")
        material_disco = Material(diffuse=[1.0, 0.8, 0.2], spec=128, ks=0.6, matType=REFLECTIVE, texture=metal_texture)
        print("Textura metal cargada para el disco dorado")
    except:
        print("No se pudo cargar metal.bmp, usando material dorado para el disco")
        material_disco = Material(diffuse=[1.0, 0.8, 0.2], spec=128, ks=0.6, matType=REFLECTIVE)
    
    # Material transparente para el triángulo
    try:
        mosaico_texture = BMPTexture("mosaico.bmp")
        material_triangulo = Material(diffuse=[0.9, 1.0, 0.9], ior=1.4, ks=0.1, spec=64, matType=OPAQUE, texture=mosaico_texture)
        print("Textura mosaico cargada para el triangulo")
    except:
        print("No se pudo cargar mosaico.bmp, usando cristal verde para el triangulo")
        material_triangulo = Material(diffuse=[0.9, 1.0, 0.9], ior=1.4, ks=0.1, spec=64, matType=TRANSPARENT)

    # Material morado para el segundo cubo con textura de cerámica
    try:
        ceramica_texture = BMPTexture("ceramica.bmp")
        material_cubo2 = Material(diffuse=[0.6, 0.2, 0.8], spec=64, ks=0.3, matType=OPAQUE, texture=ceramica_texture)
        print("Textura cerámica cargada para el segundo cubo morado")
    except:
        print("No se pudo cargar ceramica.bmp, usando material morado para el segundo cubo")
        material_cubo2 = Material(diffuse=[0.6, 0.2, 0.8], spec=64, ks=0.3, matType=OPAQUE)

    # Materiales para el cuarto
    # Material para techo y suelo (blanca.bmp)
    try:
        blanca_texture = BMPTexture("blanca.bmp")
        material_techo_suelo = Material(diffuse=[0.9, 0.9, 0.9], spec=16.0, ks=0.1, matType=OPAQUE, texture=blanca_texture)
        print("Textura blanca cargada para techo y suelo")
    except:
        print("No se pudo cargar blanca.bmp, usando material blanco sólido")
        material_techo_suelo = Material(diffuse=[0.9, 0.9, 0.9], spec=16.0, ks=0.1, matType=OPAQUE)
    
    # Material para paredes (quarzo.bmp)
    try:
        quarzo_wall_texture = BMPTexture("quarzo.bmp")
        material_paredes = Material(diffuse=[0.8, 0.8, 0.8], spec=16.0, ks=0.1, matType=OPAQUE, texture=quarzo_wall_texture)
        print("Textura quarzo cargada para paredes")
    except:
        print("No se pudo cargar quarzo.bmp, usando material gris sólido para paredes")
        material_paredes = Material(diffuse=[0.8, 0.8, 0.8], spec=16.0, ks=0.1, matType=OPAQUE)

    # Crear el cuarto cerrado con 5 planos
    
    # SUELO - Plano horizontal inferior (normal apunta hacia arriba)
    suelo = Plane(position=[0.0, -3.0, -5.0], normal=[0, 1, 0], material=material_techo_suelo)
    renderer.scene.append(suelo)
    
    # TECHO - Plano horizontal superior (normal apunta hacia abajo)
    techo = Plane(position=[0.0, 3.0, -5.0], normal=[0, -1, 0], material=material_techo_suelo)
    renderer.scene.append(techo)
    
    # PARED TRASERA - Plano vertical al fondo (normal apunta hacia adelante)
    pared_trasera = Plane(position=[0.0, 0.0, -8.0], normal=[0, 0, 1], material=material_paredes)
    renderer.scene.append(pared_trasera)
    
    # PARED IZQUIERDA - Plano vertical a la izquierda (normal apunta hacia la derecha)
    pared_izquierda = Plane(position=[-4.0, 0.0, -5.0], normal=[1, 0, 0], material=material_paredes)
    renderer.scene.append(pared_izquierda)
    
    # PARED DERECHA - Plano vertical a la derecha (normal apunta hacia la izquierda)
    pared_derecha = Plane(position=[4.0, 0.0, -5.0], normal=[-1, 0, 0], material=material_paredes)
    renderer.scene.append(pared_derecha)

    # Crear las figuras dentro del cuarto
    
    # CUBO - Posicionado a la izquierda
    cubo = Cube(position=[-1.5, -1.0, -5.0], size=1.0, material=material_cubo)
    renderer.scene.append(cubo)
    
    # DISCO - Posicionado en el centro, orientado verticalmente
    disco = Disk(position=[0.0, 0.0, -7.5], radius=2.0, normal=[0, 0, 1], material=material_disco)
    renderer.scene.append(disco)
    
    # TRIÁNGULO - Posicionado a la derecha
    triangulo = Triangle(
        v0=[1.2, -1.2, -4.0],  # Vértice inferior izquierdo
        v1=[2.0, -1.2, -4.0],  # Vértice inferior derecho
        v2=[1.6, 0.0, -4.0],   # Vértice superior
        material=material_triangulo
    )
    renderer.scene.append(triangulo)
    
    # CUBO 2 - Segundo cubo con textura de cerámica - Arriba en el centro
    cubo2 = Cube(position=[0.0, 1.2, -4.0], size=0.8, material=material_cubo2)
    renderer.scene.append(cubo2)

    # Luces
    # Luz puntual principal - Como una bombilla en el techo del cuarto
    luz_interior = PointLight(
        color=[1.0, 0.95, 0.8],  # Luz cálida (ligeramente amarillenta)
        intensity=15.0,          # Intensidad alta para iluminar todo el cuarto
        position=[0.0, 2.5, -5.0]  # Posicionada en el centro superior del cuarto
    )
    renderer.lights.append(luz_interior)
    
    # Luz ambiental suave para iluminación base
    luz_ambiental = AmbientLight(color=[0.8, 0.8, 1.0], intensity=0.2)
    renderer.lights.append(luz_ambiental)
    
    # Luz direccional secundaria suave (opcional - simula luz indirecta)
    luz_indirecta = DirectionalLight(
        color=[0.6, 0.7, 1.0], 
        intensity=0.3, 
        direction=[0.5, -0.8, -0.3]
    )
    renderer.lights.append(luz_indirecta)

    print("Iniciando renderizado...")
    print("Cuarto cerrado creado con:")
    print("- Suelo y techo con textura blanca.bmp")
    print("- 3 paredes con textura quarzo.bmp")
    print("- Dos cubos, disco y triángulo dentro del cuarto")
    print("- Luz puntual interior para iluminación realista")
    print("- Luz ambiental y direccional para detalles")

    # Renderizar
    renderer.glRenderRaytracing()
    
    # Guardar la imagen
    try:
        pygame.image.save(screen, "escena_figuras_resultado.png")
        print("Imagen guardada como 'escena_figuras_resultado.png'")
    except:
        print("No se pudo guardar la imagen")

    print("Renderizado completado!")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    # Guardar imagen al presionar 'S'
                    try:
                        pygame.image.save(screen, "escena_figuras_resultado.png")
                        print("Imagen guardada!")
                    except:
                        print("Error al guardar imagen")
    
    pygame.quit()

if __name__ == '__main__':
    main()