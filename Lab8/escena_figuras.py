import pygame
import numpy as np
from gl import Renderer
from figures import Sphere, Cube, Disk, Triangle, Plane, Capsule
from material import Material, OPAQUE, REFLECTIVE, TRANSPARENT
from lights import DirectionalLight, AmbientLight, PointLight
from camera import Camera

WIDTH, HEIGHT = 400, 400

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing - Cápsula Reflectiva')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    # Configurar recursión para reflexiones
    renderer.maxRecursionDepth = 3
    
    # Sin fondo (negro)
    renderer.glClearColor(0.0, 0.0, 0.0)
    
    # No usar textura de fondo (comentado para crear cuarto cerrado)
    # renderer.envMap = None

    # Materiales
    # Material reflectivo para la cápsula (sin textura)
    material_capsula = Material(diffuse=[0.8, 0.8, 0.8], spec=128, ks=0.9, matType=REFLECTIVE)
    print("Material reflectivo creado para la cápsula")

    # Materiales para el cuarto
    # Material para techo y suelo (blanca.bmp)
    try:
        from BMPTexture import BMPTexture
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

    # Crear la cápsula reflectiva en el centro
    
    # CÁPSULA - Posicionada en el centro del cuarto, reflectiva y sin textura
    capsula = Capsule(position=[0.0, 0.0, -5.0], radius=1.0, height=2.0, material=material_capsula)
    renderer.scene.append(capsula)

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
    print("- Una cápsula reflectiva en el centro (sin textura)")
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