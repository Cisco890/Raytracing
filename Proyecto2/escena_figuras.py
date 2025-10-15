import pygame
import numpy as np
from gl import Renderer
from figures import Sphere, Ellipsoid, Cylinder, Pentagon, Cube, Torus
from material import Material, OPAQUE, REFLECTIVE, TRANSPARENT
from lights import DirectionalLight, AmbientLight, PointLight
from camera import Camera
from BMPTexture import BMPTexture

WIDTH, HEIGHT = 800, 800

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Raytracing - Personaje Articulado')

    renderer = Renderer(screen)
    renderer.camera = Camera()
    
    # Configurar recursion para reflexiones y transparencias
    renderer.maxRecursionDepth = 3
    
    renderer.glClearColor(0.7, 0.6, 0.9)
    try:
        renderer.envMap = BMPTexture("fondo.bmp")
        print("Fondo BMP cargado como envMap")
    except Exception as e:
        print(f"No se pudo cargar 'fondo.bmp' como envMap: {e}. Se usa color solido de fondo")
    # Flip vertical del fondo,
    renderer.envMapFlipV = True
    
    # Material base naranja/marron 
    material_naranja = Material(diffuse=[0.8, 0.5, 0.3], spec=32, ks=0.3, matType=OPAQUE)

    # Texturas y materiales solicitados
    # Pentagon: dry.bmp, verde opaco
    try:
        dry_texture = BMPTexture("dry.bmp")
        print("Textura dry.bmp cargada")
    except Exception as e:
        dry_texture = None
        print(f"No se pudo cargar dry.bmp: {e}")
    material_pentagono_verde = Material(
        diffuse=[0.1, 0.8, 0.2], spec=32, ks=0.3, matType=OPAQUE, texture=dry_texture
    )

    # Torso: metal.bmp, reflectivo dorado
    try:
        metal_texture = BMPTexture("metal.bmp")
        print("Textura metal.bmp cargada")
    except Exception as e:
        metal_texture = None
        print(f"No se pudo cargar metal.bmp: {e}")
    material_torso_dorado = Material(
        diffuse=[0.85, 0.65, 0.25], spec=64, ks=0.4, matType=REFLECTIVE, texture=metal_texture
    )

    # Cubos (pies): quarzo.bmp azul opaco 
    cuarzo_tex = None
    try:
        cuarzo_tex = BMPTexture("quarzo.bmp")
        print("Textura quarzo.bmp cargada")
    except Exception as e1:
        cuarzo_tex = None
        print(f"No se pudo cargar cuarzo.bmp: {e1}")

    material_cubos_azul = Material(
        diffuse=[0.2, 0.5, 1.0], spec=32, ks=0.2, matType=OPAQUE, texture=cuarzo_tex
    )

    # Manos (torus): 
    try:
        mosaico_texture = BMPTexture("mosaico.bmp")
        print("Textura mosaico.bmp cargada")
    except Exception as e:
        mosaico_texture = None
        print(f"No se pudo cargar mosaico.bmp: {e}")
    # Material transparente claro para manos 
    material_manos_transp_clear = Material(
        diffuse=[1.0, 1.0, 1.0], ior=1.3, ks=0.1, spec=64, matType=TRANSPARENT, texture=None
    )

    # Esferas (cabeza y codos):  transparentes clear 
    try:
        blanca_texture = BMPTexture("blanca.bmp")
        print("Textura blanca.bmp cargada")
    except Exception as e:
        blanca_texture = None
        print(f"No se pudo cargar blanca.bmp: {e}")
    material_esferas_transp_clear = Material(
        diffuse=[1.0, 1.0, 1.0], ior=1.3, ks=0.1, spec=64, matType=TRANSPARENT, texture=blanca_texture
    )

    # Piernas: ceramica.bmp naranjas (opacas)
    try:
        ceramica_texture = BMPTexture("ceramica.bmp")
        print("Textura ceramica.bmp cargada")
    except Exception as e:
        ceramica_texture = None
        print(f"No se pudo cargar ceramica.bmp: {e}")
    material_piernas_ceramica_naranja = Material(
        diffuse=[0.95, 0.55, 0.2], spec=32, ks=0.2, matType=OPAQUE, texture=ceramica_texture
    )
    
    
    print("Construyendo personaje articulado...")


    # ===== CONSTRUCCIÓN DEL PERSONAJE =====

    # SOMBRERO/GORRO (Pentágono)
    sombrero = Pentagon(
        position=[0.0, 1.7, -8.0],  # \
        radius=0.35,
        normal=[0, 0, 1], \
        material=material_pentagono_verde
    )
    renderer.scene.append(sombrero)
    
    # CABEZA (Esfera)
    cabeza = Sphere(
        position=[0.0, 1.2, -8.0],
        radius=0.35,
        material=material_esferas_transp_clear
    )
    renderer.scene.append(cabeza)
    
    # CUELLO (Cilindro pequeño)
    cuello = Cylinder(
        position=[0.0, 0.7, -8.0],
        radius=0.12,
        height=0.3,
        material=material_naranja
    )
    renderer.scene.append(cuello)
    
    # TORSO (Elipsoide - cuerpo principal)
    torso = Ellipsoid(
        position=[0.0, 0.2, -8.0],
        radii=[0.6, 0.8, 0.5],
        material=material_torso_dorado
    )
    renderer.scene.append(torso)
    
    # BRAZOS SUPERIORES (Cilindros)
    brazo_superior_izq = Cylinder(
        position=[-0.9, 0.1, -8.0],
        radius=0.12,
        height=0.6,
        material=material_naranja
    )
    renderer.scene.append(brazo_superior_izq)
    
    brazo_superior_der = Cylinder(
        position=[0.9, 0.1, -8.0],
        radius=0.12,
        height=0.6,
        material=material_naranja
    )
    renderer.scene.append(brazo_superior_der)
    
    # CODOS (Esferas pequeñas)
    codo_izq = Sphere(
        position=[-0.9, -0.3, -8.0],
        radius=0.15,
        material=material_esferas_transp_clear
    )
    renderer.scene.append(codo_izq)
    
    codo_der = Sphere(
        position=[0.9, -0.3, -8.0],
        radius=0.15,
        material=material_esferas_transp_clear
    )
    renderer.scene.append(codo_der)
    
    # BRAZOS INFERIORES (Cilindros)
    brazo_inferior_izq = Cylinder(
        position=[-0.9, -0.7, -8.0],
        radius=0.12,
        height=0.5,
        material=material_naranja
    )
    renderer.scene.append(brazo_inferior_izq)
    
    brazo_inferior_der = Cylinder(
        position=[0.9, -0.7, -8.0],
        radius=0.12,
        height=0.5,
        material=material_naranja
    )
    renderer.scene.append(brazo_inferior_der)
    
    # MANOS Torus pequeños 
    mano_izq = Torus(
        position=[-0.9, -1.0, -7.5],
        major_radius=0.15,
        minor_radius=0.05,
        material=material_manos_transp_clear,
        axis='z'  
    )
    renderer.scene.append(mano_izq)
    
    mano_der = Torus(
        position=[0.9, -1.0, -7.5],
        major_radius=0.15,
        minor_radius=0.05,
        material=material_manos_transp_clear,
        axis='z'  
    )
    renderer.scene.append(mano_der)
    
    # PIERNAS (Cilindros)
    pierna_izquierda = Cylinder(
        position=[-0.4, -0.9, -8.0],
        radius=0.15,
        height=0.8,
        material=material_piernas_ceramica_naranja
    )
    renderer.scene.append(pierna_izquierda)
    
    pierna_derecha = Cylinder(
        position=[0.4, -0.9, -8.0],
        radius=0.15,
        height=0.8,
        material=material_piernas_ceramica_naranja
    )
    renderer.scene.append(pierna_derecha)
    
    # PIES (Cubos) 
    pie_izquierdo = Cube(
        position=[-0.4, -1.5, -8.0],  
        size=0.4,
        material=material_cubos_azul
    )
    renderer.scene.append(pie_izquierdo)
    
    pie_derecho = Cube(
        position=[0.4, -1.5, -8.0],  
        size=0.4,
        material=material_cubos_azul
    )
    renderer.scene.append(pie_derecho)

    # ===== ILUMINACIÓN =====
    
    # Luz principal lateral (simula iluminación de estudio)
    luz_principal = PointLight(
        color=[1.0, 0.9, 0.7],  # Luz cálida
        intensity=20.0,
        position=[-3.0, 2.0, -5.0]
    )
    renderer.lights.append(luz_principal)
    
    # Luz de relleno suave
    luz_relleno = PointLight(
        color=[0.9, 0.8, 0.6],
        intensity=8.0,
        position=[2.0, 1.0, -6.0]
    )
    renderer.lights.append(luz_relleno)
    
    # Luz ambiental cálida
    luz_ambiental = AmbientLight(
        color=[1.0, 0.8, 0.6],
        intensity=0.4
    )
    renderer.lights.append(luz_ambiental)
    
    # Luz trasera para contorno
    luz_trasera = PointLight(
        color=[1.0, 0.7, 0.5],
        intensity=10.0,
        position=[0.0, 1.0, -10.0]
    )
    renderer.lights.append(luz_trasera)

    print("Iniciando renderizado...")
    print("\nPiezas del personaje:")
    print("  * Cabeza: Esfera")
    print("  * Sombrero: Pentagono")
    print("  * Torso: Elipsoide")
    print("  * Brazos y piernas: Cilindros")
    print("  * Codos: Esferas")
    print("  * Manos: Torus")
    print("  * Pies: Cubos")

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