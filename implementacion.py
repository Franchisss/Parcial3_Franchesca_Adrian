from clases import ArchivosDicom, ImagenSencilla, Paciente
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Diccionarios globales
diccionarios_dicom = {}  # clave: nombre, valor: instancia ArchivosDicom
diccionario_pacientes = {}  # clave: nombre paciente, valor: objeto Paciente
diccionario_imagenes = {}  # clave: nombre imagen, valor: imagen procesada o DICOM

# Menú principal
def menu():
    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("a. Procesar archivos DICOM")
        print("b. Ingresar paciente")
        print("c. Ingresar imágenes JPG o PNG")
        print("d. Procesar imagen sencilla")
        print("e. Transformar imagen DICOM")
        print("f. Salir")

        opcion = input("Seleccione una opción: ").lower()

        if opcion == "a":
            procesar_dicom()
        elif opcion == "b":
            ingresar_paciente()
        elif opcion == "c":
            ingresar_imagenes_sencillas()
        elif opcion == "d":
            procesar_imagen_sencilla()
        elif opcion == "e":
            transformar_imagen_dicom()
        elif opcion == "f":
            print("Saliendo del programa...")
            break
        else:
            print("Opción no válida. Intente de nuevo.")

# a. Procesar archivos DICOM y guardarlos en diccionario
def procesar_dicom():
    clave = input("Ingrese un nombre clave para este conjunto de DICOMs: ")
    dicom = ArchivosDicom("Datos")
    dicom.cargar_dicoms()
    dicom.reconstruccion_3d()
    dicom.mostrar_cortes(salida=f"cortes_{clave}.png")
    diccionarios_dicom[clave] = dicom
    print(f"DICOM procesado y guardado bajo la clave: {clave}")

# b. Ingresar paciente desde un DICOM procesado previamente
def ingresar_paciente():
    clave = input("Ingrese la clave del DICOM previamente procesado: ")
    if clave not in diccionarios_dicom:
        print("Clave no encontrada en el diccionario de DICOMs.")
        return

    dicom = diccionarios_dicom[clave]
    if not dicom.dicoms:
        print("No hay datos DICOM cargados para esta clave.")
        return

    dataset = dicom.dicoms[len(dicom.dicoms) // 2]  # Seleccionar uno intermedio
    nombre = str(dataset.get("PatientName", "Anonimo"))
    edad = str(dataset.get("PatientAge", "00"))
    pid = str(dataset.get("PatientID", "0000"))

    paciente = Paciente(nombre, edad, pid, dicom.volumen)
    diccionario_pacientes[nombre] = paciente
    diccionario_imagenes[nombre] = dataset  # Guardar referencia

    print(f"Paciente '{nombre}' ingresado correctamente.")

# c. Ingresar imágenes JPG o PNG al diccionario global
def ingresar_imagenes_sencillas():
    carpeta = "imagenes"
    extensiones = (".png", ".jpg")
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(extensiones):
            ruta = os.path.join(carpeta, archivo)
            imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if imagen is not None:
                diccionario_imagenes[archivo] = imagen
    print("Imágenes JPG/PNG almacenadas en el diccionario.")

# d. Procesar imagen sencilla ya almacenada
def procesar_imagen_sencilla():
    if not diccionario_imagenes:
        print("No hay imágenes en el diccionario para procesar.")
        return

    print("\nImágenes disponibles:")
    for nombre in diccionario_imagenes:
        print("-", nombre)

    nombre_imagen = input("Nombre exacto de la imagen a procesar: ")
    if nombre_imagen not in diccionario_imagenes:
        print("Nombre no encontrado en el diccionario.")
        return

    try:
        tipo_umbral = int(input("Tipo de umbral (1-5): "))
        umbral_valor = int(input("Valor del umbral (ej: 127): "))
        tipo_transformacion = int(input("Tipo de transformación (1-8): "))
        kernel = int(input("Tamaño del kernel (ej: 5): "))
        forma = input("Forma (cuadrado/circulo): ").strip()

        sencilla = ImagenSencilla()
        sencilla.imagenes[nombre_imagen] = diccionario_imagenes[nombre_imagen]
        resultado = sencilla.procesar_imagen(nombre_imagen, tipo_umbral, umbral_valor,
                                             tipo_transformacion, kernel, forma)
        if resultado:
            print(f"Imagen procesada guardada como: {resultado}")
    except Exception as e:
        print("Error al procesar la imagen:", e)

# e. Transformar imagen DICOM previamente procesada
def transformar_imagen_dicom():
    clave = input("Ingrese la clave del DICOM previamente procesado: ")
    if clave not in diccionarios_dicom:
        print("Clave no encontrada.")
        return

    dx = int(input("Valor de desplazamiento horizontal (dx): "))
    dy = int(input("Valor de desplazamiento vertical (dy): "))

    dicom = diccionarios_dicom[clave]
    dicom.transformar_imagen(dx, dy, salida=f"transformada_{clave}.png")

# Ejecutar menú principal
if __name__ == "__main__":
    menu()