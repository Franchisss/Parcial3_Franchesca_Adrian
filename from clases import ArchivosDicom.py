from clases import ArchivosDicom

# Crear objeto usando la carpeta 'datos'
archivos = ArchivosDicom("datos")

# Paso 1: Cargar archivos DICOM
archivos.cargar_dicoms()

# Paso 2: Verificar que se cargaron
if not archivos.dicoms:
    print("No se cargaron archivos. Verifica la carpeta.")
else:
    print(f"{len(archivos.dicoms)} archivos cargados correctamente.")

# Paso 3: Intentar reconstrucción 3D
volumen = archivos.reconstruccion_3d()

if volumen is not None:
    print("Reconstrucción 3D exitosa.")
    print(f"Dimensiones del volumen: {volumen.shape}")

    # Paso 4: Mostrar cortes
    archivos.mostrar_cortes()

    # Paso 5: Probar transformación
    archivos.transformar_imagen(dx=20, dy=30)
else:
    print("No se pudo reconstruir el volumen.")
