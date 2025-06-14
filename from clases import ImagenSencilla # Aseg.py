from clases import ImagenSencilla
import matplotlib.pyplot as plt

# Crear el objeto y cargar imágenes desde la carpeta "imagenes"
sistema = ImagenSencilla("imagenes")
sistema.cargar_imagenes()

# Mostrar imágenes cargadas
print("Imágenes encontradas:")
for nombre in sistema.imagenes:
    print("-", nombre)

# Pedir al usuario el nombre de la imagen exacta
nombre = input("\nEscribe el nombre exacto de la imagen (ej: foto1.png): ")

# Verificar si existe la imagen
if nombre not in sistema.imagenes:
    print("La imagen no fue encontrada.")
else:
    # Parámetros de prueba (puedes cambiarlos si quieres)
    tipo_umbral = 1          # 1: binario
    umbral_valor = 127       # valor de umbral
    tipo_transformacion = 4  # 4: cierre
    kernel = 5               # tamaño de kernel
    forma = "cuadrado"       # "cuadrado" o "circulo"

    # Procesar imagen
    resultado = sistema.procesar_imagen(
        nombre,
        tipo_umbral,
        umbral_valor,
        tipo_transformacion,
        kernel,
        forma
    )

    # Mostrar resultado
    if resultado:
        print(f"\nImagen procesada y guardada como: {resultado}")
        plt.imshow(sistema.imagenes[nombre], cmap='gray')
        plt.title("Imagen original")
        plt.axis("off")
        plt.show()
    else:
        print("No se pudo procesar la imagen.")


