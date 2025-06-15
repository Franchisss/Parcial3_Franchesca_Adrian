import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from skimage.morphology import skeletonize 
from skimage.util import invert 

class Paciente:
    def __init__(self, nombre, edad, id, imagen):
        self.nombre = nombre
        self.edad = edad
        self.paciente_id = id
        self.imagen_3d = imagen

class ArchivosDicom:
    def __init__(self, carpeta = "Datos"):
        self.carpeta = carpeta
        self.dicoms = []
        self.volumen = None


    def cargar_dicoms(self):
        archivos = sorted(
            [f for f in os.listdir(self.carpeta) if f.endswith(".dcm")],
            key=lambda x: int(x.split('.')[0])
        )

        if not archivos:
            print("No se encontraron archivos .dcm en la carpeta.")
            return

        for archivo in archivos:
            ruta_completa = os.path.join(self.carpeta, archivo)
            ds = pydicom.dcmread(ruta_completa)
            self.dicoms.append(ds)

        print(f"Se cargaron {len(self.dicoms)} archivos DICOM.")

    def reconstruccion_3d(self):
        if not self.dicoms:
            print("No hay archivos DICOM cargados.")
            return None

        print(f"\nReconstruyendo volumen 3D usando {len(self.dicoms)} archivos...")

        try:
            # Ordenar por InstanceNumber si está presente
            self.dicoms.sort(key=lambda ds: int(ds.InstanceNumber))
        except AttributeError:
            print("Advertencia: Algunos archivos no tienen InstanceNumber. Se usará el orden por nombre.")

        # Verificar formas
        formas = [ds.pixel_array.shape for ds in self.dicoms]
        forma_referencia = formas[0]

        imagenes_validas = []
        for i, ds in enumerate(self.dicoms):
            try:
                pix = ds.pixel_array
                if pix.shape == forma_referencia:
                    imagenes_validas.append(pix)
            except Exception as e:
                print(f"Corte {i+1} omitido: {e}")

        if not imagenes_validas:
            print("No se encontraron imágenes válidas.")
            return None

        self.volumen = np.stack(imagenes_validas)
        print("Reconstrucción 3D completada.")
        print(f"Volumen: {self.volumen.shape}")
        return self.volumen

    def mostrar_cortes(self, salida="cortes_volumen.png"):
        if self.volumen is None:
            print("No se ha reconstruido el volumen 3D, hazlo primero.")
            return

        # Crear carpeta si no existe
        carpeta = "imagenes"
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, salida)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(self.volumen[self.volumen.shape[0] // 2], cmap='gray')
        axs[0].set_title("Corte axial")
        axs[0].axis("off")

        axs[1].imshow(self.volumen[:, self.volumen.shape[1] // 2, :], cmap='gray')
        axs[1].set_title("Corte coronal")
        axs[1].axis("off")

        axs[2].imshow(self.volumen[:, :, self.volumen.shape[2] // 2], cmap='gray')
        axs[2].set_title("Corte sagital")
        axs[2].axis("off")

        plt.tight_layout()
        fig.savefig(ruta_salida, bbox_inches='tight')
        plt.show()

        print(f"Visualización de cortes guardada como: {ruta_salida}")
    
    def transformar_imagen(self, dx, dy, salida="comparacion_transformada.png"):
        if not self.dicoms:
            print("No hay imágenes cargadas.")
            return

        imagen = self.dicoms[len(self.dicoms) // 2].pixel_array
        filas, columnas = imagen.shape

        # Crear carpeta si no existe
        carpeta = "imagenes"
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, salida)

        # Aplicar traslación
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        trasladada = cv2.warpAffine(imagen, M, (columnas, filas))

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(imagen, cmap='gray')
        axs[0].set_title("Original")
        axs[0].axis("off")

        axs[1].imshow(trasladada, cmap='gray')
        axs[1].set_title(f"Transformada ({dx},{dy})")
        axs[1].axis("off")

        plt.tight_layout()
        fig.savefig(ruta_salida, bbox_inches='tight')
        plt.show()

        print(f"Comparación guardada como: {ruta_salida}")

class ImagenSencilla: 
    def __init__(self, carpeta="imagenes"):
        self.carpeta = carpeta
        self.imagenes = {}

    def cargar_imagenes(self):
        extensiones = (".png", ".jpg")
        for archivo in os.listdir(self.carpeta):
            if archivo.lower().endswith(extensiones):
                ruta = os.path.join(self.carpeta, archivo)
                imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                if imagen is not None:
                    self.imagenes[archivo] = imagen
    
    def binarizacion(self, imagen, tipo, umbral): 
        if tipo == 1: 
            _, resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
        elif tipo == 2: 
            _, resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY_INV) 
        elif tipo == 3: 
            _, resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TRUNC) 
        elif tipo == 4: 
            _, resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TOZERO)
        elif tipo == 5: 
            _, resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TOZERO_INV)
        else: 
            resultado = imagen.copy()
        return resultado 
    
    def transformacion(self, imagen, tipo, tamano_kernel):
        kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)

        if tipo == 1:  # Erosión
            transformada = cv2.erode(imagen, kernel)
            nombre = "Erosion"
        elif tipo == 2:  # Dilatación
            transformada = cv2.dilate(imagen, kernel)
            nombre = "Dilatacion"
        elif tipo == 3:  # Apertura = erosion seguida de dilatación
            erosionada = cv2.erode(imagen, kernel)
            transformada = cv2.dilate(erosionada, kernel)
            nombre = "Apertura"
        elif tipo == 4:  # Cierre = dilatación seguida de erosión
            dilatada = cv2.dilate(imagen, kernel)
            transformada = cv2.erode(dilatada, kernel)
            nombre = "Cierre"
        elif tipo == 5:  # Gradiente = dilatación - erosión
            dilatada = cv2.dilate(imagen, kernel)
            erosionada = cv2.erode(imagen, kernel)
            transformada = cv2.subtract(dilatada, erosionada)
            nombre = "Gradiente"
        elif tipo == 6:  # Top-hat = imagen - apertura
            erosionada = cv2.erode(imagen, kernel)
            apertura = cv2.dilate(erosionada, kernel)
            transformada = cv2.subtract(imagen, apertura)
            nombre = "Top-hat"
        elif tipo == 7:  # Black-hat = cierre - imagen
            dilatada = cv2.dilate(imagen, kernel)
            cierre = cv2.erode(dilatada, kernel)
            transformada = cv2.subtract(cierre, imagen)
            nombre = "Black-hat"
        elif tipo == 8:  # Esqueletización
            invertida = invert(imagen // 255)
            esqueleto = skeletonize(invertida)
            transformada = (1 - esqueleto).astype(np.uint8) * 255
            nombre = "Esqueletizacion"
        else:
            transformada = imagen.copy()
            nombre = "Sin cambio"

        return transformada, nombre
    
    def dibujar(self, imagen, forma, umbral, kernel):
        if forma == "cuadrado":
            cv2.rectangle(imagen, (10, 10), (300, 80), 255, -1)
            cv2.putText(imagen, "Imagen binarizada", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.putText(imagen, f"Umbral: {umbral}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            cv2.putText(imagen, f"Kernel: {kernel}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        elif forma == "circulo":
            cv2.circle(imagen, (150, 100), 60, 255, -1)
            cv2.putText(imagen, "Imagen binarizada", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.putText(imagen, f"Umbral: {umbral}", (85, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.putText(imagen, f"Kernel: {kernel}", (85, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        return imagen
    
    def procesar_imagen(self, nombre, tipo_umbral, umbral_valor, tipo_transformacion, kernel, forma):
        # Verificar que exista
        if nombre not in self.imagenes:
            print("Imagen no encontrada.")
            return None

        imagen_original = self.imagenes[nombre]

        # Verificar que no esté vacía
        if imagen_original is None or imagen_original.size == 0:
            print("La imagen está vacía o dañada.")
            return None

        # Convertir tipo si no es uint8
        if imagen_original.dtype != np.uint8:
            imagen_original = imagen_original.astype(np.uint8)

        # Verificar tamaño mínimo de imagen para el kernel
        alto, ancho = imagen_original.shape
        if kernel > min(alto, ancho):
            print(f"El tamaño del kernel ({kernel}) es demasiado grande para esta imagen ({alto}x{ancho}).")
            return None

        # Procesar pasos
        imagen_binarizada = self.binarizacion(imagen_original, tipo_umbral, umbral_valor)
        imagen_transformada, nombre_trans = self.transformacion(imagen_binarizada, tipo_transformacion, kernel)
        imagen_final = self.dibujar(imagen_transformada, forma, umbral_valor, kernel)


        carpeta = "imagenes"
        os.makedirs(carpeta, exist_ok=True)

        # Ruta final
        nombre_salida = f"procesada_{nombre}"
        ruta_salida = os.path.join(carpeta, nombre_salida)

        # Guardar la imagen en la carpeta imagenes
        cv2.imwrite(ruta_salida, imagen_final)

        print(f"Imagen procesada guardada en: {ruta_salida}")
        return ruta_salida

