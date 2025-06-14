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

    def mostrar_cortes(self):
        if self.volumen is None:
            print("No se ha reconstrudi el volumen 3D, hazlo!!.")
            return

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(self.volumen[self.volumen.shape[0] // 2], cmap='gray')
        axs[0].set_title("Corte axial")
        axs[1].imshow(self.volumen[:, self.volumen.shape[1] // 2, :], cmap='gray')
        axs[1].set_title("Corte coronal")
        axs[2].imshow(self.volumen[:, :, self.volumen.shape[2] // 2], cmap='gray')
        axs[2].set_title("Corte sagital")
        plt.show()
    
    def transformar_imagen(self, dx, dy, salida="imagen_nueva.png"):
        if not self.dicoms:
            print("No hay imágenes cargadas.")
            return

        imagen = self.dicoms[len(self.dicoms) // 2].pixel_array
        filas, columnas = imagen.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        trasladada = cv2.warpAffine(imagen, M, (columnas, filas))

        plt.subplot(1, 2, 1)
        plt.imshow(imagen, cmap='gray')
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(trasladada, cmap='gray')
        plt.title(f"Trasladada ({dx},{dy})")
        plt.show()

        cv2.imwrite(salida, trasladada)
        print(f"Imagen tranformada guardada como {salida}")

class ImagenSencilla: 
    class ImagenSencilla:
        def __init__(self, carpeta="imagenes"):
            self.carpeta = carpeta
            self.imagenes = {}

        def cargar_imagenes(self):
            extensiones = (".png", ".jpg", ".jpeg", ".bmp")
            for archivo in os.listdir(self.carpeta):
                if archivo.lower().endswith(extensiones):
                    ruta = os.path.join(self.carpeta, archivo)
                    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                    if imagen is not None:
                        self.imagenes[archivo] = imagen
        
        def binarizacion(self, imagen, tipo, umbral): 
            if tipo == 1: 
                resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
            elif tipo == 2: 
                resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY_INV) 
            elif tipo == 3: 
                resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TRUNC) 
            elif tipo == 4: 
                resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TOZERO)
            elif tipo == 5: 
                resultado = cv2.threshold(imagen, umbral, 255, cv2.THRESH_TOZERO_INV)
            else: 
                resultado = imagen.copy()
            return resultado 
        
        def transformacion(self, imagen, tipo, tamano_kernel):
            kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)
            if tipo == 1:
                transformada = cv2.erode(imagen, kernel)
                nombre = "Erosion"
            elif tipo == 2:
                transformada = cv2.dilate(imagen, kernel)
                nombre = "Dilatacion"
            elif tipo == 3:
                transformada = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
                nombre = "Apertura"
            elif tipo == 4:
                transformada = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
                nombre = "Cierre"
            elif tipo == 5:
                transformada = cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)
                nombre = "Gradiente"
            elif tipo == 6:
                transformada = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
                nombre = "Top-hat"
            elif tipo == 7:
                transformada = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
                nombre = "Black-hat"
            elif tipo == 8:
                invertida = invert(imagen // 255)
                esqueleto = skeletonize(invertida)
                transformada = (1 - esqueleto).astype(np.uint8) * 255
                nombre = "Esqueletizacion"
            else:
                transformada = imagen.copy()
                nombre = "Sin cambio"
            return transformada, nombre
        
        def dibujar(self, imagen, forma, texto):
            if forma == "cuadrado":
                cv2.rectangle(imagen, (10, 10), (300, 60), 255, -1)
                cv2.putText(imagen, texto, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            elif forma == "circulo":
                cv2.circle(imagen, (150, 100), 50, 255, -1)
                cv2.putText(imagen, texto, (90, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            return imagen
        
        
