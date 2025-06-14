import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

class Paciente:
    def __init__(self, nombre, edad, id, imagen):
        self.nombre = nombre
        self.edad = edad
        self.paciente_id = id
        self.imagen_3d = imagen

class ArchivosDicom:
    def __init__(self, carpeta = "DICOMS"):
        self.carpeta = carpeta
        self.dicoms = []
        self.volumen = None
        self.dataset_ejemplo = None

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

    def seleccionar_dataset(self):
        if not self.dicoms:
            print("No hay archivos DICOM cargados.")
            return

        print("Archivos disponibles:")
        for i in range(len(self.dicoms)):
            print(f"{i + 1}. Archivo {i + 1}")

        try:
            eleccion = int(input("Ingrese el número del archivo que desea usar como ejemplo: ")) - 1
            if 0 <= eleccion < len(self.dicoms):
                self.dataset = self.dicoms[eleccion]
                print(f"Se seleccionó el archivo número {eleccion + 1}.")

                # Mostrar metadatos anonimizados
                nombre = self.dataset.get("PatientName", "No disponible")
                edad = self.dataset.get("PatientAge", "No disponible")
                pid = self.dataset.get("PatientID", "No disponible")

                print("Metadatos extraídos:")
                print(f"Nombre del paciente: {nombre}")
                print(f"Edad: {edad}")
                print(f"ID del paciente: {pid}")

            else:
                print("Número fuera de rango.")
        except ValueError:
            print("Entrada no válida. Debes ingresar un número.")
    
    def reconstruccion_3d(self):
        if not self.dicoms:
            print("No hay archivos DICOM cargados.")
            return None

        try:
            self.dicoms.sort(key=lambda ds: int(ds.InstanceNumber))
        except AttributeError:                                                          #estaparte de codigo permite saber si los cortes estan organizados para una buena reconstruccion
            print("Algunos archivos no tienen InstanceNumber. Orden incorrecto.")
            return None

        try:
            self.volumen = np.stack([ds.pixel_array for ds in self.dicoms])
            print("Reconstrucción 3D completada.")
            return self.volumen
        except Exception as e:
            print("Error durante la reconstrucción 3D:", str(e))
            return None
    
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
    
    


