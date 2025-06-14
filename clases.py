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



