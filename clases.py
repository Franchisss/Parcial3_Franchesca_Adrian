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
    def __init__(self, carpeta):
        self.carpeta = carpeta
        self.dicoms = []
        self.volumen = None
        self.dataset_ejemplo = None


