#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:00:30 2019

@author: roman
"""

import numpy as np
from astropy.io import fits
from csv import DictReader
import cv2
import math
from scipy.signal import convolve2d
from astropy.visualization import MinMaxInterval
from PIL import Image
from matplotlib import image
size=256
def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  var = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  var = var * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return var
with open("CALIFA_1_MS_basic.csv") as f:
    ravector = [row["col4"] for row in DictReader(f)]
longitud = len(ravector)
mean = 0
interval = MinMaxInterval()
varianza = 0
for i in range(937):
    data = image.imread('Imagen_Calar_Alto_%d.jpg' % i)
    data = data / 1.0
    varianza =varianza + estimate_noise(data)
varianza = varianza / 937
for i in range(800):
    data = image.imread('Imagen_Calar_Alto_%d.jpg' % i)
    data = data/1.0 #ESTA LINEA NO SE TOCA
    data2 = cv2.GaussianBlur(data,(5,5),1.5)
    gauss = np.random.normal(mean,10.0 * varianza,(size,size)) 
    data2 = data + gauss
    imagen = np.concatenate((data2,data),axis = 1)
    imagen = Image.fromarray(imagen)
    imagen = imagen.convert("L")
    imagen.save('Imagen_Calar_Alto_Train_%d.jpg' % i)
for i in range(137):
    data = image.imread('Imagen_Calar_Alto_%d.jpg' % (i+800))
    data = data/1.0 #ESTA LINEA NO SE TOCA
    data2 = cv2.GaussianBlur(data,(5,5),1.5)
    gauss = np.random.normal(mean,25.0 * varianza,(size,size)) 
    data2 = data + gauss
    imagen = np.concatenate((data2,data),axis = 1)
    imagen = Image.fromarray(imagen)
    imagen = imagen.convert("L")
    imagen.save('Imagen_Calar_Alto_Test_%d.jpg' % i)
#TEST
#print(var_original)
#print(var_blurred)
#print(var_final)
