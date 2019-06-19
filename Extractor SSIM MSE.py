#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:13:01 2019

@author: roman
"""

import numpy as np
from skimage import measure
from PIL import Image, ImageDraw, ImageFont
import math
import statistics
MSEarrayrecup = []
MSEarrayblur = []
SSIMarrayrecup = []
SSIMarrayblur = []
for i in range(137):
    im = Image.open('/home/roman/Documents/TFG propio 2/Imagenes Recuperadas Solo Ruido/Imagen_Calar_Alto_Rec_%d.jpg' % i).convert("L")
    imData = np.asarray(im)
    imorig = imData[:,0:255]
    imrecup = imData[:,256:511]
    imblur = imData[:,512:767]
    #MSE
    mse_recup = measure.compare_mse(imorig,imrecup)
    mse_blur = measure.compare_mse(imorig,imblur)
    MSEarrayrecup.append(mse_recup)
    MSEarrayblur.append(mse_blur)
    #SSIM
    ssim_recup = measure.compare_ssim(imorig, imrecup)
    ssim_blur = measure.compare_ssim(imorig, imblur)
    SSIMarrayrecup.append(ssim_recup)
    SSIMarrayblur.append(ssim_blur)
print("MSE recuperado medio "+str(statistics.mean(MSEarrayrecup)))
print("MSE borroso medio "+str(statistics.mean(MSEarrayblur)))
print("SSIM recuperado medio "+str(statistics.mean(SSIMarrayrecup)))
print("SSIM borroso medio "+str(statistics.mean(SSIMarrayblur)))
indices = np.argsort(SSIMarrayrecup)
print("0%: MSE recuperado: "+str(MSEarrayrecup[indices[0]])+" MSE blur: "+str(MSEarrayblur[indices[0]])+" SSIM recuperado: "
      +str(SSIMarrayrecup[indices[0]])+" SSIM blur: "+str(SSIMarrayblur[indices[0]])+" Imagen número: "+str(indices[0]))
print("25%: MSE recuperado: "+str(MSEarrayrecup[indices[33]])+" MSE blur: "+str(MSEarrayblur[indices[33]])+" SSIM recuperado: "
      +str(SSIMarrayrecup[indices[33]])+" SSIM blur: "+str(SSIMarrayblur[indices[33]])+" Imagen número: "+str(indices[33]))
print("50%: MSE recuperado: "+str(MSEarrayrecup[indices[68]])+" MSE blur: "+str(MSEarrayblur[indices[68]])+" SSIM recuperado: "
      +str(SSIMarrayrecup[indices[68]])+" SSIM blur: "+str(SSIMarrayblur[indices[68]])+" Imagen número: "+str(indices[68]))
print("75%: MSE recuperado: "+str(MSEarrayrecup[indices[108]])+" MSE blur: "+str(MSEarrayblur[indices[108]])+" SSIM recuperado: "
      +str(SSIMarrayrecup[indices[108]])+" SSIM blur: "+str(SSIMarrayblur[indices[108]])+" Imagen número: "+str(indices[108]))
print("100%: MSE recuperado: "+str(MSEarrayrecup[indices[136]])+" MSE blur: "+str(MSEarrayblur[indices[136]])+" SSIM recuperado: "
      +str(SSIMarrayrecup[indices[136]])+" SSIM blur: "+str(SSIMarrayblur[indices[136]])+" Imagen número: "+str(indices[136]))
for i in range(1,6):
    if i==1:
        j=indices[0]
    if i==2:
        j=indices[33]
    if i==3:
        j=indices[68]
    if i==4:
        j=indices[108]
    if i==5:
        j=indices[136]
    image = Image.open('/home/roman/Documents/TFG propio 2/Imagenes Recuperadas Solo Ruido/Imagen_Calar_Alto_Rec_%d.jpg' % j)
    font_type = ImageFont.truetype('/home/roman/Downloads/arial.ttf',15)
    draw = ImageDraw.Draw(image)
    draw.text(xy=(5+256,5),text="SSIM: "
              +str(math.trunc(100*SSIMarrayrecup[j])/100),fill=(255), font=font_type)
    draw.text(xy=(5+512,5),text="SSIM: "
              +str(math.trunc(100*SSIMarrayblur[j])/100),fill=(255), font=font_type)
    draw.text(xy=(5+256,25),text="MSE: "
              +str(math.trunc(100*MSEarrayrecup[j])/100),fill=(255), font=font_type)
    draw.text(xy=(5+512,25),text="MSE: "
              +str(math.trunc(100*MSEarrayblur[j])/100),fill=(255), font=font_type)
    image.save('/home/roman/Desktop/TFG/Solo Ruido %d.jpg' %i)