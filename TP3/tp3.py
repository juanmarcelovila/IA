"""
IA TP3 Reconocimiento de imagen con Hopfield
=========================================
Si el resultado es inverso(pixeles negados) fallo el reconocimiento.
"""
import cv2, glob
import numpy as numpy
import neurolab as neurolab
import matplotlib.pyplot as plot

IMAGE_SIZE = (48, 48, 3) #tamaño de 48x48 pixels
TEST_PATH = './datos/prueba/*.png'
TRAIN_PATH = './datos/entrena/*.png'

def img2array(name):
  img = cv2.imread(name) #lee imagen binaria
  img = img.flatten() #Retorna una copia del array colapsado en una dimension.
  img[img == 255] = 1
  return img

def array2img(array):
  array[array == -1] = 0
  array *= 255 # asignan espacios blancos
  img = numpy.reshape(array,IMAGE_SIZE) #transforma el array de una dimension en en un array multidimension
  return img

def array2float(array):
  tmp = numpy.asfarray(array)
  tmp[tmp == 0] = -1 
  return tmp

def show_images(images):
  fig = plot.figure()
  ax = fig.add_subplot(1, 2, 1)
  imgplot = plot.imshow(images[0])
  ax.set_title('Analizado')
  ax = fig.add_subplot(1, 2, 2)
  imgplot = plot.imshow(images[1])
  ax.set_title('Resultado')
  plot.show()

target = []
for file in glob.glob(TRAIN_PATH):
  array = img2array(file)
  target.append(array)
target = array2float(target)
net = neurolab.net.newhop(target) # Crea y entrena la red Hopfile
img_test = img2array('./datos/prueba/p.png')# Este es el archivo a probar
test = array2float(img_test)
out = net.sim([test]) #prueba la red
out_image = array2img(out[0]) #imagen de salida de la red
img_test = array2img(test) #imagen de prueba
show_images([img_test, out_image])