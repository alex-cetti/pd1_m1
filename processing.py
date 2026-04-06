import cv2
import numpy as np
from skimage.util import random_noise


import metricas 

def prep_img(img):
    #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array(gray, dtype=np.uint8)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_noise(img, mode='gaussian'):
    #mode='gaussian' #clip=True
    #
    
    noised = random_noise(img, mode=mode, var=0.003 )
    
    return (noised * 255).astype(np.uint8)

def hist_norm(img):

    point_start, point_count = metricas.hist(img)
    pdf = point_count / np.sum(point_count)
    cdf = np.cumsum(pdf)
    f_eq = np.round(cdf * 255).astype(np.uint8)

    img_eq = f_eq[img]
    return img_eq

def gamma_correction(img, gamma=1.7):
    
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    
    return gamma_corrected


def add_padding(img, padding_height, padding_width):
    n, m = img.shape


    ## PReenche as
    padded_img = np.zeros((n + padding_height * 2, m + padding_width * 2))
    padded_img[padding_height : n + padding_height, padding_width : m + padding_width] = img

    return padded_img


def convolution(img, kernel_name, kernel_size=3, padding=True):

    kernel = get_kernel(kernel_name, kernel_size)

    k_height, k_width = kernel.shape


    img_height, img_width = img.shape

    #Tamanho do padding, de acordo com o kernel
    pad_height = k_height // 2
    pad_width = k_width // 2


    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Executa o Padding
    else:
        padded_img = img


    output = np.zeros((img_height, img_width), dtype=float)  # Cria matriz de 0 para o output

    # Perform convolution



  

    for i in range(img_height):
       for j in range(img_width):

           ## Regiao de interesse na imagem
           reg = padded_img[i:i+k_height, j:j+k_width]
           if kernel_name == "Mediana":
             valor = np.median(reg.flatten())
           else:
             valor = np.sum(reg * kernel)

           output[i, j] =  valor



    
    return np.array(output, dtype=np.uint8)



def get_kernel(name, size=3, sigma=2):
  if name == "sobel":
    return np.array([
        [-1, 0, 1],
        [-2, 2, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
  elif name == "sharpen":
    return np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

  elif name=="Gauss":
    return gauss_create(sigma, size)

  elif name =="Media":
    return media_create(size)

  elif name == "Mediana":
    return zero_create(size)


  elif name == "LA":
    return np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float32)
  elif name == "LB":
    return np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ], dtype=np.float32)
  elif name == "LC":
    return np.array([
        [0, -1, 0],
        [-1, -4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
  elif name == "LD":
    return np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
def gauss_create(sigma, size):
  x, y = np.meshgrid(np.linspace(-1,1,size), np.linspace(-1,1,size))
  calc = 1/((2*np.pi*(sigma**2)))
  exp = np.exp(-(((x**2) + (y**2))/(2*(sigma**2))))

  return exp*calc

def media_create(size):
  return np.ones((size, size), dtype=np.float32) / (size ** 2) ## Divisao pela quantidade de pixels, para n~ao da overflow

def zero_create(size):
  return  np.zeros((size,size), dtype=float)




def main():
    print("--main")

if __name__ == "__main__":
    main()