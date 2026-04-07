import cv2
import numpy as np
from skimage.util import random_noise


import metricas 

def prep_img(img):
    #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array(gray, dtype=np.uint8)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_noise(img, mode='gaussian'):
    #mode='gaussian' #clip=True
    #
    
    noised = random_noise(img, mode=mode )
    
    return noised 

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



def conv2d_sharpening(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img

    #print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output

    # Perform convolution
    for i_img in range(img_height):  # Loop usando i
        for j_img in range(img_width):  # Loop usando j
            #calcula kernel
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    output[i_img, j_img] = output[i_img, j_img] + (padded_img[i_img+i_kernel, j_img+j_kernel] * kernel[i_kernel, j_kernel])  # Atribui valor à variável output[i, j]
            output[i_img, j_img] = int(output[i_img, j_img])

    return np.array(output, dtype=np.float32)

def medianFilter(img, kernel, padding=True):
    # Get dimensions of the kernel
    k_height, k_width = kernel.shape  # Atribui valor à variável k_height, k_width

    # Get dimensions of the image
    img_height, img_width = img.shape  # Atribui valor à variável img_height, img_width

    # Calculate padding required
    pad_height = k_height // 2  # Atribui valor à variável pad_height
    pad_width = k_width // 2  # Atribui valor à variável pad_width

    # Create a padded version of the image to handle edges
    if padding == True:
        padded_img = add_padding(img, pad_height, pad_width)  # Atribui valor à variável padded_img
    else:
        padded_img = img
    #print(padded_img)

    # Initialize an output image with zeros
    output = np.zeros((img_height, img_width), dtype=float)  # Atribui valor à variável output
    kernel_vectorized = np.zeros(k_height*k_width)
    #print(kernel_vectorized)
    # Perform convolution
    for i_img in range(img_height):  # Loop usando i
        for j_img in range(img_width):  # Loop usando j
            i_vector = 0
            for i_kernel in range(k_height):
                for j_kernel in range(k_width):
                    kernel_vectorized[i_vector] = padded_img[i_img+i_kernel, j_img+j_kernel]  # Atribui valor à variável output[i, j]
                    i_vector+=1
            kernel_vectorized = bubble_sort(kernel_vectorized)
            median_index = int(((k_height*k_width)/2)+1)
            output[i_img, j_img] = int(kernel_vectorized[median_index])

    return np.array(output, dtype=np.uint8)




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
  
  if name == "sobel_2":
    return np.array(([-1,0,1],[-2,0,2],[-1,0,1]))
  elif name =="laplac":
     return np.array(([0,1,0],[1,-4,1],[0,1,0]))
  elif name == "sobel_1":
    return np.array(([-1,-2,-1],[0,0,0],[1,2,1]))
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