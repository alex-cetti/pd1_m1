## Metricas
import numpy as np
from SSIM_PIL import compare_ssim
from PIL import Image
import sporco.metric as sm
import matplotlib.pyplot as plt



def data_table(img_a, img_b):

    data = [
        ["Metricas", "Resultado"],
        ["MSE", f"{round(mse(img_a, img_b), 3)}"],
        ["PNSR", f"{round(psnr(img_a, img_b), 3)}"],
        ["SSIM", f"{round(ssim(img_a, img_b), 3)}"],
        ["GMSE", f"{round(gmsd(img_a, img_b), 3)}"],
    ]

    return  data

def draw_hist(img, ax, title):
    x_axis, point_count = hist(img)
    
    ax.set_title(title) 
    ax.set_xlabel("Intensidade")
    ax.set_ylabel("Frequência")

    return  

def hist(img):
    intervalo_min_max = [0, 256]
    num_pontos = 256
    point_count, point_edges = np.histogram(img, num_pontos, intervalo_min_max)
    point_start = point_edges[:-1]

    return point_count, point_start


def mse(img_a, img_b):
    err =  np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
    err /= float(img_a.shape[0] * img_a.shape[1])


    return err


def psnr(original, compressed):
    mse = np.mean(( original - compressed ) ** 2)

    if(mse == 0): #Imagens Iguais
        return 100

    max_pixel = 255.0

    v_psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return v_psnr


def ssim(img_a, img_b):
    value =  compare_ssim( Image.fromarray(img_a),  Image.fromarray(img_b))

    return value


def gmsd(img_a, img_b):
    value = sm.gmsd(img_a, img_b)

    return value



def main():
    print("--main")

if __name__ == "__main__":
    main()