import medmnist  
import random 
import numpy as np

from medmnist import ChestMNIST, RetinaMNIST, BloodMNIST

def load_dataset():
    test_dataset = BloodMNIST(split="test", download=True, size=64)
    
    
    print(test_dataset[0])
    return test_dataset



def create_dataset(size=3, rand=False):
    ds = load_dataset()
    output = []
    for i in range(size):
        if rand:
            idx = random.randint(0, len(ds) - 1)
            img, lbl = ds[idx]
            a_img =  np.array(img)
            output.append(a_img) 
        else:
            img, lbl = ds[i]
            a_img =  np.array(img)
            output.append(a_img) 
        
    return output

def test():
    return "TESTADO"


def main():


    
    print(medmnist.__version__)

if __name__ == "__main__":
    main()
