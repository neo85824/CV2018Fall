import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def Histogram(img):
    
    hist = np.zeros(256)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1


    cdf = np.zeros(256)
    cdf[0] = hist[0]
    count = 0
    for i in range(1,256):
        cdf[i] += cdf[i-1]+hist[i]

    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i][j] = 255*cdf[img[i][j]]/(img.shape[0]*img.shape[1])
            hist[img[i][j]] += 1


    x = range(0,256)
    plt.plot(x,hist,label = 'histogram')
    plt.fill_between(x,hist) 
    plt.title('histogram')
    plt.savefig('histogram_equalization.png')

    cv2.imwrite('Lena_after.jpg',img)


def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    Histogram(img.copy())
    

if __name__ == '__main__':
    main()