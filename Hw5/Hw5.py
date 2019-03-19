import cv2
import numpy as np


def dilation(img, kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    result_vmap = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            v_max = 0
            for k in range(len(kernel)):
                if legal(addition(kernel[k],(i,j))):
                    e =  addition(kernel[k],(i,j))
                    if img[e[0]][e[1]] > v_max:
                        v_max = img[e[0]][e[1]]
            result[i][j] = v_max
    cv2.imwrite('dilation.jpg',result)
    return result

def erosion(img , kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    result_vmap = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            v_min = 1e100
            for k in range(len(kernel)):
                if legal(addition(kernel[k],(i,j))):
                    e =  addition(kernel[k],(i,j))
                    if img[e[0]][e[1]] < v_min:
                        v_min = img[e[0]][e[1]]
            result[i][j] = v_min
    cv2.imwrite('erosion.jpg',result)
    return result

    
def addition(a,b):
    r = []
    for i in range(len(a)):
        r.append(a[i]+b[i])
    return tuple(r)

def opening(img, kernel):
    result = erosion(img, kernel)
    result = dilation(result, kernel)
    
    cv2.imwrite('opening.jpg',result)
    
def closing(img, kernel):
    result = dilation(img, kernel)
    result = erosion(result, kernel)
    
    cv2.imwrite('closing.jpg',result)
       

def legal(a):
    for i in range(len(a)):
        if a[i] >= 512 or a[i] < 0:
            return False
    return True

def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    kernel = [(-2,0),(-2,1),(-2,-1),
            (-1,0), (-1,1),(-1,2),(-1,-1),(-1,-2),
            (0,0),(0,1),(0,2),(0,-1),(0,-2),
            (1,0),(1,1),(1,2),(1,-1),(1,-2),
            (2,0),(2,1),(2,-1)]
    J = [(0,0),(-1,0),(0,-1)]
    K = [(0,1),(1,0),(1,1)]
    dilation(img, kernel)
    erosion(img, kernel)
    opening(img, kernel)
    closing(img, kernel)


if __name__ == '__main__':
    main()