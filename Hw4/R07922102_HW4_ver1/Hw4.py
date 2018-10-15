import cv2
import numpy as np

def Binary(img):
    v_map = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128 :
                img[i][j] = 255
                v_map.append((i,j))
            else:
                img[i][j] = 0
    cv2.imwrite('binary.jpg',img)
    return img,v_map

def dilation(img, v_map, kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    result_vmap = []
    for i in range(len(kernel)):
        for j in range(len(v_map)):
            if legal(addition(kernel[i],v_map[j])):
                e =  addition(kernel[i],v_map[j])
                result[e[0]][e[1]] = 255
                result_vmap.append((e[0],e[1]))
    cv2.imwrite('dilation.jpg',result)
    return result, result_vmap

def erosion(img, v_map , kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    result_vmap = []
    for i in range(len(v_map)):
        valid = True
        for j in range(len(kernel)):
            if legal(addition(v_map[i], kernel[j])):
                e = addition(v_map[i], kernel[j])
                if img[e[0]][e[1]] != 255:
                    valid = False
                    break
        if valid == True:
            e = v_map[i]
            result[e[0]][e[1]] = 255
            result_vmap.append((e[0],e[1]))
    cv2.imwrite('erosion.jpg',result)
    return result, result_vmap

def HitAndMiss(img, v_map , J, K):
    result = np.zeros((img.shape[0],img.shape[1]))
    result_vmap = []
    for i in range(len(v_map)):
        valid = True
        for j in range(len(J)):
            if legal(addition(v_map[i], J[j])):
                e = addition(v_map[i], J[j])
                if img[e[0]][e[1]] != 255:
                    valid = False
                    break
            if valid == False:
                break
            for j in range(len(K)):
                if legal(addition(v_map[i], K[j])):
                    e = addition(v_map[i], K[j])
                    if img[e[0]][e[1]] != 0:
                        valid = False
                        break
        if valid == True:
            e = v_map[i]
            result[e[0]][e[1]] = 255
            result_vmap.append((e[0],e[1]))
    cv2.imwrite('HitAndMiss.jpg',result)

    
def addition(a,b):
    r = []
    for i in range(len(a)):
        r.append(a[i]+b[i])
    return tuple(r)

def opening(img, v_map, kernel):
    result, result_vmap = erosion(img, v_map, kernel)
    result, result_vmap = dilation(result, result_vmap, kernel)
    
    cv2.imwrite('opening.jpg',result)
    
def closing(img, v_map, kernel):
    result, result_vmap = dilation(img, v_map, kernel)
    result, result_vmap = erosion(result, result_vmap, kernel)
    
    cv2.imwrite('closing.jpg',result)
       

def legal(a):
    for i in range(len(a)):
        if a[i] >= 512 or a[i] < 0:
            return False
    return True

def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    img_bin, v_map = Binary(img.copy())
    kernel = [(-2,0),(-2,1),(-2,-1),
            (-1,0), (-1,1),(-1,2),(-1,-1),(-1,-2),
            (0,0),(0,1),(0,2),(0,-1),(0,-2),
            (1,0),(1,1),(1,2),(1,-1),(1,-2),
            (2,0),(2,1),(2,-1)]
    J = [(0,0),(-1,0),(0,-1)]
    K = [(0,1),(1,0),(1,1)]
    dilation(img_bin, v_map, kernel)
    erosion(img_bin, v_map, kernel)
    HitAndMiss(img_bin, v_map, J , K)
    opening(img_bin, v_map, kernel)
    closing(img_bin, v_map, kernel)


if __name__ == '__main__':
    main()