import cv2
import numpy as np

def Binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128 :
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img

def GaussianNoise(img, amp):
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i,j] = img[i, j] + amp * np.random.normal(0, 1, 1) 
    cv2.imwrite("GaussainNoise{}.jpg".format(str(amp)) , res)
    return res


def SaltAndPepper(img, threshold):
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.random.rand(1) < threshold:
                res[i,j] = 0
            elif np.random.rand(1) > 1-threshold:
                res[i,j] = 255
            else:
                res[i, j] = img[i,j]
    cv2.imwrite("SaltAndPepper{}.jpg".format(str(threshold)) , res)
    return res

def SNR(img_s , img_n):
    var_s = np.var(img_s)
    img_ns = img_n - img_s
    var_n = np.var(img_ns)
    return 20*np.log10(np.sqrt(var_s)/np.sqrt(var_n))

def BoxFilter(img,f_size, noise):
    res = np.zeros(img.shape)
    #===kernel===
    move = []
    steps = [i for i in range(f_size//2, -f_size//2, -1)] #steps. eg:0,1,-1,2...
    for i in steps: #get all movements from steps
        for j in steps:
            move.append((i,j))
    kernel = np.full( (f_size,f_size), 1)
    kernel = kernel/kernel.size #divide by size of kernel
    k_c = f_size//2  #kernel center index
    #===convolution===
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s = 0
            for m in move:
                r = i+m[1] #index after movement
                c = j+m[0]
                if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                    continue
                s += img[r,c] * kernel[k_c+m[1], k_c+m[0]]
            res[i, j] = s
    cv2.imwrite("BoxFilter{}_{}.jpg".format(str(f_size),str(noise)), res)        
    print(SNR(res,img))
    return res

def MedianFilter(img,f_size, noise):
    res = np.zeros(img.shape)
    #===kernel===
    move = []
    steps = [i for i in range(f_size//2, -f_size//2, -1)] 
    for i in steps:  #get all movements from steps
        for j in steps:
            move.append((i,j))
    kernel = np.full( (f_size,f_size), 1)
    k_c = f_size//2 #kernel center index
    #===convolution===
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s = []
            for m in move:
                r = i+m[1] #index after movement
                c = j+m[0]
                if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                    continue
                s.append(img[r,c] * kernel[k_c+m[1], k_c+m[0]])
            s = np.array(s) 
            res[i, j] = np.median(s)
    cv2.imwrite("MedianFilter{}_{}.jpg".format(str(f_size),str(noise)), res)        
    print(SNR(res,img))
    return res

def dilation(img, kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            v_max = 0
            for k in kernel:
                r = i+k[1] #index after movement
                c = j+k[0]
                if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                    continue 
                elif img[r,c] > v_max:
                    v_max = img[r,c]
            result[i][j] = v_max
    return result

def erosion(img , kernel):
    result = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            v_min = 1e100
            for k in kernel:
                r = i+k[1] #index after movement
                c = j+k[0]
                if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                    continue 
                elif img[r,c] < v_min:
                    v_min = img[r,c]
            result[i][j] = v_min
    return result

def opening(img, kernel):
    result = erosion(img, kernel)
    result = dilation(result, kernel)
    print(SNR(result,img))
    return result

def closing(img, kernel):
    result = dilation(img, kernel)
    result = erosion(result, kernel)
    print(SNR(result,img))
    return result
  
def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)

    GaussianNoise(img, 10)
    GaussianNoise(img, 30)
    SaltAndPepper(img, 0.05)
    SaltAndPepper(img, 0.1)

    kernel = [(-2,0),(-2,1),(-2,-1),
            (-1,0), (-1,1),(-1,2),(-1,-1),(-1,-2),
            (0,0),(0,1),(0,2),(0,-1),(0,-2),
            (1,0),(1,1),(1,2),(1,-1),(1,-2),
            (2,0),(2,1),(2,-1)]
    
    img_list = ["GaussainNoise10", "GaussainNoise30","SaltAndPepper0.05","SaltAndPepper0.1" ]
    for im in img_list:
        img = cv2.imread(im+".jpg",cv2.IMREAD_GRAYSCALE)
        BoxFilter(img, 3, im)
        BoxFilter(img, 5, im)
        MedianFilter(img, 3, im)
        MedianFilter(img, 5, im)
        cv2.imwrite("Opening&Closing_"+im+".jpg", closing(opening(img, kernel), kernel))
        cv2.imwrite("Closing&Opening"+im+".jpg", opening(closing(img, kernel), kernel))

if __name__ == '__main__':
    main()