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


def Robert(img, threshold):
    r1 = np.array([[-1,0],[0,1]])
    r2 = np.array([[0,-1],[1,0]])
    masks = [r1,r2]
    res = convolution(img, masks, threshold, "norm")
    cv2.imwrite("Robert.jpg", res)
    return res


def Prewitt(img, threshold):
    p1 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    p2 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    masks =[p1,p2]
    res = convolution(img, masks, threshold, "norm")
    cv2.imwrite("Perwitt.jpg", res)
    return res
    
def Sobel(img, threshold):
    p1 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    p2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    masks =[p1,p2]
    res = convolution(img, masks, threshold, "norm")
    cv2.imwrite("Sobel.jpg", res)
    return res

def FreiAndChen(img, threshold): 
    p1 = np.array([[-1,-(2**(1/2)),-1],[0,0,0],[1,2**(1/2),1]])
    p2 = np.array([[-1,0,1],[-(2**(1/2)),0,2**(1/2)],[-1,0,1]])
    masks =[p1,p2]
    res = convolution(img, masks, threshold, "norm")
    cv2.imwrite("FreiAndChen.jpg", res)
    return res

def Kirsch(img, threshold):
    masks = []
    masks.append(np.array([-3,-3,5,-3,0,5,-3,-3,5]))
    masks.append(np.array([-3,5,5,-3,0,5,-3,-3,-3]))
    masks.append(np.array([5,5,5,-3,0,-3,-3,-3,-3]))
    masks.append(np.array([5,5,-3,5,0,-3,-3,-3,-3]))
    masks.append(np.array([5,-3,-3,5,0,-3,5,-3,-3]))
    masks.append(np.array([-3,-3,-3,5,0,-3,5,5,-3]))
    masks.append(np.array([-3,-3,-3,-3,0,-3,5,5,5]))
    masks.append(np.array([-3,-3,-3,-3,0,5,-3,5,5]))
    for i in range(len(masks)):
        masks[i] = masks[i].reshape((3,3))
    res = convolution(img, masks, threshold, "max")
    cv2.imwrite("Kirsch.jpg", res)
    return res

def Robinson(img, threshold):
    masks = []
    masks.append(np.array([-1,0,1,-2,0,2,-1,0,1]))
    masks.append(np.array([0,1,2,-1,0,1,-2,-1,0]))
    masks.append(np.array([1,2,1,0,0,0,-1,-2,-1]))
    masks.append(np.array([2,1,0,1,0,-1,0,-1,-2]))
    masks.append(np.array([1,0,-1,2,0,-2,1,0,-1]))
    masks.append(np.array([0,-1,-2,1,0,-1,2,1,0]))
    masks.append(np.array([-1,-2,-1,0,0,0,1,2,1]))
    masks.append(np.array([-2,-1,0,-1,0,1,0,1,2]))
    for i in range(len(masks)):
        masks[i] = masks[i].reshape((3,3))
    res = convolution(img, masks, threshold, "max")
    cv2.imwrite("Robinson.jpg", res)
    return res

def NevatiaAndBabu(img, threshold):
    masks = []
    masks.append(np.array([100,100,100,100,100,100,100,100,100,100,0,0,0,0,0,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]))
    masks.append(np.array([100,100,100,100,100,100,100,100,78,-32,100,92,0,-92,-100,32,-78,-100,-100,-100,-100,-100,-100,-100,-100]))
    masks.append(np.array([100,100,100,32,-100,100,100,92,-78,-100,100,100,0,-100,-100,100,78,-92,-100,-100,100,-32,-100,-100,-100]))
    masks.append(np.array([-100,-100,0,100,100,-100,-100,0,100,100,-100,-100,0,100,100,-100,-100,0,100,100,-100,-100,0,100,100]))
    masks.append(np.array([-100,32,100,100,100,-100,-78,92,100,100,-100,-100,0,100,100,-100,-100,-92,78,100,-100,-100,-100,-32,100]))
    masks.append(np.array([100,100,100,100,100,-32,78,100,100,100,-100,-92,0,92,100,-100,-100,-100,-78,32,-100,-100,-100,-100,-100]))
    
    for i in range(len(masks)):
        masks[i] = masks[i].reshape((5,5))
    res = convolution(img, masks, threshold, "max")
    cv2.imwrite("NevatiaAndBabu.jpg", res)
    return res

    
def convolution(img, masks, threshold, mode):
    mask_size = masks[0].shape[0]
    res = np.zeros(tuple(np.subtract(img.shape, (mask_size-1,mask_size-1))))    
    move = []
    steps = [i for i in range(0, mask_size)] 
    for i in steps:  #get all movements from steps
        for j in steps:
            move.append((i,j))
    #===convolution===
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            grads = np.zeros(len(masks))
            grad_mtd = 0
            for m in move:
                r = i+m[1] #index after movement
                c = j+m[0]
                if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                    continue
                for k in range(len(masks)):
                    grads[k] += img[r,c] * masks[k][m[1], m[0]]
            if mode == "norm":
                grad_mtd = np.sqrt(np.sum(grads**2))
            elif mode == "max":
                grad_mtd = np.max(grads)
            if grad_mtd <= threshold :
                res[i,j] = 255
            else:
                res[i,j] = 0
    return res


def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    #Robert(img, 30)
    #Prewitt(img, 80)
    #Sobel(img, 100)
    #FreiAndChen(img, 100)
    Kirsch(img,400)
    Robinson(img, 100)
    NevatiaAndBabu(img, 30000)
    
if __name__ == '__main__':
    main()