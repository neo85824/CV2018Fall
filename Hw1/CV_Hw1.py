import cv2
import numpy as np

def ImgProcess(org_im):
    im = org_im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]//2):
            tmp = im[i][j].copy()
            im[i][j] = im[i][-j]
            im[i][-j] = tmp.copy()

    cv2.imwrite('lena_rightleft.jpg',im)

    im2 = org_im.copy()
    for i in range(im.shape[1]):
        for j in range(im.shape[0]//2):
            tmp = im2[j][i].copy()
            im2[j][i] = im2[-j][i]
            im2[-j][i] = tmp.copy()

    cv2.imwrite('lena_updown.jpg',im2)

    im3 = org_im.copy()
    for i in range(im.shape[0]):
        for j in range(i+1,im.shape[1]):
            tmp = im3[i][j].copy()
            im3[i][j] = im3[j][i]
            im3[j][i] = tmp.copy()

    cv2.imwrite('lena_diagonal.jpg',im3)
    

def main():
    org_im = cv2.imread('lena.bmp')
    ImgProcess(org_im)

if __name__ == '__main__':
    main()

