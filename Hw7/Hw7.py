import cv2
import numpy as np

remove = True

def Binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128 :
                img[i][j] = 255
            else:
                img[i][j] = 0
    return img


def Resize(img, shape):
    width = int(img.shape[1]/shape[1])
    height = int(img.shape[0]/shape[0])
    img_res = np.zeros( (height, width)).astype(int)
    for i in range(0,height):
        for j in range(0,width):
                img_res[i][j] = img[i*8][j*8]
    cv2.imwrite('resize.jpg',img_res)
    return img_res

def H2(b, c, d, e):
    if b == c and ( d != b or e != b):
        return 1
    else:
        return 0

def F2(a1, a2 , a3 ,a4):
    if (a1+a2+a3+a4) == 1:
        return False
    else:
        return True


def H(b, c, d, e):
    if b == c and ( d != b or e != b):
        return 'q'
    elif b == c and ( d == b and e == b):
        return 'r'
    elif b != c :
        return 's'

def F(a1, a2 , a3 ,a4):
    if a1 == a2 and a2 == a3 and a3 == a4 and a4 == 'r':
        return 5
    else:
        n = 0
        if a1 == 'q':
            n += 1
        if a2 == 'q':
            n += 1
        if a3 == 'q':
            n += 1
        if a4 == 'q':
            n += 1
        return n

def yokoi(img):
    move = [(0,0), (1,0), (0,-1), (-1,0), (0,1), (1,1), (1,-1), (-1,-1), (-1,1)] #move for getting x
    res = np.full(img.shape,-1).astype(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255: #if foreground
                x = []
                for k in range(9): #set x[i] 
                    r = i+move[k][1]
                    c = j+move[k][0]
                    if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                        x.append(0)
                    else:
                        x.append(img[r][c])
                a1 = H(x[0], x[1], x[6], x[2])
                a2 = H(x[0], x[2], x[7], x[3])
                a3 = H(x[0], x[3], x[8], x[4])
                a4 = H(x[0], x[4], x[5], x[1])
                
                res[i][j] = F(a1,a2,a3,a4)
    
    return res


def ConnectedShrink(img, img_bin):
    move = [(0,0), (1,0), (0,-1), (-1,0), (0,1), (1,1), (1,-1), (-1,-1), (-1,1)] #move for getting x
    res = img_bin
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 3: #if foreground is p 
                x = []
                for m in move: #set x[i] 
                    r = i+m[1]
                    c = j+m[0]
                    if r < 0 or r >= img.shape[0] or c < 0 or c >= img.shape[1]:
                        x.append(0)
                    else:
                        x.append(img_bin[r][c])
                a1 = H2(x[0], x[1], x[6], x[2])
                a2 = H2(x[0], x[2], x[7], x[3])
                a3 = H2(x[0], x[3], x[8], x[4])
                a4 = H2(x[0], x[4], x[5], x[1])
                if not F2(a1,a2,a3,a4): #delete 
                    res[i, j] = 0
                    global remove
                    remove = True
                
    cv2.imwrite("thinning_64.jpg", res)

    return res

    
def PairRelationship(img):
    move = [(1,0), (0,-1), (-1,0), (0,1)] #4 connectivity
    #move = [(1,0), (0,-1), (-1,0), (0,1), (1,1), (1,-1), (-1,-1), (-1,1)] #8 connectivity

    res = np.zeros(img.shape).astype(int)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r,c] == 1:
                interior = 0
                for m in move:
                    x_r = r+m[0]
                    x_c = c+m[1]
                    if not(x_r < 0 or x_r >= img.shape[0] or x_c < 0 or x_c >= img.shape[1]):
                        if img[x_r, x_c] == 1:
                            interior += 1
                if img[r, c] == 1 and interior >= 1: #border and next to interior
                    res[r, c] = 3 # 3 represent p
                elif img[r, c] != 1 or interior < 1:
                    res[r, c] = 4 # 4 represent q 
    """
    file = open("PairRelationship.txt",'w')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if res[i][j] == 0:
                file.write(' ')
            else:
                file.write(str(res[i][j]))
        file.write("\n")
    """
    return res           

def Thinning(img_resize):
    global remove
    while remove == True:
        remove = False
        #img_ib = InteriorBorder(img_resize)
        img_ib = yokoi(img_resize)
        img_pr = PairRelationship(img_ib)
        img_resize = ConnectedShrink(img_pr, img_resize)



def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    img_bin = Binary(img.copy())
    shape = (8,8)
    img_resize = Resize(img_bin, shape)
    Thinning(img_resize)



if __name__ == '__main__':
    main()