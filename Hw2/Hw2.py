import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def Binary(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= 128 :
                img[i][j] = 255
            else:
                img[i][j] = 0

    cv2.imwrite('binary.jpg',img)
    return img

def Histogram(img):
    
    hist = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1
    x = range(0,256)
    plt.plot(x,hist,label = 'histogram')
    plt.fill_between(x,hist) 
    plt.title('histogram')
    plt.savefig('histogram.png')

def TwoPass(img):
    labels = np.zeros(img.shape)
    labels = labels.astype(int)
    NextLabel = 1

    S = {} #nodes for Set  eg:S[1]=2  parent of 1 is 2
    #first pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #not background
            if img[i][j] != 0:
                #labeled neighbors (labels of neighbors)
                neighbors = findNeighbors(img, labels , i , j)

                #no neighbors
                if not neighbors:
                    labels[i][j] = NextLabel
                    S[NextLabel] = 0 #parent of node Nextlabel is 0
                    NextLabel += 1
                else :
                    L = neighbors
                    labels[i][j] = min(L) #choose the smallest label for this pixel
                    for label in L:
                        if label != labels[i][j]:
                            Union(S, labels[i][j], label) #union 

    #second pass
    records = {} #record final label, count and  4 point of compornent rectangle eg: {label:{count,max_x,max_y,min_x,min_y}}
    for i in range(img.shape[0]-1 , -1, -1):
        for j in range(img.shape[1]):
            if img[i][j] != 0 :
                labels[i][j] = findRoot(S,labels[i][j])#update label by equivalence set 
                label = labels[i][j]
                if label not in records:
                    records[label] = {"count":1, "max_x":float('-inf'), "max_y":float('-inf'), "min_x":float('inf'), "min_y":float('inf')} #record components area
                else:
                    records[label]["count"] += 1
                    if i > records[label]["max_x"]:
                        records[label]["max_x"] = i
                    if i < records[label]["min_x"]:
                        records[label]["min_x"] = i
                    if j > records[label]["max_y"]:
                        records[label]["max_y"] = j
                    if j < records[label]["min_y"]:
                        records[label]["min_y"] = j
    DrawRect(img,records)

def DrawRect(img, records):
    rbg_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #convert gray into rgb
    for l,imf in records.items() :
        if imf['count'] > 500: #if labeled pixels > 500
            center = ( (imf['max_y']+imf['min_y'])//2,  (imf['max_x']+imf['min_x'])//2 ) 
            cv2.rectangle(rbg_img, (imf['min_y'],imf['min_x']), (imf['max_y'], imf['max_x']), (0, 0, 255), 1)
            cv2.line(rbg_img, (center[0],center[1]-5), (center[0],center[1]+5), (0,0,255), 1)
            cv2.line(rbg_img, (center[0]-5,center[1]), (center[0]+5,center[1]), (0,0,255), 1)
    cv2.imwrite('ConnectedComponents_4con.jpg',rbg_img)    
    

def Union(S,a,b):
    if findRoot(S,a) != findRoot(S,b):
        S[findRoot(S,b)] = a

def findRoot(S,a):
    x = a
    while S[x] != 0:
        x = S[x]
    if x !=a : #compress 
        S[a] = x
    return x

def findNeighbors(img, label, i, j):
    #directions
    dircs = [(1,0), (-1,0), (0,1), (0,-1)]#, (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = []
    for dirc in dircs:
        dx = dirc[0]
        dy = dirc[1]
        #check out of index
        if i+dx < 0 or i+dx > img.shape[0]-1 or j+dy < 0 or j+dy > img.shape[1]-1:
            continue
        #check foreground and labeled
        elif img[i+dx][j+dy] != 0 and label[i+dx][j+dy] != 0:
            neighbors.append(label[i+dx][j+dy])
    return neighbors


def main():
    img = cv2.imread('lena.bmp',cv2.IMREAD_GRAYSCALE)
    img_bin = Binary(img.copy())
    Histogram(img.copy())
    TwoPass(img_bin) 

if __name__ == '__main__':
    main()