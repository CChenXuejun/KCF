from numpy import *
from matplotlib import pyplot as mplPlot
from PIL import Image as PIL_Image,ImageTk
import numpy as np
from PIL import Image
import os
from tkinter import *
import cv2

def gaussian_shaped_labels(sigma,sz):
    m = np.arange(1,sz[0]+1) #the thing is that if we RUN:ARANGE(1,3),it returns a array[1,2]
    n = np.arange(1,sz[1]+1)
    [rs,cs] = np.meshgrid(m - floor(sz[0]/2), n - floor(sz[1]/2))
    rs = np.transpose(rs) # Only transposition could produce the same result as NDGRID in MATLAB
    cs = np.transpose(cs)
    x_temp = np.multiply(rs,rs) + multiply(cs,cs)
    labels = np.exp(-0.5 / pow(sigma,2) * x_temp) # use dot properly
    labels = circshift(labels,int(-floor(sz[0]/2-1)),int(-floor(sz[1]/2-1)))
    assert(labels[0,0] == 1)
    #if no int here,a big problem is that -floor(sz[0]/2)+1 will give a float result which unsuitable for index
    return labels

def circshift(matrix,m,n):
    #Cyclically make matrix shift m steps down and shift n steps right
    Matrix_1 = roll(matrix,m,axis=0)
    Matrix_2 = roll(Matrix_1,n,axis=1) #axis=1 indicate that the matrix will shift in a whole row or column
    return Matrix_2

def mesh(x):
    if(isinstance(x[0,0],complex)):
        x = real(x)

    size=x.shape
    Y=np.arange(0,size[0],1)
    X=np.arange(0,size[1],1)

    X,Y=np.meshgrid(X,Y)
    fig=mplPlot.figure()
    ax=fig.gca(projection='3d')
    ax.plot_surface(X,Y,x,cmap = mplPlot.get_cmap('jet'))
    mplPlot.show()
    return 1

def rgb2gray(rgb):
    # just for picture with color
    # if you don't have PIL library,use:
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return rgb.convert('L')

def img2matrix(img):
    return np.array(img)

def get_ground_truth(path):
    #path is where the image's sequences in
    files = os.listdir(path) #ground_truth and img are in the files
    fid = open(path+files[0]) #get the path of ground_truth
    ground_truth = fid.readlines() #read lines from ground_truth
    num_groundTruth = len(ground_truth) #get the length of ground_truth
    A = zeros([num_groundTruth,4],dtype = int) #built a array for the data
    A_Row = 0

    for line in ground_truth:
        list = line.strip('\n').split(',')
        A[A_Row,:] = list[0:4]
        A_Row = A_Row + 1
    pos = A[0,0:2]
    target_sz = roll(A[1,2:4],1)

    return A,pos,target_sz

def get_subwindow(img,pos,sz):
    xs = floor(pos[1]-1) + np.arange(1,sz[1]+1) - floor((sz[1]-1)/2)
    ys = floor(pos[0]-1) + np.arange(1,sz[0]+1) - floor((sz[0]-1)/2)
    xs = xs.astype('int')
    ys = ys.astype('int')
    xs[xs < 0] = 1
    ys[ys < 0] = 1
    a,b = img.shape
    xs[xs > b-1] = b-1
    ys[ys > a-1] = a-1
    xs = mat(xs)
    ys = mat(ys)
    img_new = zeros([ys.shape[1],xs.shape[1]])
    for i in range(0,ys.shape[1]):
        for j in range(0,xs.shape[1]):
            # print('i=',i,'j=',j)
            img_new[i,j] = img[ys[0,i],xs[0,j]]
    # print('-------------------------------')
    # print('ys.shape xs.shape = ',ys.shape,xs.shape)
    return img_new

def get_features(im,cos_window):
    x = im.astype('double')
    x = (x - np.mean(x))/255
    x = np.multiply(x,cos_window)
    return x

def gaussian_correlation(xf,yf,sigma):
    N = prod(xf.shape)
    # print(xf.shape)
    xx = sum(multiply(xf,xf))/N
    yy = sum(multiply(yf,yf))/N
    xyf = multiply(xf,conj(yf))
    xy = real(fft.ifft2(xyf))
    # print('xx shape = ',xx.shape,'yy shape = ',yy.shape,'conj(yf) shape = ',a.shape,'xyf shape = ',xyf.shape,'xy shape = ',xy.shape)
    temp = xx + yy - 2 * xy
    temp[temp<0] = 0

    kf = fft.fft2(exp(-1/pow(sigma,2) * temp / N))

    return kf

def get_maxElement(mat):
    raw, column = mat.shape# get the matrix of a raw and column
    _positon = np.argmax(mat)# get the index of max in the a
    m, n = divmod(_positon, column)
    return mat[m,n]

def show_video(img_path,files,frame,fps,pos_get,target_sz,width,height):
    tk = Tk()
    canvas = Canvas(tk,width=width,height=height,bg = 'white')
    for i in range(0,frame):
        imFullName = img_path+files[i]
        img0 = cv2.imread(imFullName)
        pt1 = (int(pos_get[i,0]),int(pos_get[i,1]))
        pt2 = (pt1[0]+target_sz[1],pt1[1]+target_sz[0])
        color = (0,255,0)
        img1 = cv2.rectangle(img0,pt1,pt2,color,2) # 2 indicate the width of the line
        img2 = PIL_Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
        img3 = ImageTk.PhotoImage(image = img2)
        canvas.create_image((width/2,height/2),image = img3)
        canvas.pack()
        tk.update()
        tk.after(int(1/fps*1000))
    tk.mainloop()
    return 1