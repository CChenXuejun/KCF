from head_module import *

#-------------get the path of the video and the size of each frame-------------------
video_path = 'F:/tracking/KCF_tracker_release2/data/Benchmark/Basketball/'
img_path = video_path+'img/'
files = os.listdir(img_path)
frame = len(files) #number of frame
img = PIL_Image.open(img_path+'0001.jpg')
(x,y) = img.size #image size

#---------------read the ground_truth.txt into an array A-----------------
A,pos,target_sz = get_ground_truth(video_path)
pos_get = zeros([frame,2])
pos = pos-1
print('first position = ',pos,'\ntarget size = ',target_sz)

#------------------preprocess before the first frame---------------
padding = 1.5  #extra area surrounding the target
Lambda = 1e-4  #regularization
output_sigma_factor = 0.1  #spatial bandwidth (proportional to target)
interp_factor = 0.08
kernel_sigma = 0.2
frames = 120

resize_image = sqrt(prod(target_sz)) >= 100 #if the image's size is large then resize it
if resize_image:
    pos = floor(pos / 2) #int is necessary to use or pos will be a FLOAT TYPE which is unsuitable to index
    target_sz = floor(target_sz / 2)

window_sz = floor(target_sz * (1+padding)) #the size of search window
window_sz = window_sz.astype(int) #window_sz = [202 85]

output_sigma = sqrt(prod(target_sz)) * output_sigma_factor # output_sigma =  5.247856705360771

yf = fft.fft2(gaussian_shaped_labels(output_sigma,window_sz))
# mesh(gaussian_shaped_labels(output_sigma,window_sz))
cos_window = dot(transpose(matrix(hanning(size(yf,0)))),matrix(hanning(size(yf,1))))

# ok,the hanning function in matlab will return a vertical vector which opposed to hanning in python
# python returns a horizontal vector

#--------------------  begain to generate model and track ---------------------------

for frame in range(0,frames):
    im = PIL_Image.open(img_path+files[frame])
    if size(im,2)> 1:
        im = rgb2gray(im)
    if resize_image:
        new_sz = floor(array([size(im,0),size(im,1)])*0.5)
        new_sz = new_sz.astype('int')
        im = im.resize(new_sz)
    im = matrix(im)
    #-----------other frames-----------
    if frame > 0 :
        patch = get_subwindow(im,pos,window_sz)
        temp = get_features(patch,cos_window)
        # mesh(temp)
        zf = fft.fft2(temp)
        kzf = gaussian_correlation(zf, model_xf, kernel_sigma)
        response = real(fft.ifft2(multiply(model_alphaf,kzf)))
        # mesh(response)
        delta = argwhere(response==get_maxElement(response))

        if delta[0,0] > zf.shape[0] / 2:
            delta[0,0] = delta[0,0] - zf.shape[0]
        if delta[0,1] > zf.shape[1] / 2:
            delta[0,1] = delta[0,1] - zf.shape[1]
        pos = pos + [delta[0,0], delta[0,1]]

    patch = get_subwindow(im, pos, window_sz)
    # print('\npatch size = ',patch.shape)
    # mplPlot.imshow(patch)
    # mplPlot.show()
    xf = fft.fft2(get_features(patch,cos_window))
    kf = gaussian_correlation(xf,xf,kernel_sigma)
    alphaf = yf / (kf + Lambda)

    if frame == 0 :
        model_alphaf = alphaf
        model_xf = xf
    else:
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf
        model_xf = (1 - interp_factor) * model_xf + interp_factor * xf

    pos_get[frame,0] = pos[0]
    pos_get[frame,1] = pos[1]

# ------------------------show the result-------------------------

show_video(img_path,files,frames,10,pos_get,target_sz,x,y)



