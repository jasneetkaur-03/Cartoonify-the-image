'''workflow
1. load the image
2. create a mask
3. Reduction of noice
4. Reduce the color palette
5. Combine edge mask and color palette'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#load the image
def read(filename):
    img=cv2.imread(filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    return img

#add image to the existing folder

filename="image2.jfif"
img=read(filename)
org_image=np.copy(img)

#create edge mask
#input= input image, output= edges of the image
def edge_mask(img,line_size,blur_value):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #turning the image from rgb to gray
    gray_blur=cv2.medianBlur(gray,blur_value)
    edges=cv2.adaptiveThreshold(gray_blur, 300, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

line_size=7
blur_value=3
edges=edge_mask(img,line_size,blur_value)
plt.imshow(edges,cmap="gray")
plt.show()

#reducing the color palette....continue frm 11:40
def color_quantization(img,k): #k refers to the number of colors that we want
      #transform the image
      data=np.float32(img).reshape((-1,3))
      #determine criteria
      criteria=(cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 20,0.001)
      #implementing K-means
      ret,label,center=cv2.kmeans(data, k, None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
      center=np.uint8(center)
      result=center[label.flatten()]
      result=result.reshape(img.shape)
      return result

img=color_quantization(img, k=11)


#reduce the noise
blurred=cv2.bilateralFilter(img, d=3,sigmaColor=200,sigmaSpace=200) #d is the diameter of each pixel
plt.imshow(blurred)
plt.show()

#combine edge mask with the quantized image
def cartoon():
    c=cv2.bitwise_and(blurred,blurred,mask=edges)
    plt.imshow(c)
    plt.title("Carttonified image")
    plt.show()

    plt.imshow(org_image)
    plt.title("Original Image")
    plt.show()

cartoon()

