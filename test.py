from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as random
import numpy as np

# stolen from here
# https://samarthbhargav.wordpress.com/2014/05/05/image-processing-with-python-rgb-to-grayscale-conversion/
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def set_up_image(image):
  grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
  # get row number
  for rownum in range(len(image)):
     for colnum in range(len(image[rownum])):
        grey[rownum][colnum] = weightedAverage(image[rownum][colnum])/255.
  return grey


def convert_index(i,j,N):
  return N*i+j

image=misc.imread("adele.jpg")

c=0.1
remove=90
nsteps=500

image=set_up_image(image)

plt.imshow(image, cmap = cm.Greys_r,interpolation='none')
plt.show()

#a copy of the original
temp_image=np.zeros((image.shape[0]+2,image.shape[1]+2))
print temp_image.shape

for i in range(1,image.shape[0]):
  for j in range(1,image.shape[1]):
    temp_image[i][j]=image[i][j]

image=temp_image
image_copy=image
N=image.shape[0]

print image_copy.shape
#now randomly pick intergers
ids=random.sample(xrange(0,N*N),remove*remove)
image_copy=image_copy.flatten()
image_copy[ids]=0

non_zero_idx=np.where(image_copy>0)[0]
non_zero_value=image_copy[non_zero_idx]

for x in range(len(image_copy)):
  if image_copy[x]<0.5:
    image_copy[x]=0

a=len(image_copy)
b=len(np.where(image_copy<1.e-6)[0])
print "there are total  : " , a
print "points set to 0  : " , b
print "% blanked is     : " , float(b)/a
image_copy=image_copy.reshape((N,N))

f=image_copy
fnew=np.zeros((f.shape[0],f.shape[1]))
f_orig=f

plt.imshow(f, cmap = cm.Greys_r,interpolation='none')
plt.show()
print image_copy[13:15,13:15]
print f.shape
for k in range(0,nsteps):
  #now do the iteration
  co=0
  fnew=np.zeros((f.shape[0],f.shape[1]))
  for i in range(1,f.shape[0]-1):
    for j in range(1,f.shape[1]-1): 
      #get the converted index
      ci=convert_index(i,j,N)
      if ci not in non_zero_idx:
        co+=1
        fnew[i][j]=f[i][j]+c*(f[i+1][j]+f[i-1][j]+f[i][j-1]+f[i][j-1]-4*f[i][j])
      else:
        fnew[i][j]=f[i][j]
  
  print np.max(np.abs(f-fnew))
  f=fnew

image_copy=fnew.reshape((N,N))

fig = plt.figure()
plt.subplot(131)
plt.imshow(temp_image, cmap = cm.Greys_r,interpolation='none')
plt.subplot(132)
plt.imshow(image_copy, cmap = cm.Greys_r,interpolation='none')
plt.subplot(133)
plt.imshow(f_orig, cmap = cm.Greys_r,interpolation='none')
plt.savefig("output.pdf")
plt.show()

