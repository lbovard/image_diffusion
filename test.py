from matplotlib.colors import LogNorm
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as random
import numpy as np


#read in the image
image=misc.imread("images/GRMHD_True_Image_Greyscale.png")
image=image/float(np.max(image))
c=0.25
per_remove=0.99
thresh=0.5
nsteps=1000

#pad the edges of the image so we can apply finite difference
pad_image=np.zeros((image.shape[0]+2,image.shape[1]+2)) 
#copy the old image into the new image
nx,ny=pad_image.shape[0],pad_image.shape[1]
pad_image[1:nx-1,1:ny-1]=image[:,:]

#store the original image
orig_image=np.copy(pad_image)

#total number of points
totnp=np.shape(pad_image)[0]*np.shape(pad_image)[1]

print '{0:40} : {1:10d}'.format("total number of points", totnp)

pad_image=pad_image.flatten() 
#number of points before threshholding
print '{0:40} : {1:10f}'.format("threshold is ", thresh)

non_zero_idx=np.where(pad_image>0.0)[0]
print '{0:40} : {1:10d}'.format("num non-zero before thresholding", non_zero_idx.shape[0])

#remove all points below threshhold
pad_image[np.where(pad_image<thresh)[0]]=0.0
non_zero_idx=np.where(pad_image>0.0)[0]

print '{0:40} : {1:10d}'.format("num non-zero after thresholding", non_zero_idx.shape[0])

nl=non_zero_idx.shape[0]
npremove=int(nl*per_remove)
print '{0:40} : {1:10f}'.format("percent of points to remove", per_remove)
print '{0:40} : {1:10d}'.format("num points removed from threshold", npremove)

#select randomly those points to remove
ids=random.sample(xrange(0,nl),npremove) 
pad_image[non_zero_idx[ids]]=0

#get all non-zero points
non_zero_idx=np.where(pad_image>0.0)[0]
b=np.shape(non_zero_idx)[0]
perblank=100*float(totnp-b)/totnp
print '{0:40} : {1:10d}'.format("num non-zero points", b)
print '{0:40} : {1:10f}'.format("% of image set to 0", perblank)

oa=pad_image[non_zero_idx]
f=pad_image.reshape((nx,ny))
fnew=np.zeros((nx,ny)) 

#copy the initial state of the image
init_image=np.copy(f)

print '{0:>10} | {1:10}'.format("itr","   norm   ")
for k in range(0,nsteps):
  #now do the iteration
  fnew=np.zeros((f.shape[0],f.shape[1]))
  si,ei=1,nx-1
  #compute the second derivative using finite differences and time step
  fnew[si:ei,si:ei]=(1-4*c)*f[si:ei,si:ei]+c*(f[si+1:ei+1,si:ei]+f[si-1:ei-1,si:ei] + f[si:ei,si-1:ei-1] +f[si:ei,si+1:ei+1])

  #flatten array for this operation
  fnew=fnew.reshape(-1)
  #copy the fixed points
  fnew[non_zero_idx]=oa
  fnew=fnew.reshape((nx,ny))

  if k % 100 ==0 :
    diff=np.max(np.abs(f-fnew))
    print '{0:10d} : {1:10f}'.format(int(k),diff)
  f=fnew 

image_final=fnew

fig = plt.figure()
fig.subplots_adjust(hspace=0.1,wspace=0.1)

ax0=fig.add_subplot(1,3,1)
ax1=fig.add_subplot(1,3,2)
ax2=fig.add_subplot(1,3,3)

ax0.set_xticks([])
ax1.set_xticks([])
ax2.set_xticks([])

ax0.set_yticks([])
ax1.set_yticks([])
ax2.set_yticks([])

ax0.imshow(orig_image, cmap = cm.Greys_r,interpolation='none')
ax0.set_title("Original image")
ax1.imshow(image_final, cmap = cm.Greys_r,interpolation='none')
ax1.set_title("Recovered image")
ax1.text(0, nx, str(nsteps)+" itr", fontsize=5, color='white')
ax2.imshow(init_image, cmap = cm.Greys_r,interpolation='none')
ax2.set_title("Initial data")
ax2.text(0, nx, str(perblank)[0:6]+"%  " + str(thresh) + " thresh", fontsize=5, color='white')

#print image_final[np.where(image_final>0.1)[0]].shape[0]
norm_comp=np.sum(image_final)/float(np.sum(orig_image))
print '{0:40} : {1:10f}'.format("total sum comparison", norm_comp)
plt.savefig("output.pdf",bbox_inches='tight')
#plt.show()

