'''

 The package contain the Diffusive Image Compression (dic)

''' 
from matplotlib.colors import LogNorm
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as random
import numpy as np

class Image(object):
  ''' the image resconstruction class '''

  def __init__(self,fn):
    ''' fn contains the file to be analysed '''
    #courtant number 
    self.c = 0.25

    #must be specificed by the user
    self.per_remove=0.99
    self.thresh=0.5

    #read in the image that we store temporarily
    temp_im=misc.imread(fn)
    
    temp_im=temp_im/float(np.max(temp_im))
    
    #pad the edges of the image so we can apply finite difference
    self.image=np.zeros((temp_im.shape[0]+2,temp_im.shape[1]+2)) 
    im=self.image

    #copy the old image into the new image
    self.nx,self.ny=im.shape[0],im.shape[1]
    im[1:self.nx-1,1:self.ny-1]=temp_im[:,:]

    print "the min max of the image \t ",  np.min(im),np.max(im)
    #normalise the image
#    im=self.normalise()

    #store the original image for later
    self.orig_image=np.copy(im)
    self.initial_data_image=np.copy(im)

    #total number of points in the image
    self.totnp=np.shape(im)[0]*np.shape(im)[1]


    self.oa=[]
  def normalise(self):
    ''' normalise the image so it is between [0,1]'''
    im=self.image #pointer to the image
    return im/np.max(im) 
    
  def initialise(self,threshold,percent,print_statistics=False):
    ''' initialise the image for reconstruction '''
    self.per_remove=percent
    self.thresh=threshold
    #always use the original image
    im=self.orig_image.flatten() 

    #number of points before thresholding 
    non_zero_idx=np.where(im>0.0)[0]
    npb4thresh=non_zero_idx.shape[0]

    #remove all points below threshold
    idx=np.where(im<self.thresh)[0]
    im[idx]=0.0


    non_zero_idx=np.where(im>0.0)[0]
    nl=non_zero_idx.shape[0]
    npremove=int(nl*self.per_remove)
    

    #select randomly those points to remove
    ids=random.sample(xrange(0,nl),npremove) 
    im[non_zero_idx[ids]]=0

    #get all non-zero points
    non_zero_idx=np.where(im>0.0)[0]
    b=np.shape(non_zero_idx)[0]
    self.nzpt=b
    perblank=100*float(self.totnp-b)/self.totnp
    oa=im[non_zero_idx]
    im=im.reshape((self.nx,self.ny))
    self.oa=oa
    self.non_zero_idx=non_zero_idx
    self.initial_data_image=np.copy(im)

#    plt.imshow(self.initial_data_image, cmap = cm.Greys_r,interpolation='none')
#    plt.show()
    if print_statistics==True:
      print "%%%%%%%%%%%%%%% INITIALIZATION OF IMAGE %%%%%%%%%%%%%%%%%% "
      print '{0:40} : {1:10d}'.format("total number of points", self.totnp)
      print '{0:40} : {1:10f}'.format("threshold is ", self.thresh)
      print '{0:40} : {1:10d}'.format("num non-zero before thresholding", npb4thresh)
      print '{0:40} : {1:10d}'.format("num non-zero after thresholding", nl)
      print '{0:40} : {1:10f}'.format("percent of points to remove", self.per_remove)
      print '{0:40} : {1:10d}'.format("num points removed from threshold", npremove)
      print '{0:40} : {1:10d}'.format("num non-zero points", b)
      print '{0:40} : {1:10f}'.format("% of image set to 0", perblank)

  def reconstruct(self,nsteps):
    ''' this function reconstructions the image'''
    self.nsteps=nsteps
    f=self.initial_data_image
    for k in range(0,nsteps):
      #now do the iteration
      fnew=np.zeros((f.shape[0],f.shape[1]))
      si,ei=1,self.nx-1
      c=self.c
      #compute the second derivative using finite differences and time step
      fnew[si:ei,si:ei]=(1-4*c)*f[si:ei,si:ei]+c*(f[si+1:ei+1,si:ei]+f[si-1:ei-1,si:ei] + f[si:ei,si-1:ei-1] +f[si:ei,si+1:ei+1])

      #flatten array for this operation
      fnew=fnew.reshape(-1)
      #copy the fixed points
      fnew[self.non_zero_idx]=self.oa
      fnew=fnew.reshape((self.nx,self.ny))

    #  if k % 100 ==0 :
    #    diff=np.max(np.abs(f-fnew))
    #    print '{0:10d} : {1:10f}'.format(int(k),diff)
      f=fnew 

    self.image=fnew
    return 0


  def statistics(self):
    ''' return some statistics'''
    #light curve
    self.lc=np.sum(self.image)/np.sum(self.orig_image)
    return (self.thresh,self.per_remove,self.lc,self.nzpt,self.totnp)

  def plot_image(self):
    ''' plot the original image '''
    plt.imshow(self.image, cmap = cm.Greys_r,interpolation='none')
    plt.show()

  def save_img(self):
    ''' plot the original image '''
    fig = plt.figure()
    plt.imshow(self.image, cmap = cm.Greys_r,interpolation='none')
    plt.savefig("output_"+str(self.thresh)+".png",bbox_inches='tight')
  
  def plot_original(self):
    ''' plot the original image '''
    plt.imshow(self.orig_image, cmap = cm.Greys_r,interpolation='none')
    plt.show()
  
  def plot_initial_data(self):
    ''' plot the original image '''
    plt.imshow(self.initial_data_image, cmap = cm.Greys_r,interpolation='none')
    plt.show()
