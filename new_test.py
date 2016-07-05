import dic


#set-up the original image
A=dic.Image("images/GRMHD_True_Image_Greyscale.png")

stat_list=[]
for x in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,0.99]:
  print x
  A.initialise(threshold=0.5,percent=x,print_statistics=False)
  #A.plot_initial_data()
  A.reconstruct(nsteps=1000)
  curr_stats=A.statistics()
  stat_list.append(curr_stats)
  A.save_img()

print stat_list 
quit()
A.initialise(threshold=0.8,percent=0.99,print_statistics=True)

#A.plot_image()
A.reconstruct(nsteps=2000)
#A.plot_image()
print A.statistics()

