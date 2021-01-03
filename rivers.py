import numpy as np
import scipy as sc
import cmath 
import matplotlib.pyplot as plt 
from copy import copy
from matplotlib.patches import Circle


#print(random.random())

Nx=5#0   #lattice dimension x 
Ny=4#0   #lattice dimension y
I=1 #initial height
E=1/20 #parameter associated with water drop moving 
total_drops=10
D=10 #the decrease in height after one drop passes
M=200#the critical value for an avalanche to occur
R=100

def initialize(Nx,Ny,I):   		#so that the river flows upwards)
	grid=np.zeros((Ny,Nx))
	for i in range(Ny):
		grid[i,:]=I*i
	return grid

def erosion(grid,total_drops):   #determine how many drops in this erosion
 				#if first iteration, grid=initialize(Nx,Ny,I
	Nx=grid[:,0].size  #horizontal dimension
	Ny=grid[0,:].size  #vertical dimension
	#print(Ny,Nx,'Nx,Ny') #(Nx,Ny)=grid.shape
	#plt.imshow(grid)
#	plt.show()
	print(grid.shape)
	moves=np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype='int') 			
	for i in range(total_drops):
		xi=np.random.randint(0,Nx)
		yi=np.random.randint(0,Ny)
		wet=np.array([[yi,xi]],dtype='int') 					 #this will be a list of moves done by the droplet
		print(wet)
		position=np.array([yi,xi],dtype='int') 	
					#this will be the current position of the droplet, updated each time 
		while(position[0]>0):
			#print(position)

									#matrix that is 4 by 2 
			probabilities=np.zeros((1,4)).flatten()
			for j in range(4):
				if(position[0]+moves[j,0]>=Ny):   #	 when we are are at position[1]=Ny-1, we have to forbid position[1]+moves[j,1]=Ny
					probabilities[j]=0   #manually exclude the movement that would bring us out of the grid
				else:
					print((position[1]+moves[j,1])%Nx)
					deltahij=grid[position[0],position[1]]-grid[position[0]+moves[j,0],(position[1]+moves[j,1])%Nx]
					if(deltahij>=0):
						#print(deltahij)
						probabilities[j]=np.exp(E*deltahij)
			norm=np.sum(probabilities)
			#print(norm)
			#print(probabilities)
			probabilities/=norm
			displacement=random.choices(moves,probabilities)[0]	#a choice of a move being done		
			position[0]=position[0]+displacement[0]	
			position[1]=(position[1]+displacement[1])%Nx
			wet=np.append(wet,[position],axis=0)

		#print(wet.size)
		#print(int(wet.size/2))
		for k in range(wet.shape[0]):			#since the size of wet array is NOT the number of locations, but the total number of coordinates = 2 times the number of locations
				grid[wet[k,0],wet[k,1]]-=D
		
						
					#include by hand the case when we are at the edge of the grid by using % - addition modulo etc
				
					
				#random.choices()

			#do the transposition to interchange x and y (so that the river flows upwards)
	return grid	
	
grid=initialize(Nx,Ny,I)
#print(grid)
#plt.imshow(grid)
#plt.show()
#print(grid)
#print(erosion(grid,total_drops))
plt.imshow(erosion(grid,total_drops))
plt.show()

#[1,2]
moves=np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype='int') 
probabilities=np.array([1,2,3,40])
print(random.choices(moves,probabilities)[0])			

#print(np.zeros((4,1)))

#print(int(3)%int(4))



#def find_rivers(grid):
#probabilities=np.zeros((1,4),dtype='int')	
#print(probabilities)

#moves=np.array([[1,0],[-1,0],[0,1],[0,-1]])
#probabilities=np.array([0,0,1,4])
#position=np.array([2,16])
#print(position)
#print(random.choices(moves,probabilities))
#print(np.array(random.choices(moves,probabilities))[0])
#print(wet)
#print(moves)
#print(moves[3]+wet)
#wet=np.array([0,1],dtype='int')
#moves=np.array([[1,0],[-1,0],[0,1],[0,-1]])
#print(wet[0])	
#print(moves[0]+wet[0])
	



#out of the function aggregation i have a grid of ones and zeros
	
#grid=aggregation(N,init_radius,walkers,1/2)
#plt.savefig('prob_1/8.png')
#grid=aggregation(N,init_radius,walkers,1/4)
#plt.matshow(grid)
#plt.savefig('prob_1/8.png')
#grid=aggregation(N,init_radius,walkers,1/2)
#plt.matshow(grid)
#plt.savefig('prob_1/8.png')

#for i in range(N):
#	for j in range(N):
#		if(grid[i,j]==1):
#			counter=counter+1
#print('Counter is equal to:'+str(counter))
#plt.clf()# clear the figure
#F = plt.gcf()	# define a new one
#a = plt.gca()	#  get current axes (to operate on them)
#plt.xlim((0,boxsize))# plot dimensions
#plt.ylim((0,boxsize))
#for i in range(N):
#	for j in range(N):
#		if(grid[i,j]==1):
#			cir = Circle((i,j), radius=1 ,color='black')# put a circle at the position of the sticking particle
#			a.add_patch(cir)	# add this circle to the plot
#plt.plot()                                         # plot it		
#F.set_size_inches((30,30))            # physical size of the plot
#plt.savefig('plotaggregation_final.png') # save the figure

#if (counter%10==0):
	#plt.plot()                                         # plot it		
	#F.set_size_inches((30,30))            # physical size of the plot
	#nStr=str(counter)# convert counter to a string
	#nStr=nStr.rjust(6,'0') # pad with zeros
#plt.savefig('plot'+nStr+'.png') # save the figure	
#counter=counter+1



	
		
		
		
	










