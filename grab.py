import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from grabcut import GrabCut

def checkout(x,p):
	if x>p:
		return p
	if x<0:
		return 0
	return x

def algo_grabcut(filename,foreground=[],background=[],pos1x=1,pos1y=1,pos2x=511,pos2y=511,times=5, algo=True):
	img = cv2.imread(filename)
	mask = np.zeros(img.shape[:2],np.uint8)	
	[height,width] = img.shape[:2]
	pos1x, pos1y = checkout(pos1x,width-1), checkout(pos1y,height-1)
	pos2x, pos2y = checkout(pos2x,width-1), checkout(pos2y,height-1)
	mask[min(pos1y,pos2y):max(pos1y,pos2y)+1,min(pos1x,pos2x):max(pos1x,pos2x)+1]=3
	for y1,x1,y2,x2 in foreground:
		x1,y1 = checkout(x1,height-1), checkout(y1,width-1)
		x2,y2 = checkout(x2,height-1), checkout(y2,width-1)
		if x1==x2:
				mask[x1,min(y1,y2):max(y1,y2)+1] = 1
		else:
			k = (y1-y2)/(x1-x2)
			if (x1 < x2):
				x,y = x1,y1
			else:
				x,y = x2,y2
			while True:
				mask[x,y] = 1
				x = x+1
				y = checkout(int(round(y+k)),width-1)
				if x>max(x1,x2):
					break
	for y1,x1,y2,x2 in background:
		x1,y1 = checkout(x1,height-1), checkout(y1,width-1)
		x2,y2 = checkout(x2,height-1), checkout(y2,width-1)
		if x1==x2:
				mask[x1,min(y1,y2):max(y1,y2)+1] = 0
		else:
			k = (y1-y2)/(x1-x2)
			if (x1 < x2):
				x,y = x1,y1
			else:
				x,y = x2,y2
			while True:
				mask[x,y] = 0
				x = x+1
				y = checkout(int(round(y+k)),height-1)
				if x>max(x1,x2):
					break
	if algo is True:
		rect=(0,0,0,0)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,times,cv2.GC_INIT_WITH_MASK)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		im = img*mask2[:,:,np.newaxis]
		im += 255*(1-cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
		cv2.imwrite('out.png',im)
	else:
		rect=(0,0,0,0)
		bgdModel = np.zeros((1,65),np.float64)
		fgdModel = np.zeros((1,65),np.float64)
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,times,cv2.GC_INIT_WITH_MASK)
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
		im = img*mask2[:,:,np.newaxis]
		im += 255*(1-cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR))
		I_orig = im.copy()
		cv2.imwrite('out.png',im)

		# I_orig = cv2.imread('out.png')		
		mask1 = np.zeros(img.shape[:2],np.uint8)
		mask1[mask==1] = 3
		mask1[mask==2] = 1
		mask1[mask==3] = 2
		X = []
		Y = []		
		Z = []
		for i in [1,5,10]:
			I_seg = GrabCut(filename, mask1,i,5,50)
			print(str(i))
			ans = np.linalg.norm(I_seg - I_orig)
			perc = ans/np.linalg.norm(I_orig)
			X.append(i)
			Y.append(ans)
			Z.append((1-perc)*100)
		plt.plot(X,Y)
		plt.xlabel('Number of iterations')
		plt.ylabel('Error between OpenCV and My implementation')
		plt.show()
		plt.plot(X,Z)
		plt.xlabel('Number of Gamma Components')
		plt.ylabel('Percentage Accuracy')
		plt.show()
	return True