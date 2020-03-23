#!/home/simon/anaconda3/bin/python

import sys
print(sys.executable)
from PIL import ImageFont, ImageDraw, Image
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import figure
import cv2 as cv
from Dataset import DataProvider
from Cloud import Cloud
from ColorHelper import MplColorHelper


windowname = 'OpenCvFrame'
cv.namedWindow(windowname)
cv.moveWindow(windowname,0,00)


class Tracker(object):
    """

    docstring for Tracker

    """
    def __init__(self, path,max_dist = 10,binary=True,max_contrast=True,transform=None):
        super(Tracker, self).__init__()

        self.path = path
        self.data = DataProvider(path)
        self.max_dist = max_dist
        self.threshold = 5
        if max_contrast:
            self.data.max_contrast()
        if binary:
            self.data.binary()
    

    def label_To_index(self,label,img):
        return np.where(img == label)

    def getClouds(self,cloudlist,img):
        clouds = [Cloud(self.label_To_index(label,img)) for label,size in cloudlist]
        return clouds

    def sequentialLabeling(self,img):

        """
    
    
            Label clouds and return array of tuble (label,cloudsize)
            labels are sorted in descending order by cloudsize
            Explicit location of cloud labeld by label_A 
            can be found by np.where(img == label_A)


        """
        
        img[img >= self.threshold] = 1
        x,y = np.where(img == 1)


        collision = dict()
        label = 2

        for i,j in zip(x,y):
            i_X = slice(i-self.max_dist,i+self.max_dist)
            j_Y = slice(j-self.max_dist,j+self.max_dist)

            window = img[i_X,j_Y]

            neighbours = np.argwhere(window > 1)


            if len(neighbours) == 0:
                window[window == 1] = label
                label +=1
                img[i_X,j_Y] = window

            elif len(neighbours) == 1:
                window[window == 1] = window[neighbours[0,0],neighbours[0,1]]
                img[i_X,j_Y] = window


            # handle label collisions

            else:
                k = np.amax(window)
                img[i,j] = k
                for index in neighbours:
                    nj = window[index[0], index[1]]

                    if nj != k:
                        if k not in collision:
                            collision[k] = set()
                        collision[k].add(nj)
                        if collision[k] is None:
                            del collision[k]



        def changeLabel(elem):
            c_label = collision[elem]
            for l in c_label:
                img[img == l] = elem


        def rearangeCollisions():
            for elem in collision:
                for item in collision[elem]:
                    if item in collision:
                        collision[elem] = (collision[elem] | collision[item])
                        collision[item] = set()
                if elem in collision[elem]:
                    collision[elem].remove(elem)

        rearangeCollisions()


        for i,elem in enumerate(collision):
            if collision[elem] is None:
                continue
            changeLabel(elem)

        cloud_size = []

        for i in range(2,label):
            a = len(np.where(img == i)[0])

            if a == 0:
                continue
            cloud_size.append((i,a))
        cloud_size = sorted(cloud_size, key=lambda x: x[1],reverse = True)

        return cloud_size

    def mapToColor(self,img,cloud_size):
        COL = MplColorHelper('hsv', 0, len(cloud_size))

        colors = {}
        for i,elem in enumerate(cloud_size):
            colors[elem[1]] = COL.get_rgb(i)
        img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)

        for label,size in cloud_size:
            index = np.where(img[:,:,0] == label)
            img[index] = colors[size]

        return img

    def draw_flow(self,imgs,flow, step=8):

                h, w = imgs.shape[:2]
                y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
                fx, fy = flow[y, x].T
                lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
                #print(lines)
                lines = np.int32(lines + 0.5)
                vis = cv.cvtColor(imgs, cv.COLOR_GRAY2BGR)
                cv.polylines(vis, lines, 0, (0, 255, 0))
                #for (x1, y1), (x2, y2) in lines:
                #    cv.circle(vis, (x1, y1), 1, (255, 255, 0), -1)
                return vis

    def calcFlow(self,img0,img1,prevPts=None,nextPts=None):
        
        flow = cv.calcOpticalFlowFarneback(img0, img1, 
                                              #None,
                                              #prevPts, 
                                              nextPts,
                                              pyr_scale = 0.5, 
                                              levels = 5, 
                                              winsize = 11, 
                                              iterations = 5, 
                                              poly_n = 5, 
                                              poly_sigma = 1.1,
                                              flags=0) 
                                              #flags = cv.OPTFLOW_USE_INITIAL_FLOW)

        return flow

    def showset(self):
        def show(img):
            cv.imshow(windowname,img)
        
        for i,img in enumerate(self.data):
            cloudlist = self.sequentialLabeling(img)
            clouds = self.getClouds(cloudlist,img)

            for c in clouds:
                img = c.bbox(img)
            #img = self.mapToColor(img,cloudsize)
            show(img)
            if cv.waitKey(25) & 0XFF == ord('q'):
                break

            


        cv.destroyAllWindows()

    def create(self,inputPath, outputPath, delay, finalDelay, loop):
            cmd = "convert -delay {} {}*.png -delay {} -loop {} {}".format(
            delay, inputPath, finalDelay, loop,
            outputPath)
            print(cmd)
            os.system(cmd)

    def showFlow(self,create_gif=False,name="clouds.gif",nbr_imgs=0):
        img_list = []
        flow = None 

        if create_gif: 
            folder = "GIF/"
            if not os.path.exists(folder):
                os.mkdir(folder)

        for i,img in enumerate(self.data):
            if len(img_list) == 0:
                mask = np.zeros_like(img)


            cloudlist = self.sequentialLabeling(img)
            clouds = self.getClouds(cloudlist,img)

            #clouds as points for tracking
            #pts = [ cloud.points for cloud in clouds]
            #center of mass as points for tracking
            pts = [ cloud.center_of_mass for cloud in clouds]

            # need to reshape pts
            pts = np.array(pts)
            x,y = pts.shape
            pts = pts.reshape(y,1,x) 
            
            if len(img_list) >= 2:
                img_list.pop(0)
            
            img_list.append((img,pts) )

            if len(img_list) < 2:
                continue
            
            flow = self.calcFlow(img_list[0][0],img_list[1][0],pts)
            vis = self.draw_flow(mask,flow)
            mask = cv.cvtColor(vis,cv.COLOR_RGB2GRAY)
            frame = np.concatenate((img_list[0][0],mask),axis=1)
            
            cv.imshow(windowname,frame)
            print(i,end="\r")


            if cv.waitKey(25) & 0XFF == ord('q'):
                break


            if create_gif and len(img_list) == 2:
                filename = "{0:0>5}".format(i)
                cv.imwrite(os.path.join(folder,filename+".png"),frame)
                if i == nbr_imgs and nbr_imgs != 0:
                    break
        if create_gif:
            self.create(folder,name,20,250,0)
                





t = Tracker("../PNG")
#t.showset()
t.showFlow(create_gif=False,name="clouds_as_center_of_mass.gif")
cv.destroyAllWindows()
