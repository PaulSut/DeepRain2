
import numpy as np
import cv2 as cv

class Cloud():
    def __init__(self,points,size = None):
        
        self.points = points
        if size is None:
            self.size = len(points[0])
        else:
            self.size = size
        self.color = [255,255,255]
        self.center_of_mass = int(np.sum(points[0])/len(points[0])),int(np.sum(points[1])/len(points[1]))
        self.min_x,self.max_x = points[0].min(),points[0].max()
        self.min_y,self.max_y = points[1].min(),points[1].max()  
            
    def __lt__(self, other):
        return self.size < other.size

    def __eq__(self, other):
        return self.size == other.size
    
    def __str__(self):
        return "Cloudsize: {}\nCenter : {}\n".format(self.size,self.center_of_mass)
    
    def addText(self,img,coord,text):

        font                   = cv.FONT_HERSHEY_COMPLEX_SMALL
        fontScale              = 0.5
        fontColor              = (255,255,255)
        lineType               = 1

        cv.putText(img,
                   text, 
                   coord,
                   font, 
                   fontScale,
                   fontColor,
                   lineType)


        return img

    def bounding_box(self,img):


        if self.size <= 1:
            return img
  
        ret = img
        ret[self.min_x:self.max_x,self.min_y] = 255
        ret[self.min_x:self.max_x,self.max_y] = 255
        ret[self.min_x,self.min_y:self.max_y] = 255
        ret[self.max_x,self.min_y:self.max_y] = 255
            


        ret = self.addText(ret,
                        (self.max_y,self.max_x+50),
                        "SWP: ("+str(self.center_of_mass[0])+","+str(self.center_of_mass[1])+")")
            
        ret[self.max_x:self.max_x+40,self.max_y] = 255

        return ret
    
    
    def bbox(self,img):
        return self.bounding_box(img)
    
    def paintcolor(self,img):
        img[self.points] = self.color
        return img
        
    def points_to_2D(self):
        ret = np.array(self.points)
        x,y = ret.shape
        return ret.reshape(y,1,x)

    def dist(self,cloud):
        """
        
            returns euclidean dist between center of ma
        
        """
        
        return np.sqrt((cloud.center_of_mass[0] - self.center_of_mass[0])**2 +
                       (cloud.center_of_mass[1] - self.center_of_mass[1])**2)