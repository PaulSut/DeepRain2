from PIL import Image
import numpy as np
import os
class DataProvider(object):
    """
        
        path: string
            Path to directory were the files are stored.
    
    """
    
    def __init__(self,path,openIMG=True,pattern="scaled_17051.*"):
        def get_files(pwd,pattern=""):
    
            """

                Read all files from path which match a pattern and return a list with the names

            """

            import re
            files = []
            for file in os.listdir(pwd):
                matches = re.search(pattern, file)
                if matches:
                    files.append(matches[0])
            
            files.sort()
            return files

        self.files = get_files(path,pattern)
        self.path = path
        self.openIMG = openIMG
        self.transform = []

    def binary(self):
        self.transform.append(self.binary_)


    def max_contrast(self):
        self.transform.append(self.max_contrast_)

        
    def binary_(self,img,threshold=5):
        
        img[img > threshold ] = 255
        img[img <= threshold] = 0
        
        return img

    def transform(function):
        self.transform.append(function)

    def max_contrast_(self,img):
        
            """

                maximize contrast of images
                also deleting "edges"

            """

            img[img == img[0,0]] = 0

            mi,ma = img.min(),img.max()
            if ma == 0:
                return img

            img -= mi
            img[img == ma -mi] = 0
            ma = img.max()
            img = np.array(((img / ma) * 255),dtype='uint8')
            return img
    
    def _openIMG(self,pwd):
        img = np.array(Image.open(pwd))
        if len(self.transform) > 0:
            for f in self.transform:
                img = f(img)

        return img
        
    def __getitem__(self,i):
        if self.openIMG:
            return self._openIMG(os.path.join(self.path,self.files[i]))
        else:
            return os.path.join(self.path,self.files[i])
        
        
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        self.n = -1
        return self
        
    def __next__(self):
        self.n += 1
        if self.n < len(self.files):
            if self.openIMG:
                return self._openIMG(os.path.join(self.path,self.files[self.n]))
            else:
                return os.path.join(self.path,self.files[self.n])
        else:
            raise StopIteration