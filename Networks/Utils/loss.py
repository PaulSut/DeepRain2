from __future__ import absolute_import
import tensorflow as tf 
from keras import backend as K

class SSIM(tf.keras.losses.Loss):
    """docstring for ClassName"""
    def __init__(self, kernel_size=3,k1=0.01,k2=0.03,l=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = [kernel_size,kernel_size]
        self.k1 = k1
        self.k2 = k2
        self.l = l
        self.c1=(l*self.k1)**2
        self.c2=(l*self.k2)**2


    @tf.function
    def __int_shape(self,x):
        return tf.shape(x)


    @tf.function
    def __call__(self,y_pred,y_true,sample_weight=None):

        img_pred = tf.image.extract_patches(images=y_pred, 
                                            sizes=[1, self.kernel_size, self.kernel_size, 1], 
                                            strides=[1, self.kernel_size, self.kernel_size, 1], 
                                            rates=[1, 1, 1, 1], 
                                            padding='VALID')

        img_true = tf.image.extract_patches(images=y_true, 
                                            sizes=[1, self.kernel_size, self.kernel_size, 1], 
                                            strides=[1, self.kernel_size, self.kernel_size, 1], 
                                            rates=[1, 1, 1, 1], 
                                            padding='VALID')

        bs, w, h, c = img_true.shape

        if bs is None:
            bs = 1
        


        
        img_true = tf.reshape(img_true,[bs,w,h,-1])
        img_pred = tf.reshape(img_pred,[bs,w,h,-1])
        
        
        u_x = tf.reduce_mean(img_true,axis=(1,2))
        u_y = tf.reduce_mean(img_pred,axis=(1,2))

        
        
        
        
        v_x = tf.math.reduce_std(img_true,axis=(1,2))
        v_y = tf.math.reduce_std(img_pred,axis=(1,2))

        cov = tf.reduce_mean(img_pred * img_true,axis=(1,2)) - (u_x*u_y)
        
        ssim_top = ((2* u_x * u_y) + self.c1)*((2*cov) + self.c2)
        ssim_bot = (tf.square(u_x) + tf.square(u_y) + self.c1)*(tf.square(v_x) + tf.square(v_y) + self.c2)
        ssim = ssim_top / ssim_bot
        
        return tf.reduce_mean((1-ssim)/2.0,axis=-1)

        

        