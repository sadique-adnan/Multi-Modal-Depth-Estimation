import tensorflow as tf
import tensorflow.keras.backend as K

def depth_loss_function(y_true, y_pred, maxDepthVal=10.0/1.0):
    
    # Point-wise depth
    absdiff = K.abs(y_pred-y_true)
    C = 0.2*K.max(absdiff)
    l_depth =  K.mean(tf.where(absdiff < C, absdiff,(absdiff*absdiff+C*C)/(2*C) ))



    #l_depth = K.mean(K.square(y_pred - y_true), axis=-1)
    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0


    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (K.mean(l_depth))
 
