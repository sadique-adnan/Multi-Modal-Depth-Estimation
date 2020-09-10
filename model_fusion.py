
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate
from tensorflow.keras import Model
from densenet import DenseNet169_multi_channel


def create_model():
    model = DenseNet169_new(input_shape = (None, None, 4), include_top=False, weights='imagenet')
    print(model.summary())
    print('model loaded.')
    
    # Starting point for decoder
    model_output_shape = model.layers[-1].output.shape
    for layer in model.layers: layer.trainable = True

    decode_filters = int(model_output_shape[-1]/2)


        # Upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = UpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
        up_i = Concatenate(name=name+'_concat')([up_i, model.get_layer(concat_with).output]) # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=model_output_shape, name='conv2')(model.output)
    decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
    
    

    # Last Layer
    conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

    # Creation of the model
    model = Model(inputs=model.input, outputs=conv3)

    print('Model created.')

    return model
