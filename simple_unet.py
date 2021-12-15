import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Conv2D, Conv2DTranspose, 
    MaxPooling2D, Dropout, 
    concatenate, Reshape)
from keras import Model
from keras_unet_collection.losses import dice

def conv_block(input, filt):
    C_1 = Conv2D(filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input)
    C_1 = Dropout(0.1)(C_1)
    C_1 = Conv2D(filt, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C_1)
    P_1 = MaxPooling2D((2, 2))(C_1)
    return C_1, P_1

def define_model(outchannels, img_height, img_width, img_channels):
    inputs = tf.keras.layers.Input((img_height, img_width, img_channels))
    filters = 64
    c1, p1 = conv_block(inputs, filters)
    c2, p2 = conv_block(p1, filters*2)
    c3, p3 = conv_block(p2, filters*4)
    c4, p4 = conv_block(p3, filters*8)

    ## BRIDGE/BOTTLENECK
    c5 = Conv2D(
        filters*16, (3, 3), 
        activation='relu', 
        kernel_initializer='he_normal', 
        padding='same')(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(
        filters*16, (3, 3), 
        activation='relu', 
        kernel_initializer='he_normal', 
        padding='same')(c5)

    ## Decoder
    u6 = Conv2DTranspose(
        filters=filters*8,
        kernel_size=(2,2),
        strides=(2,2),
        padding=('same'))(c5)
    # c4 = Reshape(
    #     target_shape=u6.shape[1:]
    # )(c4)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(filters*8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(
        filters=filters*4,
        kernel_size=(2,2),
        strides=(2,2),
        padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(
        filters=filters*2,
        kernel_size=(2,2),
        strides=(2,2),
        padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(filters*4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    
    u9 = Conv2DTranspose(
        filters=filters,
        kernel_size=(2,2),
        strides=(2,2),
        padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(outchannels, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# if __name__ == "__main__":
#     model = define_model(1, int(320*0.8), 320, 1)

#     model.compile(
#         optimizer='adam', 
#         loss='binary_crossentropy', 
#         metrics=[dice]
#     )

#     print(model.summary())
