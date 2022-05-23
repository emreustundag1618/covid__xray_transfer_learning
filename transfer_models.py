from tensorflow.keras.applications import InceptionV3, DenseNet121, ResNet50, Xception


def get_models():
    
    models_ = dict(
                    
                    # this is used for ChexNet
                    DenseNet121=dict(
                        input_shape=(224, 224, 3),
                        module_name="densenet",
                        last_conv_layer="conv5_block16_concat",
                    ),
                    ResNet50=dict(
                        input_shape=(224, 224, 3),
                        module_name="resnet50",
                        last_conv_layer="conv5_block3_out",
                    ),
                    InceptionV3=dict(
                        input_shape=(299, 299, 3),
                        module_name="inception_v3",
                        last_conv_layer="mixed10",
                    ),
                    Xception=dict(
                        input_shape=(299, 299, 3),
                        module_name="xception",
                        last_conv_layer="block14_sepconv2_act",
                    )
                
                )
    
    return models_
