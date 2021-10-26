from tensorflow.keras.applications import VGG16, VGG19, InceptionV3, NASNetMobile, NASNetLarge, DenseNet121, ResNet50, Xception, InceptionResNetV2

models_ = dict(
                VGG16 = dict(
                    input_shape = (224,224,3),
                    module_name = "vgg16",
                    last_conv_layer = "block5_conv3",
                ),
                VGG19 = dict(
                    input_shape = (224,224,3),
                    module_name = "vgg19",
                    last_conv_layer = "block5_conv4",
                ),
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
                InceptionResNetV2=dict(
                    input_shape=(299, 299, 3),
                    module_name="inception_resnet_v2",
                    last_conv_layer="conv_7b_bn",
                ),
                NASNetMobile=dict(
                    input_shape=(224, 224, 3),
                    module_name="nasnet",
                    last_conv_layer="normal_concat_12",
                ),
                NASNetLarge=dict(
                    input_shape=(331, 331, 3),
                    module_name="nasnet",
                    last_conv_layer="normal_concat_18",
                ),
                Xception=dict(
                    input_shape=(299, 299, 3),
                    module_name="xception",
                    last_conv_layer="block14_sepconv2_act",
                ),
            
            )
