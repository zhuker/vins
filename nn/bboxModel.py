from keras.layers import Input, UpSampling2D, Deconv2D,Dense, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
import os
import keras.backend as K
currPath = os.path.dirname(os.path.abspath(__file__))


def bboxModel(test = False):
    inp = Input((160, 320, 1))

    c1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(inp)))
    m1 = MaxPooling2D((2, 2))(c1)

    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(m1)))
    c2 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(64, kernel_size=3, activation='linear', padding='same')(c2)))
    m2 = MaxPooling2D((2, 2))(c2)

    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, activation='linear', padding='same')(m2)))
    c3 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(96, kernel_size=3, activation='linear', padding='same')(c3)))
    m3 = MaxPooling2D((2, 2))(c3)

    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, activation='linear', padding='same')(m3)))
    c4 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(128, kernel_size=3, activation='linear', padding='same')(c4)))
    m4 = MaxPooling2D((2, 2))(c4)

    c5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(m4)))
    c5 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(256, kernel_size=3, activation='linear', padding='same')(c5)))
    m5 = MaxPooling2D((2, 2))(c5)

    c6 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(512, kernel_size=3, activation='linear', padding='same')(m5)))
    c6 = LeakyReLU(alpha=0.2)(BatchNormalization()(Conv2D(1024, kernel_size=3, activation='linear', padding='same')(c6)))

    flat = GlobalAveragePooling2D()(c6)
    d1 = LeakyReLU(alpha=0.2)(BatchNormalization()(Dense(256, activation='linear')(flat)))

    out = Dense(4,activation='sigmoid', name='bbox_out')(d1)

    def iou_loss(true, pred):
        tl_pred_x = pred[:, 0] - (pred[:, 2] / 2.)
        tl_pred_y = pred[:, 1] - (pred[:, 3] / 2.)

        br_pred_x = pred[:, 0] + (pred[:, 2] / 2.)
        br_pred_y = pred[:, 1] + (pred[:, 3] / 2.)

        tl_true_x = true[:, 0] - (true[:, 2] / 2.)
        tl_true_y = true[:, 1] - (true[:, 3] / 2.)

        br_true_x = true[:, 0] + (true[:, 2] / 2.)
        br_true_y = true[:, 1] + (true[:, 3] / 2.)

        inter_upleft_x = K.maximum(tl_pred_x, tl_true_x)
        inter_upleft_y = K.maximum(tl_pred_y, tl_true_y)

        inter_botright_x = K.minimum(br_pred_x, br_true_x)
        inter_botright_y = K.minimum(br_pred_y, br_true_y)

        inter_h = K.relu(inter_botright_x - inter_upleft_x)
        inter_w = K.relu(inter_botright_y - inter_upleft_y)

        inter = inter_h * inter_w

        area_pred = (br_pred_x - tl_pred_x) * (br_pred_y - tl_pred_y)
        area_gt = (br_true_x - tl_true_x) * (br_true_y - tl_true_y)

        union = area_pred + area_gt - inter
        iou = inter / union
        Miou = K.mean(1 - iou)
        #mse = K.mean(K.square(true - pred))
        return Miou

    if test:
        model =  Model(input = inp, output=out)
        model.compile('adam', iou_loss)
    else:
        model = Model(input = inp, outputs=[out, out])
        model.compile('adam', [iou_loss, 'mse'])
    return model