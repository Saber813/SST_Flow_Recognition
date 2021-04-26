import keras
import keras.backend as K
from keras.layers import Input
from keras.layers.convolutional import Conv3D
from keras.layers.core import Activation, Dense, Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import (AveragePooling3D, GlobalAveragePooling3D,
                                  MaxPooling3D)
from keras.models import Model
from keras.regularizers import l2


# 卷积操作，按照论文的说法，这里应该是一个组合函数，分别为： / 经典的DenseBlock 和 有bottleneck 层的卷积
# BatchNormalization -> ReLU -> 3x3 Conv
# 输入(l*k channels) -> BN + ReLU + 1×1卷积(4k个) → 输出 → BN + RuLU + 3×3卷积 -> (k channels)
# pseudo-3D 模块 拆分 3*3*3conv -> 3*3*1conv & 1*1*3 conv 提高训练效率
def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    """

    @param ip: Input keras 张量
    @param nb_filter: filter的数量
    @param bottleneck: add bottleneck block
    @param dropout_rate: dropout rate
    @param weight_decay: weight decay factor // 权重衰减: L2正则化来防止个别权重对总体结果产生过大的影响
    @return x : keras tensor with batch_norm, relu and convolution3d added (optional bottleneck)
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # 在每一个批次的数据中标准化前一层的激活项， 即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。
    # axis 冻结特征轴 epsilon 增加到方差的小的浮点数，以避免除以零
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4

        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    # pseudo-3D
    x = Conv3D(nb_filter, (3, 3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = Conv3D(nb_filter, (1, 1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


# channel_wise_mean usebefore attention
def channel_wise_mean(x):
    """

    @param x: tensor
    @return: tensor after channel_wise_mean
    """
    mid = K.mean(x, axis=-1)
    return mid


# 使用循环实现了dense_block的密集连接
def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    """

    @param x:keras tensor
    @param nb_layers:
        每个denseblock层里面__conv_block的数量
    @param nb_filter: number of filters
    @param growth_rate: growth rate
    @param bottleneck: bottleneck block
    @param dropout_rate: dropout rate
    @param weight_decay: weight decay factor
    @param grow_nb_filters:
        @todo flag to decide to allow number of filters to grow
    @param return_concat_list:
        @todo return the list of feature maps along with the actual output
    @return:
        keras tensor with nb_layers of conv_block appened
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck,
                          dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter

    return


# 构造attention_block
def Attention_block(input_tensor, spatial_attention=True, temporal_attention=True):
    """

    @param input_tensor: 输入张量
    @param spatial_attention: flag to decide to allow spatial_attention
    @param temporal_attention: flag to decide to allow temporal_attention
    @return: tensor after attention
    """
    tem = input_tensor
    # 将任意表达式封装为一个layer对象
    x = Lambda(channel_wise_mean)(input_tensor)
    x = keras.layers.Reshape([K.int_shape(input_tensor)[1], K.int_shape(
        input_tensor)[2], K.int_shape(input_tensor)[3], 1])(x)

    nbSpatial = K.int_shape(input_tensor)[1] * K.int_shape(input_tensor)[2]
    nbTemporal = K.int_shape(input_tensor)[-2]

    if spatial_attention:
        spatial = AveragePooling3D(
            pool_size=[1, 1, K.int_shape(input_tensor)[-2]])(x)
        spatial = keras.layers.Flatten()(spatial)
        spatial = Dense(nbSpatial)(spatial)
        spatial = Activation('sigmoid')(spatial)
        spatial = keras.layers.Reshape(
            [K.int_shape(input_tensor)[1], K.int_shape(input_tensor)[2], 1, 1])(spatial)

        tem = keras.layers.multiply([input_tensor, spatial])

    if temporal_attention:
        temporal = AveragePooling3D(pool_size=[K.int_shape(input_tensor)[
                                                   1], K.int_shape(input_tensor)[2], 1])(x)
        temporal = keras.layers.Flatten()(temporal)
        temporal = Dense(nbTemporal)(temporal)
        temporal = Activation('sigmoid')(temporal)
        temporal = keras.layers.Reshape(
            [1, 1, K.int_shape(input_tensor)[-2], 1])(temporal)

        tem = keras.layers.multiply([temporal, tem])

    return tem


# 过渡层，用来连接两个dense_block ,并降低特征图的大小
# 按照论文的说法，过渡层由四部分组成：
# BatchNormalization -> ReLU -> 1x1Conv -> 2x2Maxpooling。
# 在最后一个dense_block的尾部不需要使用过渡层。
def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    """

    @param ip:keras tensor
    @param nb_filter:number of filters
    @param compression:
        压缩系数小于1: DenseNet-C 压缩系数小于1+bottleneck :DenseNet-BC
        caculated as 1 - reduction. Reduces the number of features maps in the transition block
    @param weight_decay:weight decay factor
    @return x : keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x


# 创建完整的3DCM 可以选择输出 if include_top =1
def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax', attention=True, spatial_attention=True,
                       temporal_attention=True):
    """

    @param nb_classes: 分类数量
    @param img_input:  tuple of shape (channels, rows, columns) or (rows, columns, channels)
    @param include_top: flag to include the final Dense layer
    @param depth:
        @todo number or layers . why?
    @param nb_dense_block: number of dense blocks to add to end (generally = 3)
    @param growth_rate: number of filters to add per dense block
    @param nb_filter: initial number of filters
    @param nb_layers_per_block: number of layers in each dense block
    @param bottleneck: add bottleneck blocks
    @param reduction: eduction factor of transition blocks. Note : reduction value is inverted to compute compression
    @param dropout_rate: dropout rate
    @param weight_decay: weight decay rate
    @param subsample_initial_block:
        @todo Set to True to subsample the initial convolution and
                    add a MaxPool3D before the dense blocks are added.
    @param activation:
        Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
    @param attention:...
    @param spatial_attention:...
    @param temporal_attention:...
    @return:
    """
    # 如果channel 在第一维的话 那此值为1 否则(tensorflow默认)为-1
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # layers in each dense block
    # 两种方式来确定layer的数量 1. depth确定 2. 直接通过规定nb_layers_per_block 来确定
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block), 'If list, nb_layer is used as provided. ' \
                                                   'Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (
                           depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)

            if bottleneck:
                count = count // 2

            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # compression factor in transition layer
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'
    compression = 1.0 - reduction

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # Initial convolution
    # todo : 2. 三次卷积插值
    if subsample_initial_block:
        initial_kernel = (5, 5, 3)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 1)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # Add dense blocks 在这里调换attention的位置
    for block_idx in range(nb_dense_block - 1):
        # add attention_block
        if attention:
            x = Attention_block(
                x, spatial_attention=spatial_attention, temporal_attention=temporal_attention)

        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(
            x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    if attention:
        x = Attention_block(
            x, spatial_attention=spatial_attention, temporal_attention=temporal_attention)
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)

    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


# 总架构
def sst_emotionnet(input_width, specInput_length, temInput_length, depth_spec,
                   depth_tem, gr_spec, gr_tem, nb_dense_block, nb_layers_per_block,
                   attention=False, spatial_attention=True, temporal_attention=True, nb_class=3):
    '''
        Model Input: [Spatial-Spectral Stream Input, Spatial-Temporal Stream Input]
    '''

    '''
    原始模型:
    # Input用于实例化张量
    # Spatial-Spectral Stream
    specInput = Input([input_width, input_width, specInput_length, 1])
    x_s = __create_dense_net(img_input=specInput, depth=depth_spec, nb_dense_block=nb_dense_block,
                             growth_rate=gr_spec, nb_classes=nb_class, reduction=0.5, bottleneck=True,
                             include_top=False, attention=attention, spatial_attention=spatial_attention,
                             temporal_attention=temporal_attention)
    # Spatial-Temporal Stream
    temInput = Input([input_width, input_width, temInput_length, 1])
    x_t = __create_dense_net(img_input=temInput, depth=depth_tem, nb_dense_block=nb_dense_block,
                             growth_rate=gr_tem, nb_classes=nb_class, bottleneck=True, include_top=False,
                             subsample_initial_block=True, attention=attention)
    '''
    # Spectral-Temporal Stream
    specInput = Input([input_width, input_width, specInput_length, 1])
    x_s = __create_dense_net(img_input=specInput, depth=depth_spec, nb_dense_block=nb_dense_block,
                             growth_rate=gr_spec, nb_classes=nb_class, reduction=0.5, bottleneck=True,
                             include_top=False, attention=attention, spatial_attention=spatial_attention,
                             temporal_attention=temporal_attention, nb_layers_per_block=nb_layers_per_block)
    # Spatial-Temporal Stream
    temInput = Input([input_width, input_width, temInput_length, 1])
    x_t = __create_dense_net(img_input=temInput, depth=depth_tem, nb_dense_block=nb_dense_block,
                             growth_rate=gr_tem, nb_classes=nb_class, bottleneck=True, include_top=False,
                             attention=attention, spatial_attention=spatial_attention,
                             temporal_attention=temporal_attention, nb_layers_per_block=nb_layers_per_block)

    y = keras.layers.concatenate([x_s, x_t], axis=-1)
    y = keras.layers.Dense(50)(y)
    y = keras.layers.Dropout(0.5)(y)

    if nb_class == 2:
        y = keras.layers.Dense(nb_class, activation='sigmoid')(y)
    else:
        y = keras.layers.Dense(nb_class, activation='softmax')(y)

    model = Model([specInput, temInput], y)

    return model
