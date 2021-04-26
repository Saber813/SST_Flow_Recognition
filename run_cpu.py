'''
    此文件为cpu下训练文件
    相比于gpu
    引入包 缺少 from keras.utils import multi_gpu_model
    也就是多gpu训练的函数 model = multi_gpu_model(model, gpus=2) (用于model=sst_emotion之后)
    run() 开头 缺少
        :   # 控制GPU资源使用 运行过程中动态申请内存
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

'''
import argparse
import configparser
import os
import keras
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
# from keras.utils import multi_gpu_model
# import tensorflow as tf

from model import model as sst_model

train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None

test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None

result_path = None
model_save_path = None

input_width = None
specInput_length = None
temInput_length = None

depth_spec = None
depth_tem = None
nb_layers_per_block = None
gr_spec = None
gr_tem = None
nb_dense_block = None
nb_class = None

nbEpoch = None
batch_size = None
lr = None


def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path, encoding='utf-8')

    global train_specInput_root_path, train_tempInput_root_path, train_label_root_path, test_specInput_root_path, test_tempInput_root_path, test_label_root_path
    train_specInput_root_path = conf['path']['train_specInput_root_path']
    train_tempInput_root_path = conf['path']['train_tempInput_root_path']
    train_label_root_path = conf['path']['train_label_root_path']
    test_specInput_root_path = conf['path']['test_specInput_root_path']
    test_tempInput_root_path = conf['path']['test_tempInput_root_path']
    test_label_root_path = conf['path']['test_label_root_path']

    global result_path, model_save_path
    result_path = conf['path']['result_path']
    model_save_path = conf['path']['model_save_path']

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    global input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, nb_class, nb_layers_per_block
    input_width = int(conf['model']['input_width'])
    specInput_length = int(conf['model']['specInput_length'])
    temInput_length = int(conf['model']['temInput_length'])
    depth_spec = int(conf['model']['depth_spec'])
    depth_tem = int(conf['model']['depth_tem'])
    nb_layers_per_block = int(conf['model']['nb_layers_per_block'])
    gr_spec = int(conf['model']['gr_spec'])
    gr_tem = int(conf['model']['gr_tem'])
    nb_dense_block = int(conf['model']['nb_dense_block'])
    nb_class = int(conf['model']['nb_class'])

    global nbEpoch, batch_size, lr
    nbEpoch = int(conf['training']['nbEpoch'])
    batch_size = int(conf['training']['batch_size'])
    lr = float(conf['training']['lr'])


def run():
    global history

    # 训练模式 :0 测试模式 :1
    K.set_image_data_format('channels_last')
    K.set_learning_phase(0)

    all_result_file = open(os.path.join(result_path, 'all_result.txt'), "w")
    all_result_file.close()

    # 这里我不能单被试进行实验 分类样本太少! 只能通过去完基线的数据 一起进行实验.
    # 所以我的训练集就是 63个样本 每个样本有时域特征和频域特征
    train_specInput = np.load(train_specInput_root_path)
    train_tempInput = np.load(train_tempInput_root_path)
    train_label = np.load(train_label_root_path)

    # 封装数据时混乱
    # 包装的时候 用第零维代表个数
    index = np.arange(train_specInput.shape[0])
    np.random.shuffle(index)

    train_specInput = train_specInput[index]
    train_tempInput = train_tempInput[index]
    train_label = train_label[index]

    # 编码为 0 1 2 ; x if 编码为 1 2 3
    train_label = [x - 1 for x in train_label]
    train_label = to_categorical(train_label)

    # Evaluate
    test_specInput = np.load(test_specInput_root_path)
    test_tempInput = np.load(test_tempInput_root_path)
    test_label = np.load(test_label_root_path)

    # 编码为 0 1 2 ; x if 编码为 1 2 3
    test_label = [x - 1 for x in test_label]
    test_label = to_categorical(test_label)

    # 可以调整用depth 还是用nb_layers_per_block
    model = sst_model.sst_emotionnet(input_width=input_width, specInput_length=specInput_length,
                                     temInput_length=temInput_length,
                                     depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem,
                                     nb_dense_block=nb_dense_block, nb_class=nb_class,
                                     nb_layers_per_block=nb_layers_per_block)
    # 训练模式
    if K.learning_phase == 0:
        adam = keras.optimizers.Adam(
            lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, verbose=1)

        save_model = ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True)

        history = model.fit([train_specInput, train_tempInput], train_label, epochs=nbEpoch, batch_size=batch_size,
                            validation_data=([test_specInput, test_tempInput], test_label),
                            callbacks=[early_stopping, save_model], verbose=1, validation_split=0.25)
    # 测试模式
    if K.learning_phase == 1:
        model = load_model(model_save_path)
        loss, accuracy = model.evaluate(
            [test_specInput, test_tempInput], test_label)
        print('\ntest loss', loss)
        print('accuracy', accuracy)

        # Result Processing

        f = open(result_path, "w")
        print(history.history, file=f)
        f.close()

        maxAcc = max(history.history['val_accuracy'])
        print("maxAcc = " + str(maxAcc))
        all_result_file = open(os.path.join(result_path, 'all_result.txt'), "a")
        print(str(accuracy), file=all_result_file)
        all_result_file.close()

    keras.backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument of running SST-EmotionNet.')
    parser.add_argument(
        '-c', type=str, help='Config file path.', required=True)
    args = parser.parse_args()
    read_config(args.c)

    # plot model
    model = sst_model.sst_emotionnet(input_width=input_width, specInput_length=specInput_length,
                                     temInput_length=temInput_length,
                                     depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem,
                                     nb_dense_block=nb_dense_block, nb_class=nb_class,
                                     nb_layers_per_block=nb_layers_per_block)
    plot_model(model, to_file='model_auth.png', show_shapes=True)

    # run()
