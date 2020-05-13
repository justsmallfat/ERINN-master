# 預測電阻綠 (測試模型)
import os
import re
from tqdm import tqdm
import importlib

import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam

from erinn.python.generator import PredictGenerator
from erinn.python.metrics import r_squared
from erinn.python.utils.io_utils import get_pkl_list, read_pkl, write_pkl, read_config_file

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

# setting
# config_file = os.path.join('..', 'config', 'config.yml')
pkl_dir_test = os.path.join('..', 'data', '1w', 'test')#下拉
model_dir = os.path.join('..', 'models', 'add_noise')#下拉

weights_dir = os.path.join(model_dir, 'weights')#文字
predictions_dir = os.path.join(model_dir, 'predictions', 'raw_data')#文字

pkl_list_test = get_pkl_list(pkl_dir_test)
input_shape = (210, 780, 1)
output_shape = (30, 140, 1)

preprocess_generator = {'add_noise':
                            {'perform': False,
                             'kwargs':
                                 {'ratio': 0.1}
                             },
                        'log_transform':
                            {'perform': True,
                             'kwargs':
                                 {'inverse': False,
                                  'inplace': True}
                             }
                        }
# data generator
testing_generator = PredictGenerator(pkl_list_test, input_shape, output_shape,
                                     batch_size=64, **preprocess_generator)


# load custom keras model
# reference: https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
pattern = re.compile(r'\'([^\']+)\'')
config_file = os.path.join('..', 'config', 'config.yml')
config = read_config_file(config_file)
module_name, py_file = re.findall(pattern, config['custom_NN'])#
loader = importlib.machinery.SourceFileLoader(module_name, py_file)
spec = importlib.util.spec_from_loader(module_name, loader)
module = importlib.util.module_from_spec(spec)
loader.exec_module(module)
model = getattr(module, module_name)()

model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[r_squared])
model.load_weights(os.path.join(weights_dir, 'trained_weight.h5'))

print('\nPredict.')
predict = model.predict_generator(testing_generator, workers=os.cpu_count(), verbose=True)

os.makedirs(predictions_dir, exist_ok=True)
with tqdm(total=len(pkl_list_test), desc='write pkl') as pbar:
    for i, pred in enumerate(predict):
        data = read_pkl(pkl_list_test[i])
        data['synth_V'] = data.pop('inputs').reshape(input_shape[0:2])
        data['synth_log_rho'] = data.pop('targets').reshape(output_shape[0:2])
        data['pred_log_rho'] = pred.reshape(output_shape[0:2])

        suffix = re.findall(r'\d+.pkl', pkl_list_test[i])[0]
        write_pkl(data, os.path.join(predictions_dir, f'result_{suffix}'))
        pbar.update()
