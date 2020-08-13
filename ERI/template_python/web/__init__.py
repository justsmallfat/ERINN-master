import json
import os
from time import sleep

from flask import Flask, request, jsonify, send_from_directory
import yaml
import datetime
import magic
from werkzeug.utils import secure_filename
from keras.callbacks import EarlyStopping

from ERI.template_python.web.EarlyStoppingAtMinLoss import EarlyStoppingAtMinLoss
from erinn.python.FW2_5D.fw2_5d_ext import make_dataset
from erinn.python.FW2_5D.fw2_5d_ext import get_forward_para
from erinn.python.utils.io_utils import read_config_file
from erinn.python.utils.urf import URF

app = Flask(__name__)
global progressData
progressData = {}
progressData['generateData'] = {'name':'Null','value':'None','message':''}
progressData['training'] = {'name':'Null','value':'None','message':''}
progressData['predictResistivity'] = {'name':'Null','value':'None','message':''}
progressData['log'] = {'name':'Null','value':'None','message':''}
nowTime = datetime.datetime.now()

@app.route('/')
def hello_world():
    return 'ERINN!'

@app.route('/getConfigs', methods=['POST'])
def getConfigs():
    from os import walk
    print('getConfigs')
    config_dir = os.path.join('..', 'config')
    files = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        files.extend(filenames)
        break
    return ','.join([ele for ele in files if 'yml' in ele])

@app.route('/getConfigData', methods=['POST'])
def getConfigData():
    print('getConfigData')
    requestPostDictionary = request.values
    configFileName = requestPostDictionary.get("configFileName")
    config_dir = os.path.join('..', 'config', configFileName)
    stream = open(config_dir, "r")
    yaml_data = yaml.safe_load(stream)
    return json.dumps(yaml_data)

@app.route('/getTrainingDataList', methods=['POST'])
def getTrainingDataList():
    from os import walk
    print('getTrainingDataList')
    config_dir = os.path.join('..', 'data')
    dirs = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        dirs.extend(dirnames)
        break
    return ','.join([ele for ele in dirs])

@app.route('/getReportsList', methods=['POST'])
def getReportsList():
    from os import walk
    print('getReportsList')
    config_dir = os.path.join('..', 'reports')
    dirs = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        dirs.extend(dirnames)
        break
    return ','.join([ele for ele in dirs])

@app.route('/getReportImgsList', methods=['POST'])
def getReportImgsList():
    from os import walk
    print(f'getReportImgsList 1')
    jsonTest = request.json
    figs_dir_path = jsonTest.get("figs_dir")
    config_dir = os.path.join('..', 'reports', figs_dir_path, 'testing_figs_raw')
    print(f'getReportImgsList {config_dir}')
    files = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        files.extend(filenames)
        break

    print(f'getReportImgsList {files}')
    return ','.join([ele for ele in files if 'png' in ele])

@app.route('/getModelList', methods=['POST'])
def getModelList():
    print('getModelList')
    from os import walk
    config_dir = os.path.join('..', 'models')
    dirs = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        dirs.extend(dirnames)
        break
    return ','.join([ele for ele in dirs])


@app.route('/getDLModels', methods=['POST'])
def getDLModels():
    print('getDLModels')
    from os import walk
    config_dir = os.path.join('..', 'config')
    files = []
    for (dirpath, dirnames, filenames) in walk(config_dir):
        files.extend(filenames)
        break
    return ','.join([ele for ele in files if 'py' in ele])

@app.route('/generateData', methods=['POST'])
def generateData():
    jsonTest = request.json
    newConfigFileName = jsonTest.get("newConfigFileName")
    # print(f"generateData start {newConfigFileName}")
    progressData['generateData']['name'] = newConfigFileName
    progressData['generateData']['value'] = 'Start!'
    progressData['generateData']['message'] = ''
    newConfigFilePath = os.path.join('..', 'config', newConfigFileName+'.yml')
    f = open(newConfigFilePath, "w")
    for k, v in jsonTest.items():
        f.write(f"{k} : {v}\n")
    f.close()

    try:
        initStopedConfig(newConfigFileName)
        make_dataset(newConfigFilePath, progressData)
    except Exception as e:
        print(f'Exception {e}')
        progressData['generateData']['value'] = 'Error!'
        progressData['generateData']['message'] = f'{e}'
        progressData['log']['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d %X")}'
        progressData['log']['value'] = 'generateData'
        progressData['log']['message'] = f'{e}'

    return jsonify(request.values)
    # return "GenerateData finish"

@app.route('/training', methods=['GET', 'POST'])
def training():
    import importlib
    import os
    import re
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.keras.utils import multi_gpu_model
    from erinn.python.generator import DataGenerator
    from erinn.python.metrics import r_squared
    from erinn.python.utils.io_utils import get_pkl_list, read_config_file
    from erinn.python.utils.os_utils import OSPlatform

    try:
        # Allowing GPU memory growth
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)

        # setting
        print("training start")
        jsonTest = request.json
        newConfigFileName = jsonTest.get("newConfigFileName")
        progressData['training']['name'] = newConfigFileName
        progressData['training']['value'] = 'Start!'
        progressData['training']['message'] = ''
        newConfigFilePath = os.path.join('..', 'config', newConfigFileName+'.yml')
        f = open(newConfigFilePath, "w")
        for k, v in jsonTest.items():
            f.write(f"{k} : {v}\n")

        f.close()
        # setting
        config_file = os.path.join('..', 'config', newConfigFileName+'.yml')
        initStopedConfig(newConfigFileName)

        # config_file = os.path.join('..', 'config', 'config.yml')
        config = read_config_file(config_file)
        pkl_dir_train = config['train_dir']
        pkl_dir_valid = config['valid_dir']
        model_dir = config['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        weights_dir = os.path.join(model_dir, 'weights')
        tb_log_dir = os.path.join(model_dir, 'logs')
        pre_trained_weight_h5 = config['pre_trained_weights']  # training from this weights.
        trained_weight_h5 = os.path.join(weights_dir, 'trained_weight.h5')  # save trained weights to this file.
        gpus = config['num_gpu']
        batch_size = config['batch_size']
        epochs = config['num_epochs']
        optimizer = config['optimizer']
        learning_rate = config['learning_rate']
        optimizer = getattr(importlib.import_module('tensorflow.python.keras.optimizers'), optimizer)(lr=learning_rate)
        preprocess_generator = config['preprocess_generator']
        loss = config['loss']
        use_multiprocessing = False
        # when use_multiprocessing is True, training would be slow. Why?
        # _os = OSPlatform()  # for fit_generator's keyword arguments `use_multiprocessing`
        # if _os.is_WINDOWS:
        #     use_multiprocessing = False
        # else:
        #     use_multiprocessing = True

        # load custom keras model
        # reference: https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3
        pattern = re.compile(r'\'([^\']+)\'')
        module_name, py_file = re.findall(pattern, config['custom_NN'])
        loader = importlib.machinery.SourceFileLoader(module_name, py_file)
        spec = importlib.util.spec_from_loader(module_name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        model = getattr(module, module_name)()
        # use custom keras model to define shape
        pkl_list_train = get_pkl_list(pkl_dir_train)
        pkl_list_valid = get_pkl_list(pkl_dir_valid)
        input_shape = model.input_shape[1:]
        output_shape = model.output_shape[1:]

        # data generator
        progressData['training']['value'] = 'training_generator!'
        training_generator = DataGenerator(pkl_list_train, input_shape, output_shape,
                                           batch_size=batch_size, shuffle=True, **preprocess_generator)
        progressData['training']['value'] = 'validation_generator'
        validation_generator = DataGenerator(pkl_list_valid, input_shape, output_shape,
                                             batch_size=batch_size, **preprocess_generator)

        # TODO: custom callbacks
        tensorboard = keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                                                  write_graph=True, write_images=False, update_freq='epoch')
        def setTrainProgress(epoch, logs):
            progressData['training']['value'] = f'Epoch {epoch}/{epochs} '

        batch_print_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: setTrainProgress(epoch, logs))
        callbacks = [tensorboard, batch_print_callback, EarlyStoppingAtMinLoss(config_file)]

        # training
        if gpus <= 1:
            # 1 gpu
            if not model._is_compiled:
                model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])
            if os.path.isfile(pre_trained_weight_h5):
                model.load_weights(pre_trained_weight_h5)
            print("3")
            original_weights = keras.backend.batch_get_value(model.weights)
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          epochs=epochs, use_multiprocessing=use_multiprocessing,
                                          callbacks=callbacks, workers=os.cpu_count())
            # check weights
            print(f"history {history}")
            weights = keras.backend.batch_get_value(model.weights)
            if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
                print('Weights in the template model have not changed')
            else:
                print('Weights in the template model have changed')
        else:
            # 2 gpu or more
            print("4")
            if os.path.isfile(pre_trained_weight_h5):
                model.load_weights(pre_trained_weight_h5)
            original_weights = keras.backend.batch_get_value(model.weights)
            parallel_model = multi_gpu_model(model, gpus=gpus, cpu_relocation=False, cpu_merge=True)
            if not model._is_compiled:
                parallel_model.compile(optimizer=optimizer, loss=loss, metrics=[r_squared])
            else:
                parallel_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics,
                                       loss_weights=model.load_weights, sample_weight_mode=model.sample_weight_mode,
                                       weighted_metrics=model._compile_weighted_metrics)
            history = parallel_model.fit_generator(generator=training_generator,
                                                   validation_data=validation_generator,
                                                   epochs=epochs, use_multiprocessing=use_multiprocessing,
                                                   callbacks=callbacks, workers=os.cpu_count())
            # check weights
            # references: https://github.com/keras-team/keras/issues/11313
            weights = keras.backend.batch_get_value(model.weights)
            parallel_weights = keras.backend.batch_get_value(parallel_model.weights)

            if all([np.all(w == ow) for w, ow in zip(weights, original_weights)]):
                print('Weights in the template model have not changed')
            else:
                print('Weights in the template model have changed')

            if all([np.all(w == pw) for w, pw in zip(weights, parallel_weights)]):
                print('Weights in the template and parallel model are equal')
            else:
                print('Weights in the template and parallel model are different')


        # save weights
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(trained_weight_h5)
        config = read_config_file(config_file)
        result = config['trainingStop']
        if 'true' == result:
            progressData['training']['value'] = 'User Stop !'
        else:
            progressData['training']['value'] = 'Finish!'
    except Exception as e:
        print(f'Exception {e}')
        progressData['training']['value'] = 'Error!'
        progressData['training']['message'] = f'{e}'
        progressData['log']['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d %X")}'
        progressData['log']['value'] = 'training'
        progressData['log']['message'] = f'{e}'

    return "training"


# draw
import numpy as np
from erinn.python.utils.io_utils import read_pkl, write_pkl
from erinn.python.FW2_5D.fw2_5d_ext import forward_simulation, make_dataset
def _forward_simulation(pkl_name, config, config_file):
    data = read_pkl(pkl_name)
    shape_V = data['synth_V'].shape
    sigma = 1 / np.power(10, data['pred_log_rho']).T
    data['pred_V'] = forward_simulation(sigma, config).reshape(shape_V)
    write_pkl(data, pkl_name)
    tempConfig = read_config_file(config_file)
    return tempConfig['predictStop']

@app.route('/predictResistivity', methods=['POST'])
def predictResistivity():
    print("predictResistivity start")
    jsonTest = request.json
    import os
    import re
    import importlib
    import tensorflow as tf
    import multiprocessing as mp
    from tqdm import tqdm
    from functools import partial
    from tensorflow.python.keras.optimizers import Adam
    from erinn.python.generator import PredictGenerator
    from erinn.python.metrics import r_squared
    from erinn.python.utils.io_utils import get_pkl_list, read_pkl, write_pkl, read_config_file
    from erinn.python.FW2_5D.fw2_5d_ext import get_forward_para, forward_simulation
    from erinn.python.utils.io_utils import get_pkl_list, read_pkl, write_pkl
    from erinn.python.utils.io_utils import read_config_file, read_urf
    from erinn.python.utils.vis_utils import plot_result_synth


    userStop=False
    try:
        # Allowing GPU memory growth
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)

        # setting
        pkl_dir_Name = jsonTest.get("pkl_dir_test")
        model_dir_Name = jsonTest.get("model_dir")
        weights_dir_Name = jsonTest.get("weights_dir")
        predictions_dir_Name = jsonTest.get("predictions_dir")
        figs_dir_path = jsonTest.get("figs_dir")
        pkl_dir_test = os.path.join('..', 'data', pkl_dir_Name, 'test')#下拉
        model_dir = os.path.join('..', 'models', model_dir_Name)#下拉
        weights_dir = os.path.join(model_dir, weights_dir_Name)#文字
        predictions_dir = os.path.join(model_dir, predictions_dir_Name, 'raw_data')#文字



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
        newConfigFileName = jsonTest.get("newConfigFileName")
        initStopedConfig(newConfigFileName.replace(".yml", ""))
        pattern = re.compile(r'\'([^\']+)\'')
        config_file = os.path.join('..', 'config', newConfigFileName)#config??需要選擇?
        config = read_config_file(config_file)
        progressData['predictResistivity']['name'] = newConfigFileName.replace(".yml", "")
        progressData['predictResistivity']['value'] = 'Start!'
        progressData['predictResistivity']['message'] = ''
        module_name, py_file = re.findall(pattern, config['custom_NN'])#需要選擇?
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
                config = read_config_file(config_file)
                result = config['predictStop']
                if 'true' == result:
                    progressData['predictResistivity']['value'] = 'User Stop !'
                    break
                data = read_pkl(pkl_list_test[i])
                data['synth_V'] = data.pop('inputs').reshape(input_shape[0:2])
                data['synth_log_rho'] = data.pop('targets').reshape(output_shape[0:2])
                data['pred_log_rho'] = pred.reshape(output_shape[0:2])
                progressData['predictResistivity']['value'] = f'predict_resistivity {i+1}/ {len(pkl_list_test)}'
                suffix = re.findall(r'\d+.pkl', pkl_list_test[i])[0]
                write_pkl(data, os.path.join(predictions_dir, f'result_{suffix}'))
                pbar.update()
        config = read_config_file(config_file)
        result = config['predictStop']

        # print(f'result : {result}')
        if 'true' == result:
            progressData['predictResistivity']['value'] = 'User Stop !'
            userStop = True

        #作圖1
        print(f"DRAW 1 start {config_file}")
        # config_file = os.path.join('..', 'config', 'config.yml')#需要選擇?

        pkl_list_result = get_pkl_list(predictions_dir)#跟pkl_list_test是否一樣?

        os.makedirs(predictions_dir, exist_ok=True)
        config = get_forward_para(config_file)
        par = partial(_forward_simulation, config=config, config_file=config_file)
        pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
        i=0
        results = pool.imap_unordered(par, pkl_list_result)
        for result in results:
            # print(f'result : {result} {i} ')
            progressData['predictResistivity']['value'] = f'predict_potential_over_current {i}/ {len(pkl_list_result)}'
            progressData['generateData']['message'] = ''
            i=i+1
            if 'true' == result:
                progressData['predictResistivity']['value'] = 'User Stop !'
                progressData['predictResistivity']['message'] = ''
                progressData['log']['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d %X")}'
                progressData['log']['value'] = 'predict'
                progressData['log']['message'] = f'User Stop'
                # shutil.rmtree(dir)
                # shutil.rmtree(config['dataset_dir'])
                userStop = True
                break
        pool.close()
        pool.join()

        #作圖2

        figs_dir = os.path.join(figs_dir_path)#加欄位
        num_figs = np.inf  # np.inf  # use np.inf to save all figures

        os.makedirs(figs_dir, exist_ok=True)
        iterator_pred = os.scandir(predictions_dir)
        geo_urf = config['geometry_urf']

        # electrode coordinates in the forward model
        _, _, _, coord, _ = read_urf(geo_urf)
        xz = coord[:, 1:3]
        xz[:, 0] += (config['nx'] - coord[:, 1].max()) / 2
        if not userStop:
            print(f"iterator_pred {iterator_pred}")
            userStop = plot_result_synth(iterator_pred, num_figs, xz, progressData, config_file, save_dir=figs_dir)

        if userStop:
            progressData['predictResistivity']['value'] = 'User Stop !'
            progressData['predictResistivity']['message'] = ''
        else:
            progressData['predictResistivity']['value'] = 'Finish!'
            progressData['predictResistivity']['message'] = ''

    except Exception as e:
        print(f'RequestException {e}')
        progressData['predictResistivity']['value'] = 'Error!'
        progressData['predictResistivity']['message'] = f'{e}'
        progressData['log']['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d %X")}'
        progressData['log']['value'] = 'predictResistivity'
        progressData['log']['message'] = f'{e}'

    return jsonify(request.values)

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    dataTest = request.values
    print(f'download {dataTest}')
    figs_dir_path = dataTest.get("figs_dir")
    uploads = os.path.join('..', 'reports', figs_dir_path, 'testing_figs_raw')
    return send_from_directory(directory=uploads, filename=filename)

@app.route('/uploadModel', methods=['GET', 'POST'])
def uploadModel():
    print(request.values)
    print(request.data)
    print(request.form)
    print(request.files)

    file = request.files.get("upload_file")
    print(f'upload_file {file}')

    file = request.files['file']
    print(f'file {file}')
    # print(f'file first {file.stream.read()}')
    # print(f'file first 2 {file.stream.read()}')
    if file and is_allowed_file(file):
        filename = secure_filename(file.filename)
        # print(f'is_allowed_file {file.stream.read()}')
        print(filename)
        # print(f'is_allowed_file secure_filename {file.stream.read()}')
        file.save(os.path.join('../config', filename))
        return "Success"
    return "error"

@app.route('/uploadData', methods=['GET', 'POST'])
def uploadData():
    print(request.values)
    print(request.data)
    print(request.form)
    print(request.files)

    file = request.files.get("upload_file")
    print(f'upload_file {file}')

    file = request.files['file']
    print(f'file {file}')
    if file and is_allowed_file(file):
        filename = secure_filename(file.filename)
        dir = os.path.join('../data', file.filename.rsplit('.', 1)[0].lower())
        os.makedirs(dir, exist_ok=True)
        file.save(os.path.join(dir, filename))
        # sigma, suffix_num = zip_item
        testUrf = URF(os.path.join(dir, filename))
        print(f' data {testUrf.I}')

        newConfigFilePath = os.path.join('..', 'config', '123.yml')
        config = read_config_file(newConfigFilePath)
        config = get_forward_para(config)
        dobs = forward_simulation(testUrf.I, config)
        # pickle dump/load is faster than numpy savez_compressed(or save)/load
        for dir_name, num_samples in (('train', 1),
                                  ('valid', 1),
                                  ('test', 1)):
            dirSub = os.path.join(dir, dir_name)
            print(f"dir : {dir} dirSub : {dirSub}")

            os.makedirs(dirSub, exist_ok=True)
            pkl_name = os.path.join(dir, dir_name, f'raw_data_1.pkl')
            write_pkl({'inputs': dobs, 'targets': np.log10(1 / 1)}, pkl_name)



        # pkl_name = os.path.join(dir, 'test', f'raw_data_1.pkl')
        # write_pkl({'inputs': dobs, 'targets': np.log10(1 / 1)}, pkl_name)

        return "Success"
    return "error"

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'txt', 'py', 'urf'}
# ALLOWED_MIME_TYPES = {'image/jpeg'}
def is_allowed_file(file):
    # print(f'is_allowed_file 1 {file.stream.read()}')
    if '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
    else:
        return False

    # mime_type = magic.from_buffer(file.stream.read(), mime=True)
    # print(f'mime_type : {mime_type}')
    # print(file.stream.read())
    # if (# mime_type in ALLOWED_MIME_TYPES and ext in ALLOWED_EXTENSIONS):

    if (ext in ALLOWED_EXTENSIONS):
        # print(f'is_allowed_file 2 {file.stream.read()}')
        return True

    return False

@app.route('/getProgress', methods=['GET', 'POST'])
def getProgress():
    lastLog = request.json
    print(lastLog)
    global progressData
    serverLog = progressData['log']
    print(f'lastLog {lastLog} serverLog {serverLog}')
    if lastLog == serverLog:
        print(f'=================')
        progressData['log']['name'] = ''
        progressData['log']['value'] = ''
        progressData['log']['message'] = ''
    # lastLog = jsonTest.get("log")
    return json.dumps(progressData)

@app.route('/stopProcess', methods=['GET', 'POST'])
def stopProcess():
    jsonData = request.json
    print(jsonData)
    fileName = jsonData['fileName']
    action = jsonData['action']
    print(f'fileName {fileName} action {action}')
    stopKey = f'{action}Stop'
    newConfigFilePath = os.path.join('..', 'config', fileName+'.yml')

    with open(newConfigFilePath) as f:
         list_doc = yaml.load(f)
    for data in list_doc:
        print(f'data : {data}')
        if  data == stopKey:
            print(f'get it data : {data}')
            list_doc[data] = 'true'

    with open(newConfigFilePath, "w") as f:
        yaml.dump(list_doc, f)
    f.close()

    # lastLog = jsonTest.get("log")
    return json.dumps(progressData)

@app.route('/getServerVersion', methods=['GET', 'POST'])
def getServerVersion():
    return '1.0.0'

# 讓以前沒有這參數的config可以有
def initStopedConfig(fileName):
    print(f'initStopedConfig fileName {fileName}')
    stopKey1 = f'generateDataStop'
    stopKey2 = f'trainingStop'
    stopKey3 = f'predictStop'
    newConfigFilePath = os.path.join('..', 'config', fileName+'.yml')
    hasGenerateDataStop = False
    hasTrainingStop = False
    hasPredictStop = False

    with open(newConfigFilePath) as f:
         list_doc = yaml.load(f, Loader=yaml.FullLoader)
    for data in list_doc:
        if  data == stopKey1:
            list_doc[data] = 'false'
            hasGenerateDataStop = True
        if  data == stopKey2:
            list_doc[data] = 'false'
            hasTrainingStop = True
        if  data == stopKey3:
            list_doc[data] = 'false'
            hasPredictStop = True

    if not hasGenerateDataStop:
        list_doc['generateDataStop']="false"
    if not hasTrainingStop:
        list_doc['trainingStop']="false"
    if not hasPredictStop:
        list_doc['predictStop']="false"

    with open(newConfigFilePath, "w") as f:
        yaml.dump(list_doc, f)
    f.close()

    print(f'initStopedConfig fileName {fileName} sccuess')
if __name__ == '__main__':
    app.run(host='0.0.0.0')
