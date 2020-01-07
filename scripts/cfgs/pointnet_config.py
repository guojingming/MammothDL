import os
import socket
from easydict import EasyDict as edict
import utils.shapenet_provider as sp


def get_pointnet_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data/modelnet40_ply_hdf5_2048/')
    log_dir = os.path.join(base_dir, 'train_outputs/pointnet/log/')
    model_dir = os.path.join(base_dir, 'train_outputs/pointnet/model/')
    train_files = sp.getDataFiles(os.path.join(data_dir, 'train_files.txt'))
    test_files = sp.getDataFiles(os.path.join(data_dir, 'test_files.txt'))
    label_names = sp.getLabelNameDictionary(os.path.join(data_dir, 'shape_names.txt'))
    for i in range(len(train_files)):
        train_files[i] = os.path.join(base_dir, train_files[i])

    cfg = edict()
    cfg.base_lr = 0.001
    cfg.final_lr = 0.00001
    cfg.batch_size = 32
    cfg.decay_step = 200000
    cfg.decay_rate = 0.7
    cfg.max_epoch = 250
    cfg.point_num = 1024
    cfg.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log/pointnet/')
    cfg.gpu_idx = 0
    cfg.model = 'pointnet_cls'
    cfg.hostname = socket.gethostname()
    cfg.base_dir = base_dir
    cfg.data_dir = data_dir
    cfg.model_dir = model_dir
    cfg.log_dir = log_dir
    cfg.train_files = train_files
    cfg.test_files = test_files
    cfg.label_names = label_names
    return cfg
