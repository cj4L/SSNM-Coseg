import os
import torch
from pycocotools import coco
import queue
import threading
from model import build_model, weights_init
from tools import custom_print
from data_processed import train_data_producer
from train import train
import time
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # train_val_config
    annotation_file = '/home/chenjin/dataset/COCO/COCO2017/annotations/instances_train2017.json'
    coco_item = coco.COCO(annotation_file=annotation_file)

    train_datapath = '/home/chenjin/dataset/COCO/COCO2017/train2017/'

    val_datapath = ['./cosegdatasets/iCoseg8',
                    './cosegdatasets/MSRC7',
                    './cosegdatasets/Internet_Datasets300',
                    './cosegdatasets/PASCAL_VOC']

    vgg16_path = './weights/vgg16_bn_feat.pth'
    npy = './utils/new_cat2imgid_dict4000.npy'

    # project config
    project_name = 'SSNM-Coseg'
    device = torch.device('cuda:0')
    img_size = 224
    lr = 1e-5
    lr_de = 20000
    epochs = 100000
    batch_size = 4
    group_size = 5
    log_interval = 100
    val_interval = 1000

    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    models_train_last = os.path.join(models_root, project_name + '_last.pth')
    models_train_best = os.path.join(models_root, project_name + '_best.pth')

    net = build_model(device).to(device)
    net.train()
    net.apply(weights_init)
    net.base.load_state_dict(torch.load(vgg16_path))

    # continute load checkpoint
    # net.load_state_dict(torch.load('./models/SSNM-Coseg_last.pth', map_location='cuda:0'))

    q = queue.Queue(maxsize=40)

    p1 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p2 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p3 = threading.Thread(target=train_data_producer, args=(coco_item, train_datapath, npy, q, batch_size, group_size, img_size))
    p1.start()
    p2.start()
    p3.start()
    time.sleep(2)

    train(net, device, q, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)



