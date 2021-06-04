from torchvision import transforms
import torch
import os
from PIL import Image
from tools import *


def validation(net, datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
               img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png']):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.449], std=[0.226])])
    net.eval()
    with torch.no_grad():
        ave_p, ave_j = [], []
        for p in range(len(datapath)):
            all_p, all_j = [], []
            all_class = os.listdir(os.path.join(datapath[p], img_dir_name))
            image_list, gt_list = list(), list()
            for s in range(len(all_class)):
                image_path = os.listdir(os.path.join(datapath[p], img_dir_name, all_class[s]))
                image_list.append(list(map(lambda x: os.path.join(datapath[p], img_dir_name, all_class[s], x), image_path)))
                gt_list.append(list(map(lambda x: os.path.join(datapath[p], gt_dir_name, all_class[s], x.replace(img_ext[p], gt_ext[p])), image_path)))
            for i in range(len(image_list)):
                cur_class_all_image = image_list[i]
                cur_class_all_gt = gt_list[i]

                cur_class_gt = torch.zeros(len(cur_class_all_gt), img_size, img_size)
                for g in range(len(cur_class_all_gt)):
                    gt_ = Image.open(cur_class_all_gt[g]).convert('L')
                    gt_ = gt_transform(gt_)
                    gt_[gt_ > 0.5] = 1
                    gt_[gt_ <= 0.5] = 0
                    cur_class_gt[g, :, :] = gt_

                cur_class_rgb = torch.zeros(len(cur_class_all_image), 3, img_size, img_size)
                for m in range(len(cur_class_all_image)):
                    rgb_ = Image.open(cur_class_all_image[m])
                    if rgb_.mode == 'RGB':
                        rgb_ = img_transform(rgb_)
                    else:
                        rgb_ = img_transform_gray(rgb_)
                    cur_class_rgb[m, :, :, :] = rgb_

                cur_class_mask = torch.zeros(len(cur_class_all_image), img_size, img_size)
                divided = len(cur_class_all_image) // group_size
                rested = len(cur_class_all_image) % group_size
                if divided != 0:
                    for k in range(divided):
                        group_rgb = cur_class_rgb[(k * group_size): ((k + 1) * group_size)]
                        group_rgb = group_rgb.to(device)
                        _, pred_mask = net(group_rgb)
                        cur_class_mask[(k * group_size): ((k + 1) * group_size)] = pred_mask
                if rested != 0:
                    group_rgb_tmp_l = cur_class_rgb[-rested:]
                    group_rgb_tmp_r = cur_class_rgb[:group_size-rested]
                    group_rgb = torch.cat((group_rgb_tmp_l, group_rgb_tmp_r), dim=0)
                    group_rgb = group_rgb.to(device)
                    _, pred_mask = net(group_rgb)
                    cur_class_mask[(divided * group_size): ] = pred_mask[:rested]

                for q in range(cur_class_mask.size(0)):
                    single_p, single_j = calc_precision_and_jaccard(cur_class_mask[q, :, :].numpy(), cur_class_gt[q, :, :].numpy())
                    all_p.append(single_p)
                    all_j.append(single_j)

            dataset_p = np.mean(all_p)
            dataset_j = np.mean(all_j)

            ave_p.append(dataset_p)
            ave_j.append(dataset_j)

    return ave_p, ave_j