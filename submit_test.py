"""

测试脚本
"""

import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from models.ResUNet import ResUNet
# from models.unet.unet import UNet
from utils.metrics import Metirc

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# 定义网络并加载参数
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
# net = torch.nn.DataParallel(UNet()).cuda()
# net.load_state_dict(torch.load(para.module_path),map_location=device)
state_dict = torch.load(para.liver_module_path)
new_state_dict = collections.OrderedDict()
for k,v in state_dict.items():
    name = 'module.' + k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()

for file_index, file in enumerate(os.listdir(para.test_liver_ct_path)):

    start = time()

    # file_name.append(file)

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(para.test_liver_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    # print(ct_array.shape) #(75, 512, 512)

    origin_shape = ct_array.shape

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max 归一化
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3) #(,256,256)

    # 对slice过少的数据使用padding填充，如果切片数量小于48，要对切片就行边界填充，为(48,256,256)
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale))) * para.lower  # (48,256,256)
        temp[0: depth] = ct_array
        ct_array = temp
        too_small = True

    #  将原始CT影像分割成长度为48的一系列的块，如0~47, 48~95, 96~143, .....
    start_slice = 0
    end_slice = start_slice + para.size - 1 # 47
    # 用来统计原始CT影像中的每一个像素点被预测了几次
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    # 用来存储每个像素点的预测值
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()  # [0,48]
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)  # shape变为: (1, 1, 48, 256, 256)

            outputs = net(ct_tensor)
            # outputs = outputs[3]
            # print(outputs.shape) #torch.Size([1, 1, 48, 512, 512])

            count[start_slice: end_slice + 1] += 1
            # print(count)
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())
            # print(probability_map.shape) #(75, 512, 512)
            # exit()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            del outputs

            # 滑动窗口取样预测,
            start_slice += para.stride  # 12,24,36
            end_slice = start_slice + para.size - 1  # 59,71,83

        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            # outputs = outputs[3]

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs

        pred_seg = np.zeros_like(probability_map)  # 创建同shape的全0矩阵
        pred_seg[probability_map >= (para.threshold * count)] = 1

        if too_small:
            temp = np.zeros((depth, 512, 512), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp


    liver_seg = pred_seg.astype(np.uint8)

    pred_seg = sitk.GetImageFromArray(liver_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_liver_path, file.replace('test-volume', 'test-segmentation')))

    speed = time() - start
    # time_pre_case.append(speed)

    print(file_index, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


