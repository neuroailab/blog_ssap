import glob

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import dorsalventral.data.tdw as tdw
import dorsalventral.data.utils as data_utils

## RAFT
import os, sys, copy
RAFT_DIR = os.path.expanduser("~/RAFT-TDW/")
sys.path.append(RAFT_DIR)
sys.path.append(os.path.join(RAFT_DIR, 'core'))

import train
import raft
from datasets import TdwFlowDataset

def preprocess(img):
    """preprocess

         入力画像の形式を統一する.

        Args:
            img (ndarray): 入力画像.
                行列の形式は(height, width, ch)

        Returns:
            ndarray: 入力画像.
                行列の形式は(height, width, ch)

    """
    # 白黒ならcolorに
    if len(img.shape) != 3:
        img = np.tile(img[:, :, None], (1, 1, 3))
    # αチャンネルは削除
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return img


class Mydatasets(torch.utils.data.Dataset):
    """Mydatasets

     学習に用いるデータを取得.

    """
    def __init__(self, t_color, img_path, img_t_path, img_t_ins_path,
                 img_size, aff_r, mean, std):
        """__init__

            Args:
                t_color (list[list[int]]): 各クラスに対応したSegmentationの
                    色が格納されている.
                img_path (str): 入力画像を格納しているdirectory.
                img_t_path (str): 正解Semantic Segmentationを
                    格納しているdirectory.
                img_t_ins_path (str): 正解Instance Segmentationを
                    格納しているdirectory.
                img_size (int): 切り取る画像の大きさ.
                    前処理(resize, crop)により画像が正方形になっていることを
                    前提とする.
                aff_r (int): Affinityを計算する Window size.
                mean (list): 学習データの各chの平均値.
                std (list): 学習データの各chの標準偏差.

        """
        self.img_size = img_size
        self.aff_r = aff_r
        # numpyをTensorにする処理
        self.transform = transforms.ToTensor()
        self.transform_img \
            = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=mean,
                                                       std=std)])
        self.data = sorted(glob.glob(img_path + "*"))
        self.data_t = sorted(glob.glob(img_t_path + "*"))
        self.data_t_aff = sorted(glob.glob(img_t_ins_path + "*"))
        self.datalen = len(self.data)
        self.labels = t_color

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        """__getitem__

            Args:
                idx (int): 学習に使用する画像のIndex.

            Returns:
                Tensor: img_sizeに切り取られた画像
                    行列の形式は(n_batch, ch, height, width)
                Tensor: out_dataに対応した、クラスごとの正解Semantic Segmentation画像
                    行列の形式は(n_batch, class数, height, width)
                Tensor: out_dataに対応した正解Affinity
                    行列の形式は(n_batch, 階層数, aff_r**2, height, width)

        """
        img = np.array(Image.open(self.data[idx]))
        img_t = np.array(Image.open(self.data_t[idx]))
        img_t_aff = np.array(Image.open(self.data_t_aff[idx]))

        img_t_cls = np.zeros((img.shape[0], img.shape[1], len(self.labels)))
        # Semantic Segmentationをclass毎に作成する
        for i in range(len(self.labels)):
            img_t_cls[:, :, i] \
                = np.where((img_t[:, :, 0] == self.labels[i][0])
                           & (img_t[:, :, 1] == self.labels[i][1])
                           & (img_t[:, :, 2] == self.labels[i][2]), 1, 0)

        img = preprocess(img)

        out_data = torch.zeros((3,
                                self.img_size, self.img_size))
        out_t = torch.zeros((len(self.labels),
                             self.img_size, self.img_size))
        out_t_aff = torch.zeros((self.aff_r, self.aff_r**2,
                                 self.img_size, self.img_size))

        img = self.transform_img(img)
        img_t = self.transform(img_t_cls)
        out_data = img
        out_t = img_t

        for mul in range(5):
            img_t_aff_mul = img_t_aff[0:self.img_size:2**mul,
                                      0:self.img_size:2**mul]
            img_size = self.img_size // (2**mul)

            # 上下左右2pixelずつ拡大
            img_t_aff_mul_2_pix = np.zeros((img_size
                                            + (self.aff_r//2)*2,
                                            img_size
                                            + (self.aff_r//2)*2, 3))
            img_t_aff_mul_2_pix[self.aff_r//2:
                                img_size+self.aff_r//2,
                                self.aff_r//2:
                                img_size+self.aff_r//2] \
                = img_t_aff_mul

            img_t_aff_compare = np.zeros((self.aff_r**2,
                                         img_size, img_size, 3))
            # 1pixelずつずらす
            for i in range(self.aff_r):
                for j in range(self.aff_r):
                    img_t_aff_compare[i*self.aff_r+j] \
                        = img_t_aff_mul_2_pix[i:i+img_size,
                                              j:j+img_size]

            # 同じ色ならAffinity=1(同じ物体)/同じ色でなければAffinity=0(別の物体)
            aff_data = np.where((img_t_aff_compare[:, :, :, 0]
                                 == img_t_aff_mul[:, :, 0])
                                & (img_t_aff_compare[:, :, :, 1]
                                   == img_t_aff_mul[:, :, 1])
                                & (img_t_aff_compare[:, :, :, 2]
                                   == img_t_aff_mul[:, :, 2]), 1, 0)
            aff_data = self.transform(aff_data.transpose(1, 2, 0))
            out_t_aff[mul, :, 0:img_size, 0:img_size] = aff_data

        return out_data, out_t, out_t_aff

class TdwAffinityDataset(TdwFlowDataset):

    def __init__(self, aff_r, num_levels=5,
                 raft_ckpt=os.path.join(RAFT_DIR, 'models', 'raft-sintel.pth'),
                 raft_args={'test_mode': True, 'iters': 24},
                 full_supervision=False,
                 flow_thresh=0.5,
                 *args, **kwargs):

        super(TdwAffinityDataset, self).__init__(
            get_gt_segments=True,
            *args, **kwargs)
        self.aff_r = aff_r
        self.K = self.aff_r**2
        self.num_levels = num_levels
        self.is_test = False

        self.raft = self._load_raft(raft_ckpt)
        self.raft_args = copy.deepcopy(raft_args)

        self.full_supervision = full_supervision
        self.flow_thresh = flow_thresh

    def _load_raft(self, ckpt):
        if ckpt is None:
            return None
        raft = train.load_model(
            load_path=ckpt,
            smalle=False,
            cuda=True,
            train=False)
        return raft

    def _get_foreground(self, flow):

        flow_mag = flow.float().square().sum(-3, True).sqrt()
        is_moving = (flow_mag > self.flow_thresh).float()
        return is_moving

    def __getitem__(self, idx):

        img1, img2, flow, gt_segments = super().__getitem__(idx)
        print("segments", gt_segments.dtype, gt_segments.shape)
        if self.raft is not None:
            _, flow = self.raft(
                img1[None].cuda(), img2[None].cuda(),
                **self.raft_args)
            flow = flow.squeeze(0) # remove batch dim
            print("raft flow", flow.shape)

        foreground = self._get_foreground(flow if not self.full_supervision else gt_segments)
        print("foreground", foreground.shape)
        print("images", img1.shape, img2.shape)



        # return image, segments, affinities
        return (img1, foreground, None)

if __name__ == '__main__':

    dataset = TdwAffinityDataset(
        aff_r=5,
        root='/data5/dbear/tdw_datasets/playroom_large_v3copy/',
        filepattern="*[0-8]",
        raft_ckpt=None
    )
    print(len(dataset))
    inp = dataset[0]

    args = train.get_args("")
    net = train.RAFT(args)
