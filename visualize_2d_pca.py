import sklearn
import sklearn.cluster
import numpy as np
import torch
import os
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

import torchvision.transforms as transforms
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of DINOv2 on ScanNet.')
    parser.add_argument('--pc_path', type=str, help='/workspace/data/processed/replica_processed/lexicon3d/dinov2/dinov2_points/office0.npy')
    parser.add_argument('--feat_path', type=str, help='/workspace/data/processed/replica_processed/lexicon3d/dinov2/dinov2_features/office0.pt')
    parser.add_argument('--output_dir', type=str, help='/workspace/data/processed/replica_processed/replica_3d/train/')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()

    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]

    img_dim = (640,360) #(600,340)? (640,360), dog(600,490)
    # img_dim = (600,490)
    n_patch = [round(img_dim[0] / 14), round(img_dim[1] / 14)]
    img_dim_resized = (n_patch[0] * 14,n_patch[1] * 14) # 644/14=46, 364/14=26のようにパッチ数14で割り切れるサイズに変更
    transform = transforms.Compose(
                    [
                        transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=transforms_mean, std=transforms_std),
                    ]
                )


    img_dir = '/workspace/data/processed/replica_processed/replica_2d/room2/color/0.jpg'
    #img_dir = '/workspace/happy-puppy-welsh-corgi-14-600nw-2270841247.jpg'
    image_original = Image.open(img_dir).convert('RGB')
    image = transform(image_original).unsqueeze(0).to('cuda')

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda() # 238, 322 --> 17, 23
    evaluator = model

    with torch.no_grad():
        # パッチレベルで正規化された特徴量にアクセスする, 'x_norm_clstoken'とx_norm_patchtokensから成り立っている
        feat_2d = evaluator.forward_features(image)["x_norm_patchtokens"] # 1バッチ数, 391(26*46)パッチ数, 1024

    feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, n_patch[1], n_patch[0])
    # resize the feat_2d from 17x23 to 240x320
    feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(img_dim[1],img_dim[0]), mode='bicubic', align_corners=False).squeeze(0) # 1024, 240, 320
    print(feat_2d.shape)
    feat_2d = feat_2d.reshape(1024, -1).T

    pca = sklearn.decomposition.PCA(n_components=3)
    pca_feat = pca.fit_transform(feat_2d.to('cpu').detach().numpy())
    pca_feat_bg = pca_feat[:, 0] > 0.99 # from first histogram
    pca_feat_fg = ~pca_feat_bg
    pca_feat_left = pca.fit_transform(feat_2d.to('cpu').detach().numpy()[pca_feat_fg])
    # pca_feat = (pca_feat - pca_feat.min(axis=0)) / (pca_feat.max(axis=0) - pca_feat.min(axis=0))
    pca_feat_left = (pca_feat_left - pca_feat_left.min(axis=0)) / (pca_feat_left.max(axis=0) - pca_feat_left.min(axis=0))

    pca_features_rgb = pca_feat.copy()
    # for black background
    pca_features_rgb[pca_feat_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_feat_fg] = pca_feat_left

    pca_feat = pca_features_rgb.reshape(img_dim[1], img_dim[0],3)

    # pca_feat = pca_features_rgb.reshape(n_patch[1], n_patch[0],3)

    print(pca_feat.shape)
    # pca_feat = pca_feat.reshape(img_dim[1], img_dim[0],3)
    #plt.imshow(np.array(image_original))
    plt.imshow(pca_feat)
    plt.savefig('room_fg_16.png')



    print(feat_2d.shape)
    c

    pc_pos = np.load(args.pc_path)
    pc_feat = torch.load(args.feat_path)

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    n_clusters = 10