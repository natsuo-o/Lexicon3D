import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
import sklearn.cluster
import matplotlib.pyplot as plt
import json

# add the parent directory to the path
import sys
sys.path.append('..')

import torchvision.transforms as transforms
from PIL import Image
from fusion_util import PointCloudToImageMapper, save_fused_feature, adjust_intrinsic, make_intrinsic


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of DINOv2 on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--object_name', type=str, help='')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='dinov2', help='prefix for the output file')


    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args


def process_one_scene(data_path, out_dir, args):
    # short hand
    # office0.pth
    scene_id = data_path.split('/')[-1].split('_vh')[0]

    feat_dim = args.feat_dim
    point2img_mapper = args.point2img_mapper
    depth_scale = args.depth_scale
    evaluator = args.evaluator # modelを指す
    transform = args.transform

    # load 3D data (point cloud)
    # locs_in = torch.load(data_path)[0]
    locs_in = torch.load(data_path)[0]
    rgb_in = torch.load(data_path)[1]

    # save_point_cloud_to_ply(locs_in, rgb_in, '/workspace/data/Replica/office0.ply')
    n_points = locs_in.shape[0]
    '''
    if exists(join(out_dir, args.prefix+'_points', scene_id.split('.')[0] + '.npy')):
        print(scene_id.split('.')[0] +'.pt' + ' already exists, skip!')
        return 1
    '''
    # short hand for processing 2D features
    # /workspace/data/processed/replica_processed/lexicon3d/dinov2/dinov2_features
    scene = join(args.data_root_2d, scene_id) # /workspace/data/replica_processed/replica_2d/office0.pth
    img_dirs = sorted(glob(join(scene[:-4], 'color/*')), key=lambda x: int(os.path.basename(x)[:-4])) # /workspace/data/replica_processed/replica_2d/office0/color/0.jpg
    num_img = len(img_dirs)
    device = torch.device('cpu')

    # 3D特徴量の初期化
    n_points_cur = n_points
    counter = torch.zeros((n_points_cur, 1), device=device)
    # 　各3Dポイントごとの特徴ベクトルの合計を保持するテンソル
    sum_features = torch.zeros((n_points_cur, feat_dim), device=device)

    ################ Feature Fusion ###################
    # N,num_img, それぞれの画像におけるpoint cloudのmask情報を保存
    vis_id = torch.zeros((n_points_cur, num_img), dtype=int, device=device)
    for img_id, img_dir in enumerate(tqdm(img_dirs)):
        # load pose
        # poseはcamera_to_worldであり、外部パラメータの逆行列をとったものだよ
        # 外部パラメータはwolrd_to_cameraのため
        posepath = img_dir.replace('color', 'pose').replace('.png', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        # (640,360)画像サイズと一致
        depth = imageio.v2.imread(img_dir.replace('color', 'depth')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        # pose: camera_to_world
        # 画像内に対応するpoint cloudから変換されたx,yだけ座標が入っている(x,y,mask)
        # maskは０ or 1で画像内のピクセルに対応するpoint cloudだと1対応しないと0
        # pose = np.linalg.inv(pose)
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        # N, num_img
        vis_id[:, img_id] = mask
        image = Image.open(img_dir).convert('RGB')
        image_shape = np.array(image).shape
        # height, widthの順番になっている
        # argsのところとかの設定はwidth, heightの順番になっている
        image_array = np.array(image)[args.cropped_coordinate[0]:args.cropped_coordinate[1],args.cropped_coordinate[2]:args.cropped_coordinate[3]]
        plt.imshow(image_array)
        plt.savefig('cropped_cars.png')
        image = Image.fromarray(image_array)
        image = transform(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            # パッチレベルで正規化された特徴量にアクセスする, 'x_norm_clstoken'とx_norm_patchtokensから成り立っている
            feat_2d = evaluator.forward_features(image)["x_norm_patchtokens"] # 1バッチ数, 391(26*46)パッチ数, 1024

        feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, args.n_patch[1], args.n_patch[0])
        # resize the feat_2d from 17x23 to 240x320
        feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(args.img_dim[1],args.img_dim[0]), mode='bicubic', align_corners=False).squeeze(0) # [1024, 360, 640]
        feat_original_2d = torch.zeros(feat_2d.shape[0], image_shape[0], image_shape[1])
        feat_original_2d[:, args.cropped_coordinate[0]:args.cropped_coordinate[1],args.cropped_coordinate[2]:args.cropped_coordinate[3]] = feat_2d

        '''
        # pcaを実行
        feat_2d = feat_2d.reshape(1024, -1).T

        pca = sklearn.decomposition.PCA(n_components=3)
        pca_feat = pca.fit_transform(feat_2d.to('cpu').detach().numpy())
        pca_feat = (pca_feat - pca_feat.min(axis=0)) / (pca_feat.max(axis=0) - pca_feat.min(axis=0))

        pca_feat = pca_feat.reshape(args.img_dim[1], args.img_dim[0],3)
        print(pca_feat.shape)

        # pca_feat = pca_features_rgb.reshape(n_patch[1], n_patch[0],3)

        print(pca_feat.shape)
        # pca_feat = pca_feat.reshape(img_dim[1], img_dim[0],3)
        #plt.imshow(np.array(image_original))
        plt.imshow(pca_feat)
        plt.savefig('cars_pca.png')
        print(img_dir)
        plt.imshow(Image.open(img_dir).convert('RGB'))
        plt.savefig('cars_original.png')
        c
        '''
        # mapping[:,1]はshapeがN それぞれのpoint cloudの画像座標面上でのx座標を指す
        # feat_2dはそれぞれのピクセルに特徴量が格納されている
        # feat_2d_3d:N,1024になる。　それぞれのpoint cloudにおける特徴量を計算する
        #print(feat_2d.shape)
        feat_2d_3d = feat_original_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)
        #feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)

        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    # N,1024
    feat_bank = sum_features/counter
    # point_ids には vis_id 内で少なくとも1回でも可視であった3Dポイントの一意なインデックスが含まれる
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature(feat_bank, point_ids, locs_in, n_points, out_dir, scene_id, args)

def calc_resize_size(path):
    top, bottom, left, right = 100000,0,100000,0
    for img_file in os.listdir(path):
        img = np.array(Image.open(join(path,img_file)))
        indices = np.argwhere(img != 0)
        # 各方向の境界を取得
        if indices.size > 0:
            if indices[:, 0].min() <= top: # 最上の1の行インデックス
                top = indices[:, 0].min()
            if indices[:, 0].max() >= bottom:
                bottom = indices[:, 0].max()
            if indices[:, 1].min() <= left:
                left = indices[:, 1].min()
            if indices[:, 1].max() >= right:  # 最右の1の列インデックス
                right = indices[:, 1].max()
    return (right - left, bottom - top), [top, bottom, left, right]

def get_original_img_size(path):
    for img_file in os.listdir(path):
        img = np.array(Image.open(join(path,img_file)))
        return img.shape[1], img.shape[0] # width, heightの順にする。元々がそのような順番になっていたため

def load_intrinstics_param(path):
    json_open = open(path, 'r')
    json_load = json.load(json_open)
    return (json_load['K'][0],json_load['K'][4], json_load['K'][6], json_load['K'][7])



def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    visibility_threshold = 0.25 # threshold for the visibility check
    depth_scale = 1000
    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary

    split = args.split
    data_dir = args.data_dir

    #data_root = join(data_dir, 'scannet_3d')
    #data_root_2d = join(data_dir,'scannet_2d')
    data_root = join(data_dir, args.object_name, 'replica_3d')
    data_root_2d = join(data_dir,args.object_name, 'replica_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    args.feat_dim = 1024 # 512 CLIP feature dimension, 768/1024 DINOv2 feature dimension
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 300000
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000



    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]
    #######################################


    ###############################
    #### load the DINOv2 model ####

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda() # 238, 322 --> 17, 23
    args.evaluator = model


    data_paths = sorted(glob(join(data_root, split, '*.pth')))
    print(data_root)
    total_num = len(data_paths)

    id_range = None
    # process_id_rangeが具体的にどのデータを用いるのか
    # office0.pth, office1.pth, office2.pth...あるけど具体的にどれを使うのか
    if process_id_range is not None:
        id_range = [int(process_id_range[0].split(',')[0]), int(process_id_range[0].split(',')[1])]

    for i in trange(total_num):
        if id_range is not None and \
           (i<id_range[0] or i>id_range[1]):
            print('skip ', i, data_paths[i])
            continue

        data_path = data_paths[i] # /workspace/data/processed/wildrgbd_processed/cars/replica_3d/train/scene_088.pth


        #!### Dataset specific parameters #####
        #/workspace/data/processed/wildrgbd_processed/cars/replica_2d/scene_088/masks
        img_dim, cropped_coordinate = calc_resize_size(join(data_root_2d, data_path.split('/')[-1].split('.')[0], 'masks'))
        args.cropped_coordinate = cropped_coordinate
        n_patch = [round(img_dim[0] / 14), round(img_dim[1] / 14)]
        img_dim_resized = (n_patch[0] * 14,n_patch[1] * 14) # 644/14=46, 364/14=26のようにパッチ数14で割り切れるサイズに変更

        args.img_dim = img_dim
        args.n_patch = n_patch
        args.img_dim_resized = img_dim_resized
        data_root_2d

        fx, fy, mx, my = load_intrinstics_param(join(data_root_2d, data_path.split('/')[-1].split('.')[0], 'metadata'))

        args.transform = transforms.Compose(
                    [
                        transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=transforms_mean, std=transforms_std),
                    ]
                )

        # calculate image pixel-3D points correspondances
        intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        #intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[args.img_dim[0], args.img_dim[1]], image_dim=img_dim)
        original_img_dim = get_original_img_size(join(data_root_2d, data_path.split('/')[-1].split('.')[0], 'masks'))
        # intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[original_img_dim[1], original_img_dim[0]], image_dim=img_dim)

        args.point2img_mapper = PointCloudToImageMapper(
                image_dim=original_img_dim, intrinsics=intrinsic,
                visibility_threshold=visibility_threshold,
                cut_bound=args.cut_num_pixel_boundary)
        '''
        args.point2img_mapper = PointCloudToImageMapper(
                image_dim=img_dim, intrinsics=intrinsic,
                visibility_threshold=visibility_threshold,
                cut_bound=args.cut_num_pixel_boundary)
        '''

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
