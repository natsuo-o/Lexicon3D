import os
import torch
import imageio
import argparse
from os.path import join, exists
import numpy as np
from glob import glob
from tqdm import tqdm, trange
# add the parent directory to the path
import sys
sys.path.append('..')

import torchvision.transforms as transforms
from PIL import Image
from fusion_util import PointCloudToImageMapper, save_fused_feature_with_locs, adjust_intrinsic, make_intrinsic


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of DINOv2 on ScanNet.')
    parser.add_argument('--data_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--output_dir', type=str, help='Where is the base logging directory')
    parser.add_argument('--split', type=str, default='val', help='split: "train"| "val"')
    parser.add_argument('--process_id_range', nargs='+', default=None, help='the id range to process')
    parser.add_argument('--prefix', type=str, default='dinov2', help='prefix for the output file')


    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()
    return args

def save_point_cloud_to_ply(points, colors, filename):
    # pointsとcolorsはnumpy配列を想定
    num_points = len(points)
    
    with open(filename, 'w') as f:
        # ヘッダーの書き込み
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 点群データの書き込み
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

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

    if exists(join(out_dir, args.prefix+'_points', scene_id + '.npy')):
        print(scene_id +'.pt' + ' already exists, skip!')
        return 1

    # short hand for processing 2D features
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
        posepath = img_dir.replace('color', 'pose').replace('.jpg', '.txt')
        pose = np.loadtxt(posepath)

        # load depth and convert to meter
        # (640,360)画像サイズと一致
        depth = imageio.v2.imread(img_dir.replace('color', 'depth').replace('jpg', 'png')) / depth_scale

        # calculate the 3d-2d mapping based on the depth
        mapping = np.ones([n_points, 4], dtype=int)
        # pose: camera_to_world
        # 画像内に対応するpoint cloudから変換されたx,yだけ座標が入っている(x,y,mask)
        # maskは０ or 1で画像内のピクセルに対応するpoint cloudだと1対応しないと0
        mapping[:, 1:4] = point2img_mapper.compute_mapping(pose, locs_in, depth)
        if mapping[:, 3].sum() == 0: # no points corresponds to this image, skip
            continue

        mapping = torch.from_numpy(mapping).to(device)
        mask = mapping[:, 3]
        # N, num_img
        vis_id[:, img_id] = mask

        image = Image.open(img_dir).convert('RGB')
        image = transform(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            # パッチレベルで正規化された特徴量にアクセスする, 'x_norm_clstoken'とx_norm_patchtokensから成り立っている
            feat_2d = evaluator.forward_features(image)["x_norm_patchtokens"] # 1バッチ数, 391(17x23)パッチ数, 1024

        feat_2d = feat_2d.squeeze(0).permute(1, 0).view(-1, 17, 23)
        # resize the feat_2d from 17x23 to 240x320
        feat_2d = torch.nn.functional.interpolate(feat_2d.unsqueeze(0), size=(240, 320), mode='bicubic', align_corners=False).squeeze(0) # 1024, 240, 320
        # mapping[:,1]はshapeがN それぞれのpoint cloudの画像座標面上でのx座標を指す
        # feat_2dはそれぞれのピクセルに特徴量が格納されている
        # feat_2d_3d:N,1024になる。　それぞれのpoint cloudにおける特徴量を計算する
        feat_2d_3d = feat_2d[:, mapping[:, 1], mapping[:, 2]].permute(1, 0).to(device)


        counter[mask!=0]+= 1
        sum_features[mask!=0] += feat_2d_3d[mask!=0]

    counter[counter==0] = 1e-5
    # N,1024
    feat_bank = sum_features/counter
    # point_ids には vis_id 内で少なくとも1回でも可視であった3Dポイントの一意なインデックスが含まれる
    point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

    save_fused_feature_with_locs(feat_bank, point_ids, locs_in, n_points, out_dir, scene_id, args)


def main(args):
    seed = 1457
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #!### Dataset specific parameters #####
    img_dim = (320, 240)
    img_dim_resized = (322, 238)
    depth_scale = 1000.0
    fx = 577.870605
    fy = 577.870605
    mx=319.5 # : 画像の横方向（x方向）の光学中心の座標
    my=239.5
    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]
    #######################################
    visibility_threshold = 0.25 # threshold for the visibility check

    args.depth_scale = depth_scale
    args.cut_num_pixel_boundary = 10 # do not use the features on the image boundary

    split = args.split
    data_dir = args.data_dir

    #data_root = join(data_dir, 'scannet_3d')
    #data_root_2d = join(data_dir,'scannet_2d')
    data_root = join(data_dir, 'replica_3d')
    data_root_2d = join(data_dir,'replica_2d')
    args.data_root_2d = data_root_2d
    out_dir = args.output_dir
    args.feat_dim = 1024 # 512 CLIP feature dimension, 768/1024 DINOv2 feature dimension
    os.makedirs(out_dir, exist_ok=True)
    process_id_range = args.process_id_range

    if split== 'train': # for training set, export a chunk of point cloud
        args.n_split_points = 300000
    else: # for the validation set, export the entire point cloud instead of chunks
        args.n_split_points = 2000000


    ###############################
    #### load the DINOv2 model ####

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda() # 238, 322 --> 17, 23
    args.evaluator = model


    args.transform = transforms.Compose(
                [
                    transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=transforms_mean, std=transforms_std),
                ]
            )

    # calculate image pixel-3D points correspondances
    intrinsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
    intrinsic = adjust_intrinsic(intrinsic, intrinsic_image_dim=[640, 480], image_dim=img_dim)


    args.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=intrinsic,
            visibility_threshold=visibility_threshold,
            cut_bound=args.cut_num_pixel_boundary)

    data_paths = sorted(glob(join(data_root, split, '*.pth')))
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

        process_one_scene(data_paths[i], out_dir, args)

if __name__ == "__main__":
    args = get_args()
    print("Arguments:")
    print(args)
    main(args)
