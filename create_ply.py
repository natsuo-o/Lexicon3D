import open3d as o3d
import numpy as np
import argparse
import torch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='pthfile to plyfile')
    parser.add_argument('--input_path', type=str, help='/workspace/data/processed/replica_processed/replica_3d/train/office0.pth')
    parser.add_argument('--output_path', type=str, help='data/processed/replica_processed/replica_3d/train/office0.ply')
    args = parser.parse_args()

    xyz, rgb, _ = torch.load(args.input_path)

    # ポイントクラウドの作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)  # 座標情報の設定
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # 色情報の設定 (0-1 に正規化)

    # plyファイルに保存
    o3d.io.write_point_cloud(args.output_path, pcd, write_ascii=True)
    print(f"Point cloud saved to {args.output_path}")