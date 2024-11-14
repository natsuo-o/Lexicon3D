import sklearn
import sklearn.cluster
import numpy as np
import torch
import os
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


# ミニバッチk-meansの実行
def run_minibatch_kmeans(data, n_clusters=8, chunk_size=10000):
    # ミニバッチk-meansの初期化
    n_samples = len(data)
    
    # 初期クラスタ中心を計算するための最初のチャンク
    #init_chunk = data[:min(chunk_size * 3, n_samples)]
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1024,
        max_iter=100,
        random_state=43,
        n_init='auto',
        init='k-means++'
    )
    
    # 初期クラスタ中心の計算
    batch_size = 30000
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        kmeans.partial_fit(batch)
    # kmeans.partial_fit(data)
    
    # 残りのデータを処理
    #for i in range(0, n_samples, chunk_size):
    #    chunk = data[i:min(i + chunk_size, n_samples)]
    #    kmeans.partial_fit(chunk)
    #    print(f"処理済み: {min(i + chunk_size, n_samples)}/{n_samples}")
    
    return kmeans


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multi-view feature fusion of DINOv2 on ScanNet.')
    parser.add_argument('--pc_path', type=str, help='/workspace/data/processed/replica_processed/lexicon3d/dinov2/dinov2_points/office0.npy')
    parser.add_argument('--feat_path', type=str, help='/workspace/data/processed/replica_processed/lexicon3d/dinov2/dinov2_features/office0.pt')
    parser.add_argument('--output_dir', type=str, help='/workspace/data/processed/replica_processed/replica_3d/train/')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()

    pc_dir = 'lexicon3d/dataset/lexicon3d/dinov2/dinov2_points'
    feat_dir = 'lexicon3d/dataset/lexicon3d/dinov2/dinov2_features'

    pc_pos = np.load(args.pc_path)
    pc_feat = torch.load(args.feat_path)

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    n_clusters = 10
        # ミニバッチk-meansの実行
    #kmeans =  run_minibatch_kmeans(
    #    pc_feat['feat'].numpy(),
    #    n_clusters=10,
    #    chunk_size=30000
    #)
    '''
    dbscan =sklearn.cluster.Birch(
    n_clusters=10,          # 近傍の範囲
    threshold=0.5,      # 枝分かれの閾値
    branching_factor=50 # 分岐係数
    )
    clusters = dbscan.fit_predict(pc_feat['feat'].numpy())
    print(clusters)

    # kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(pc_feat['feat'][:50000].numpy())
    # clusters = kmeans.labels_ # cluster index for each point
    unique_colors = plt.get_cmap('tab10')(np.linspace(0, 1, n_clusters))[:, :3]
    # unique_colors = plt.get_cmap('rainbow')(np.linspace(0, 1, num_clusters))[:, :3]
    cluster_colors = np.array([unique_colors[cluster] for cluster in clusters])
    pc_with_feature_kmeans = o3d.geometry.PointCloud()
    pc_with_feature_kmeans.points = o3d.utility.Vector3dVector(pc_pos)
    pc_with_feature_kmeans.colors = o3d.utility.Vector3dVector(cluster_colors)

    # plyファイルに保存
    # output_path = "output.ply"
    output_file_name = args.feat_path.split('/')[-1].split('.')[0] + '_kmeans10.ply'
    output_path = args.output_dir + output_file_name
    o3d.io.write_point_cloud(output_path, pc_with_feature_kmeans, write_ascii=True)
    c
    '''


    pca = sklearn.decomposition.PCA(n_components=3)
    print(pc_feat['feat'].numpy().shape)
    pca_feat = pca.fit_transform(pc_feat['feat'].numpy())
    pca_feat = (pca_feat - pca_feat.min(axis=0)) / (pca_feat.max(axis=0) - pca_feat.min(axis=0))
    print(pca_feat.shape)


    # ポイントクラウドの作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_pos)  # 座標情報の設定
    pcd.colors = o3d.utility.Vector3dVector(pca_feat)


    # plyファイルに保存
    # output_path = "output.ply"
    output_file_name = args.feat_path.split('/')[-1].split('.')[0] + '_pca.ply'
    output_path = args.output_dir + output_file_name
    print(output_path)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)
    print(f"Point cloud saved to {output_path}")
