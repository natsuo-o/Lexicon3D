# Wildrgbdデータ(物体単位のデータ)を用いて可視化

### データ準備
`/workspace/third_party/wildrgbd`ディクトリで実行する
1. データのダウンロード`python download.py --cat <category_name>`を実行
    - データサイズがかなりでかいため、ダウンロード時注意
2. `python3 cp_wildrgbd.py`を実行し、ディレクトリの作成とコピーをする
3. `python3 created_pose.py`を実行し、pose(カメラの外部パラメータ)ディレクトリを作成
4. `python3 mv_pose.py`を実行し、適切なファイル名に変更する
5. Depthデータからpoint cloudを作成する
    - `python3 wildrgbd_generate_point_cloud.py --path /workspace/data/processed/wildrgbd_processed/cars/replica_2d/scene_088 --save_dir /workspace/data/processed/wildrgbd_processed/cars/replica_3d/scene_088`
    - `cars/replica_3d/scene_088`部分を変更することで作成するデータの作成する
    -  `/workspace/data/processed/wildrgbd_processed/cars/replica_3d/train`以下に作成されたpoint cloudデータが存在する

### 推論
6. `cd /workspace/lexicon3d`に移動
7. 推論をする
    - `python fusion_wildrgbd_dinov2.py  --data_dir /workspace/data/processed/wildrgbd_processed  --output_dir  /workspace/data/processed/wildrgbd_processed/lexicon3d/dinov2 --object_name cars --split train --prefix dinov2`
    - `--object_name`でカテゴリーの指定
    - `--split`は`train`で固定
    - `/workspace/data/processed/wildrgbd_processed/lexicon3d/dinov2`にpoint cloudごとの特徴量とpoint cloudの座標が保存される

### PCAを用いた可視化
8. cd 'workspace'に移動する
9. 以下を実行する
    - `python3 visualize_pca.py --pc_path /workspace/data/processed/wildrgbd_processed/lexicon3d/dinov2/dinov2_points/scene_088.npy --feat_path /workspace/data/processed/wildrgbd_processed/lexicon3d/dinov2/dinov2_features/scene_088.pt --output_dir /workspace/data/processed/wildrgbd_processed/cars/replica_3d/train/`
    - `scene_088.npy`のように7.で作成したファイルをパスで指定する
    - `workspace/data/processed/wildrgbd_processed/cars/replica_3d/train/`に`scene_088_pca.ply`という名前で保存される





