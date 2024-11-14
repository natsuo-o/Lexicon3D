import os
import os.path as osp

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

import torch
import torchvision.transforms as transforms
from PIL import Image


def main(img_path="/workspace/data/processed/replica_processed/replica_2d/room2/color/0.jpg", 
        ):
    os.makedirs(output_folder, exist_ok=True)

    # assert img_size % 14 == 0, "The image size must be exactly divisible by 14"

    # Load DINO model
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_vits14.eval()


    transforms_mean = [0.48145466, 0.4578275, 0.40821073]
    transforms_std = [0.26862954, 0.26130258, 0.27577711]

    img_dim = (224,224) #(600,340)? (640,360)
    n_patch = [round(img_dim[0] / 14), round(img_dim[1] / 14)]
    img_dim_resized = (n_patch[0] * 14,n_patch[1] * 14) # 644/14=46, 364/14=26のようにパッチ数14で割り切れるサイズに変更
    transform = transforms.Compose(
                    [
                        transforms.Resize((img_dim_resized[1], img_dim_resized[0])),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=transforms_mean, std=transforms_std),
                    ]
                )
    
    patch_h = n_patch[0]
    patch_w = n_patch[1]

    # Load and process the single image
    image = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
    image_plot = ((image.cpu().numpy() * 0.5 + 0.5) * 255).transpose(0, 2, 3, 1).astype(np.uint8)[0]

    # Extract DINO features
    with torch.no_grad():
        embeddings = dinov2_vits14.forward_features(image)
        x_norm_patchtokens = embeddings["x_norm_patchtokens"].cpu().numpy()

    # Reshape for PCA
    x_norm_1616_patches = x_norm_patchtokens.reshape(patch_h * patch_w, -1)
    print(x_norm_1616_patches.shape)

    # Apply PCA
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(x_norm_1616_patches)
    pca_features = minmax_scale(pca_features)  # Scale features to [0,1]
    pca_features = pca_features.reshape(patch_h, patch_w, 3)

    # Plot and save result
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image_plot)

    plt.imshow(pca_features, extent=(0, img_size, img_size, 0), alpha=0.5)
    plt.savefig("raw_pca_result.jpg")
    plt.close()

if __name__ == "__main__":
    img_path = "/workspace/data/processed/replica_processed/replica_2d/room2/color/0.jpg"
    img_size = 448
    output_folder = "outputs"
    main(img_path)