import numpy as np

for batch_idx in range(50000):
    cam_maps = np.load(f"./cam_npy/{batch_idx}.npy", allow_pickle=True)
    weighted_cam_map = cam_maps[-1]

    # 根据加权热力图寻找mask
    contirbution = 0
    window = np.zeros((50,50))
    mask_x = 0
    mask_y = 0
    # 滑动窗口寻找贡献度最大的部分
    for i in range(weighted_cam_map.shape[0]-window.shape[0]):
        for j in range(weighted_cam_map.shape[1]-window.shape[1]):
            window = weighted_cam_map[i:i+window.shape[0],j:j+window.shape[1]]
            current_contribution = np.sum(window)
            if current_contribution > contirbution:
                contirbution = current_contribution
                mask_x = i
                mask_y = j
    # 根据坐标制作mask
    # 将patch放入和图片同大小的矩阵中
    # rot = np.random.choice(4)
    x = np.zeros((1, 3, 224, 224))
    for i in range(3):
        # patch[0][i] = np.rot90(patch[0][i], rot)
        x[0][i][mask_x: mask_x+window.shape[0], mask_y: mask_y+window.shape[1]] = np.ones_like(window)
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    np.save(f"./cam_npy/{batch_idx}_mask.npy", mask)
    print(batch_idx)