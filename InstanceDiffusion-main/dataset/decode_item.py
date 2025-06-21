import cv2

def saliency_sampling(mask, k):
    """
    mask: 2D numpy array, 单通道二值mask
    k:   期望采样点个数

    返回按显著性加密采样的点列表 [ [y1, x1], [y2, x2], ... ]
    """
    mask = mask[:,:,0] if mask.ndim == 3 else mask
    # 1. 计算边缘显著性
    edges = cv2.Canny((mask*255).astype(np.uint8), 100, 200)
    edge_points = np.transpose(np.nonzero(edges))
    # 2. 计算曲率显著性（可选，可用边缘点近似）
    # 3. 采样策略：优先在边缘点采样，剩余点均匀/随机采样
    n_edge = int(k * 0.7)
    n_rest = k - n_edge

    result = []
    if len(edge_points) > 0:
        idx = np.random.choice(len(edge_points), min(n_edge, len(edge_points)), replace=False)
        result.extend([edge_points[i] for i in idx])

    # 在非边缘区域随机补齐
    nonzero_coords = np.transpose(np.nonzero(mask))
    if len(nonzero_coords) > 0 and n_rest > 0:
        idx = np.random.choice(len(nonzero_coords), min(n_rest, len(nonzero_coords)), replace=False)
        result.extend([nonzero_coords[i] for i in idx])

    # 如果还不够，补零
    while len(result) < k:
        result.append([0,0])
    # 返回格式 [y1, x1, ...]
    xy_points = []
    for p in result[:k]:
        xy_points.append(float(p[1]))
        xy_points.append(float(p[0]))
    return xy_points