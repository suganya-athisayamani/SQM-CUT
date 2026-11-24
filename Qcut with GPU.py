import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from skimage.segmentation import slic, mark_boundaries
from skimage.color import rgb2lab
from skimage.util import img_as_float
from pyqubo import Array, Constraint
import neal  # Simulated Annealing Solver

# ==========================================
# 0. GPU DEVICE SETUP
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing pipeline on device: {device}")


# ==========================================
# 1. GPU DEEP FEATURE EXTRACTOR
# ==========================================
class DeepFeatureExtractorGPU(nn.Module):
    def __init__(self):
        super(DeepFeatureExtractorGPU, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Extract features up to Layer 2
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:6])
        self.eval()

    def forward(self, img_tensor):
        with torch.no_grad():
            # Pass image tensor directly on CUDA
            features = self.feature_extractor(img_tensor) # Shape: (1, C, H_feat, W_feat)
        return features


# ==========================================
# 2. SUPERPIXEL RAG BUILDER
# ==========================================
class SuperpixelRAGBuilderGPU:
    def __init__(self, n_segments=200, compactness=10.0):
        self.n_segments = n_segments
        self.compactness = compactness

    def process_frame(self, image_rgb, deep_features):
        img_float = img_as_float(image_rgb)
        lab_img = rgb2lab(image_rgb)
        
        # 1. SLIC Superpixel Segmentation (CPU / GPU-compatible)
        segments = slic(img_float, n_segments=self.n_segments, compactness=self.compactness, start_label=0)
        num_nodes = np.max(segments) + 1

        # 2. Resize Deep Feature Maps on GPU
        _, c, h_f, w_f = deep_features.shape
        deep_features_resized = torch.nn.functional.interpolate(
            deep_features, size=image_rgb.shape[:2], mode='bilinear', align_corners=False
        ).squeeze(0) # Shape: (C, H, W) on GPU

        # Convert feature map and lab image to GPU Tensors
        deep_features_gpu = deep_features_resized.permute(1, 2, 0) # Shape: (H, W, C)
        lab_gpu = torch.from_numpy(lab_img).float().to(device)

        # 3. Calculate Mean Node Descriptors using GPU acceleration
        node_features = []
        node_areas = []
        for i in range(num_nodes):
            mask_np = (segments == i)
            mask_gpu = torch.from_numpy(mask_np).to(device)
            area = np.sum(mask_np)
            
            mean_color = lab_gpu[mask_gpu].mean(dim=0).cpu().numpy()
            mean_coords = np.argwhere(mask_np).mean(axis=0) / np.array(image_rgb.shape[:2])
            mean_deep_feat = deep_features_gpu[mask_gpu].mean(dim=0).cpu().numpy()
            
            node_features.append({
                'color': mean_color,
                'coord': mean_coords,
                'deep': mean_deep_feat
            })
            node_areas.append(area)

        # 4. Construct Region-Adjacency Weight Matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))
        rows, cols = segments.shape
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                u, v = segments[r, c], segments[r+1, c]
                if u != v and adj_matrix[u, v] == 0:
                    dist_color = np.linalg.norm(node_features[u]['color'] - node_features[v]['color'])
                    dist_coord = np.linalg.norm(node_features[u]['coord'] - node_features[v]['coord'])
                    dist_deep = np.linalg.norm(node_features[u]['deep'] - node_features[v]['deep'])

                    w_ij = np.exp(-dist_color / 20.0) * np.exp(-dist_coord / 0.5) * np.exp(-dist_deep / 2.0)
                    adj_matrix[u, v] = w_ij
                    adj_matrix[v, u] = w_ij

        return segments, adj_matrix, node_areas


# ==========================================
# 3. QUBO FORMULATION & QUANTUM SOLVER
# ==========================================
class QuantumGraphCutSolver:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def solve_qubo(self, adj_matrix, node_areas, target_area_ratio=0.15):
        num_nodes = adj_matrix.shape[0]
        x = Array.create('x', shape=num_nodes, vartype='BINARY')
        
        # 1. Boundary Min-Cut Energy Term
        cut_objective = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_matrix[i, j] > 0:
                    cut_objective += adj_matrix[i, j] * (x[i] - x[j])**2

        # 2. Area Balance Constraint
        total_image_area = sum(node_areas)
        target_area = total_image_area * target_area_ratio
        selected_area = sum(node_areas[i] * x[i] for i in range(num_nodes))
        
        area_constraint = Constraint(
            self.gamma * (selected_area - target_area)**2, 
            label="area_penalty"
        )

        # 3. Compile Total QUBO
        H = cut_objective + area_constraint
        model = H.compile()
        qubo, offset = model.to_qubo()

        # 4. Solve via Annealer Sampler
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_reads=50)
        
        solution = response.first.sample
        binary_result = np.array([solution[f'x[{i}]'] for i in range(num_nodes)])
        
        return binary_result


# ==========================================
# 4. PIPELINE EXECUTION
# ==========================================
def run_q_sgp_gpu_pipeline(image_path, target_size=(352, 352), num_superpixels=200):
    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Move Image Tensor directly to GPU
    transform_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # Step 1: Deep Feature Extraction on GPU
    print("[1/4] Extracting Features on GPU...")
    feature_extractor = DeepFeatureExtractorGPU().to(device)
    deep_features = feature_extractor(transform_tensor)

    # Step 2: Superpixel RAG Construction
    print(f"[2/4] Constructing Superpixel RAG (N={num_superpixels})...")
    rag_builder = SuperpixelRAGBuilderGPU(n_segments=num_superpixels, compactness=10.0)
    segments, adj_matrix, node_areas = rag_builder.process_frame(img_rgb, deep_features)

    # Step 3: QUBO Solver
    print("[3/4] Solving QUBO Hamiltonian...")
    qpu_solver = QuantumGraphCutSolver(gamma=1e-5)
    node_binary_labels = qpu_solver.solve_qubo(adj_matrix, node_areas, target_area_ratio=0.15)

    # Step 4: Mask Reconstruction
    print("[4/4] Reconstructing Dense Segmentation Mask...")
    binary_mask = np.zeros_like(segments, dtype=np.uint8)
    for node_idx, label in enumerate(node_binary_labels):
        binary_mask[segments == node_idx] = label

    # Display Results
    superpixel_boundaries = mark_boundaries(img_rgb, segments, color=(1, 0, 0))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Frame")
    axes[0].axis("off")

    axes[1].imshow(superpixel_boundaries)
    axes[1].set_title(f"Superpixels (N={np.max(segments)+1})")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Q-SGP GPU Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    sample_image_path = "/content/polyp_sample.jpg"
    if not os.path.exists(sample_image_path):
        dummy_img = np.ones((352, 352, 3), dtype=np.uint8) * 120
        cv2.circle(dummy_img, (176, 176), 60, (50, 80, 200), -1)
        cv2.imwrite(sample_image_path, dummy_img)

    run_q_sgp_gpu_pipeline(sample_image_path, target_size=(352, 352), num_superpixels=200)