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
import neal  # Simulated Annealing Solver (Emulates Quantum Annealer)

# ==========================================
# 1. DEEP FEATURE EXTRACTOR BACKBONE
# ==========================================
class DeepFeatureExtractor(nn.Module):
    """
    Extracts deep semantic feature maps from input colonoscopy images
    using a pre-trained ResNet backbone.
    """
    def __init__(self):
        super(DeepFeatureExtractor, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # Extract features up to Layer 2 to retain high spatial resolution
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:6])
        self.eval()

    def forward(self, img_tensor):
        with torch.no_grad():
            features = self.feature_extractor(img_tensor) # Shape: (1, C, H_feat, W_feat)
        return features


# ==========================================
# 2. SUPERPIXEL RAG BUILDER
# ==========================================
class SuperpixelRAGBuilder:
    """
    Over-segments the input image into superpixels (SLIC) and constructs
    a Region-Adjacency Graph (RAG) incorporating color, position, and deep features.
    """
    def __init__(self, n_segments=200, compactness=10.0):
        self.n_segments = n_segments
        self.compactness = compactness

    def process_frame(self, image_rgb, deep_features):
        img_float = img_as_float(image_rgb)
        lab_img = rgb2lab(image_rgb)
        
        # 1. SLIC Superpixel Segmentation
        segments = slic(img_float, n_segments=self.n_segments, compactness=self.compactness, start_label=0)
        num_nodes = np.max(segments) + 1

        # 2. Extract Deep Feature Maps & Interpolate to Image Dimensions
        _, c, h_f, w_f = deep_features.shape
        deep_features_resized = torch.nn.functional.interpolate(
            deep_features, size=image_rgb.shape[:2], mode='bilinear', align_corners=False
        ).squeeze(0).cpu().numpy().transpose(1, 2, 0) # Shape: (H, W, C)

        # 3. Calculate Mean Node Descriptors
        node_features = []
        node_areas = []
        for i in range(num_nodes):
            mask = (segments == i)
            area = np.sum(mask)
            mean_color = lab_img[mask].mean(axis=0)
            mean_coords = np.argwhere(mask).mean(axis=0) / np.array(image_rgb.shape[:2]) # Normalized coordinates
            mean_deep_feat = deep_features_resized[mask].mean(axis=0)
            
            node_features.append({
                'color': mean_color,
                'coord': mean_coords,
                'deep': mean_deep_feat
            })
            node_areas.append(area)

        # 4. Construct Region-Adjacency Weight Matrix
        adj_matrix = np.zeros((num_nodes, num_nodes))
        rows, cols = segments.shape
        
        # Identify adjacent superpixel pairs
        for r in range(rows - 1):
            for c in range(cols - 1):
                u, v = segments[r, c], segments[r+1, c]
                if u != v and adj_matrix[u, v] == 0:
                    dist_color = np.linalg.norm(node_features[u]['color'] - node_features[v]['color'])
                    dist_coord = np.linalg.norm(node_features[u]['coord'] - node_features[v]['coord'])
                    dist_deep = np.linalg.norm(node_features[u]['deep'] - node_features[v]['deep'])

                    # Affinity Weight Formulation
                    w_ij = np.exp(-dist_color / 20.0) * np.exp(-dist_coord / 0.5) * np.exp(-dist_deep / 2.0)
                    adj_matrix[u, v] = w_ij
                    adj_matrix[v, u] = w_ij

        return segments, adj_matrix, node_areas


# ==========================================
# 3. QUBO FORMULATION & QUANTUM SOLVER
# ==========================================
class QuantumGraphCutSolver:
    """
    Translates the Superpixel RAG into a QUBO Matrix Hamiltonian 
    and solves the Min-Cut / Max-Cut optimization problem via Quantum Annealing.
    """
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

        # 2. Area Balance Regularization Penalty
        total_image_area = sum(node_areas)
        target_area = total_image_area * target_area_ratio
        selected_area = sum(node_areas[i] * x[i] for i in range(num_nodes))
        
        area_constraint = Constraint(
            self.gamma * (selected_area - target_area)**2, 
            label="area_penalty"
        )

        # 3. Compile Total QUBO Hamiltonian
        H = cut_objective + area_constraint
        model = H.compile()
        qubo, offset = model.to_qubo()

        # 4. Solve via Simulated Quantum Annealer (D-Wave Neal Sampler)
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_reads=50)
        
        solution = response.first.sample
        binary_result = np.array([solution[f'x[{i}]'] for i in range(num_nodes)])
        
        return binary_result


# ==========================================
# 4. FULL PIPELINE & VISUALIZATION
# ==========================================
def run_q_sgp_pipeline(image_path, target_size=(352, 352), num_superpixels=200):
    # Load and Preprocess Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr, target_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PyTorch Tensor for Backbone
    transform_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # Step 1: Deep Feature Extraction
    print("[1/4] Extracting Deep Semantic Features...")
    feature_extractor = DeepFeatureExtractor()
    deep_features = feature_extractor(transform_tensor)

    # Step 2: Superpixel Over-segmentation & RAG
    print(f"[2/4] Constructing Superpixel RAG (N={num_superpixels})...")
    rag_builder = SuperpixelRAGBuilder(n_segments=num_superpixels, compactness=10.0)
    segments, adj_matrix, node_areas = rag_builder.process_frame(img_rgb, deep_features)

    # Step 3: QUBO Formulation & Quantum Annealing Solve
    print("[3/4] Compiling QUBO Hamiltonian & Solving via Simulated Quantum Annealer...")
    qpu_solver = QuantumGraphCutSolver(gamma=1e-5)
    node_binary_labels = qpu_solver.solve_qubo(adj_matrix, node_areas, target_area_ratio=0.15)

    # Step 4: Reconstruct Dense Pixel Segmentation Mask
    print("[4/4] Reconstructing Dense Polyp Segmentation Mask...")
    binary_mask = np.zeros_like(segments, dtype=np.uint8)
    for node_idx, label in enumerate(node_binary_labels):
        binary_mask[segments == node_idx] = label

    # Visualization
    superpixel_boundaries = mark_boundaries(img_rgb, segments, color=(1, 0, 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Colonoscopy Frame")
    axes[0].axis("off")

    axes[1].imshow(superpixel_boundaries)
    axes[1].set_title(f"Superpixels (N={np.max(segments)+1})")
    axes[1].axis("off")

    axes[2].imshow(binary_mask, cmap="gray")
    axes[2].set_title("Q-SGP Predicted Segmentation Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # Test on an example image path
    sample_image_path = "/content/polyp_sample.jpg"
    
    # Create a dummy test image if file does not exist
    if not os.path.exists(sample_image_path):
        dummy_img = np.ones((352, 352, 3), dtype=np.uint8) * 120
        # Draw simulated polyp region
        cv2.circle(dummy_img, (176, 176), 60, (50, 80, 200), -1)
        cv2.imwrite(sample_image_path, dummy_img)

    run_q_sgp_pipeline(sample_image_path, target_size=(352, 352), num_superpixels=200)