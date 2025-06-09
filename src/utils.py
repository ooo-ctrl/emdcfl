import networkx as nx
import numpy as np
import os
import socket
import torch

from datetime import datetime
from itertools import permutations

from sklearn.cluster import AgglomerativeClustering, KMeans
from torch.utils.tensorboard import SummaryWriter

def get_summarywriter(out_dir, **params):

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        out_dir, params["method"] + "_" + current_time + "_" + socket.gethostname()
    )
    writer = SummaryWriter(log_dir=log_dir)

    if params is not None:
        num_centers = params["num_centers"]
        num_labels = params["label_len"]
        layout = {
            "Federation": {
                "loss/train": ["Multiline", [f"Loss/train/{i}" for i in range(num_centers)] + ["Loss/train/-1"]],
                "loss/val": ["Multiline", [f"Loss/val/{i}" for i in range(num_centers)] + ["Loss/val/-1"]],
                "acc/val": ["Multiline", [f"Acc/val/{i}" for i in range(num_centers)] + ["Acc/val/-1"]],
            },
        }

        for i in range(num_centers):
            layout["Federation"][f"probs/train/{i}"] = ["Multiline", [f"Probs/train/{i}/class{c}" for c in range(num_labels)]]

        writer.add_custom_scalars(layout)

    return writer

def cluster_clients(S):
    clustering = AgglomerativeClustering(metric="precomputed", linkage="complete").fit(-S)

    c1 = np.argwhere(clustering.labels_ == 0).flatten() 
    c2 = np.argwhere(clustering.labels_ == 1).flatten() 
    return c1, c2

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(centers):
    # Precompute flattened tensors and stack them into a matrix
    flattened = torch.stack([flatten(source.get_dW()) for source in centers])
    norms = torch.norm(flattened, dim=1, keepdim=True) + 1e-12  # Compute norms as a column vector
    
    # Compute pairwise dot products
    dot_products = torch.mm(flattened, flattened.T)  # Matrix multiplication for all pairwise dot products
    norm_products = norms @ norms.T                 # Outer product of norms for normalization
    
    # Compute the angles matrix
    angles = dot_products / norm_products

    return angles.numpy()

def compute_max_update_norm(centers):
    return np.max([torch.norm(flatten(center.get_dW())).item() for center in centers])


def compute_mean_update_norm(centers):
    return torch.norm(torch.mean(torch.stack([flatten(center.get_dW()) for center in centers]), 
                                    dim=0)).item()



# based on neighborhoods
def find_neighborhoods(matrix, symmetric=True, verbose=True):

    matrix = np.array(matrix)
    if symmetric:
        matrix = matrix * matrix.T # this makes sure that edges are unilaterally broken
        G = nx.Graph(matrix)
    else:
        G = nx.DiGraph(matrix)

    neighborhoods = [frozenset(list(G.neighbors(n))) for n in range(len(matrix))]

    if verbose:
        print(neighborhoods)
    return neighborhoods

def cluster_neighborhoods(neighborhoods, centers):
    seen_neighborhoods = {}
    for c_id, neighborhood in enumerate(neighborhoods):
        if neighborhood in set(seen_neighborhoods.keys()):
            cluster_id = seen_neighborhoods[neighborhood]
        else:
            cluster_id = len(seen_neighborhoods)
            seen_neighborhoods[neighborhood] = cluster_id

        centers[c_id].set_cluster(cluster_id)

    return seen_neighborhoods


def spectral_clustering_and_matching(gradient_profile_matrix, num_centers, num_clients, estimated_cluster_ids_old=None, clustering_algorithm='KMeans'):
    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    reduced_G = np.matmul(np.transpose(P[:, :num_centers]), gradient_profile_matrix)

    estimated_part_ids = None
    if clustering_algorithm == "KMeans":
        kmeans = KMeans(n_clusters=num_centers, init="k-means++", random_state=42).fit(np.transpose(reduced_G))
        cluster_centers = kmeans.cluster_centers_
        estimated_part_ids = kmeans.labels_

    if estimated_cluster_ids_old is not None:
        best_estimated_part_ids = estimated_part_ids
        best_n_consistency = 0
        for model_order in list(permutations([m_idx for m_idx in range(num_centers)])):
            temp_estimated_part_ids = np.array(model_order)[estimated_part_ids]
            n_consistency = np.sum((estimated_cluster_ids_old - temp_estimated_part_ids) == 0)
            if best_n_consistency < n_consistency:
                best_estimated_part_ids = temp_estimated_part_ids
                best_n_consistency = n_consistency
    else:
        best_estimated_part_ids = estimated_part_ids

    info = {
        "estimated_cluster_ids": best_estimated_part_ids,
        "reduced_gradient_profile_matrix": reduced_G,
        "singular_values": singular_values,
    }
    return info

def get_gt_clustering(data, num_centers, num_clusters, unassigned_center_ids=None):

    all_clusters = []
    if data != 'pacs500':
        cluster_len = num_centers // num_clusters
        for c_id in range(num_centers):
            start = cluster_len * (c_id // cluster_len)
            end = cluster_len * (1 + c_id // cluster_len)
            clusters = np.zeros(shape=(num_centers,), dtype=int)
            clusters[start:end] = 1

            all_clusters.append(clusters)
    else:
        cluster_lens = [4, 4, 3, 7]
        total_size = sum(cluster_lens)

        # Initialize the matrix with zeros
        matrix = np.zeros((total_size, total_size), dtype=int)

        # Fill the matrix with block-diagonal ones
        start_idx = 0
        for length in cluster_lens:
            end_idx = start_idx + length
            matrix[start_idx:end_idx, start_idx:end_idx] = 1
            start_idx = end_idx
        all_clusters = matrix.tolist()

    all_clusters = np.array(all_clusters)
    if unassigned_center_ids is not None:
        for c_id in unassigned_center_ids:
            all_clusters[c_id] = np.zeros(shape=(num_centers,), dtype=int)
            all_clusters[c_id][c_id] = 1

    all_clusters = all_clusters * all_clusters.T

    return all_clusters