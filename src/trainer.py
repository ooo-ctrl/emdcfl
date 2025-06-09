import copy
import csv
import cvxpy as cp
import math
import numpy as np
import os
import ot
import sys
import torch

from pprint import pprint
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from src.data import DatasetWithIndices, load_labels
from src.utils import cluster_clients, compute_max_update_norm, get_gt_clustering,compute_mean_update_norm, flatten, pairwise_angles, find_neighborhoods, cluster_neighborhoods, spectral_clustering_and_matching
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Iterable 

np.set_printoptions(precision=3)

is_interactive = sys.stdout.isatty()
PIN_MEMORY = False

class Trainer():
    def __init__(self, batch_size=None, num_workers=None, epochs=None, prefit_epochs=None, local_epochs=None, save_every=None, soft_penalty=False, tau=0, writer=None):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.prefit_epochs = prefit_epochs
        self.local_epochs = local_epochs
        self.writer = writer
        self.save_every = save_every
        self.soft_penalty = soft_penalty
        self.tau = tau

    def average_weights(self, weights, n_clients):
        """
        Arguments:
        weights -- list of weights.
        n_clients -- list of dataset sizes
        Returns:
        w_avg -- the average of the weights.
        """

        n_total = sum(n_clients)
        w_avg = copy.deepcopy(weights[0])

        for key in w_avg.keys():
            w_avg[key] = w_avg[key] * 0.
            for i, w in enumerate(weights):
                w_avg[key] += (n_clients[i] / n_total) * w[key]

        return w_avg

    def average_weights_clusterwise(self, seen_neighborhoods, server, centers, n_clients):
        servers = []
        for centers_of_cluster, cluster_id in seen_neighborhoods.items():
            centers_of_cluster = list(centers_of_cluster)
            local_weights_cluster = [centers[c_id].get_weights() for c_id in centers_of_cluster]
            local_n_cluster = [n_clients[c_id] for c_id in centers_of_cluster]

            weights = self.average_weights(local_weights_cluster, local_n_cluster)
            new_server = copy.deepcopy(server) # just to have a valid setup               
            new_server.cluster_id = cluster_id
            new_server.update_weights(weights)
            
            servers.append(new_server)
        return servers


    def save_models(self, method, centers, servers=None, epoch=None):

        membership = [method]
        if servers is not None:
            for c_id, center in enumerate(centers):
                membership.append(center.cluster_id)
            log_to = os.path.join(self.writer.log_dir, f"memberships.csv") 
            with open(log_to, 'w') as f:
                f.write(','.join(map(str, membership)))

            for cluster_id, server in enumerate(servers):
                save_to = os.path.join(self.writer.log_dir, f"model-cluster{cluster_id}.pt")
                if epoch is not None:
                    save_to = os.path.join(self.writer.log_dir, f"model-cluster{cluster_id}-epoch{epoch}.pt")

                torch.save(server.get_weights(), save_to)
        else:
            for c_id, center in enumerate(centers):
                membership.append(c_id)
                save_to = os.path.join(self.writer.log_dir, f"model-center{c_id}.pt")
                if epoch is not None:
                    save_to = os.path.join(self.writer.log_dir, f"model-center{c_id}-epoch{epoch}.pt")

                log_to = os.path.join(self.writer.log_dir, f"memberships.csv") 
                torch.save(center.get_weights(), save_to)
            with open(log_to, 'w') as f:
                f.write(','.join(map(str, membership)))


    def find_distance_threshold(self, iidness, num_clusters, epoch):

        num_centers = len(iidness)
        cluster_len = num_centers // num_clusters
        log_to = os.path.join(self.writer.log_dir, f"iidness_{epoch}.csv") 

        with open(log_to, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["c_id", "min_cluster_iidness", "max_cluster_iidness", "min_ood_iidness", "max_ood_iidness"])
            for c_id, iidness_of_center in enumerate(iidness):
                true_cluster_id = c_id // cluster_len 
                cluster_start = true_cluster_id * cluster_len
                cluster_end = (true_cluster_id + 1) * cluster_len # cluster does not include this anymore

                cluster_iidness = iidness_of_center[cluster_start:cluster_end]
                ood_iidness = iidness_of_center[:cluster_start] + iidness_of_center[cluster_end:]

                min_cluster_iidness = np.nanmax(cluster_iidness) # this is the farthest distance within cluster
                max_cluster_iidness = np.nanmin(cluster_iidness) # this is the closest distance within cluster
                max_ood_iidness = 0
                min_ood_iidness = 0
                if len(ood_iidness) > 0:
                    max_ood_iidness = np.nanmin(ood_iidness) # this is the closest distance outside of cluster
                    min_ood_iidness = np.nanmax(ood_iidness) # this is the farthest distance outside of cluster

                writer.writerow([c_id, min_cluster_iidness, max_cluster_iidness, min_ood_iidness, max_ood_iidness])


    def find_memberships(self, method, centers, epoch):
        membership = [method]
        log_to = os.path.join(self.writer.log_dir, f"memberships_{epoch}.csv") 
        for c_id, center in enumerate(centers):
            membership.append(center.cluster_id)
        with open(log_to, 'w') as f:
            f.write(','.join(map(str, membership)))


    def fit_fedavg(self, server, centers, center_trains, center_vals):

        center_step = [0 for i in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for center in centers:
            center.set_cluster(0)

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            for center in centers:
                center.update_weights(server.get_weights())
            
            local_weights = []
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                local_weights.append(center.get_weights())

            weights = self.average_weights(local_weights, n_clients)
            server.update_weights(weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedavg', [server for _ in range(len(centers))], epoch=g_epoch + 1) 

        self.save_models('fedavg', centers, servers=[server])


    def fit_fedprox(self, server, centers, center_trains, center_vals):

        center_step = [0 for i in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for center in centers:
            center.set_cluster(0)

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            for center in centers:
                center.update_weights(server.get_weights())
                center.save_weights(server.get_weights())

            local_weights = []
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_fedprox_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                local_weights.append(center.get_weights())

            weights = self.average_weights(local_weights, n_clients)
            server.update_weights(weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedprox', [server for _ in range(len(centers))], epoch=g_epoch + 1) 

        self.save_models('fedprox', centers, servers=[server])


    def fit_ifca(self, servers, centers, center_trains, center_vals):
        
        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for g_epoch in range(self.epochs):
            print(f"Global ep och {1+g_epoch} / {self.epochs}")

            # associate each center with a cluster
            for c_id, center in enumerate(centers):
                
                loss_select = []
                for server in servers:
                    center.update_weights(server.get_weights())
                    
                    loss, hits = self.test(center, center_vals[c_id], verbose=True)
                    loss_select.append(loss)

                if g_epoch == -1:
                    center.set_cluster(center.id // 2)
                else:
                    center.set_cluster(np.argmin(loss_select))
                print(f"Center {c_id} cluster:", center.cluster_id)

            # go into actual local training loop
            local_weights_cluster = [[] for _ in servers]
            local_n_cluster = [[] for _ in servers]

            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                local_weights_cluster[cluster_id].append(center.get_weights())
                local_n_cluster[cluster_id].append(n_clients[c_id])
            self.find_memberships('ifca', centers, g_epoch)

            # update cluster weights
            for i, server in enumerate(servers):
                if len(local_weights_cluster[i]) == 0:
                    print("Skipping since no update to cluster")
                    continue
                weights = self.average_weights(local_weights_cluster[i], local_n_cluster[i])
                server.update_weights(weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('ifca', centers, servers, epoch=g_epoch + 1) 

        self.save_models('ifca', centers, servers)
        

    def fit_cfl(self, server, centers, center_trains, center_vals):

        center_step = [0 for _ in range(len(centers))]
        EPS_1 = 0.4
        EPS_2 = 1.6

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # initialise clusters with a single cluster
        servers = [copy.deepcopy(server)]
        centers_of_cluster = [[]]
        for c_id, center in enumerate(centers):       
            center.set_cluster(0)
            centers_of_cluster[0].append(c_id)

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # go into local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())
                center.save_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            # compute similarities and update clustering
            new_centers_of_cluster = []
            for cluster_id in range(len(centers_of_cluster)):

                similarities = pairwise_angles([centers[c_id] for c_id in centers_of_cluster[cluster_id]])
                max_norm = compute_max_update_norm([centers[c_id] for c_id in centers_of_cluster[cluster_id]])
                mean_norm = compute_mean_update_norm([centers[c_id] for c_id in centers_of_cluster[cluster_id]])
            
                if mean_norm<EPS_1 and max_norm>EPS_2 and len(centers_of_cluster[cluster_id])>2 and g_epoch>0:
                              
                    c1, c2 = cluster_clients(similarities)
                    c1 = [centers_of_cluster[cluster_id][id] for id in c1]
                    c2 = [centers_of_cluster[cluster_id][id] for id in c2]

                    new_centers_of_cluster += [c1, c2]
                    print("Splitting cluster", cluster_id)
                else:
                    new_centers_of_cluster += [centers_of_cluster[cluster_id]]

            centers_of_cluster = new_centers_of_cluster
            print(centers_of_cluster)

            # update cluster weights
            servers = []
            for cluster_id in range(len(centers_of_cluster)):
                for c_id in centers_of_cluster[cluster_id]:
                    centers[c_id].set_cluster(cluster_id)
                local_weights_cluster = [centers[c_id].get_weights() for c_id in centers_of_cluster[cluster_id]]
                local_n_cluster = [n_clients[c_id] for c_id in centers_of_cluster[cluster_id]]

                weights = self.average_weights(local_weights_cluster, local_n_cluster)
                new_server = copy.deepcopy(server) # just to have a valid setup
                new_server.cluster_id = cluster_id
                new_server.update_weights(weights)
                servers.append(new_server)
            self.find_memberships('cfl', centers, g_epoch)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('cfl', centers, servers, epoch=g_epoch + 1) 

        self.save_models('cfl', centers, servers)


    def fit_emdcfl(self, server, centers, center_trains, center_vals, tolerance, num_clusters, embs_folder, proj_ratio):
        
        center_step = [0 for _ in range(len(centers))]
        clustering_epochs = set([0])

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # initialise clusters to match number of clients
        servers = [copy.deepcopy(server) for _ in range(len(centers))]
        centers_of_cluster = [[] for _ in range(len(centers))]
        for c_id, center in enumerate(centers):
            center.set_cluster(c_id)
            centers_of_cluster[c_id].append(c_id)

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # go into local training loop
            idness = [[] for _ in range(len(centers))]
            mean_distances = [[] for _ in range(len(centers))]
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                if g_epoch in clustering_epochs:
                    feature_extractor = copy.deepcopy(center.model)
                    feature_extractor.feature_extraction = True

                    own_wd = self.get_wd(feature_extractor, center_trains[c_id], center_vals[c_id], device=center.device, max_sample=512, metric='cosine')
                    
                    if embs_folder is not None:
                        indices = torch.randperm(len(center_trains[c_id]))[:512]
                        own_train = Subset(center_trains[c_id], indices)
                        embs = self.get_embeddings(model=feature_extractor, val=own_train, device=center.device)
                        np.save(f"{embs_folder}/embs_c{c_id}_ep{g_epoch}.npy", embs.numpy())
                        del embs

                    print(own_wd)
                    tau = own_wd

                    # compute scores for each center, threshold and combine
                    for other_c_id, other_val in enumerate(tqdm(center_vals, disable=not is_interactive)):
                        
                        other_wd = self.get_wd(feature_extractor, center_trains[c_id], other_val, device=center.device, max_sample=512, metric='cosine', proj_ratio=proj_ratio)
                        mean_distance = (other_wd - tau)
                        mean_distances[c_id].append(round(mean_distance, 4))

                        idness[c_id].append(round(mean_distance, 4))

                        # getting embeddings
                        if embs_folder is not None:
                            embs = self.get_embeddings(model=feature_extractor, val=other_val, device=center.device)
                            np.save(f"{embs_folder}/embs_c{c_id}_o{other_c_id}_ep{g_epoch}.npy", embs.numpy())
                            del embs
                    print(idness[c_id])
                    del feature_extractor
                center.model.to('cpu')
                torch.cuda.empty_cache()

            if g_epoch in clustering_epochs:
                print(idness)
                self.find_distance_threshold(idness, num_clusters, g_epoch)

                for c_id in range(len(idness)):
                    idness[c_id] = [val < tolerance for val in idness[c_id]]

                neighborhoods = find_neighborhoods(idness) # each node's neighborhood
                save_neighborhoods = copy.deepcopy(neighborhoods)
            neighborhoods = save_neighborhoods    
            seen_neighborhoods = cluster_neighborhoods(neighborhoods, centers)
            self.find_memberships(method='emdcfl', centers=centers, epoch=g_epoch)

            pprint(seen_neighborhoods)
            servers = self.average_weights_clusterwise(seen_neighborhoods, server, centers, n_clients)
            print("neighbourhoods:", len(seen_neighborhoods))
            print("servers:", len(servers))
            print("memberships:", [center.cluster_id for center in centers])

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('emdcfl', centers, servers, epoch=g_epoch + 1) 

        if self.epochs > 0:
            self.save_models('emdcfl', centers, servers)


    def fit_emdcfl_partial(self, server, centers, center_trains, center_vals, tolerance, num_clusters, p_num, proj_ratio):
        
        center_step = [0 for _ in range(len(centers))]
        clustering_epochs = set([i for i in range(self.epochs)])

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # initialise clusters to match number of clients
        servers = [copy.deepcopy(server) for _ in range(len(centers))]
        centers_of_cluster = [[] for _ in range(len(centers))]
        for c_id, center in enumerate(centers):
            center.set_cluster(c_id)
            centers_of_cluster[c_id].append(c_id)

        # need this to take into account unparticipating centers
        idness = [[True if i == j else False for i in range(len(centers))] for j in range(len(centers))]
        unassigned_center_ids = set(range(len(centers)))

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # select p_rate proportion of all centers
            center_ids = np.arange(len(centers))
            selected_center_ids = np.random.choice(center_ids, size=p_num, replace=False)
            selected_center_ids = np.sort(selected_center_ids)
            print("Participants:", selected_center_ids)
            unassigned_center_ids = unassigned_center_ids - set(selected_center_ids)            

            # go into local training loop
            mean_distances = [[np.nan for i in range(len(centers))] for j in range(len(centers))]

            for c_id, center in enumerate(centers):
                
                # skip center if not selected
                if c_id not in set(selected_center_ids):
                    continue

                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader
                center.model.to('cpu')
                torch.cuda.empty_cache()

            for c_id, center in enumerate(centers):

                if g_epoch in clustering_epochs and c_id not in unassigned_center_ids:
                    feature_extractor = copy.deepcopy(center.model)
                    feature_extractor.feature_extraction = True

                    own_wd = self.get_wd(feature_extractor, center_trains[c_id], center_vals[c_id], device=center.device, max_sample=512, metric='cosine')

                    print(own_wd)
                    tau = own_wd

                    # compute scores for each center, threshold and combine
                    for other_c_id, other_val in enumerate(tqdm(center_vals, disable=not is_interactive)):
                        
                        # only consider centers that are participating / have participated in the distance eval
                        if other_c_id in unassigned_center_ids:
                            continue

                        other_wd = self.get_wd(feature_extractor, center_trains[c_id], other_val, device=center.device, max_sample=512, metric='cosine', proj_ratio=proj_ratio)
                        mean_distance = (other_wd - tau)
                        
                        mean_distances[c_id][other_c_id] = round(mean_distance, 4)
                        idness[c_id][other_c_id] = mean_distance < tolerance
                        
                        # getting embeddings as evaluated by center 0
                        # if c_id == 0 and g_epoch == 1:
                        if False:
                            embs = self.get_embeddings(model=feature_extractor, val=maha_eval, device=center.device)
                            np.save(f"tsne/embs_c{c_id}_o{other_c_id}_ep{g_epoch}.npy", embs.numpy())
                    print(mean_distances[c_id])
                    del feature_extractor

            if g_epoch in clustering_epochs:
                print(idness)
                self.find_distance_threshold(mean_distances, num_clusters, g_epoch) # need to handle nans

                neighborhoods = find_neighborhoods(idness) # each node's neighborhood
                save_neighborhoods = copy.deepcopy(neighborhoods)

            neighborhoods = save_neighborhoods    
            seen_neighborhoods = cluster_neighborhoods(neighborhoods, centers)
            self.find_memberships(method='emdcflpp', centers=centers, epoch=g_epoch)

            pprint(seen_neighborhoods)
            servers = self.average_weights_clusterwise(seen_neighborhoods, server, centers, n_clients)
            print("neighbourhoods:", len(seen_neighborhoods))
            print("servers:", len(servers))
            print("memberships:", [center.cluster_id for center in centers])

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('emdcflpp', centers, servers, epoch=g_epoch + 1) 

        if self.epochs > 0:
            self.save_models('emdcflpp', centers, servers)


    def fit_cflgp(self, servers, centers, center_trains, center_vals):

        def flatten_tensor(source):
            if type(source) is tuple:
                ft = []
                for value in source:
                    ft.append(value.flatten())
                ft = torch.cat(ft)
            else:
                ft = torch.cat([value.flatten() for value in source.values()])
            return ft

        # ====================================
        # params that were in self before
        num_params_compressed_gradient = sum(param.numel() for param in servers[0].model.state_dict().values()) # no compression assumed
        num_centers = len(servers)
        num_clients = len(centers)
        clustering_period = 1
        # ====================================
        # more legacy params
        gradient_profile_matrix = np.zeros(shape=(num_centers * num_params_compressed_gradient, num_clients))
        criterion_model_index = 0
        # ====================================

        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        center_step = [0 for _ in range(num_clients)]
        centers_of_cluster = [[] for _ in range(num_centers)]
        for c_id, center in enumerate(centers):
            center.set_cluster(0)
            centers_of_cluster[0].append(c_id)
        
        estimated_cluster_ids = None
        
        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())
                center.save_weights(servers[cluster_id].get_weights())

                criterion_model_index = center.cluster_id

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):

                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                if g_epoch % clustering_period == 0:
                    
                    model_info = center.get_dW()
                    vectorized_model_info = flatten_tensor(model_info).clone().cpu().detach().numpy()

                    # Cumulative Averaging
                    beta = (1 / (np.floor((g_epoch + 1) / (num_centers * clustering_period)) + 1))
                    gradient_profile_matrix[(criterion_model_index) * num_params_compressed_gradient : (criterion_model_index + 1) * num_params_compressed_gradient, c_id] = gradient_profile_matrix[
                                    (criterion_model_index) * num_params_compressed_gradient : (criterion_model_index + 1) * num_params_compressed_gradient, c_id] * (1 - beta) \
                                    + (beta) * vectorized_model_info

            # ====================================
            # Clustering & Matching
            # ====================================

            info = spectral_clustering_and_matching(gradient_profile_matrix, num_centers=num_centers, num_clients=num_clients, estimated_cluster_ids_old=estimated_cluster_ids)
            estimated_cluster_ids_new = info["estimated_cluster_ids"]  # update cluster ids.
            reduced_gradient_profile_matrix = info["reduced_gradient_profile_matrix"]
            singular_values = info["singular_values"]

            estimated_cluster_ids = estimated_cluster_ids_new  # np.zeros(shape=self.n_clients, dtype=int)

            print("cluster ids: {}".format(estimated_cluster_ids))

            centers_of_cluster = [[] for _ in range(num_centers)]
            for c_id, center in enumerate(centers):
                cluster_id = estimated_cluster_ids[c_id]
                center.set_cluster(cluster_id)
                centers_of_cluster[cluster_id].append(c_id)
            self.find_memberships('cflgp', centers, g_epoch)

            # ====================================
            # Model update
            # ====================================
            # update cluster weights
            new_servers = []
            for cluster_id in range(len(centers_of_cluster)):
                local_weights_cluster = [centers[c_id].get_weights() for c_id in centers_of_cluster[cluster_id]]
                local_n_cluster = [n_clients[c_id] for c_id in centers_of_cluster[cluster_id]]

                weights = self.average_weights(local_weights_cluster, local_n_cluster)
                new_server = copy.deepcopy(servers[0]) # just to have a valid setup
                new_server.cluster_id = cluster_id
                new_server.update_weights(weights)
                new_servers.append(new_server)
            servers = new_servers

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('cflgp', centers, servers, epoch=g_epoch + 1) 

        self.save_models('cflgp', centers, servers)


    def fit_fedsoft(self, servers, centers, center_trains, center_vals):
        
        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            n_min_loss = [] # for every client, n given to each center: shape num_clients x num_clusters
            # associate each center with a cluster
            for c_id, center in enumerate(centers):

                val_loader = DataLoader(center_vals[c_id], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
                
                n_min_loss_cluster = [0] * len(servers)

                # for each batch evaluate every cluster model
                for data in tqdm(val_loader, disable=not is_interactive):
                    min_loss = 999999
                    best_cluster_id = len(servers) + 1
                    for cluster_id, server in enumerate(servers):
                        center.update_weights(server.get_weights())
                        loss, _ = center.local_test_step(data)
                        if loss < min_loss:
                            min_loss = loss
                            best_cluster_id = cluster_id

                    # assign best_cluster the samples as it was min loss
                    n_min_loss_cluster[best_cluster_id] += len(data[0])
                
                n_min_loss_cluster = np.array(n_min_loss_cluster)
                # counter smoother had a weight of at least 0.001 for every cluster, i.e. force assigning at least some samples to each cluster
                n_min_loss_cluster[n_min_loss_cluster == 0] += max(1, int(0.001 * len(val_loader.dataset)))

                n_min_loss.append(n_min_loss_cluster)
                personal_weights = self.average_weights([s.get_weights() for s in servers], n_min_loss_cluster)
                center.update_weights(personal_weights)

            local_n_cluster = np.array(n_min_loss).T # for every center, n coming from every client: shape num_clusters x num_clients
            print(np.round(local_n_cluster / local_n_cluster.sum(axis=1, keepdims=True), decimals=3)) # print for every center, percentage coming from every client

            # go into actual local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            # update cluster weights
            print(local_n_cluster)
            for i, server in enumerate(servers):
                weights = self.average_weights([c.get_weights() for c in centers], local_n_cluster[i])
                server.update_weights(weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedsoft', centers, epoch=g_epoch + 1) 

        self.save_models('fedsoft', centers)


    def fit_fedce(self, servers, centers, center_trains, center_vals):
        
        def get_loss_vector(center, val):
            
            loss_vector = []
            val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            for data in tqdm(val_loader, disable=not is_interactive):
                loss, hits = center.local_test_step(data)
                loss_vector.append(loss.detach().cpu().item())
            loss_vector = np.array(loss_vector)
            return loss_vector

        delta = 0.5

        center_step = [0 for _ in range(len(centers))]
        hist_assoc = np.zeros((len(centers), len(servers)), dtype=np.float32)

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # associate each center with a cluster
            for c_id, center in enumerate(centers):
                
                loss_select = []
                for server in servers:
                    center.update_weights(server.get_weights())
                    
                    loss, hits = self.test(center, center_vals[c_id], verbose=True)
                    loss_select.append(loss)
                    
                    # best cluster and update hist assoc for best cluster only
                    best_cluster_id = np.argmin(loss_select)
                    center.set_cluster(best_cluster_id)

                    hist_assoc[c_id][best_cluster_id] += 1

                # use best cluster to get loss vector
                # get all other loss vector
                # update hist assoc for other cluster models

                center.update_weights(servers[best_cluster_id].get_weights())
                best_cluster_loss_vec = get_loss_vector(center, center_vals[c_id])
                for i, server in enumerate(servers):
                    if i == best_cluster_id:
                        continue
                    # Suppose we have stored or can compute the batch-wise loss vector for cluster i:
    
                    center.update_weights(server.get_weights())
                    current_cluster_loss_vec = get_loss_vector(center, center_vals[c_id])

                    # d(s_k1, s_kj) = sqrt( sum( (l_k_batch_s_k1 - l_k_batch_s_kj)^2 ) )
                    # (Ref eq. (3) in the paper)
                    dist_val = np.sqrt(np.sum((best_cluster_loss_vec - current_cluster_loss_vec) ** 2))
                    # eq. (5): h(t)_s_kj += exp(-delta * sqrt(d(...)))
                    hist_assoc[c_id][i] += np.exp(-delta * dist_val)

            # # go into actual local training loop
            local_weights = []

            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                local_weights.append(center.get_weights())

            self.find_memberships('fedce', centers, g_epoch)

            for i, server in enumerate(servers):
                hist_assoc_server = hist_assoc.T[i]
                alpha_scores_server = np.exp(hist_assoc_server) / np.sum(np.exp(hist_assoc_server))
                weights = self.average_weights(local_weights, alpha_scores_server)                
                server.update_weights(weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedce', centers, servers, epoch=g_epoch + 1) 

        self.save_models('fedce', centers, servers)


    def fit_fedem(self, servers, centers, center_trains, center_vals):
        
        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes and make datasets return indices
        n_clients = []
        for i, center_train in enumerate(center_trains):
            n_clients.append(len(center_train))
            center_trains[i] = DatasetWithIndices(center_trains[i])

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")


            local_weights_cluster = [[] for _ in range(len(servers))]
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")

                # 1. E-step: estimate probability that a given sample belongs to a server
                # for each server, get the instancewise loss for this center
                all_losses = []
                for server in servers:
                    center.update_weights(server.get_weights())
                    
                    instancewise_loss = self.get_instancewise_loss(center, center_trains[c_id]) # Marfoq paper also actually uses train
                    all_losses.append(instancewise_loss)
                all_losses = torch.stack(all_losses)
                center.update_q(all_losses)

                # 2. M-step pi: update cluster probabilities
                center.update_pi() 

                # 3. M-step theta: local training
                for cluster_id, server in enumerate(servers):
                    center.update_weights(server.get_weights())

                    train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                    for l_epoch in range(self.local_epochs):
                        
                        for data in tqdm(train_loader, disable=not is_interactive):
                            batch_c_loss = center.local_weighted_step(data, cluster_id)

                            self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                            center_step[c_id] += 1

                        _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                    del train_loader

                    local_weights_cluster[cluster_id].append(center.get_weights())

            # update cluster models
            for cluster_id, server in enumerate(servers):
                weights = self.average_weights(local_weights_cluster[cluster_id], n_clients)
                server.update_weights(weights)

            print([c.get_cluster_weights() for c in centers])

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedem', centers, epoch=g_epoch + 1) 

        self.save_models('fedem', centers)


    def fit_pfedgraph(self, centers, center_trains, center_vals):

        lmbda = 0.8 # taken from paper
        def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):

            n = model_difference_matrix.shape[0]
            p = np.array(fed_avg_freqs)
            P = lamba * np.identity(n)
            P = cp.atoms.affine.wraps.psd_wrap(P)
            G = - np.identity(n)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.ones(1)
            for i in range(model_difference_matrix.shape[0]):
                model_difference_vector = model_difference_matrix[i]
                d = model_difference_vector
                q = d - 2 * lamba * p
                x = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                        [G @ x <= h,
                        A @ x == b]
                        )
                prob.solve()

                graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
            return graph_matrix

        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # init collaboration graph with diagonal zero - i dont think this actually matters
        G = torch.ones(len(centers), len(centers))
        # G = G - torch.eye(len(centers))
        # G /= len(centers) - 1

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # go into local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                center.get_optimizer()
                center.save_weights(center.get_weights()) # save_weights to form dW after training and to do fedprox

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_pfedgraph_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            # compute similarities
            cos_sims = -pairwise_angles(centers)
            G = optimizing_graph_matrix_neighbor(graph_matrix=G, 
                                                 index_clientid=[i for i in range(len(centers))], 
                                                 model_difference_matrix=cos_sims, 
                                                 lamba=lmbda, 
                                                 fed_avg_freqs=[n / sum(n_clients) for n in n_clients])

            print(cos_sims)
            print(G)

            # update center weights
            for c_id, center in enumerate(centers):
                aggregation_weights = G[c_id] # for other models only

                all_weights = [c.get_weights() for c in centers]

                new_weights = self.average_weights(all_weights, aggregation_weights)
                center.update_weights(new_weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('pfedgraph', centers, epoch=g_epoch + 1) 

        self.save_models('pfedgraph', centers)


    def fit_flexcfl(self, centers, center_trains, center_vals, num_clusters):
        
        def intergroup_learning(source_w, target_ws, intergroup_lr):

            w_avg = copy.deepcopy(source_w)
            target_norms = [torch.norm(flatten(target_w)) for target_w in target_ws]

            for key in w_avg.keys():
                for i, w in enumerate(target_ws):
                    w_avg[key] = w_avg[key].float() + intergroup_lr * w[key].float() / target_norms[i]

            return w_avg

        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # make sure centers have common init; saving weights for dW later
        for center in centers:
            center.save_weights(center.get_weights())

        # cold start apply hard clustering
        print("cold start")
        servers = [] # this will hold the clusters

        for c_id, center in enumerate(centers):
            print(f"---Center {1+c_id} / {len(centers)}")

            center.get_optimizer()
            train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
            for l_epoch in range(self.local_epochs):
                
                for data in tqdm(train_loader, disable=not is_interactive):
                    batch_c_loss = center.local_step(data)

                    self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                    center_step[c_id] += 1

                _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
            del train_loader

        # get all of the model updates
        dWs = []
        for center in centers:
            dWs.append(flatten(center.get_dW()))
        dWs = torch.stack(dWs) # num_clients x model_params

        # apply TruncatedSVD
        svd = TruncatedSVD(n_components=num_clusters)
        decomp_updates = svd.fit_transform(dWs.numpy().T) # should be model_params x num_clusters

        # get pairwise similarities between dWs and each of the num_cluster components to cluster via KMeans
        decomposed_cossim_matrix = cosine_similarity(dWs, decomp_updates.T) # should be num_clients x num_clusters
        result = KMeans(num_clusters, max_iter=20).fit(decomposed_cossim_matrix)        

        local_weights_cluster = [[] for _ in range(num_clusters)]
        local_n_cluster = [[] for _ in range(num_clusters)]

        clients_of_cluster = [[] for _ in range(num_clusters)]
        for c_id, cluster_id in enumerate(result.labels_):
            clients_of_cluster[cluster_id].append(c_id) # for printing only
            centers[c_id].set_cluster(cluster_id)
            local_weights_cluster[cluster_id].append(centers[c_id].get_weights())
            local_n_cluster[cluster_id].append(n_clients[c_id])
        print(clients_of_cluster)

        for cluster_id, weights in enumerate(local_weights_cluster):
            new_server = copy.deepcopy(centers[0]) # just to get a valid setup
            new_server.set_cluster(cluster_id)
            new_server_weights = self.average_weights(weights, [1 for _ in range(len(weights))]) # equal weighting 
            new_server.update_weights(new_server_weights)
            servers.append(new_server)
    
        # enter actual training
        for g_epoch in range(1,self.epochs):

            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            local_weights_cluster = [[] for _ in range(num_clusters)]
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                local_weights_cluster[cluster_id].append(center.get_weights())

            # intragroup aggregation
            for cluster_id, weights in enumerate(local_weights_cluster):
                new_server_weights = self.average_weights(weights, local_n_cluster[cluster_id]) # fedavg 
                servers[cluster_id].update_weights(new_server_weights)

            # intergroup aggregation
            all_servers_weights = [server.get_weights() for server in servers]
            for cluster_id, server in enumerate(servers):
                other_servers_weights = [w for i, w in enumerate(all_servers_weights) if i != cluster_id]
                new_server_weights = intergroup_learning(server.get_weights(), other_servers_weights, intergroup_lr=5.)
                server.update_weights(new_server_weights)

            self.find_memberships('flexcfl', centers, g_epoch)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('flexcfl', centers, servers, epoch=g_epoch + 1) 

        self.save_models('flexcfl', centers, servers)


    def fit_fedsac(self, centers, center_trains, center_vals):

        # assumptions
        k_principal = 3

        lambda_1 = 0.9 # taken from paper for complementarity (0.5 if low compl known)
        lambda_2 = 1.4 # taken from paper for similarity (1.6 if low compl known)
        def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix, model_difference_matrix, lambda_1, lambda_2, fed_avg_freqs):
            n = model_difference_matrix.shape[0]
            p = np.array(fed_avg_freqs)
            P = np.identity(n)
            P = cp.atoms.affine.wraps.psd_wrap(P)
            G = - np.identity(n)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.ones(1)
            for i in range(model_difference_matrix.shape[0]):
                model_complementary_vector = model_complementary_matrix[i]
                model_difference_vector = model_difference_matrix[i]
                s = model_difference_vector
                c = model_complementary_vector
                q = lambda_1*c + lambda_2*s - 2 * p
                x = cp.Variable(n)
                prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                        [G @ x <= h,
                        A @ x == b]
                        )
                prob.solve()

                graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
            return graph_matrix


        def compute_principal_angles(A, B):
            assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"
            
            k = A.shape[0]
            norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
            norm_B = np.linalg.norm(B, axis=1)
            dot_product = np.dot(A, B.T)
            cosine_matrix = dot_product / (norm_A * norm_B)
            cos_phi_values = []

            for _ in range(k):
                i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)
                cos_phi_values.append(cosine_matrix[i, j])
                cosine_matrix[i, :] = -np.inf
                cosine_matrix[:, j] = -np.inf
            phi = np.arccos(np.clip(cos_phi_values, -1, 1))

            return phi

        def cal_complementary(num_clients, principal_list):
            model_complementary_matrix = np.zeros((num_clients, num_clients))
            k = principal_list[0].shape[0]
            for i in range(num_clients):
                for j in range(i, num_clients):
                    phi = compute_principal_angles(principal_list[i], principal_list[j])
                    principal_angle = np.cos((1 / k) * np.sum(phi))
                    model_complementary_matrix[i][j] = principal_angle
                    model_complementary_matrix[j][i] = principal_angle
            return model_complementary_matrix

        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # init collaboration graph with diagonal zero - i dont think this actually matters
        G = torch.ones(len(centers), len(centers))

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            principal_list = []
            # go into local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                center.get_optimizer()
                center.save_weights(center.get_weights()) # save_weights to form dW after training and to do fedprox

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_pfedgraph_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

                # extract features and do PCA
                feature_extractor = copy.deepcopy(center.model)
                feature_extractor.feature_extraction = True
                embeddings = self.get_embeddings(feature_extractor, center_trains[c_id], device=center.device)
                pca = PCA(n_components=k_principal)
                pca.fit_transform(embeddings.numpy())
                orthogonal_basis = pca.components_ # these should be n_components x embed_dim
                principal_list.append(orthogonal_basis)
            
            # compute similarities
            cos_sims = -pairwise_angles(centers)
            model_complementary_matrix = cal_complementary(num_clients=len(centers), principal_list=principal_list)
            G = optimizing_graph_matrix_neighbor(graph_matrix=G, 
                                                 index_clientid=[i for i in range(len(centers))], 
                                                 model_difference_matrix=cos_sims,
                                                 model_complementary_matrix=model_complementary_matrix,
                                                 lambda_1=lambda_1,
                                                 lambda_2=lambda_2, 
                                                 fed_avg_freqs=[n / sum(n_clients) for n in n_clients])

            print(cos_sims)
            print(model_complementary_matrix)
            print(torch.round(G,decimals=3))

            # update center weights
            for c_id, center in enumerate(centers):
                aggregation_weights = G[c_id] # for other models only

                all_weights = [c.get_weights() for c in centers]

                new_weights = self.average_weights(all_weights, aggregation_weights)
                center.update_weights(new_weights)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedsac', centers, epoch=g_epoch + 1) 

        self.save_models('fedsac', centers)


    def fit_fedrc(self, servers, centers, center_trains, center_vals):
        
        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes and make datasets return indices
        n_clients = []
        for i, center_train in enumerate(center_trains):
            n_clients.append(len(center_train))
            center_trains[i] = DatasetWithIndices(center_trains[i])

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")


            local_weights_cluster = [[] for _ in range(len(servers))]
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")

                # 1. E-step: estimate probability that a given sample belongs to a server
                # for each server, get the instancewise loss for this center
                all_losses = []
                for server in servers:
                    center.update_weights(server.get_weights())
                    
                    instancewise_loss = self.get_instancewise_loss(center, center_trains[c_id]) # Marfoq paper also actually uses train
                    all_losses.append(instancewise_loss)
                all_losses = torch.stack(all_losses)
                center.update_q(all_losses)

                # 2. M-step pi: update cluster probabilities
                center.update_pi() 
                center.update_cluster_labels_weights(center_trains[c_id])

                # 3. M-step theta: local training
                for cluster_id, server in enumerate(servers):
                    center.update_weights(server.get_weights())

                    train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                    for l_epoch in range(self.local_epochs):
                        
                        for data in tqdm(train_loader, disable=not is_interactive):
                            batch_c_loss = center.local_weighted_step(data, cluster_id)

                            self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                            center_step[c_id] += 1

                        _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                    del train_loader

                    local_weights_cluster[cluster_id].append(center.get_weights())

            # update cluster models
            for cluster_id, server in enumerate(servers):
                weights = self.average_weights(local_weights_cluster[cluster_id], n_clients)
                server.update_weights(weights)

            print([c.get_cluster_weights() for c in centers])

            num_classes = centers[0].num_classes # hacky
            cluster_label_weights = [[0] * num_classes for _ in range(len(servers))]
            cluster_weights = [0 for _ in range(len(servers))]

            for c_id, center in enumerate(centers):
                for i in range(len(center_trains[c_id])):
                    for j in range(len(cluster_label_weights)):
                            cluster_weights[j] += center.samples_weights[j][i]
            print(cluster_weights)

            for center in centers:
                client_cluster_labels_weights = center.cluster_labels_weights
                for j in range(len(cluster_label_weights)):
                    for k in range(num_classes):
                        cluster_label_weights[j][k] += client_cluster_labels_weights[j][k]
            for j in range(len(cluster_label_weights)):
                for i in range(len(cluster_label_weights[j])):
                    if cluster_label_weights[j][i] < 1e-8:
                        cluster_label_weights[j][i] = 1e-8
                cluster_label_weights[j] = [i / sum(cluster_label_weights[j]) for i in cluster_label_weights[j]]

            # I think this should be each g_epoch
            for c_id, center in enumerate(centers):
                center.update_labels_weights(cluster_label_weights, center_trains[c_id])

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedrc', centers, epoch=g_epoch + 1) 

        self.save_models('fedrc', centers)


    def fit_fedclust(self, server, centers, center_trains, center_vals):

        assert self.tau > 0
        center_step = [0 for _ in range(len(centers))]
        lmbda = self.tau

        def recursive_subclustering(centers, distances, lmbda):
            # Initialize with a single cluster containing all centers
            centers_of_cluster = [list(range(len(centers)))]  # List of lists, each representing a cluster

            while True:
                new_centers_of_cluster = []

                # Flag to check if we performed any split
                did_split = False

                # Iterate over current clusters
                for cluster_id in range(len(centers_of_cluster)):
                    # Get the distances for the current cluster
                    centers_within_cluster = [c_id for c_id in centers_of_cluster[cluster_id]]
                    distances_within_cluster = distances[centers_within_cluster, :][:, centers_within_cluster]
                    print(distances_within_cluster)
                    # Split the cluster into two subclusters
                    if len(distances_within_cluster) > 1:
                        clustering = AgglomerativeClustering(metric="precomputed", linkage="complete", n_clusters=None, distance_threshold=lmbda).fit(distances_within_cluster)

                        cluster_labels = np.unique(clustering.labels_)
                        cs = [np.argwhere(clustering.labels_ == lbl).flatten() for lbl in cluster_labels]
                        print(cs)
                        if len(cs) > 1:
                            for i, c in enumerate(cs):
                                cs[i] = [centers_of_cluster[cluster_id][id] for id in c]

                            # Add the new subclusters to the list
                            new_centers_of_cluster += cs

                            # Indicate that a split was made
                            did_split = True
                            print(f"Splitting cluster {cluster_id} into two subclusters.")
                    
                    if not did_split:
                        # If no split, retain the original cluster
                        new_centers_of_cluster.append(centers_of_cluster[cluster_id])

                # If no clusters were split, we can stop
                if not did_split:
                    print("No more clusters to split. Stopping.")
                    break

                # Update the cluster list with the new set of clusters
                centers_of_cluster = new_centers_of_cluster

            return centers_of_cluster

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))


        # pretraining for hierarchical clustering
        local_weights = []
        for c_id, center in enumerate(centers):
            print(f"---Center {1+c_id} / {len(centers)}")
            center.update_weights(server.get_weights())

            train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)
            for l_epoch in range(self.local_epochs):
                
                for data in tqdm(train_loader, disable=not is_interactive):
                    batch_c_loss = center.local_step(data)

                    self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                    center_step[c_id] += 1

            del train_loader

            local_weights.append(flatten(center.get_weights()).cpu())
        local_weights = torch.stack(local_weights)

        distances = euclidean_distances(local_weights.numpy())

        # start with single cluster
        centers_of_cluster = recursive_subclustering(centers, distances, lmbda)
        print(centers_of_cluster)        

        # update cluster weights
        servers = []
        for cluster_id in range(len(centers_of_cluster)):
            for c_id in centers_of_cluster[cluster_id]:
                centers[c_id].set_cluster(cluster_id)
            local_weights_cluster = [centers[c_id].get_weights() for c_id in centers_of_cluster[cluster_id]]
            local_n_cluster = [n_clients[c_id] for c_id in centers_of_cluster[cluster_id]]

            weights = self.average_weights(local_weights_cluster, local_n_cluster)
            new_server = copy.deepcopy(server) # just to have a valid setup
            new_server.cluster_id = cluster_id
            new_server.update_weights(weights)
            servers.append(new_server)

        for g_epoch in range(1,self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # go into local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())
                center.save_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            self.find_memberships('fedclust', centers, g_epoch)
            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fedclust', centers, servers, epoch=g_epoch + 1) 

        self.save_models('fedclust', centers, servers)


    def fit_fesem(self, servers, centers, center_trains, center_vals):
        
        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            local_weights_cluster = [[] for _ in servers]
            local_n_cluster = [[] for _ in servers]
            # initialise clusters using KMeans
            if g_epoch == 0:
                center_models = [flatten(center.get_weights()) for center in centers]
                pairwise_l2 = euclidean_distances(center_models)
                result = KMeans(len(servers), max_iter=20).fit(pairwise_l2)

                for c_id, cluster_id in enumerate(result.labels_):                 
                    centers[c_id].set_cluster(cluster_id)
                    local_weights_cluster[cluster_id].append(centers[c_id].get_weights())
                    local_n_cluster[cluster_id].append(n_clients[c_id])
            else:
                # E-step: associate each center with a cluster
                for c_id, center in enumerate(centers):
                    center_model = flatten(center.get_weights()).cpu()
                    min_distance = 999999
                    best_cluster = None
                    for cluster_id, server in enumerate(servers):
                        cluster_model = flatten(server.get_weights())
                        l2_distance = torch.norm(cluster_model - center_model, p=2)
                        if l2_distance < min_distance:
                            best_cluster = cluster_id
                            min_distance = l2_distance

                    assert best_cluster is not None
                    center.set_cluster(best_cluster)
                    local_weights_cluster[best_cluster].append(center.get_weights())
                    local_n_cluster[best_cluster].append(n_clients[c_id])
                    print(f"Center {c_id} cluster:", center.cluster_id)

            # M-step: update all servers
            for cluster_id, server in enumerate(servers):
                if len(local_weights_cluster[cluster_id]) == 0:
                    print("Skipping since no update to cluster")
                    continue
                weights = self.average_weights(local_weights_cluster[cluster_id], local_n_cluster[cluster_id])
                server.update_weights(weights)
            
            # go into actual local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())
                center.save_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_fedprox_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            self.find_memberships('fesem', centers, g_epoch)
            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('fesem', centers, servers, epoch=g_epoch + 1) 

        self.save_models('fesem', centers, servers)


    def fit_gt(self, server, centers, center_trains, center_vals, dataset, num_clusters, p_num):

        center_step = [0 for i in range(len(centers))]
        save_case = 'gt'
        if p_num < len(centers):
            save_case = 'gtpp'

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # initialise clusters to match number of clients
        servers = [copy.deepcopy(server) for _ in range(len(centers))]
        centers_of_cluster = [[] for _ in range(len(centers))]
        for c_id, center in enumerate(centers):
            center.set_cluster(c_id)
            centers_of_cluster[c_id].append(c_id)

        unassigned_center_ids = set(range(len(centers)))
        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")
            
            # select participating centers
            center_ids = np.arange(len(centers))
            selected_center_ids = np.random.choice(center_ids, size=p_num, replace=False)
            selected_center_ids = np.sort(selected_center_ids)
            print("Participants:", selected_center_ids)
            unassigned_center_ids = unassigned_center_ids - set(selected_center_ids)

            for c_id, center in enumerate(centers):

                # skip center if not selected
                if c_id not in set(selected_center_ids):
                    continue

                print(f"---Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])

                del train_loader
                # move model back to cpu
                center.model.cpu()

            all_clusters = get_gt_clustering(dataset, len(centers), num_clusters, unassigned_center_ids)
            print(all_clusters)

            neighborhoods = find_neighborhoods(all_clusters) # each node's neighborhood
            seen_neighborhoods = cluster_neighborhoods(neighborhoods, centers)

            servers = self.average_weights_clusterwise(seen_neighborhoods, server, centers, n_clients)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models(save_case, centers, servers, epoch=g_epoch + 1) 

        if self.epochs > 1:
            self.save_models(save_case, centers, servers)


    def fit_pacfl(self, server, centers, center_trains, center_vals, dataset):

        assert self.tau > 0
        def calculating_adjacency(clients_idxs, U): 
                
            nclients = len(clients_idxs)
            
            sim_mat = np.zeros([nclients, nclients])
            for idx1 in range(nclients):
                for idx2 in range(nclients):
                    #print(idx1)
                    #print(U)
                    #print(idx1)
                    U1 = copy.deepcopy(U[clients_idxs[idx1]])
                    U2 = copy.deepcopy(U[clients_idxs[idx2]])
                    
                    #sim_mat[idx1,idx2] = np.where(np.abs(U1.T@U2) > 1e-2)[0].shape[0]
                    #sim_mat[idx1,idx2] = 10*np.linalg.norm(U1.T@U2 - np.eye(15), ord='fro')
                    #sim_mat[idx1,idx2] = 100/np.pi*(np.sort(np.arccos(U1.T@U2).reshape(-1))[0:4]).sum()
                    mul = np.clip(U1.T@U2 ,a_min =-1.0, a_max=1.0)
                    sim_mat[idx1,idx2] = np.min(np.arccos(mul))*180/np.pi
                
            return sim_mat


        def flatten(items):
            """Yield items from any nested iterable; see Reference."""
            for x in items:
                if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                    for sub_x in flatten(x):
                        yield sub_x
                else:
                    yield x

        def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
            '''
            Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
            rows and columns replacing the minimum elements. It is working on adjacency matrix. 
            
            :param: A (adjacency matrix), thresh (stopping threshold)
            :type: A (np.array), thresh (int)
            
            :return: clusters
            '''
            label_assg = {i: i for i in range(A.shape[0])}
            
            step = 0
            while A.shape[0] > 1:
                np.fill_diagonal(A,-np.NINF)
                #print(f'step {step} \n {A}')
                step+=1
                ind=np.unravel_index(np.argmin(A, axis=None), A.shape)

                if A[ind[0],ind[1]]>thresh:
                    print('Breaking HC')
                    break
                else:
                    np.fill_diagonal(A,0)
                    if linkage == 'maximum':
                        Z=np.maximum(A[:,ind[0]], A[:,ind[1]])
                    elif linkage == 'minimum':
                        Z=np.minimum(A[:,ind[0]], A[:,ind[1]])
                    elif linkage == 'average':
                        Z= (A[:,ind[0]] + A[:,ind[1]])/2
                    
                    A[:,ind[0]]=Z
                    A[:,ind[1]]=Z
                    A[ind[0],:]=Z
                    A[ind[1],:]=Z
                    A = np.delete(A, (ind[1]), axis=0)
                    A = np.delete(A, (ind[1]), axis=1)

                    if type(label_assg[ind[0]]) == list: 
                        label_assg[ind[0]].append(label_assg[ind[1]])
                    else: 
                        label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

                    label_assg.pop(ind[1], None)

                    temp = []
                    for k,v in label_assg.items():
                        if k > ind[1]: 
                            kk = k-1
                            vv = v
                        else: 
                            kk = k 
                            vv = v
                        temp.append((kk,vv))

                    label_assg = dict(temp)

            clusters = []
            for k in label_assg.keys():
                if type(label_assg[k]) == list:
                    clusters.append(list(flatten(label_assg[k])))
                elif type(label_assg[k]) == int: 
                    clusters.append([label_assg[k]])
                    
            return clusters

        center_step = [0 for _ in range(len(centers))]

        # get the dataset sizes
        n_clients = []
        for center_train in center_trains:
            n_clients.append(len(center_train))

        # adapted from https://github.com/MMorafah/PACFL/blob/main/main_pacfl.py
        K = 3
        U_clients = []
        for c_id in range(len(centers)):
            
            val_data = center_vals[c_id]       
            labels_local = [d[1] for d in val_data]

            uni_labels, cnt_labels = np.unique(labels_local, return_counts=True)
            
            # print(f'Labels: {uni_labels}, Counts: {cnt_labels}')
            
            nlabels = len(load_labels(dataset))
            U_temp = []
            for j in range(nlabels):
            
                filtered_data = [img for img, lbl in val_data if lbl == j]
                if len(filtered_data) == 0:
                    continue
                filtered_data = torch.concat(filtered_data)
                filtered_data = filtered_data.reshape(len(filtered_data), -1).T.numpy()
                # print(filtered_data.shape)
                    
                if K > 0: 
                    u1_temp, sh1_temp, vh1_temp = np.linalg.svd(filtered_data, full_matrices=False)
                    u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0)
                    U_temp.append(u1_temp[:, 0:K])
                
                
            U_clients.append(copy.deepcopy(np.hstack(U_temp)))
            
        adj_mat = calculating_adjacency([i for i in range(len(centers))], U_clients)
        centers_of_cluster = hierarchical_clustering(copy.deepcopy(adj_mat), thresh=self.tau, linkage='average')
        # initialise clusters with same copy
        servers = [copy.deepcopy(server) for _ in range(len(centers_of_cluster))]

        for cluster_id, c_ids in enumerate(centers_of_cluster):
            for c_id in c_ids:
                centers[c_id].set_cluster(cluster_id)

        for g_epoch in range(self.epochs):
            print(f"Global epoch {1+g_epoch} / {self.epochs}")

            # go into local training loop
            for c_id, center in enumerate(centers):
                print(f"---Epoch {1+g_epoch} --- Center {1+c_id} / {len(centers)}")
                cluster_id = center.cluster_id
                center.update_weights(servers[cluster_id].get_weights())

                train_loader = DataLoader(center_trains[c_id], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                for l_epoch in range(self.local_epochs):
                    
                    for data in tqdm(train_loader, disable=not is_interactive):
                        batch_c_loss = center.local_step(data)

                        self.writer.add_scalar(f"Loss/train/{center.id}", batch_c_loss, center_step[c_id])
                        center_step[c_id] += 1

                    _ = self.test(center, center_vals[c_id], verbose=True, writer=self.writer, n_iter=center_step[c_id])
                del train_loader

            # update cluster weights
            servers = []
            for cluster_id in range(len(centers_of_cluster)):
                local_weights_cluster = [centers[c_id].get_weights() for c_id in centers_of_cluster[cluster_id]]
                local_n_cluster = [n_clients[c_id] for c_id in centers_of_cluster[cluster_id]]

                weights = self.average_weights(local_weights_cluster, local_n_cluster)
                new_server = copy.deepcopy(server) # just to have a valid setup
                new_server.cluster_id = cluster_id
                new_server.update_weights(weights)
                servers.append(new_server)
            self.find_memberships('pacfl', centers, g_epoch)

            if g_epoch + 1 < self.epochs and (g_epoch + 1) % self.save_every == 0:
                self.save_models('pacfl', centers, servers, epoch=g_epoch + 1) 

        self.save_models('pacfl', centers, servers)

    def test(self, center, val, verbose=False, writer=None, n_iter=None):
        val_loss = 0
        val_hits = 0
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for data in tqdm(val_loader, disable=not is_interactive):
            loss, hits = center.local_test_step(data)
            val_loss += loss
            val_hits += hits 
        val_loss = val_loss / len(val_loader)
        val_acc = val_hits / len(val_loader.dataset) 

        if verbose:
            print(f"Center {center.id} - Val loss: {val_loss}, Acc: {val_acc}")

        if writer is not None:
            self.writer.add_scalar(f"Loss/val/{center.id}", val_loss, n_iter)
            self.writer.add_scalar(f"Acc/val/{center.id}", val_acc, n_iter)

        return (val_loss.item(), val_acc.item())        

    def get_instancewise_loss(self, center, val):
        losses = []
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        for data in val_loader:
            loss = center.local_instancewise_loss_step(data)
            losses.append(loss.detach().cpu())
        losses = torch.concat(losses)

        return losses        

    def get_embeddings(self, model, val, device):
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        embs = []
        model.to(device)
        for data in val_loader:
            x, _ = data
            x = x.to(device)
            emb = model(x).detach().cpu()
            embs.append(emb)
        embs = torch.cat(embs)
        return embs
    
    def get_wd(self, model, source, target, device, proj_ratio=1., max_sample=None, metric='cosine'):

        if max_sample is not None:
            if max_sample < len(source):
                indices = torch.randperm(len(source))[:max_sample]
                source = Subset(source, indices)
            if max_sample < len(target):
                indices = torch.randperm(len(target))[:max_sample]
                target = Subset(target, indices)

        source_zs = self.get_embeddings(model, source, device).to(device)
        target_zs = self.get_embeddings(model, target, device).to(device)

        embed_dim = source_zs.shape[1]
        proj_dim = int(proj_ratio * embed_dim)
        R = torch.randn(embed_dim, proj_dim).to(device) / math.sqrt(embed_dim)
        R = torch.nn.functional.normalize(R, p=2, dim=1)

        n, m = source_zs.shape[0], target_zs.shape[0]
        a = np.ones(n) / n  # Uniform distribution over points in X
        b = np.ones(m) / m  # Uniform distribution over points in Y
        M = ot.dist((source_zs @ R).cpu().numpy(), (target_zs @ R).cpu().numpy(), metric=metric)
        wd = ot.emd2(a, b, M)

        return wd