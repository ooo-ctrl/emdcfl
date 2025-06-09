import copy
import torch
import torch.nn.functional as F

class Center():
    def __init__(self, id, model, optimizer, lr, device, cluster_id=None) -> None:

        self.id = id
        self.model = model
        self.old_model = None
        self._optimizer = optimizer
        self.lr = lr
        self.device = device
        self.cluster_id = cluster_id

    def get_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6)

    def set_cluster(self, cluster_id):
        self.cluster_id = cluster_id

    def local_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)

        loss = F.cross_entropy(out, y)
        loss.backward()
        self.optimizer.step()

        return loss

    def get_weights(self):
        return copy.deepcopy(self.model.state_dict())

    def update_weights(self, weights):
        self.model.load_state_dict(copy.deepcopy(weights))
        self.get_optimizer()

    def save_weights(self, weights):
        self.old_model = copy.deepcopy(self.model).cpu()
        self.old_model.load_state_dict(copy.deepcopy(weights))

    def get_dW(self):

        dW = self.get_weights()

        for key in dW.keys():
            dW[key] = dW[key] * 0.
            dW[key] = self.get_weights()[key].cpu() - self.old_model.state_dict()[key]

        return dW

    @torch.no_grad()
    def local_test_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)

        loss = F.cross_entropy(out, y)
        top_p, top_class = out.softmax(dim=1).topk(1, dim=1)
        hits = torch.sum(top_class.squeeze() == y)

        self.model.train()
        return loss, hits

    @torch.no_grad()
    def local_logit(self, data, max=False):
        self.model.to(self.device)
        self.model.eval()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        logit = self.model(x)
        if max:
            logit, top_class = logit.topk(1, dim=1)

        self.model.train()
        return logit
    
class CenterFedProx(Center):
    def __init__(self, id, model, optimizer, lr, device, cluster_id=None, mu=0.) -> None:
        super().__init__(id, model, optimizer, lr, device, cluster_id)
        self.mu = mu

    def local_fedprox_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)

        loss = F.cross_entropy(out, y)

        proximal_term = 0.0
        self.old_model.to(self.device)
        for w, w_t in zip(self.model.parameters(), self.old_model.parameters()):
            proximal_term += (w - w_t).norm(2)

        loss += self.mu * proximal_term

        loss.backward()
        self.optimizer.step()

        return loss

class CenterEM(Center):

    def __init__(self, id, model, optimizer, lr, device, cluster_id=None, num_clusters=1) -> None:
        super().__init__(id, model, optimizer, lr, device, cluster_id)
        self.cluster_weights = torch.ones(num_clusters) / num_clusters # pi - should be shape num_cluster x 1
        self.samples_weights = None # should be shape num_cluster x num_train_samples

    def update_q(self, all_losses):
        # calculate q
        # all_losses shape num_cluster x num_train_samples
        self.samples_weights = F.softmax((torch.log(self.cluster_weights) - all_losses.T), dim=1).T # should be shape num_cluster x num_train_samples

    def update_pi(self):
        self.cluster_weights = self.samples_weights.mean(dim=1)


    def local_weighted_step(self, data, cluster_id):
        self.model.to(self.device)
        self.model.train()

        x, y, indices = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)

        samples_weights = self.samples_weights[cluster_id][indices].to(self.device)
        loss = F.cross_entropy(out, y, reduction='none') # get the instancewise loss
        loss = (loss @ samples_weights) / loss.size(0) # weigh every instance by likelihood it belongs to cluster_id and average

        loss.backward()
        self.optimizer.step()

        return loss
    
    @torch.no_grad()
    def local_instancewise_loss_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        x, y, _ = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='none') # no reduction to get the instancewise loss!

        self.model.train()
        return loss

    def get_cluster_weights(self):
        return self.cluster_weights
    
class CenterPFedGraph(Center):
    def __init__(self, id, model, optimizer, lr, device, cluster_id=None, mu=0.) -> None:
        super().__init__(id, model, optimizer, lr, device, cluster_id)
        self.mu = mu

    def local_pfedgraph_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)

        loss = F.cross_entropy(out, y)

        flat_cur = []
        flat_old = []
        for w, w_t in zip(self.model.parameters(), self.old_model.parameters()):
            flat_cur.append(w.view(-1))
            flat_old.append(w_t.view(-1))

        flat_cur = torch.cat(flat_cur)
        flat_old = torch.cat(flat_old).to(self.device)

        loss -= self.mu * torch.dot(flat_cur, flat_old) / (torch.linalg.norm(flat_cur) * torch.linalg.norm(flat_old))

        loss.backward()
        self.optimizer.step()

        return loss
    
class CenterRC(Center):

    def __init__(self, id, model, optimizer, lr, device, cluster_id=None, num_clusters=1, num_classes=10, n_train=None) -> None:
        super().__init__(id, model, optimizer, lr, device, cluster_id)
        self.cluster_weights = torch.ones(num_clusters) / num_clusters # pi - should be shape num_cluster x 1
        self.samples_weights = None # should be shape num_cluster x num_train_samples
        self.cluster_labels_weights = torch.ones(num_clusters, num_classes) / num_classes # for each cluster, how much emphasis on each class cluster0: 1/2 1/2 for two classes 
        self.labels_weights = torch.ones(num_clusters, n_train) / num_clusters # for each cluster, how much of sample belongs to cluster cluster0: 1/4 ... 1/4 for four clusters  

        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.n_train = n_train

    def update_q(self, all_losses):
        # calculate q
        # all_losses shape num_cluster x num_train_samples
        L = - all_losses.T - torch.log(self.labels_weights.T)
        self.mean_I = torch.exp(torch.log(self.cluster_weights) - all_losses.T).T
        self.mean_I = torch.mean(torch.sum(self.mean_I,dim=1))

        new_samples_weights = F.softmax(torch.log(self.cluster_weights) + L, dim=1).T
        self.samples_weights = new_samples_weights

    def update_pi(self):
        self.cluster_weights = self.samples_weights.mean(dim=1)

    def update_cluster_labels_weights(self, data):
        self.cluster_labels_weights = torch.zeros(self.num_clusters, self.num_classes) / self.num_classes
        for i, (x, y, idx) in enumerate(data):
            for j in range(self.num_clusters):
                self.cluster_labels_weights[j][y] += self.samples_weights[j][i]

    def update_labels_weights(self, labels_weights, data):
        for i, (x ,y, idx) in enumerate(data):
            for j in range(self.num_clusters):
                self.labels_weights[j][i] = labels_weights[j][y]
        # for i, learner in enumerate(self.learners_ensemble.learners):
        #     learner.labels_weights = labels_weights[i]

    def local_weighted_step(self, data, cluster_id):
        self.model.to(self.device)
        self.model.train()

        x, y, indices = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(x)

        samples_weights = self.samples_weights[cluster_id][indices].to(self.device)
        loss = F.cross_entropy(out, y, reduction='none') # get the instancewise loss
        loss = (loss @ samples_weights) / loss.size(0) # weigh every instance by likelihood it belongs to cluster_id and average

        loss.backward()
        self.optimizer.step()

        return loss
    
    @torch.no_grad()
    def local_instancewise_loss_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        x, y, _ = data
        x = x.to(self.device)
        y = y.to(self.device)

        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction='none') # no reduction to get the instancewise loss!

        self.model.train()
        return loss

    def get_cluster_weights(self):
        return self.cluster_weights