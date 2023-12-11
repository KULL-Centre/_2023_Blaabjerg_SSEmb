import json
import numpy as np
import tqdm, random
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric
import torch_cluster
import esm
from scipy.spatial import distance_matrix

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class CATHDataset:
    '''
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.
    
    Has attributes `self.total, self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.
    
    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    '''
    def __init__(self, path, splits_path, minlen=0, maxlen=10000):
        
        self.splits_path = splits_path
        self.total = []
        self.maxlen = maxlen
        self.minlen = minlen

        with open(path) as f:
            lines = f.readlines()
        
        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry['name']
            coords = entry['coords']
            
            entry['coords'] = list(zip(
                coords['N'], coords['CA'], coords['C'], coords['O']
            ))
            
            self.total.append(entry)

    def split(self):         
        with open(self.splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits['train'], \
            dataset_splits['validation'], dataset_splits['test']

        self.train, self.val, self.test = [], [], []

        for entry in self.total:
            if entry["name"] in train_list:
                self.train.append(entry)
            elif entry["name"] in val_list:
                self.val.append(entry)
            elif entry["name"] in test_list:
                self.test.append(entry)

class BatchSampler(data.Sampler):
    '''
    Batch sampler that samples a single protein at a time
    '''
    def __init__(self, node_counts, shuffle=True):    
        self.indices = [i for i in range(len(node_counts))]
        self.shuffle = shuffle
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.indices)
        indices = self.indices
        for idx in indices:
            self.batches.append([idx])

    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch

class ProteinGraphData(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the
    manuscript.
    '''
    def __init__(self, data_list, num_positional_embeddings=16,
                 top_k=20, dist_cutoff=12, num_rbf=16, device="cpu"):
        super(ProteinGraphData, self).__init__()

        self.data_list = data_list
        self.dist_cutoff = dist_cutoff
        self.num_rbf = num_rbf
        self.contacts_bins = np.array([0, 10, 20, 30, 40, 50])
        self.num_positional_embeddings = num_positional_embeddings
        self.top_k = top_k
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, "<MASK>": 20}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}

    def __len__(self): return len(self.data_list)

    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])

    def _featurize_as_graph(self, protein):
        name = protein['name']
        with torch.no_grad():
            coords = torch.tensor(np.array(protein["coords"]).astype(np.float32), device=self.device)
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)
           
            # Mask positions without coords
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            # Compute edges in graph
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)            

            # Compute distance mask for row attention
            dist_mask = torch.zeros((seq.size()[0], seq.size()[0])).bool()
            dist_mask[edge_index[1,:], edge_index[0,:]] = True
            dist_mask[np.diag_indices(seq.size()[0]), np.diag_indices(seq.size()[0])] = True
            dist_mask = ~dist_mask # Reverse so that non-contacts are set to zero in MSA row attention map

            # Compute graph node and edge features
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)

            dihedrals = self._dihedrals(coords)
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)

            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)

            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))

            if "msa" in protein:
                msa = protein["msa"]
            else:
                msa = None

            if "label" in protein:
                label = protein["label"]
            else:
                label = None
           
            if "emb" in protein:
                emb = protein["emb"]
            else:
                emb = None

        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                         node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,
                                         edge_index=edge_index,
                                         mask=mask,
                                         msa=msa,
                                         dist_mask=dist_mask,
                                         label=label,
                                         emb=emb,
                                         )

        return data

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features


    def _positional_embeddings(self, edge_index,
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
