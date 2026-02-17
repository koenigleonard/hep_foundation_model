import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FactorizedOutputHead(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 num_features = 3,
                 num_bins = (42, 32, 32)
                 ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.num_bins = num_bins

        self.sum_bins = int(np.sum(num_bins))
        #add single linear layer for embedding(d = 256) --> (d = 256 * sum_bins)
        self.linear = nn.Linear(hidden_dim, hidden_dim * self.sum_bins)

        self.activation = nn.Softplus() #constraints the output of the singlelayer to always positive / similiar to Relu but smooth
        self.logit_rescale = nn.Parameter(torch.tensor(1.0)) #adds a single learnable parameter for the rescaling of the final logit

    def forward(self, h):
        batch_size, num_const, _hidden_dim = h.shape 
        assert _hidden_dim == self.hidden_dim

        #project hidden state (d = 256) into (d = 256 * sum_bins) with linear layer
        bin_emb_flat = self.activation(self.linear(h))

        #reshape state (d = 256 * sum_bins) into matrix with ( sum_bins x 256)
        bin_emb = bin_emb_flat.view(batch_size, num_const, self.sum_bins, self.hidden_dim)
        #split it into 3 different matrices for pt, eta and phi along dimension with size of sum_bins
        pt_emb, eta_emb, phi_emb = torch.split(bin_emb, self.num_bins, dim = 2)

        #now recombine into a single particle vector
        #first get a single vector for all pt, eta combinations so 42*32 possibilities
        pt_eta_emb = pt_emb.unsqueeze(2) * eta_emb.unsqueeze(3) # shape = (42 x 32 x 256)
        #reshape into ((42*32)x256)
        pt_eta_emb = pt_eta_emb.view(batch_size, num_const, -1, self.hidden_dim) #shape= ((42*32)x256)

        #combine with phi into final logit
        logits_3d = pt_eta_emb @ phi_emb.transpose(2, 3) #((42*32)x256)@(256 x 32) = ((42*32)x32)
        #flatten again and scale --> shape = batchsize x num_const x (42*32*32)

        logits = self.logit_rescale.exp() * logits_3d.view( batch_size, num_const, -1 )
        #logits have dimension n_pt*n_eta*n_phi
        return logits

class JetTransformer(nn.Module):
    def __init__(self,
                 hidden_dim = 256,
                 num_layers = 10,
                 num_heads = 4,
                 num_features = 3,
                 num_bins = (40, 30, 30),
                 dropout = 0.1,
                 add_start = True,
                 add_stop = True,
                 causal_mask = True):
        
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_features = num_features
        self.num_bins = num_bins
        self.causal_mask = causal_mask

        if add_start:
            self.num_bins = [x + 1 for x in self.num_bins]
            print("Added start token!")
        if add_stop:
            self.num_bins = [x + 1 for x in self.num_bins]
            print("Added stop token!")

        print(f"Num_bins: {self.num_bins}")

        num_bins = self.num_bins

        #this total voc size includes all physical bins + start + stop + pad --> (43,33,33)
        #this is only used for the input embedding --> the output does not make room for padding because we dont want to predict padding
        self.total_voc_bins = [x + 2 for x in self.num_bins] #+2 because num_bin describes the max value (incl) so 0...40 (41) + start + stop + pad = 44 bins
        self.total_voc_size = np.prod(self.total_voc_bins)

        self.voc_bins = [x + 1 for x in self.num_bins]
        self.voc_size = np.prod(self.voc_bins)

        print(f"Total_voc_bin: {self.total_voc_bins}")

        #chooses pad bin to total_bins +1 in each feature 
        self.PAD_BIN = [bins + 1 for bins in self.num_bins]
        self.START_BIN = [0 for i in range(self.num_features)]
        self.STOP_BIN = self.num_bins

        self.PAD_IDX = -1

        print(f"Bins reserved for PAD: {self.PAD_BIN}")

        #----- adding all different layers to the network
        #we want to embedd all features seperately so we a embedding layer with 3 parrallel embeddings
        self.feature_embeddings = nn.ModuleList(
            [
                nn.Embedding(embedding_dim=self.hidden_dim, num_embeddings=self.total_voc_bins[i])
                for i in range(num_features)
            ]
        )

        #add the #num_layers of TransformerEncoder Layers
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model = hidden_dim, #expected input dimension
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                    norm_first=True,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

        self.out_norm = nn.LayerNorm(hidden_dim)
        #this is only used during training and randomly drops a percentage (0.1) of the neurons
        #this prevents overfitting, so the model learns not to rely on specific single neurons
        self.dropout = nn.Dropout(dropout)

        self.output_layer = FactorizedOutputHead(hidden_dim = self.hidden_dim,
                                                 num_features = self.num_features,
                                                 num_bins = self.voc_bins, #here we do not count the padding bin
                                                )

        #define output layer
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.PAD_IDX)

    def forward(self,
                x, #has shape (batch_size, num_const, num_features) ---> logits with shape (batch_size, num_const, total_bins)
                ):

        batch_size = x.shape[0]
        num_const = x.shape[1] #saves number of const. used in this data
        num_features = x.shape[2] 
        assert num_features == self.num_features

        #creates a padding_mask to mask invalid particles
        padding_mask = (x[:, :, 0] < 0) #shows every where an invalid particle is does not exclude start and stop needs shape (batch_size, num_const)

        #replaces the -1 as padding bin with a valid pad bin which is stop_bin + 1
        for i in range(self.num_features):
            x[:, : , i] = torch.where(x[:, :, i] < 0, self.PAD_BIN[i], x[:, :, i])

        #extract features from x
        #embed seperately and add together 
        # (padding tokens are mapped to a padding embedding but never used because padding mask will exclude them)
        # they are also excluded while calculating the loss
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        #create a casual mask
        attention_mask = None
        if self.causal_mask:
            #creates a triangular casual mask
            attention_mask = torch.triu(
                torch.ones(num_const, num_const, device = x.device),
                diagonal = 1
            ).bool()

        #transformer layer
        for transformer_layer in self.layers:
            emb = transformer_layer(src = emb, 
                                    src_mask = attention_mask, 
                                    src_key_padding_mask=padding_mask,
                                    is_causal = self.causal_mask
                                    )
        
        #apply last norm
        emb = self.out_norm(emb)
        #apply last droptout
        emb = self.dropout(emb)

        #print("forward: Transformer layer + out_norm + dropout successfull!")
        #project to final logit
        logits = self.output_layer(emb)
        #print("forward: Output layer successfull!")

        return logits

    #this computes autoregressive loss for gpt style training
    def loss(self, logits_joint, targets):
        #computes cross entropy over full dictonary of size prod(num_bins)
        #ignores padding -1

        #discard the final logit, there is no target for this logit
        logits_joint = logits_joint[:,:-1].reshape(-1, self.voc_size) 

        #compute target ids from target tokens
        target_ids = self.tuple_to_index(targets[..., 0], targets[..., 1], targets[..., 2], self.num_bins)

        #shift targets to the right, because t_id1 contains what logit_0 should predict
        target_ids = target_ids[:, 1:].reshape(-1) 
        #print("------ target_ids-----")
        #print(target_ids)

        #this computes cross entropy loss only if the target particle is valid (ignores the padding because our network does not produce padding)
        #this now computes the autoregressive loss we need for the gpt style loss
        loss = self.criterion(logits_joint, target_ids)

        return loss
    
    #returns the probability of a target jet relative to the logits of the network
    #ideally the logits come from a forward pass from the same targets
    # (thats basically the same from what the loss function is calculated)
    def probability(
            self,
            logits_joint, 
            targets, 
            logarithmic = False,
            topk = None,

    ):
        batch_size, seq_length, voc_size = logits_joint.shape

        #discard the final logit, there is no target for this logit
        logits_joint = logits_joint[:,:-1]
        #apply softmax to get probs from logits
        probs = torch.softmax(logits_joint, dim = -1)
        # --- TOP-K FILTERING ---
        if topk is not None:
            # find topk probs at each (B,S)
            topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

            # mask: True for entries in topk
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)

            # suppress everything else
            probs = probs.masked_fill(~mask, 1.0)

        #compute target ids from target tokens
        target_ids = self.tuple_to_index(targets[..., 0], targets[..., 1], targets[..., 2], self.num_bins)
        
        #shift targets to the right, because t_id1 contains what logit_0 shWould predict
        target_ids = target_ids[:, 1:]

        padding_mask = target_ids == self.PAD_IDX
        #compute the prob that each particle id in target_ids shows up
        target_ids = target_ids.masked_fill(padding_mask, 0)
        probs = probs.gather(
            dim = -1, #gather along last dimension
            index= target_ids.unsqueeze(-1)
        ).squeeze(-1)

        probs = probs.masked_fill(padding_mask, 1.0)

        if logarithmic:
            prob = torch.log(probs).sum(dim = -1)
        else:
            prob = probs.prod(dim = -1)

        return prob

    def tuple_to_index(self, pt, eta, phi, num_bins):
        """
        Map (pt, eta, phi) bin indices into single vocabulary index.

        and padding -1 to -1 in index

        index = pt + n_pt * eta + (n_pt*n_eta) * phi
        """
        padding_mask = (pt == -1)

        joint_id = pt + eta * num_bins[0] + (num_bins[0]*num_bins[1])* phi
        joint_id[padding_mask] = self.PAD_IDX

        return joint_id

    def index_to_tuple(self, index, num_bins):
        """
        Inverse mapping: dictionary index -> (pt, eta, phi)
        """

        padding_mask = (index == self.PAD_IDX)

        pt = index % num_bins[0]
        eta = (index // num_bins[0]) % num_bins[1]
        phi = index // (num_bins[0] * num_bins[1])

        pt[padding_mask] = -1
        eta[padding_mask] = -1
        phi[padding_mask] = -1

        return pt, eta, phi
    
