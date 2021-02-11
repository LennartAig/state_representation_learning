from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from srl_framework.srl.models.base import BaseModelSRL
from srl_framework.utils.networks import make_cnn, make_mlp
from srl_framework.utils.encoder import ResNetEncoder, CnnEncoder


# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}

class DeepInfoMax(BaseModelSRL):
    """
    """
    def __init__(self, img_channels = 3, state_dim=128, params = None, normalized_obs=True, img_size = 84):
        super(DeepInfoMax, self).__init__()
        self.normalized_latent = True if params.NORMALIZED_LATENT and not params.CNN.NORMALIZED_LATENT else False
        self.squashed_latent = True if params.SQUASHED_LATENT and not params.CNN.SQUASHED_LATENT else False
        if params.CNN.ARCHITECTURE == 'impala':
            self.encoder = ResNetEncoder(
                img_channels = img_channels,
                feature_dim=state_dim,
                params=params.CNN,
                img_size = img_size,
                architecture=params.CNN.ARCHITECTURE,
                normalized_obs=normalized_obs
                )
        else:
            self.encoder = CnnEncoder(
                img_channels = img_channels,
                feature_dim=state_dim,
                params=params.CNN,
                img_size = 84,
                architecture=params.CNN.ARCHITECTURE,
                normalized_obs=normalized_obs
                )

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=params.LEARNING_RATE)
        self.train

    def forward(self, x):
        out = self.encoder(x,fmaps=True)
        state = out['out']
        f5 = out['f5']
        if self.normalized_latent: state = self.layer_norm(state) 
        if self.squashed_latent: state = torch.tanh(state)       
        return state, f5
    
    @property
    def local_layer_depth(self):
        return self.encoder.local_layer_depth

    def get_state(self, obs, grad=True):
        """
        Input:
        ------
            - obs (torch tensor)
            - grad (bool): Set if gradient is required in latter calculations
        Return:
        ------
            - state (torch tensor)
        """
        if grad:
            state = self.encoder(obs,fmaps=False)
        else:
            with torch.no_grad():
                state = self.encoder(obs,fmaps=False)
        if self.normalized_latent: state = self.layer_norm(state) 
        if self.squashed_latent: state = torch.tanh(state)
        return state

    def loss(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        
        # y: (64, 64), M: (64, 128, 26, 26), M_prime: (64*128*26*26)
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 26, 26)   # y_exp: 64*64*26*26
        
        y_M = torch.cat((M, y_exp), dim=1)    # y_M: 64*192*26*26
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR
        
class GlobalDiscriminator(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, input_size=9):
        super().__init__()
        self.c0 = nn.Conv2d(local_layer_depth, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * (input_size-4) * (input_size-4) + state_dim, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, input_channels=82):
        super().__init__()
        self.c0 = nn.Conv2d(input_channels, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, state_dim = 50):
        super().__init__()
        self.l0 = nn.Linear(state_dim, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
    
    
class DeepInfoMaxLoss(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, device='cpu', alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator(local_layer_depth, state_dim)
        self.local_d = LocalDiscriminator(local_layer_depth+state_dim)
        self.prior_d = PriorDiscriminator(state_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.train()
        self.to(device)

    def forward(self, y, M, M_prime):
        """
        Inputs:
        ------
            - state
            - 
        Outputs:
        ------
        """

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        
        # y: (64, 64), M: (64, 128, 26, 26), M_prime: (64*128*26*26)
        B, C, H, W = M.size()
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, H, W)   # y_exp: 64*64*26*26
        
        y_M = torch.cat((M, y_exp), dim=1)    # y_M: 64*192*26*26
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


   
class DIM_Loss(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, device='cpu'):
        super().__init__()
        self.classifier1 = nn.Linear(state_dim, local_layer_depth)
        self.device = device
        self.train()
        self.to(device)

    def forward(self, feat_global, feat_local_map):
        sy = feat_local_map.size(2)
        sx = feat_local_map.size(3)
        N = feat_global.size(0)
        loss = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(feat_global)
                positive = feat_local_map[:, :, y, x]
                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                loss += step_loss
        loss = loss / (sx * sy)    
        return loss

class ST_DIM_Loss(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, device='cpu'):
        super().__init__()
        self.classifier1 = nn.Linear(state_dim, local_layer_depth)
        self.classifier2 = nn.Linear(local_layer_depth, local_layer_depth)
        self.device = device
        self.train()
        self.to(device)

    def forward(self, feat_local_map_t, feat_local_map_tp1, feat_global_tp1):
        sy = feat_local_map_t.size(2)
        sx = feat_local_map_t.size(3)
        N = feat_global_tp1.size(0)
        loss_global_local = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier1(feat_global_tp1)
                positive = feat_local_map_t[:, :, y, x]
                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                loss_global_local += step_loss
        loss_global_local = loss_global_local / (sx * sy)

        loss_local_local = 0.
        for y in range(sy):
            for x in range(sx):
                predictions = self.classifier2(feat_local_map_tp1[:, :, y, x])
                positive = feat_local_map_t[:, :, y, x]
                logits = torch.matmul(predictions, positive.t())
                step_loss = F.cross_entropy(logits, torch.arange(N).to(self.device))
                loss_local_local += step_loss
        loss_local_local = loss_local_local / (sx * sy)     
        loss = loss_global_local + loss_local_local
        return loss

class JSD_ST_DIM_Loss(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, alpha=0.5, beta=1.0, gamma=0.1, device='cpu'):
        super().__init__()
        self.classifier1 = nn.Linear(state_dim, local_layer_depth)
        self.classifier2 = nn.Linear(local_layer_depth, local_layer_depth)
        self.state_dim = state_dim
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device
        self.train()
        self.to(device)

    def forward(self, feat_local_map_t, feat_local_map_tp1, feat_global_tp1, feat_local_map_t_hat):
        sy = feat_local_map_t.size(1)
        sx = feat_local_map_t.size(2)
        feat_global_map_tp1 = feat_global_tp1.unsqueeze(1).unsqueeze(1).expand(-1, sy, sx, self.state_dim)

        target = torch.cat((torch.ones_like(feat_global_map_tp1[:, :, :, 0]),
                            torch.zeros_like(feat_global_map_tp1[:, :, :, 0])), dim=0).to(self.device)

        x1 = torch.cat([feat_global_map_tp1, feat_global_map_tp1], dim=0)
        x2 = torch.cat([feat_local_map_t, feat_local_map_t_hat], dim=0)
        
        shuffled_idxs = torch.randperm(len(target))
        x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
        
        loss1 = self.loss_fn(self.classifier1(x1, x2).squeeze(), target)

        x1_p = torch.cat([feat_local_map_tp1, feat_local_map_tp1], dim=0)
        x2_p = torch.cat([feat_local_map_t, feat_local_map_t_hat], dim=0)

        x1_p, x2_p = x1_p[shuffled_idxs], x2_p[shuffled_idxs]
        loss2 = self.loss_fn(self.classifier2(x1_p, x2_p).squeeze(), target)

        loss = loss1 + loss2

        return loss


class DRIML_Loss(nn.Module):
    def __init__(self, local_layer_depth=32, state_dim=50, device='cpu', envtype='discrete', action_dim = 2):
        super().__init__()
        self.classifier1 = nn.Linear(state_dim, local_layer_depth)
        self.classifier2 = nn.Linear(local_layer_depth, local_layer_depth)
        self.state_dim = state_dim
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device
        self.train()
        self.to(device)

    def forward(self, feat_local_map_t, feat_local_map_tp1, feat_global_tp1, feat_local_map_t_hat):
        sy = feat_local_map_t.size(1)
        sx = feat_local_map_t.size(2)
        feat_global_map_tp1 = feat_global_tp1.unsqueeze(1).unsqueeze(1).expand(-1, sy, sx, self.state_dim)

        target = torch.cat((torch.ones_like(feat_global_map_tp1[:, :, :, 0]),
                            torch.zeros_like(feat_global_map_tp1[:, :, :, 0])), dim=0).to(self.device)

        x1 = torch.cat([feat_global_map_tp1, feat_global_map_tp1], dim=0)
        x2 = torch.cat([feat_local_map_t, feat_local_map_t_hat], dim=0)
        
        shuffled_idxs = torch.randperm(len(target))
        x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]
        
        loss1 = self.loss_fn(self.classifier1(x1, x2).squeeze(), target)

        x1_p = torch.cat([feat_local_map_tp1, feat_local_map_tp1], dim=0)
        x2_p = torch.cat([feat_local_map_t, feat_local_map_t_hat], dim=0)

        x1_p, x2_p = x1_p[shuffled_idxs], x2_p[shuffled_idxs]
        loss2 = self.loss_fn(self.classifier2(x1_p, x2_p).squeeze(), target)

        loss = loss1 + loss2

        return loss


#####
def InfoNCE_no_action_loss(model,s_t, a_t, r_t, s_t_p_1,args,s_t_p_k=None,target=None):
    device = args['device']
    score_fn = globals()[args['score_fn']]

    local_encoder = model.local_encoder # Local encoder
    global_encoder = model.global_encoder # Global encoder

    # Extract features phi(s)
    s_t_local = local_encoder(s_t.float())
    s_t_global = global_encoder(s_t_local)
    s_t_p_1_local = local_encoder(s_t_p_1.float())
    s_t_p_1_global = global_encoder(s_t_p_1_local)
    if args['score_fn'] not in ('nce_scores_log_softmax','nce_scores_log_softmax_expanded'):
        s_t_p_k_local = local_encoder(shuffle_joint(s_t_p_1.float()))
        s_t_p_k_global = global_encoder(s_t_p_k_local)
    else:
        s_t_p_k_local = None
        s_t_p_k_global = None

    encoder_shape = list(model.psi_local_LL(s_t_p_1_local).shape)

    # Local -> Local
    if args['lambda_LL'] != 0:
        psi_local_LL_t = model.psi_local_LL
        psi_local_LL_t_p_1 = model.psi_local_LL
        nce_L_L, reg_L_L = abstract_scores_action(psi_local_LL_t,psi_local_LL_t_p_1,
                            s_t_local,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_L_L = torch.zeros(1).to(device)
        reg_L_L = torch.zeros(1).to(device)

    # Local -> Global
    if args['lambda_LG'] != 0:
        psi_local_LG = model.psi_local_LG
        psi_global_LG = model.psi_global_LG
        nce_L_G, reg_L_G = abstract_scores_action(psi_local_LG,psi_global_LG,
                            s_t_local,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_L_G = torch.zeros(1).to(device)
        reg_L_G = torch.zeros(1).to(device)

    # Global -> Local
    if args['lambda_GL'] != 0:
        psi_local_GL = model.psi_local_GL
        psi_global_GL = model.psi_global_GL
        nce_G_L, reg_G_L = abstract_scores_action(psi_global_GL,psi_local_GL,
                            s_t_global,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_G_L = torch.zeros(1).to(device)
        reg_G_L = torch.zeros(1).to(device)

    # Global -> Global
    if args['lambda_GG'] != 0:
        psi_global_GG_t = model.psi_global_GG
        psi_global_GG_t_p_1 = model.psi_global_GG
        nce_G_G, reg_G_G = abstract_scores_action(psi_global_GG_t,psi_global_GG_t_p_1,
                            s_t_global,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_G_G = torch.zeros(1).to(device)
        reg_G_G = torch.zeros(1).to(device)

    return {'nce_L_L':nce_L_L,
            'nce_L_G':nce_L_G,
            'nce_G_L':nce_G_L,
            'nce_G_G':nce_G_G,
            'reg_L_L':reg_L_L,
            'reg_L_G':reg_L_G,
            'reg_G_L':reg_G_L,
            'reg_G_G':reg_G_G
    }

def temporal_DIM_scores(reference,positive,clip_val=20):
    """
    reference: n_batch × n_rkhs × n_locs
    positive: n_batch x n_rkhs x n_locs
    """
    reference = reference.permute(2,0,1)
    positive = positive.permute(2,1,0)
    # reference: n_loc × n_batch × n_rkhs
    # positive: n_locs × n_rkhs × n_batch
    pairs = torch.matmul(reference, positive)
    # pairs: n_locs × n_batch × n_batch
    pairs = pairs / reference.shape[2]**0.5
    pairs = clip_val * torch.tanh((1. / clip_val) * pairs)
    shape = pairs.shape
    scores = F.log_softmax(pairs, 2)
    # scores: n_locs × n_batch × n_batch
    mask = torch.eye(shape[2]).unsqueeze(0).repeat(shape[0],1,1)
    # mask: n_locs × n_batch × n_batch
    scores = scores * mask
    # scores: n_locs × n_batch × n_batch
    return scores


