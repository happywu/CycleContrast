import torch
import torch.nn as nn
import torch.nn.functional as F
import cycle_contrast.resnet as models


class CycleContrast(nn.Module):
    """
    Build a CycleContrast model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/2105.06463
    """
    def __init__(self, arch, dim=128,
                 K=65536, m=0.999,
                 T=0.07, mlp=False,
                 soft_nn=False, soft_nn_support=-1,
                 sep_head=False, cycle_neg_only=True,
                 soft_nn_T=-1.,
                 cycle_back_cls=False,
                 cycle_back_cls_video_as_pos=False,
                 cycle_K=None,
                 moco_random_video_frame_as_pos=False
                 ):
        super(CycleContrast, self).__init__()

        self.K = K
        if cycle_K is not None:
            self.cycle_K = cycle_K
        else:
            self.cycle_K = K
        self.m = m
        self.T = T

        if soft_nn_T != -1:
            self.soft_nn_T = soft_nn_T
        else:
            self.soft_nn_T = T

        self.soft_nn = soft_nn
        self.soft_nn_support = soft_nn_support
        self.sep_head = sep_head
        self.cycle_back_cls = cycle_back_cls
        self.cycle_neg_only = cycle_neg_only
        self.cycle_back_cls_video_as_pos = cycle_back_cls_video_as_pos
        self.moco_random_video_frame_as_pos = moco_random_video_frame_as_pos

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = models.__dict__[arch](num_classes=dim,
                                               return_inter=True)
        self.encoder_k = models.__dict__[arch](num_classes=dim,
                                               return_inter=sep_head)

        # sep head
        if self.sep_head:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.q_cycle_fc = nn.Linear(dim_mlp, dim)
            self.k_cycle_fc = nn.Linear(dim_mlp, dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

            if self.sep_head:
                dim_mlp = self.q_cycle_fc.weight.shape[1]
                self.q_cycle_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.q_cycle_fc)
                self.k_cycle_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.k_cycle_fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.sep_head:
            for param_q, param_k in zip(self.q_cycle_fc.parameters(), self.k_cycle_fc.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        if self.sep_head:
            self.register_buffer("queue_cycle", torch.randn(dim, self.cycle_K))
            self.queue_cycle = nn.functional.normalize(self.queue_cycle, dim=0)
            self.register_buffer("queue_cycle_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_labels", torch.zeros(1, K, dtype=torch.long))

        self.register_buffer("queue_indices", torch.zeros(1, K, dtype=torch.long))

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self.sep_head:
            for param_q, param_k in zip(self.q_cycle_fc.parameters(), self.k_cycle_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_cycle=None, labels=None, indices=None, cluster=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        if keys_cycle is not None:
            keys_cycle = concat_all_gather(keys_cycle)
        if labels is not None:
            labels = concat_all_gather(labels)
        if indices is not None:
            indices = concat_all_gather(indices)
        if cluster is not None:
            cluster = concat_all_gather(cluster)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        # if some gpu of the machine is dead, K might not able to be divided by batch size
        assert self.K % batch_size == 0, 'K {} can not be divided by batch size {}'.format(self.K, batch_size)  # for simplicity

        if self.sep_head:
            cycle_ptr = int(self.queue_cycle_ptr[0])
            # if some gpu of the machine is dead, K might not able to be divided by batch size
            assert self.cycle_K % batch_size == 0, \
                'K {} can not be divided by batch size {}'.format(self.cycle_K,
                                                                  batch_size)  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        try:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        except Exception:
            print('enqueue size', ptr, batch_size, self.K)
            enqueue_size = min(self.K-ptr, batch_size)
            self.queue[:, ptr:ptr + batch_size] = keys[:enqueue_size].T
        try:
            if self.sep_head and keys_cycle is not None:
                self.queue_cycle[:, cycle_ptr:cycle_ptr + batch_size] = keys_cycle.T
        except Exception:
            print('enqueue size', ptr, keys_cycle.shape[0], self.K)
            enqueue_size = min(self.K - ptr, keys_cycle.shape[0])
            self.queue_cycle[:, ptr:ptr + keys_cycle.shape[0]] = keys_cycle[:enqueue_size].T
        try:
            if labels is not None:
                self.queue_labels[:, ptr:ptr + batch_size] = labels
            if indices is not None:
                self.queue_indices[:, ptr:ptr + batch_size] = indices
            if cluster is not None:
                self.queue_cluster[:, ptr:ptr + batch_size] = cluster.T
        except Exception:
            enqueue_size = min(self.K-ptr, batch_size)
            if labels is not None:
                self.queue_labels[:, ptr:ptr + batch_size] = labels[:enqueue_size]
            if indices is not None:
                self.queue_indices[:, ptr:ptr + batch_size] = indices[:enqueue_size]
            if cluster is not None:
                self.queue_cluster[:, ptr:ptr + batch_size] = cluster[:enqueue_size].T

        ptr = (ptr + batch_size) % self.K  # move pointer
        assert ptr < self.K, 'ptr: {}, batch_size: {}, K: {}'.format(ptr, batch_size, self.K)

        self.queue_ptr[0] = ptr

        if self.sep_head:
            cycle_ptr = (cycle_ptr + batch_size) % self.cycle_K  # move pointer
            assert cycle_ptr < self.cycle_K, 'cycle ptr: {}, batch_size: {}, cycle K: {}'.format(
                cycle_ptr, batch_size, self.cycle_K)

            self.queue_cycle_ptr[0] = cycle_ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None,
                cls_labels=None, indices=None,
                cls_candidates=None
                ):

        # compute query features
        unnormalize_q = self.encoder_q(im_q)  # queries: NxC
        unnormalize_q, q_avgpool = unnormalize_q
        if self.sep_head:
            q_cycle = self.q_cycle_fc(q_avgpool)
            q_cycle = nn.functional.normalize(q_cycle, dim=1)
        q = nn.functional.normalize(unnormalize_q, dim=1)

        meta = {}

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if (self.cycle_back_cls or self.moco_random_video_frame_as_pos) and cls_candidates is not None:
                n, k_can = cls_candidates.shape[:2]
                concat_im = torch.cat((im_k, cls_candidates.flatten(0, 1)), dim=0)
                im_k, idx_unshuffle = self._batch_shuffle_ddp(concat_im)
                k = self.encoder_k(im_k)  # keys: NxC
                if self.sep_head:
                    k, k_avgpool = k
                    k_cycle = self.k_cycle_fc(k_avgpool)
                    k_cycle = nn.functional.normalize(k_cycle, dim=1)
                    k_cycle = self._batch_unshuffle_ddp(k_cycle, idx_unshuffle)
                    k_cycle, can_cycle = torch.split(k_cycle, [n, n * k_can], dim=0)
                    can_cycle = can_cycle.view(n, k_can, -1)
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k, can = torch.split(k, [n, n * k_can], dim=0)
                can = can.view(n, k_can, -1)
            else:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k)  # keys: NxC
                if self.sep_head:
                    k, k_avgpool = k
                    k_cycle = self.k_cycle_fc(k_avgpool)
                    k_cycle = nn.functional.normalize(k_cycle, dim=1)
                    k_cycle = self._batch_unshuffle_ddp(k_cycle, idx_unshuffle)
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        if self.sep_head:
            l_pos_cycle = torch.einsum('nc,nc->n', [q_cycle, k_cycle]).unsqueeze(-1)
            # negative logits: NxK
            l_neg_cycle = torch.einsum('nc,ck->nk', [q_cycle, self.queue_cycle.clone().detach()])
        else:
            l_pos_cycle = l_pos
            l_neg_cycle = l_neg

        if self.soft_nn:
            if self.cycle_neg_only:
                nn_embs, perm_indices = \
                    self.cycle_back_queue_without_self(l_neg_cycle, self.queue_cycle \
                        if self.sep_head else self.queue)
            else:
                pos_neigbor = q_cycle if self.sep_head else q
                nn_embs, perm_indices = \
                    self.cycle_back_queue_with_self(l_pos_cycle, l_neg_cycle,
                                                    pos_neigbor,
                                                    self.queue_cycle if self.sep_head else self.queue,
                                                    meta=meta,
                                                    q=q, k=k
                                                    )
            nn_embs = F.normalize(nn_embs, dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if self.cycle_back_cls:
            assert cls_candidates is not None, "cls candidates can not be None"
            back_logits, cycle_back_labels = \
                self.get_logits_cycle_back_cls_video_as_pos(nn_embs,
                                                            can_cycle if self.sep_head else can,
                                                            self.queue_cycle if self.sep_head else self.queue,
                                                            perm_indices,
                                                            indices=indices)

        # random sample one frame from the same video of im_q / im_k as positive, i.e. intra-video objective
        if self.moco_random_video_frame_as_pos:
            #  can: n x k x 128
            pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
            pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])
            logits_neg = torch.matmul(q, self.queue.clone().detach())
            logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)
            logits = torch.cat([logits_pos, logits_neg], dim=1)
            logits /= self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k,
                                  keys_cycle=k_cycle if self.sep_head else None,
                                  labels=cls_labels, indices=indices,
                                  )

        if self.soft_nn:
            return logits, labels, back_logits, cycle_back_labels, meta
        else:
            return logits, labels, meta

    def cycle_back_queue_without_self(self, l_neg_cycle, queue_cycle):
        if self.soft_nn_support != -1:
            perm_indices = torch.randperm(self.cycle_K if self.sep_head else self.K)
            sampled_indices = perm_indices[:self.soft_nn_support]
            sampled_l_neg = l_neg_cycle[:, sampled_indices]
            weights = nn.functional.softmax(sampled_l_neg / self.soft_nn_T, dim=1)
            nn_embs = torch.matmul(weights, queue_cycle.clone().detach().transpose(1, 0)[sampled_indices])
        else:
            weights = nn.functional.softmax(l_neg_cycle / self.soft_nn_T, dim=1)
            nn_embs = torch.matmul(weights, queue_cycle.clone().detach().transpose(1, 0))
            perm_indices = None
        return nn_embs, perm_indices

    def cycle_back_queue_with_self(self, l_pos_cycle, l_neg_cycle, q_cycle, queue_cycle, meta=None, q=None, k=None):
        if self.soft_nn_support != -1:
            perm_indices = torch.randperm(self.cycle_K if self.sep_head else self.K)
            sampled_indices = perm_indices[:self.soft_nn_support]
            sampled_l_neg = l_neg_cycle[:, sampled_indices]
            logits_cycle = torch.cat([l_pos_cycle, sampled_l_neg], dim=1)
        else:
            logits_cycle = torch.cat([l_pos_cycle, l_neg_cycle], dim=1)
            perm_indices = None

        weights = nn.functional.softmax(logits_cycle / self.soft_nn_T, dim=1)
        num_neg = self.soft_nn_support if self.soft_nn_support != -1 \
            else (self.cycle_K if self.sep_head else self.K)
        weights_pos, weights_neg = torch.split(weights, [1, num_neg], dim=1)
        nn_embs_pos = weights_pos * q_cycle
        if self.soft_nn_support != -1:
            nn_embs_neg = torch.matmul(weights_neg, queue_cycle.clone().detach().transpose(1, 0)[sampled_indices])
        else:
            nn_embs_neg = torch.matmul(weights_neg, queue_cycle.clone().detach().transpose(1, 0))
        nn_embs = nn_embs_pos + nn_embs_neg  #
        return nn_embs, perm_indices

    def get_logits_cycle_back_cls_video_as_pos(self, nn_embs, can_cycle, queue_cycle, perm_indices, indices=None):
        if self.soft_nn_support == -1:
            back_logits_neg = torch.matmul(nn_embs, queue_cycle.clone().detach())
        else:
            remain_indices = perm_indices[self.soft_nn_support:]
            back_logits_neg = torch.matmul(nn_embs, queue_cycle.clone().detach()[:, remain_indices])
        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



