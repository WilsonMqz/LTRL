import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def forward_loss(f, K, labels, device):
    Q = torch.ones(K, K) * 1 / (K - 1)
    Q = Q.to(device)
    for k in range(K):
        Q[k, k] = 0
    q = torch.mm(F.softmax(f, 1), Q)
    return F.nll_loss(q.log(), labels.long())


def pc_loss(f, K, labels, device):
    sigmoid = nn.Sigmoid()
    fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
    loss_matrix = sigmoid(-1. * (f - fbar))  # multiply -1 for "complementary"
    M1, M2 = K * (K - 1) / 2, K - 1
    pc_loss = torch.sum(loss_matrix) * (K - 1) / len(labels) - M1 + M2
    return pc_loss


def non_k_softmax_loss(f, K, labels, device):
    Q_1 = 1 - F.softmax(f, 1)
    Q_1 = F.softmax(Q_1, 1)
    labels = labels.long()
    return F.nll_loss(Q_1.log(), labels.long())


def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = -torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss


def rc_loss(outputs, confidence, index):
    logsm_outputs = F.log_softmax(outputs, dim=1)
    final_outputs = logsm_outputs * confidence[index, :]
    average_loss = -((final_outputs).sum(dim=1)).mean()
    return average_loss


def log_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = - ((k - 1) / (k - can_num) * torch.log(final_outputs.sum(dim=1))).mean()
    return average_loss


def exp_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n

    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = ((k - 1) / (k - can_num) * torch.exp(-final_outputs.sum(dim=1))).mean()
    return average_loss


def c_loss(output, label, K, device):
    loss = nn.MSELoss(reduction='mean')
    one_hot = F.one_hot(label.to(torch.int64), K) * 2 - 1
    sig_out = output * one_hot
    y_label = torch.ones(sig_out.size())
    y_label = y_label.to(device)
    output = loss(sig_out, y_label)
    return output


def mse_loss(outputs, Y):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.MSELoss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, Y.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def mae_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    loss_fn = nn.L1Loss(reduction='none')
    loss_matrix = loss_fn(sm_outputs, partialY.float())
    sample_loss = loss_matrix.sum(dim=-1)
    return sample_loss


def ce_loss(outputs, Y):
    logsm = nn.LogSoftmax(dim=1)
    logsm_outputs = logsm(outputs)
    final_outputs = logsm_outputs * Y
    sample_loss = - final_outputs.sum(dim=1)
    return sample_loss


def gce_loss(outputs, Y):
    q = 0.7
    sm_outputs = F.softmax(outputs, dim=1)
    pow_outputs = torch.pow(sm_outputs, q)
    sample_loss = (1 - (pow_outputs * Y).sum(dim=1)) / q  # n
    return sample_loss


def c_TF_loss(output_F, target_SL_F, K, device):
    target_SL_F = target_SL_F.T
    A_num = target_SL_F.size()[0]
    loss = 0
    for i in range(K):
        label_i = torch.zeros(target_SL_F.size()[1]) + i
        loss += c_loss(output_F, label_i, K, device)
    for i in range(A_num):
        loss -= c_loss(output_F, target_SL_F[i], K, device)
    return loss


def SL_loss(output, target_TF, target_SL, K, model, images, alpha=0):
    device = output.device
    index_T = []
    index_F = []
    for i in range(target_TF.size()[0]):
        if target_TF[i] == K:
            index_F.append(i)
        else:
            index_T.append(i)
    index_F = torch.tensor(index_F).long()
    index_T = torch.tensor(index_T).long()
    index_T = index_T.to(device)
    index_F = index_F.to(device)

    image_T = torch.index_select(images, dim=0, index=index_T)
    target_TL = torch.index_select(target_TF, dim=0, index=index_T)
    output_TL = torch.index_select(output, dim=0, index=index_T)
    target_SL = torch.index_select(target_SL, dim=0, index=index_F)
    output_SL = torch.index_select(output, dim=0, index=index_F)

    partialY = 1 - target_SL
    can_num = partialY.sum(dim=1).float()  # n
    can_num = 1.0 / can_num

    criterion = nn.CrossEntropyLoss().to(device)
    target_TL_loss = criterion(output_TL, target_TL)
    target_SL_loss = can_num * gce_loss(output_SL, partialY)

    target_SL_loss = target_SL_loss.sum() / partialY.shape[0]

    if alpha == 0:
        loss = target_TL_loss + target_SL_loss
    elif alpha == 1 or alpha == 2:
        loss_mix_up = mix_up(model, criterion, image_T, target_TL, alpha)
        loss = target_TL_loss + target_SL_loss + loss_mix_up
    elif alpha == 3 or alpha == 4:
        loss_mix_up = manifold_mix_up(model, criterion, image_T, target_TL, alpha-2)
        loss = target_TL_loss + target_SL_loss + loss_mix_up
    return loss


def mix_up(model, criterion, image, label, alpha=1.0):
    r"""
    References:
        Zhang et al., mixup: Beyond Empirical Risk Minimization, ICLR
    """
    device = image.device
    alpha = float(alpha)
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0))
    image_a, image_b = image, image[idx]
    label_a, label_b = label, label[idx]
    mixed_image = l * image_a + (1 - l) * image_b
    label_a = label_a.to(device)
    label_b = label_b.to(device)
    mixed_image = mixed_image.to(device)

    output = model(mixed_image)
    loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

    return loss


def manifold_mix_up(model, criterion, image, label, alpha):
    r"""
    References:
        Verma et al., Manifold Mixup: Better Representations by Interpolating Hidden States, ICML 2019.

    Specially, we apply manifold mixup on only one layer in our experiments.
    The layer is assigned by param ``self.manifold_mix_up_location''
    """
    device = image.device
    alpha = float(alpha)
    l = np.random.beta(alpha, alpha)
    idx = torch.randperm(image.size(0))
    label_a, label_b = label, label[idx]
    label_a = label_a.to(device)
    label_b = label_b.to(device)
    image = image.to(device)
    output = model(image)
    loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
    return loss


def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
