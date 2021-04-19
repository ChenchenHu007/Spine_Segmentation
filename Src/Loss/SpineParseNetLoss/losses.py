"""
copied from
https://github.com/pangshumao/SpineParseNet/blob/e069246e4e430d6e5bc73112f9eaedbde0555f6c/networks/losses.py
"""

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss


def compute_per_channel_dice(input_, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a predicted mask

    # input and target shapes must match

    assert input_.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input_ = input_ * mask
        target = target * mask

    input_ = flatten(input_)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input_ * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input_ + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    input_ : (b, num_classes, D, H, W)
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input_, target):  # input_:()
        # get probabilities from logits
        input_ = self.normalization(input_)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]
        target = expand_as_one_hot(target, num_channels=input_.size()[1], ignore_index=self.ignore_index)

        per_channel_dice = compute_per_channel_dice(input_, target, epsilon=self.epsilon,
                                                    ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)


class FPFNLoss(nn.Module):
    def __init__(self, lamda=0.1):
        super(FPFNLoss, self).__init__()
        self.lamda = lamda
        self.activation = nn.Softmax(dim=1)

    def forward(self, input_, target, weights=None):
        target = expand_as_one_hot(target, num_channels=input_.size()[1])  # batch, C, d, h, w
        input_ = self.activation(input_)
        if weights is not None:
            weights = torch.unsqueeze(weights, dim=1)
            weights = Variable(weights, requires_grad=False)
        else:
            weights = 1.0
        fp = torch.sum(weights * (1 - target) * input_, dim=(1, 2, 3, 4))  # batch
        fn = torch.sum(weights * (1 - input_) * target, dim=(1, 2, 3, 4))  # batch

        loss = self.lamda * fp + fn
        return torch.mean(loss)


class LSLoss(nn.Module):
    """Computes Level Set Loss.
    """

    def __init__(self, epsilon=0.05, lamda=4e-4):
        super(LSLoss, self).__init__()
        self.epsilon = epsilon
        self.lamda = lamda
        self.activation = nn.Softmax(dim=1)
        self.ceLoss = nn.CrossEntropyLoss()

    def forward(self, input_, target):
        eps = 1e-8
        ceLoss = self.ceLoss(input_, target)
        target = expand_as_one_hot(target, num_channels=input_.size()[1])  # batch, C, d, h, w

        target = self.activation(target)
        sdm = target - 0.5
        h = 0.5 * (1. + torch.tanh(sdm / self.epsilon))
        h_sum = torch.sum(h, dim=(2, 3, 4)) + eps  # batch, C
        h_sum_ = torch.sum(1. - h, dim=(2, 3, 4)) + eps  # batch, C
        cin = torch.div(torch.sum(torch.mul(target, h), dim=(2, 3, 4)), h_sum)  # batch, C
        cout = torch.div(torch.sum(torch.mul(target, 1. - h), dim=(2, 3, 4)), h_sum_)  # batch, C

        cin = cin.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(target.size())
        cout = cout.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(target.size())

        inLoss = torch.sum(torch.mul(torch.pow(target - cin, 2), h), dim=(2, 3, 4))  # batch, C
        outLoss = torch.sum(torch.mul(torch.pow(target - cout, 2), 1 - h), dim=(2, 3, 4))  # batch, C
        lsLoss = torch.mean(inLoss + outLoss)
        loss = ceLoss + self.lamda * lsLoss

        return loss


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input_, target):
        # get probabilities from logits
        input_ = self.normalization(input_)

        assert input_.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input_ = input_ * mask
            target = target * mask

        input_ = flatten(input_)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input_ * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input_ + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)


class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input_, target):
        class_weights = self._class_weights(input_)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input_, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input_):
        # normalize the input first
        input_ = F.softmax(input_, _stacklevel=5)
        flattened = flatten(input_)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class BCELossWrapper:
    """
    Wrapper around BCE loss functions allowing to pass 'ignore_index' as well as 'skip_last_target' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input_, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]

        assert input_.size() == target.size()

        masked_input = input_
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            masked_input = input_ * mask
            masked_target = target * mask

        return self.loss_criterion(masked_input, masked_target)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, target, weights):
        """
        :param input_: tensor with shape of [N, C, D, H, W]
        :param target: tensor with shape of [N, D, H, W]
        :param weights: tensor with shape of [N, D, H, W]
        :return:
        """
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input_)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, num_channels=input_.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(1)  # [N, 1, D, H, W]
        weights = weights.expand_as(input_)  # [N, C, D, H, W]

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input_.size()[1]).float().to(input_.device)
            self.register_buffer('class_weights', class_weights)

        # resize class_weights to be broadcastable into the weights
        # class_weights = self.class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        # weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class MSEWithLogitsLoss(MSELoss):
    """
    This loss combines a `Sigmoid` layer and the `MSELoss` in one single class.
    """

    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_, target):
        return super().forward(self.sigmoid(input_), target)


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss


def square_angular_loss(input_, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.
    :param input_: 5D input tensor (N,C, D, H, W)
    :param target: 5D target tensor (N, C, D, H, W)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input_.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input_ = input_ / torch.norm(input_, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input_ * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def expand_as_one_hot(input_, num_channels, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW or NxHxW label image to NxCxHxW, where each label gets converted to
    its corresponding one-hot vector
    :param input_: 4D input image (NxDxHxW) or 3D input image (NxHxW)
    :param num_channels: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW) or 4D output image (NxCxHxW)
    """
    assert input_.dim() in [3, 4]

    shape = input_.size()
    shape = list(shape)
    shape.insert(1, num_channels)
    shape = tuple(shape)

    # expand the input tensor to 1xNxDxHxW
    # index = input.unsqueeze(0)

    # expand the input tensor to Nx1xDxHxW
    index = input_.unsqueeze(1)

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_index = index.expand(shape)
        mask = expanded_index == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        index = index.clone()
        index[index == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input_.device).scatter_(1, index, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input_.device).scatter_(1, index, 1)
