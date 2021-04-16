# This function modified from https://github.com/pangshumao/SpineParseNet/blob/master/augment/transforms.py
def Normalize(img, mean, std, eps=1e-4):
    """
    Normalizes a given input tensor to be 0-mean and 1-std.
    mean and std parameter have to be provided explicitly.
    """

    return (img - mean) / (std + eps)
