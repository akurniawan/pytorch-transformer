def label_smoothing(labels, eps=0.1):
    C = labels.size(1)
    return ((1 - eps) * labels) + (eps / C)

