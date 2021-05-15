import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from utils import FunctionNegativeTripletSelector, random_hard_negative
import math
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
       # anchor1=[]
        #positive1=[]
       # negative1=[]
      #   for i in range(0, 32):
       #     anchor[i] /= (anchor.pow(2).sum(1).pow(.5))[i]
       #     positive[i]/=(positive.pow(2).sum(1).pow(.5))[i]
       #     negative[i]/=(negative.pow(2).sum(1).pow(.5))[i]
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        #pred = (distance_negative - distance_positive).cpu().data
        #acc = (pred > 0).sum() * 1.0 / distance_positive.size()[0]
        return losses.mean()# acc

class Accuracy(nn.Module):
    def __init__(self, mar):
        super(Accuracy, self).__init__()
        self.mar = mar
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        pred = (distance_negative - distance_positive-self.mar).cpu().data.double()
        acc = (pred > 0).sum().double()/ distance_positive.size()[0]
        return acc

class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin1, margin2, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.triplet_selector = triplet_selector    #FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=True)

    def forward(self, embeddings, confidence, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        confidence = F.softmax(confidence, dim=1)
        #weight_relation = torch.zeros([len(triplets)], dtype=torch.int64)
        #weight_negative = torch.zeros([len(triplets)], dtype=torch.int64)
        weight_relation = torch.exp(confidence[triplets[:, 0], target[triplets[:, 2]]], out=None) * torch.exp(confidence[triplets[:, 2], target[triplets[:, 0]]], out=None)
        weight_negative = torch.exp(confidence[triplets[:, 0], target[triplets[:, 3]]], out=None) * torch.exp(confidence[triplets[:, 3], target[triplets[:, 0]]], out=None)
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        ar_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 3]]).pow(2).sum(1)
        losses = F.relu(ap_distances - ar_distances + weight_relation * self.margin1) + F.relu(ar_distances - an_distances + weight_negative *self.margin2)

        return losses.mean(), len(triplets)
