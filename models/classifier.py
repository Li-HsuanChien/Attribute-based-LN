import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self,input_dim, config):
        super(Classifier, self).__init__()
        self.pre_classifier = nn.Linear(input_dim, config.pre_classifier_dim)
        self.classifier = nn.Linear(config.pre_classifier_dim, config.num_labels) # weight: (dim, classes)

    def forward(self, hidden):
        pre_hidden = torch.tanh(self.pre_classifier(hidden))
        logits = self.classifier(pre_hidden)
        return logits, pre_hidden, torch.transpose(self.classifier.weight, 0, 1)# logits, pre_hidden, label_embedding: (classes, dim)

class MLP(nn.Module):
    def __init__(self, input_dim, pre_classifier_dim, num_classes):
        super(MLP, self).__init__()
        self.depth_perceptron = 2
        self.pre_classifier = nn.Linear(input_dim, pre_classifier_dim)
        if self.depth_perceptron > 1:
            self.pre_classifiers = nn.ModuleList(
                [nn.Linear(pre_classifier_dim, pre_classifier_dim) for _ in range(1, self.depth_perceptron)]
            )
        self.classifier = nn.Linear(pre_classifier_dim, num_classes)

    def forward(self, usr, prd):
        pre_hidden = torch.tanh(self.pre_classifier(torch.cat([usr, prd], -1)))
        if self.depth_perceptron > 1:
            for pre_classifier in self.pre_classifiers:
                pre_hidden = torch.tanh((pre_classifier(pre_hidden)))
        logits = self.classifier(pre_hidden)
        return logits, pre_hidden


