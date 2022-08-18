from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
import numpy as np
from jiant.modules.simple_modules import Classifier
from jiant.tasks.tasks import Task

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
#weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class FactualityModule(nn.Module):
    def _make_span_extractor(self):
        if self.span_pooling == "attn":
            return SelfAttentiveSpanExtractor(self.d_inp)
        else:
            return EndpointSpanExtractor(self.d_inp, combination=self.span_pooling)

    def __init__(self, task, d_inp, task_params):
        super(FactualityModule, self).__init__()
        self.task = task
        self.span_pooling = task_params["cls_span_pooling"]
        # input dimension of task_specific modules, defined in models.py L:291
        self.d_inp = d_inp

        self.span_extractor = self._make_span_extractor()
        clf_input_dim = self.span_extractor.get_output_dim()
        self.classifier = Classifier.from_params(clf_input_dim, 3, task_params)
        self.smoothl1loss = FocalLoss()


    def get_raw_logits(self, batch: Dict, sent_embs: torch.Tensor, sent_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass and return only raw logits, for running IntegratedGradients."""
        span_mask = batch["span1s"][:, :, 0] != -1
        spans_embs = self.span_extractor(sent_embs, batch["span1s"],
                                         sequence_mask=sent_mask.long(),
                                         span_indices_mask=span_mask.long())

        # [batch_size, n_targets_per_sent, 1]
        raw_logits = self.classifier(spans_embs)
        return raw_logits


    def forward(self,
                batch: Dict,
                sent_embs: torch.Tensor,
                sent_mask: torch.Tensor,
                task: Task,
                predict: bool
                ) -> Dict:
        """
        Run forward pass.
        :param batch: Dict. Expects it to have the following entries:
            'input1' : [batch_size, max_len] xxx
            'labels' : [batch_size, n_targets_per_sent] of label indices
            'span1s' : [batch_size, n_targets_per_sent, 2], span indices
        :param sent_embs:  [batch_size, max_len, d_inp]
        :param sent_mask: [batch_size, max_len, 1]
        :param task: Task object
        :param predict:
        """
        batch_size = sent_embs.shape[0]
        n_targets_per_sent = batch["labels"].shape[1]
        span_mask = batch["span1s"][:, :, 0] != -1
        # total number targets in the batch, sum of number of targets in each item in the batch
        n_targets_total = span_mask.sum()
        out = {"preds": [], "logits": [],
               "n_inputs": batch_size,
               "n_targets": n_targets_total,
               "n_exs": n_targets_total,
               "mask": span_mask}
        spans_embs = self.span_extractor(sent_embs, batch["span1s"],
                                         sequence_mask=sent_mask.long(),
                                         span_indices_mask=span_mask.long())

        # [batch_size, n_targets_per_sent, 1]
        raw_logits = self.classifier(spans_embs)

        # Flatten logits and labels to have shape [n_targets_total]
        logits = raw_logits[span_mask]
        labels = batch["labels"][span_mask].long()
        gold = labels.detach().cpu().numpy()
        # if len(np.bincount(gold) == 3):
        #     weights = (1/np.bincount(gold))
        #     weights[1] = 1
        #     weights = np.float32(weights)
        #     weights = torch.tensor(weights).cuda()
        # else:
        #     weights = torch.tensor(np.float32(np.array([1,1,1]))).cuda()
        # self.smoothl1loss.weight = weights
        out["loss"] = self.smoothl1loss(logits, labels)
        task.update_metrics(logits.detach().cpu().numpy().argmax(-1), labels.detach().cpu().numpy())
        if predict:
            out["preds"] = list(self.unbind_predictions(raw_logits, span_mask))

        return out

    def unbind_predictions(self, preds: torch.Tensor, masks: torch.Tensor):
        """ Unpack preds to varying-length numpy arrays.

        Args:
            preds: [batch_size, num_targets, ...]
            masks: [batch_size, num_targets] boolean mask

        Yields:
            np.ndarray for each row of preds, selected by the corresponding row
            of span_mask.
        """
        preds = preds.detach().cpu()
        masks = masks.detach().cpu()
        for pred, mask in zip(torch.unbind(preds, dim=0), torch.unbind(masks, dim=0)):
            yield pred[mask].squeeze(dim=-1).numpy()  # only non-masked predictions
