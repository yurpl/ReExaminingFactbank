from typing import Dict

import torch
import torch.nn as nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from jiant.modules.simple_modules import Classifier
from jiant.tasks.tasks import Task
import torch.nn.functional as F
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.stats import pearsonr, spearmanr, gmean
import numpy as np

bins = 50


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=20., gamma=1):
    loss = F.smooth_l1_loss(inputs, targets, reduce=False)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


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
        self.smoothl1loss = nn.SmoothL1Loss()
        self.span_extractor = self._make_span_extractor()
        clf_input_dim = self.span_extractor.get_output_dim()
        self.classifier = Classifier.from_params(clf_input_dim, task.n_classes, task_params)

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

        logits = raw_logits[span_mask].squeeze(dim=-1)
        labels = batch["labels"][span_mask]
        gold = labels.detach().cpu().numpy()
        value_lst, bins_edges = np.histogram(gold, bins=bins, range=(-3.0, 3.))

        def get_lds_kernel_window(kernel, ks, sigma):
            assert kernel in ['gaussian', 'triang', 'laplace']
            half_ks = (ks - 1) // 2
            if kernel == 'gaussian':
                base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
                kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
                    gaussian_filter1d(base_kernel, sigma=sigma))
                # kernel = gaussian(ks)
            elif kernel == 'triang':
                kernel_window = triang(ks)
            else:
                laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
                kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
                    map(laplace, np.arange(-half_ks, half_ks + 1)))
            return kernel_window

        def get_bin_idx(label):
            if label == 3.:
                return bins - 1
            else:
                return np.where(bins_edges > label)[0][0] - 1

        bin_index_per_label = [get_bin_idx(label) for label in gold]

        Nb = max(bin_index_per_label) + 1
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)

        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = [np.float32(1 / x) for x in eff_num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        weights = torch.tensor(weights).cuda()
        out["loss"] = weighted_focal_l1_loss(logits, labels, weights=weights)
        task.update_metrics(logits.detach().cpu().numpy(), labels.detach().cpu().numpy())

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
