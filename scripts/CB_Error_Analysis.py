import os

import pandas as pd
import scipy.stats
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import jsonlines
import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load("en_core_web_lg")


labels = {
    0: 0,
    1: 0,
    2: 0,
    3: 1
}

gold_fb = []
pred_fb = []


path = "/Users/john/PycharmProjects/summer21/scripts/MODEL_OUTPUT/FB.json"
os.chdir("/Users/john/PycharmProjects/summer21/scripts/MODEL_OUTPUT")
with jsonlines.open(path) as reader1:
    for obj in reader1:
        doc = nlp(obj['orig_text'])
        nouns = []
        for token in doc:
          if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            nouns.append(str(token))
        for t in obj["targets"]:
            if (t['span_text']) in nouns:
                gold_fb.append(np.array(t['label']))
with jsonlines.open(path) as reader:
    for obj in reader:
        doc = nlp(obj['orig_text'])
        nouns = []
        for token in doc:
          if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            nouns.append(str(token))
        for t in obj["targets"]:
            if t['span_text'] in nouns:
                pred_fb.append(np.array(t['prediction']))
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# print(f1_score(gold_fb, pred_fb, average='macro'))
# print(f1_score(gold_fb, pred_fb, average=None))
# print(precision_recall_fscore_support(gold_fb, pred_fb, average = None))


gold_cb = []
pred_cb = []

#
path = "/Users/john/PycharmProjects/summer21/scripts/MODEL_OUTPUT/FB_MTL_reg_test.json"
os.chdir("/Users/john/PycharmProjects/summer21/scripts/MODEL_OUTPUT")
with jsonlines.open(path) as reader1:
    for obj in reader1:
        doc = nlp(obj['orig_text'])
        nouns = []
        for token in doc:
          if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            nouns.append(str(token))
        for t in obj["targets"]:
            if t['span_text'] not in nouns:
                gold_cb.append((t['label']))
with jsonlines.open(path) as reader:
    for obj in reader:
        doc = nlp(obj['orig_text'])
        nouns = []
        for token in doc:
          if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
            nouns.append(str(token))
        for t in obj["targets"]:
            if t['span_text'] not in nouns:
                pred_cb.append(np.array(t['prediction']))

preds = np.concatenate((pred_fb, pred_cb), axis=None)
golds = np.concatenate((gold_fb, gold_cb), axis=None)
print(preds, golds)
print(pearsonr(golds, preds))
print(mean_absolute_error(golds, preds))

# s = []
# for i, j in zip(pred_fb, pred_cb):
#     for ii, jj in zip(i['targets'], j['targets']):
#         if ii['prediction'] != jj['prediction']:
#             s.append({'text': i['orig_text'], 'span':
#                 ii['orig_span1'], 'label':ii['label'], 'spantext':jj['span_text'], 'fb_pred': ii['prediction'], 'cb_pred': jj['prediction']})
#
# for i in s:
#     print(i)
#
# cm = cm_fb - cm_cb
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_cb,
#                              display_labels=["CT-", "PR-", "UU", "PR+", "CT+"])
# disp.plot()
# plt.title("FB+CB 5-way")
# plt.show()
