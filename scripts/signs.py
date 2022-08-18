import os
import jsonlines

# 0 neg
# 1 unknown
# 2 positive
neg_dict_three = {
    2: 0,
    1: 1,
    0: 2
}
pos_dict_three = {
    0: 2,
    1: 1,
    2: 0
}

labels = {
     3.0 : 4,
     2.0 : 3,
     1.0: 3,
     0.0 : 2,
    -1.0 : 1,
    -2.0 : 1,
    -3.0: 0,
}
three_way_labels = {
    3.0: 2,
    2.0: 1,
    1.0: 1,
    0.0: 0,
    -1.0: 1,
    -2.0: 1,
    -3.0: 2
}
neg_dict = {
    4: 0,
    3: 1,
    2: 2,
    1: 1,
    0: 0
}


pos_dict = {
    4: 4,
    3: 3,
    2: 2,
    1: 3,
    0: 4
}

neg_dict_reg = {
    3.0: -3.0,
    2.0: -2.0,
    1.0: -1.0,
    0.0: 0.0,
    -1.0: -1.0,
    -2.0: -2.0,
    -3.0: -3.0,
}



pos_dict_reg = {
    3.0: 3.0,
    2.0: 2.0,
    1.0: 1.0,
    0.0: 0.0,
    -1.0: 1.0,
    -2.0: 2.0,
    -3.0: 3.0
}

three_way_labels = {
    3.0: 2,
    2.0: 1,
    1.0: 1,
    0.0: 0,
    -1.0: 1,
    -2.0: 1,
    -3.0: 2
}







os.chdir("/Users/john/PycharmProjects/summer21/scripts/MODEL_OUTPUT")
corrections = 0
path = "FB_ldc_five_test.json"
import numpy as np
true = []
pred = []
with jsonlines.open(path) as reader1:
    for obj in reader1:
        for t in obj["targets"]:
            true.append(t['label'])
with jsonlines.open(path) as reader:
    for obj in reader:
        for t in obj["targets"]:
            pred.append(np.array(t['prediction']).argmax(-1))

print("Error previous: ", np.sum(np.array(true) != np.array(pred)))

counter = 0
with jsonlines.open('ALL_POL_det_test.json') as reader, jsonlines.open(path) as reader2:
    ls = []
    for obj, obj2 in zip(reader, reader2):
        counter += 1
        for t1, t2 in zip(obj['targets'], obj2['targets']):
            t1['prediction'] = np.array(t1['prediction']).argmax(-1)
            t2['prediction'] = np.array(t2['prediction']).argmax(-1)

            if t1['prediction'] == 1: #negative
                ls.append(neg_dict[t2['prediction']])
            elif t1['prediction'] == 0: #UU aka 0

                ls.append(2)

            elif t1['prediction'] == 2: #Positive
                ls.append(pos_dict[t2['prediction']])


gold = []
with jsonlines.open(path) as reader1:
    for obj in reader1:
        for t in obj["targets"]:
            gold.append(t['label'])
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, f1_score

print(f1_score(gold, ls,average="macro"))
print(f1_score(gold, ls,average=None))
