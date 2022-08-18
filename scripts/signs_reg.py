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
    2: 4,
    1: 3,
    0: 2
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
'''
with jsonlines.open('POL_test.json') as reader, jsonlines.open("LDCCB_test.json", 'r') as reader2, jsonlines.open("signs.jsonl", 'w') as writer:
    for obj, obj2 in zip(reader, reader2):
        for t1, t2 in zip(obj['targets'], obj2['targets']):
            if t1['prediction'] == 0: #negative
                if t2['label'] < 0.0:
                    t2['label'] = t2['label']
                else:
                    t2['label'] = - t2['label']
                    print(t2['label'])
            if t1['prediction'] == 1: #UU aka 0
                t2['label'] = 0.0
            elif t1['prediction'] == 2: #Positive
                t2['label'] = abs(t2['label'])
        writer.write(obj2)
'''
path = 'FB_CB_MV_reg_test.json'
counter = 0
import math
with jsonlines.open('FB_AOAS_one_reg_test.json') as reader, jsonlines.open(path) as reader2:
    ls = []
    for obj, obj2 in zip(reader, reader2):
        counter += 1
        for t1, t2 in zip(obj['targets'], obj2['targets']):
            if round(t1['prediction']) < 0: #negative
                if t2['prediction'] < 0:
                    ls.append(t2['prediction'])
                else:
                    ls.append(-t2['prediction'])

            elif round(t1['prediction']) == 0.: #UU aka 0
                ls.append(0.0)

            elif (t1['prediction']) > 0: #Positive

                ls.append(abs(t2['prediction']))
gold = []
with jsonlines.open(path) as reader1:
    for obj in reader1:
        for t in obj["targets"]:
            gold.append(t['label'])
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
print("Correlation: ", pearsonr(gold, ls)[0])
print("MAE: ", mean_absolute_error(gold, ls))
