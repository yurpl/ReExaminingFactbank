import os
import jsonlines
import numpy as np
from nltk.tokenize import SpaceTokenizer
os.chdir("../factuality_bert/glue_data/ST_GE")
clf = {
    2.0: 2,
    1.0: 1,
    0.0: 0,
    3.0: 3,
    4.0: 4,
    5.0: 5,
    6.0: 6
}
labs = []
spans = []
tk = SpaceTokenizer()

# n = [0,1,2,3,4,5,6,7,8,9,'.','$']
with jsonlines.open('train.jsonl') as reader, jsonlines.open("train2.jsonl", 'w') as writer:
    counter = 0
    for obj in reader:
        txt = []
        ls = []
        for idx, t in enumerate(tk.tokenize(obj['text'])):
            for i in obj['targets']:
                if type(i['label']) is float:
                    i['label'] = clf[i['label']]
                    ls.append(i)
                    txt.append(i['span_text'])
            if not any(t in s for s in txt) and idx != len(tk.tokenize(obj['text'])) - 1:
                ls.append({"span1": [idx, idx+1], "label": 4, "span_text": t})


        obj.pop('targets')
        obj['targets'] = ls
        if obj:

            obj["idx"] = counter
            obj['file_idx'] = None
            #del obj["file_idx"]
            writer.write(obj)
            counter += 1
