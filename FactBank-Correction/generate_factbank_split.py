import pandas as pd
import json
import jsonlines
import numpy as np
from ast import literal_eval

# Data Generation for Self Attentive Span Extractor
# input: filename - csv file in factbank labels format
# output: jsonl file of texts and spans

s = 6
np.random.seed(s)
print('seed = ', s)
test = []

labels_dict = {
    'CT+': 3.0,
    'PR+': 2.0,
    'PS+': 1.0,
    'Uu': 0.0,
    'PS-': -1.0,
    'PR-': -2.0,
    'CT-': -3.0,
    'CTu': 0.0,
    'NA': 0.0
}

### input: csv file in factbank labels format
### creates: jsonl file of all factbank sentences
### returns: counts of sentences

def extract_jsonl(filename: str):
    tokens = pd.read_csv(filename)
    sentence_count = 0 
    # list of dictionary of list of dictionaries:
    # {text, targets}
    # text: sentence
    # targets: [{span1, label, span_text}]
    texts = []

    # One pass through the data, get text and targets
    i = 0
    # while i < len(tokens):
    articles = set()
    while i < len(tokens):
        # article = str(tokens['file'][i])
        # print(article)
        # Get first word in sentence:
        sentence = [str(tokens['text'][i])]

        # Create targets list
        targets = []
        
        # Has to compare to "'Event'" because of the way the Tags are written in the CSV
        if tokens["tmlTag"][i] == "'EVENT'" and type(tokens["factValue"][i]) == str:
            #tokens['text'][i] = literal_eval(tokens['text'][i])
            #(tokens['text'][i]) = "\"* " + tokens['text'][i] + "\""
            targets_dict = {}
            targets_dict['span1'] = [int(tokens['tokLoc'][i]), int(tokens['tokLoc'][i]) + 1]
            targets_dict['label'] = labels_dict[str(tokens['factValue'][i]).strip("''\n")]
            targets_dict['span_text'] = str(tokens['text'][i]).strip("''\n").strip("\"")
            targets.append(targets_dict)
        '''
        else:
            if tokens['tokLoc'][i] == max(tokens['tokLoc'].to_list()):
                #print(tokens['text'][i])
                pass
            else:
                targets_dict = {}
                targets_dict['span1'] = [int(tokens['tokLoc'][i]), int(tokens['tokLoc'][i]) + 1]
                targets_dict['label'] = 3
                targets_dict['span_text'] = str(tokens['text'][i]).strip("''\n").strip("\"")
                targets.append(targets_dict)
        '''

        i += 1
        while i < len(tokens) and tokens["tokLoc"][i] > 0:
            # Has to compare to "'Event'" because of the way the Tags are written in the CSV
            if tokens["tmlTag"][i] == "'EVENT'" and type(tokens["factValue"][i]) == str:
                #tokens['text'][i] = literal_eval(tokens['text'][i])
                #(tokens['text'][i]) = "\"* " + tokens['text'][i] + "\""
                targets_dict = {}
                targets_dict['span1'] = [int(tokens['tokLoc'][i]), int(tokens['tokLoc'][i]) + 1]
                targets_dict['label'] = labels_dict[str(tokens['factValue'][i]).strip("''\n")]
                targets_dict['span_text'] = str(tokens['text'][i]).strip("''\n").strip("\"")
                targets.append(targets_dict)
            '''
            else:
                if tokens['tokLoc'][i] == max(tokens['tokLoc'].to_list()):

                    pass
                else:
                    targets_dict = {}
                    targets_dict['span1'] = [int(tokens['tokLoc'][i]), int(tokens['tokLoc'][i]) + 1]
                    targets_dict['label'] = 3
                    targets_dict['span_text'] = str(tokens['text'][i]).strip("''\n").strip("\"")
                    targets.append(targets_dict)
            '''
            sentence.append(tokens["text"][i])
            i += 1

        s = []
        for index in sentence:
            s.append(literal_eval(index)) 

        # print(tokens['file'][i-1])
        articles.add(tokens['file'][i-1])
        text_dict = {'text': ' '.join(s)}
        text_dict['targets'] = targets
        text_dict['article'] = tokens['file'][i-1]
        text_dict['file_idx'] = i
        text_dict['idx'] = i
        if not text_dict['targets']:
            pass
        else:
            texts.append(text_dict)
            sentence_count += len(text_dict['targets'])

    with open("fb_spans.jsonl", 'w') as f:
        for text in texts:
            f.write(json.dumps(text) + '\n')
    
    # print(sentence_count)
    return sentence_count

def generate_splits(filename, count):
    # percentage of splits
    num_train = 0.70*count
    num_dev = 0.20*count
    num_test = 0.10*count

    train = []
    dev = []
    test = []
    texts = []

    with jsonlines.open(filename) as reader:
        for obj in reader:
            texts.append(obj)

    np.random.shuffle(texts)
    articles = {}

    for text in texts:     
        if text['article'] in articles:
            articles[text['article']] += len(text['targets'])
        else:
            articles[text['article']] = len(text['targets'])

    sum = 0
    train_count = 0
    while sum < num_train:
        for a in articles:
            sum += articles[a]
            train.append(a)
            train_count += 1
            if sum >= num_train:
                break 

    # print(train_count, sum)

    sum = 0
    dev_count = 0
    while sum < num_dev:
        for a in articles: 
            if a not in train:
                sum += articles[a]
                dev.append(a)
                dev_count += 1
                if sum >= num_dev:
                    break

    # print(dev_count, sum)

    sum = 0
    test_count = 0
    while sum < num_test:
        for a in articles:
            if a not in train and a not in dev:
                sum += articles[a]
                test.append(a)
                test_count += 1
                if sum >= num_test:
                    break

    # print(test_count, sum)

    train_data = []
    dev_data = []
    test_data = []

    for text in texts:
        if text['article'] in train:
            train_data.append(text)
        elif text['article'] in dev:
            dev_data.append(text)
        elif text['article'] in test:
            test_data.append(text)

    with open("train.jsonl", "w") as f:
        for text in train_data:
            f.write(json.dumps(text) + '\n')

    with open("dev.jsonl", "w") as f:
        for text in dev_data:
            f.write(json.dumps(text) + '\n')
        
    with open("test.jsonl", "w") as f:
        for text in test_data:
            f.write(json.dumps(text) + '\n')

if __name__ == "__main__":
    filename = "fb_labels.csv"
    jsonlines_filename = "fb_spans.jsonl"
    sentence_count = extract_jsonl(filename)
    generate_splits(jsonlines_filename, sentence_count)