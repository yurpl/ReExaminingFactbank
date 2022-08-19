import pandas as pd


def label_head(sentence: list, index: int):
    new_sentence = sentence.copy()
    new_sentence[index] = '* ' + new_sentence[index]
    return new_sentence


def extract_sentence_labels(filename: str):
    tokens = pd.read_csv(filename)

    # Array of labeled sentences which will be tuples of:
    # (sentence, label)
    labeled_sentences = []

    # One pass through the data, only care about text, tmlTag, factValue, and tokLoc
    # sentences are comprised of text, Tag is used to see what labels we attach
    # factValue are the label values we need, and tokLoc lets us know when the sentence is completed.
    i = 0
    while i < len(tokens):
        labels = {}
        sentence = [tokens["text"][i]]

        # Has to compare to "'Event'" because of the way the Tags are written in the CSV
        if tokens["tmlTag"][i] == "'EVENT'" and type(tokens["factValue"][i]) == str:
            labels[tokens['tokLoc'][i]] = str(tokens['factValue'][i])

        i += 1

        while i < len(tokens) and tokens["tokLoc"][i] > 0:
            # Has to compare to "'Event'" because of the way the Tags are written in the CSV
            if tokens["tmlTag"][i] == "'EVENT'" and type(tokens["factValue"][i]) == str:
                sentence.append(tokens["text"][i])
                labels[tokens['tokLoc'][i]] = str(tokens['factValue'][i])
            else:
                sentence.append(tokens["text"][i])
            i += 1

        for loc in labels:
            labeled_sentence = label_head(sentence, loc)
            labeled_sentence = (labeled_sentence, labels[loc])
            labeled_sentences.append(labeled_sentence)

    labeled_sentences_df = pd.DataFrame(labeled_sentences, columns=["sentence", "label"])
    labeled_sentences_df.to_csv("Labeled Sentences.csv", index=False)


if __name__ == "__main__":
    filename = "FB_labels.csv"
    extract_sentence_labels(filename)