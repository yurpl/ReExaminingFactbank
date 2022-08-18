import os
import jsonlines

#os.chdir("/Users/john/PycharmProjects/summer21/factuality_bert/glue_data/FB_MTL_clf")
# clf = {
#     3.0: 4,
#     2.0: 3,
#     1.0: 3,
#     0.0: 2,
#     -1.0: 1,
#     -2.0: 1,
#     -3.0: 0,
# }
clf = {
    3.0: 6,
    2.0: 5,
    1.0: 4,
    0.0: 3,
    -1.0: 2,
    -2.0: 1,
    -3.0: 0,
}

pol = {
    3.0: 2,
    2.0: 2,
    1.0: 2,
    0.0: 1,
    -1.0: 0,
    -2.0: 0,
    -3.0: 0
}

pol_one = {
    3.0: 1.0,
    2.0: 1.0,
    1.0: 1.0,
    0.0: 0.0,
    -1.0: -1.0,
    -2.0: -1.0,
    -3.0: -1.0
}
os.chdir("/Users/john/PycharmProjects/summer21/factuality_bert/glue_data/FB_seven")

labs = []
with jsonlines.open('dev.jsonl', 'r') as reader, jsonlines.open("dev2.jsonl", 'w') as writer:
    counter = 0
    for obj in reader:
        if not (bool(obj['targets'])):
            del obj
            continue
        else:
            for t in obj["targets"]:
                if type(t['label']) is float:
                     t['label'] = clf[round(t['label'])]
                if "file_idx" in t:
                    t.pop("file_idx")
                    t.pop("idx")

        obj["idx"] = counter
        obj['file_idx'] = None

        writer.write(obj)
        counter += 1
#
# with jsonlines.open('dev.jsonl', 'r') as reader, jsonlines.open("dev2.jsonl", 'w') as writer:
#     counter = 0
#     for obj in reader:
#         if not (bool(obj['targets'])):
#             del obj
#             continue
#         else:
#             for t in obj["targets"]:
#                 if type(t['label']) is float:
#                     t['label'] = clf[round(t['label'])]
#                 if "file_idx" in t:
#                     t.pop("file_idx")
#                     t.pop("idx")
#
#         obj["idx"] = counter
#         obj['file_idx'] = None
#
#         writer.write(obj)
#         counter += 1
#
# with jsonlines.open('test.jsonl', 'r') as reader, jsonlines.open("test2.jsonl", 'w') as writer:
#     counter = 0
#     for obj in reader:
#         if not (bool(obj['targets'])):
#             del obj
#             continue
#         else:
#             for t in obj["targets"]:
#                 if type(t['label']) is float:
#                     t['label'] = clf[round(t['label'])]
#                 if "file_idx" in t:
#                     t.pop("file_idx")
#                     t.pop("idx")
#
#         obj["idx"] = counter
#         obj['file_idx'] = None
#
#         writer.write(obj)
#         counter += 1
#
