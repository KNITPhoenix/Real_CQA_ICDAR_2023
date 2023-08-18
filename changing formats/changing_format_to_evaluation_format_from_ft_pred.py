import os
import json
import pandas as pd
import time
from collections import defaultdict

dicts_by_name=defaultdict(list)
pred_file = "C:/Users/pande/Desktop/predictions_ftvlt5_aftervega.json"
formatted_file = "C:/Users/pande/Desktop/predictions_ftvlt5_aftervega_formatted.json"

df = pd.read_csv("C:/Users/pande/Desktop/cqa_22_with_id/data_test_qid.csv")

with open("C:/Users/pande/Desktop/cqa_22_with_id/test.json",'r') as f:
    js_corpus = json.load(f)
for i in js_corpus:
    dicts_by_name[i['qa_id']] = i

with open(pred_file,'r') as f:
    js_pred = json.load(f)

format_lst = list()
counter = 0
for _,i in df.iterrows():
    start_time = time.time()
    format_dict = dict()
    format_dict["image_index"] = str(i['Image Index'])
    format_dict["question_id"] = i['Question ID']
    format_dict["predicted_answer"] = str(js_pred[str(i['Question ID'])])
    format_dict["qa_id"] = i['qaid']
    format_dict["answer_bbox"] = list()
    format_dict["answer_id"] = None
    inde = dicts_by_name[i['qaid']]
    counter+=1
    print(counter)
    format_dict['qid'] = inde['QID']
    format_dict['question_string'] = inde['question']
    format_dict['answer_type'] = inde['answer_type']
    format_dict['answer'] = inde['answer']
    format_dict["taxonomy id"] = inde["taxonomy id"]
    format_lst.append(format_dict)
    print("--- %s seconds ---" % (time.time() - start_time))

with open(formatted_file,'w') as f:
    json.dump(format_lst,f,indent=4)