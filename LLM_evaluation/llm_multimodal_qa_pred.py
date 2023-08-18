import os 
import json 
import pandas as pd
import time
import shutil
from IPython import display
import csv 
from transformers import pipeline
import torch
seed=123
# torch.use_deterministic_algorithms(True)
torch.cuda.manual_seed(seed)

images_dir = "/home/spandey8/llm/llm_multi_modal_data/images/"
test_dir = "/home/spandey8/llm/llm_multi_modal_data/pmctest22/"
save_dir = "/home/spandey8/llm/output_pred_llm/multi_modal/"

# models = ["naver-clova-ix/donut-base-finetuned-docvqa","impira/layoutlm-document-qa"]
# models = ["naver-clova-ix/donut-base-finetuned-docvqa"]
# print(multi_modal(images_dir+"PMC1831497___g006.jpg","How many major ticks are there on the independent axis of the chart?"))
models = ['microsoft/layoutxlm-base']
start = time.time()
count_of_files = 0
saved_files = 0
for m in models:
    number_of_ignored = 0
    multi_modal = pipeline("document-question-answering",model=m,device=3,max_length=512)
    sd = os.path.join(save_dir, m)
    if os.path.exists(sd):
        shutil.rmtree(sd)
    os.makedirs(sd)
    for chart in os.listdir(test_dir)[]:
        print(chart)
        count_of_files+=1
        qajson = json.load(open(os.path.join(test_dir, chart)))
        oparr = []
        for q in qajson :
            op_js = {'qa_id': q['qa_id']}
            query = q['question']
            res = multi_modal(images_dir+os.path.splitext(chart)[0]+'.jpg',query,truncate = True)
            op_js.update({"predicted_answer" : res['answer']})
            # print(op_js)
            oparr.append(op_js)
        with open(os.path.join(sd, chart), 'w') as f : 
            json.dump(oparr, f)
            saved_files+=1
        # print("Multi Modal Predictions Ongoing")
        # print("Completed Files:[",count_of_files,"]")
        # print("Saved Files:[",saved_files,"]")
        display.clear_output(wait=True)
    print(m,":",number_of_ignored)

end = time.time()
print("time for both models:",end-start)

# print(os.listdir(test_dir)[116])