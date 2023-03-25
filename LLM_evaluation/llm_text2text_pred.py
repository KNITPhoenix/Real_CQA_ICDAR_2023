import os 
import json 
import pandas as pd
import time
import shutil
from IPython import display
import csv 
from transformers import pipeline

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_dir = "/home/spandey8/llm/cqa_22_with_id/pmctest22"
table_dir = "/home/spandey8/llm/flatten_tables/"
save_dir = "/home/spandey8/llm/output_pred_llm/text2text/"


models = ['google/flan-t5-base']
t = "text2text-generation"
start = time.time()
count_of_files = 0
saved_files = 0
for m in models :
    oracle = pipeline(t,model=m,device=2)
    sd = os.path.join(save_dir, m)
    if os.path.exists(sd):
        shutil.rmtree(sd)
    os.makedirs(sd)
    for chart in os.listdir(test_dir):
        count_of_files+=1
        if os.path.exists(table_dir+chart):
            with open(table_dir+chart,'r') as f:
                table = json.load(f)
                table = table[os.path.splitext(chart)[0]]
        else:
            table = None
        if table is not None : 
            qajson = json.load(open(os.path.join(test_dir, chart)))
            oparr = []
            for q in qajson :
                op_js = {'qa_id': q['qa_id']}
                query = q['question']
                input_string = "question: "+ query +"context: "+table
                if len(input_string)>512:
                    input_string = input_string[:512]
                res = oracle(input_string)
                op_js.update({"predicted_answer" : res[0]['generated_text']})
                print(op_js)
                oparr.append(op_js)
            with open(os.path.join(sd, chart), 'w') as f : 
                json.dump(oparr, f)
                saved_files+=1
        print("Question Answering Ongoing")
        print("Completed Files:[",count_of_files,"]")
        print("Saved Files:[",saved_files,"]")
        display.clear_output(wait=True)

end = time.time()
print("time for both models:",end-start)



# with open(table_dir+"PMC1831497___g006.json",'r') as f:
#     table = json.load(f)
# table = table["PMC1831497___g006"]

# text2text_generator = pipeline("text2text-generation",model="google/flan-t5-xl")
# input_string = "question: How many major ticks are there on the independent axis of the chart? context: "+table
# if len(input_string)>512:
#     input_string = input_string[:512]
# print(text2text_generator(input_string))
