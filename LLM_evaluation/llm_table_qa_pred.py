import os 
import json 
import pandas as pd
import time
import shutil
from IPython import display
import csv 
from transformers import pipeline

# model = "distilbert-base-cased-distilled-squad" , task ="question-answering"
# model = "deepset/roberta-base-squad2" , task ="question-answering"
# model = "google/flan-t5-xl" , task ="text2textgeneration"
# model = "google/flan-t5-base" , task ="text2textgeneration"
# + gpt, roberta, tapas already mentioned in the file.

test_dir = "/home/spandey8/llm/cqa_22_with_id/pmctest22"
table_dir = "/home/spandey8/llm/tables/"
save_dir = "/home/spandey8/llm/output_pred_llm/"
td = os.listdir(table_dir)

# models = ['gpt2', 'TapasForQuestionAnswering']
# text2text_generator = pipeline(model=m, task ="text2text-generation")
# text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")

def get_table(chart) :
    t = None
    c_ = chart[:-4]+'csv'
    if c_ in td : 
        t = pd.read_csv(os.path.join(table_dir, c_))
        # t = pd.read_csv("/home/spandey8/llm/tables/PMC5855060___11.csv")    #this csv is empty
        if not t.empty: 
            t = t.applymap(lambda x: str(x))
            # for col in t.columns:
                # t[col] = t[col].astype(str)
            # print(t)
        else:
            t=None
    return t 

models = ['google/tapas-large-finetuned-wtq', 'google/tapas-base-finetuned-wtq']
t = "table-question-answering"
start = time.time()
count_of_files = 0
saved_files = 0
for m in models :
    oracle = pipeline(model=m, task=t)
    sd = os.path.join(save_dir, m)
    if os.path.exists(sd):
        shutil.rmtree(sd)
    os.makedirs(sd)
    sd = os.path.join(sd, t)
    if os.path.exists(sd):
        shutil.rmtree(sd)
    os.makedirs(sd)
    for chart in os.listdir(test_dir):
        count_of_files+=1
        table = get_table(chart)
        # print(chart , 'got table', table)
        if table is not None : 
            qajson = json.load(open(os.path.join(test_dir, chart)))
            oparr = []
            for q in qajson :
                op_js = {'qa_id': q['qa_id']}
                query = q['question']
                # print('query', query)
                res = oracle(query=query, table=table)
                op_js.update({"predicted_answer" : res['answer']})
                # print(op_js)
                oparr.append(op_js)
            with open(os.path.join(sd, chart), 'w') as f : 
                json.dump(oparr, f)
                saved_files+=1
        print("Table QA Ongoing")
        print("Completed Files:[",count_of_files,"]")
        print("Saved Files:[",saved_files,"]")
        display.clear_output(wait=True)

end = time.time()
print("time for both models:",end-start)
    




# table = {
#     "Repository": ["Transformers", "Datasets", "Tokenizers"],
#     "Stars": ["36542", "4512", "3934"],
#     "Contributors": ["651", "77", "34"],
#     "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
# }
# oracle(query="How many stars does the transformers repository have?", table=table)
# oracle(query="How many stars does the transformers repository have?", table=table)



# tabvqa = pipeline("question-answering")