# import os 
# import json 
# import pandas as pd
# import time
# import shutil
# from IPython import display
# import csv 
# from transformers import pipeline

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# test_dir = "/home/spandey8/llm/cqa_22_with_id/pmctest22"
# # table_dir = "/home/spandey8/llm/tables/"
# table_dir = "/home/spandey8/llm/flatten_tables/"
# save_dir = "/home/spandey8/llm/output_pred_llm/question_answering_models/"

# models = ['distilbert-base-cased-distilled-squad', 'deepset/roberta-base-squad2']
# t = "question-answering"
# start = time.time()
# count_of_files = 0
# saved_files = 0
# for m in models :
#     oracle = pipeline(t,model=m,device=1)
#     sd = os.path.join(save_dir, m)
#     if os.path.exists(sd):
#         shutil.rmtree(sd)
#     os.makedirs(sd)
#     for chart in os.listdir(test_dir):
#         count_of_files+=1
#         if os.path.exists(table_dir+chart):
#             with open(table_dir+chart,'r') as f:
#                 table = json.load(f)
#                 table = table[os.path.splitext(chart)[0]]
#         else:
#             table = None
#         if table is not None : 
#             qajson = json.load(open(os.path.join(test_dir, chart)))
#             oparr = []
#             for q in qajson :
#                 op_js = {'qa_id': q['qa_id']}
#                 query = q['question']
#                 res = oracle(question=query, context=table)
#                 op_js.update({"predicted_answer" : res['answer']})
#                 print(op_js)
#                 oparr.append(op_js)
#             with open(os.path.join(sd, chart), 'w') as f : 
#                 json.dump(oparr, f)
#                 saved_files+=1
#         print("Question Answering Ongoing")
#         print("Completed Files:[",count_of_files,"]")
#         print("Saved Files:[",saved_files,"]")
#         display.clear_output(wait=True)

# end = time.time()
# print("time for both models:",end-start)
    


# table = get_table("PMC1831497___g006.json")
# with open(table_dir+"PMC1831497___g006.json",'r') as f:
#     table = json.load(f)
# print(table['PMC1831497___g006'])

# oracle = pipeline("question-answering",model="distilbert-base-cased-distilled-squad")

# print(oracle(question="How many major ticks are there on the independent axis of the chart?", context=table))
# oracle(query="How many stars does the transformers repository have?", table=table)

# question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad",device = 2)
# ques = "Given the following information: "+table['PMC1831497___g006']+", "+"How many major ticks are there on the independent axis of the chart?"
# # print(ques)
# print(question_answerer(question=ques, context=table['PMC1831497___g006']))

# tabvqa = pipeline("question-answering")


from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2',device=2)
set_seed(42)

print(generator("Given the following table in a comma separated format, where row is separated by new line and each value is separated by comma : \n \
Effector: Target Ratio,B721-H-Ras G12V,B721-H-Ras WT,B721-GFP \n \
0.37184667053118764,0.42376087779039595,0.7863172133043008,0.36408946290429256 \n \
0.7467816209876925,23.51872871736661,28.79939334711743,7.490645749656631 \n \
1.5045298092185713,38.66818009837305,31.373956711199998,14.875667874674516 \n \
2.9655979494895672,55.9011224618489,40.51582090151027,18.366878696256617 \n \
Based on this the name of data series in the table that has the highest mean is", max_length=300, num_return_sequences=1))

