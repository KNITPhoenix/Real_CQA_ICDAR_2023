import pandas as pd
import json

table_di = "/home/spandey8/llm/flatten_tables/"
df = pd.read_csv("/home/spandey8/llm/data.csv")

df_table_dict = dict()
for _,i in df.iterrows():
    if i['Image Index'] not in df_table_dict.keys():
        df_table_dict [str(i['Image Index'])] = str(i['Input'][i['Input'].find("Table:"):-1])


for i in df_table_dict:
    di = dict()
    di[i] = df_table_dict[i]
    with open(table_di+i+".json",'w') as f:
        json.dump(di,f,indent = 4)