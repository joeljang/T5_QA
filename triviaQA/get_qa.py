import json
import numpy as np
import pandas as pd

with open('unfiltered-web-test-without-answers.json') as json_file:
    json_data = json.load(json_file)

data = json_data['Data']

qa_pair = []
for entry in data:
    question = entry['Question']
    answers = entry['Answer']['Value']
    qa_pair.append([question,answers])

pd.DataFrame(np.asarray(qa_pair), columns=['question', 'answer']).to_csv('triviaQA_train_qa.csv')
#np.savetxt("triviaQA_train_qa.csv", np.asarray(qa_pair), delimiter=',', fmt="%s", encoding='utf-8')