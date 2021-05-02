from nlp import load_dataset
import os
import pandas as pd
import numpy as np

train = load_dataset('trivia_qa', 'rc', split='train')
dev = load_dataset('trivia_qa', 'rc', split='validation')

print(len(train))
print(len(dev))

def get_docs(dataset, doc_list):
    for i in range(len(dataset)):
        print(i)
        wiki_context = dataset[i]['entity_pages']['wiki_context']
        if wiki_context:
            for context in wiki_context:
                doc_list.append(context)
    return doc_list

doc_lst = []

doc_lst = get_docs(train, doc_lst)
print(f'Length of doc_lst after train set : {len(doc_lst)}')
doc_lst = get_docs(dev, doc_lst)
print(f'Length of doc_lst after dev set : {len(doc_lst)}')

pd.DataFrame(np.asarray(doc_lst), columns=['context']).to_csv('triviaQA/context.csv')