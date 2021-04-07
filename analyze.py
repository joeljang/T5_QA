import pandas as pd
import numpy as np
import string
import re

train = pd.read_csv('triviaQA/trivia_qa.csv')
#correct_valid = pd.read_csv('triviaQA_correctvalid.csv')
#correct_valid = pd.read_csv('outputs/t5_small_nossm/triviaQA_correctvalid.csv')
#correct_valid = pd.read_csv('outputs/t5_small_ssm/triviaQA_correctvalid.csv')
correct_valid = pd.read_csv('outputs/t5_large_nossm/triviaQA_correctvalid.csv')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

train_answer = np.array(train['answer'])

for i in range(len(train_answer)):
    train_answer[i] = normalize_answer(train_answer[i])

no_overlap=0
for index, row in correct_valid.iterrows():
    prediction = normalize_answer(row['prediction'])
    #print(prediction)
    if prediction not in train_answer:
        no_overlap+=1
        print(prediction)

print(no_overlap)