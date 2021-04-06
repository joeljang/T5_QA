from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import re
import math 
from transformers import pipeline

def get_dataset(file_path):
    data=[]
    with open(file_path) as f:
        for fileobj in f:
            line = json.loads(fileobj)
            data.append(line)
    return data

from nlp import load_dataset

class Finetune(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, print_text=False):         
        self.name = args.dataset
        self.args = args
        if self.name == 'triviaQA':
            self.dataset = load_dataset('trivia_qa', 'unfiltered.nocontext', split=type_path)
        elif self.name == 'naturalQA':
            if type_path == 'train':
                self.dataset = get_dataset('NQ/nq_train.json')
            else:
                self.dataset = get_dataset('NQ/nq_dev.json')
        else:
            raise NameError('Name a correct dataset!')
        if args.valid_on_recentQA:     
            if type_path=='validation':
                self.dataset = pd.read_csv("recentQA/recentqa_qa.csv", delimiter=',')
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        self.type_path = type_path

    def __len__(self):
        if self.name== 'naturalQA':
            return len(self.dataset)
        else:
            return self.dataset.shape[0]

    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        return text
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['question']))
        input_ = self.clean_text(example_batch['question'])
        if self.type_path=='validation' and self.args.valid_on_recentQA:
            target_ = self.clean_text(example_batch['answer'])
        else:
            if self.name == 'triviaQA':
                target_ = self.clean_text(example_batch['answer']['value'])
            elif self.name =='naturalQA':
                target_ = self.clean_text(example_batch['answer'][0])
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        return source, targets

    def __getitem__(self, index):
        if self.args.valid_on_recentQA:   
            if self.type_path=='validation':
                source, targets = self.convert_to_features(self.dataset.iloc[index])
            else:
                source, targets = self.convert_to_features(self.dataset[index])
        else:
            source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

class Pretrain(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, print_text=False):
        self.args = args
        if 'ssm' in self.args.output_dir:
            self.ssm = True
            self.nlp = pipeline("ner")
        else:
            self.ssm = False
        if self.args.dataset == 'recentQA':
            self.dataset = self.split_into_segment(pd.read_csv("recentQA/recentqa_context.csv", delimiter='\t'),input_length)
        else:
            raise NameError('Select the correct Dataset!')
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        self.sentinels = sentinels
  
    def split_into_segment(self, ds, input_length):
        new_rows = []
        input_length = int(input_length * 0.7)
        for index, row in ds.iterrows():
            if len(row['context'].split()) > input_length:
                word_list = row['context'].split()
                seg1 = word_list[:input_length]
                segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
                segment2 = seg2_a + (' '.join(word_list[input_length:]))
                ds.loc[index, 'context'] = segment1
                while(len(segment2.split()) > input_length):
                    word_list = segment2.split()
                    seg1_ = word_list[:input_length]
                    if '.' in ' '.join(seg1_):
                        segment1_, seg2_a_ = (' '.join(seg1_)).rsplit('.',1)
                        segment2 = seg2_a_ + (' '.join(word_list[input_length:]))
                    else:
                        segment1_ = ' '.join(seg1_)
                        segment2 = (' '.join(word_list[input_length:]))
                    new_rows.append(segment1_)
                new_rows.append(segment2)
        ds2 = pd.DataFrame(new_rows, columns=['context'])
        ds = ds.append(ds2)
        return ds

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        text = re.sub("\\[.*\\]",'',text)
        return text

    def span_corruption_mask(self, text, noise_span_length=3, noise_density=.15):
        max_index = len(text.split())
        mask = max_index * [0]
        span_num = math.ceil(( max_index * noise_density ) / 3 )
        exclude=[max_index-2, max_index-1]
        for i in range(span_num):
            while True:
                rand_num = np.random.randint(low=0, high=max_index) #Getting random number for mask index
                if rand_num not in exclude:
                    span = [rand_num, rand_num+1, rand_num+2]
                    for s in span:
                        mask[s] = 1
                        exclude.append(s)
                    if rand_num==1:
                        exclude.append(rand_num-1)
                    elif rand_num==2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                    elif rand_num>2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                        exclude.append(rand_num-3)
                    if not rand_num==max_index-3:
                        exclude.append(span[-1]+1)
                    break
                else:
                    continue
        return mask
    
    def salient_span_corruption_mask(self, text, noise_span_length=3, noise_density=.15):
        tokens = text.split()
        max_index = len(tokens)
        mask = max_index * [0]
        span_num = math.ceil(( max_index * noise_density ) / 3 )
        exclude=[max_index-2, max_index-1]
        NER = self.nlp(text)
        NERs=[]
        disected=False
        for n in NER:
            t = n['word']
            if '#' in t:
                if disected==False:
                    try:
                        restore = NERs.pop()
                    except:
                        restore=''
                        start_index = n['start']
                        end_index = n['end']
                        NERs.append(text[start_index:end_index])
                    text = text[:start_index] + restore + text[end_index:]
                    disected=True
                end_index = n['end']
            else:
                if disected==True:
                    disected=False
                    NERs.append(text[start_index:end_index])
                    text = text[:start_index] + ''.join('*' for i in range(end_index - start_index)) + text[end_index:]
                    start_index = n['start']
                    end_index = n['end']
                    NERs.append(text[start_index:end_index])
                    text = text[:start_index] + ''.join('*' for i in range(end_index - start_index)) + text[end_index:]
                else:
                    start_index = n['start']
                    end_index = n['end']
                    NERs.append(text[start_index:end_index])
                    text = text[:start_index] + ''.join('*' for i in range(end_index - start_index)) + text[end_index:]
        tokens = text.split()
        for i in range(len(tokens)):
            if '*' in tokens[i]:
                mask[i] = 1
        return mask
    
    def noise_span_to_unique_sentinel(self, text, mask):
        tokens = text.split()
        text_ = []
        one_count=0
        sentinel_cnt=0
        first=True
        for i in range(len(tokens)):
            if mask[i] == 1:
                if first:
                    text_.append(self.sentinels[sentinel_cnt])
                    sentinel_cnt+=1
                    first = False
            else:
                text_.append(tokens[i])
                first = True
        text_ = ' '.join(text_)
        return text_

    def nonnoise_span_to_unique_sentinel(self, text, mask):
        tokens = text.split()
        text_ = []
        zero_first=True
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 0:
                if zero_first:
                    text_.append(self.sentinels[sentinel_cnt])
                    zero_first=False
                    sentinel_cnt+=1
            else:
                zero_first=True
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['context']))
        text = self.clean_text(example_batch['context'])
        if self.ssm:
            mask = self.salient_span_corruption_mask(text)
        else:
            mask = self.span_corruption_mask(text)
        input_ = self.noise_span_to_unique_sentinel(text,mask)
        target_ = self.nonnoise_span_to_unique_sentinel(text,mask)
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset.iloc[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}