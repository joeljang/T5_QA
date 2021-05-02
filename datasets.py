from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import re
import math 
from transformers import pipeline
import os
import random
os.environ['TRANSFORMERS_CACHE'] = '/mnt/joel/'

from nlp import load_dataset
import pprint

class Finetune(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args, print_text=False):         
        self.name = args.dataset
        self.args = args
        if args.valid_on_recentQA:     
            if type_path=='validation':
                #self.dataset = pd.read_csv("triviaQA/no_overlap_cloze.csv", delimiter='|')
                self.dataset = pd.read_csv("recentQA/recentqa_cloze.csv", delimiter='|')
        else:
            if self.name == 'triviaQA':
                self.dataset = load_dataset('trivia_qa', 'unfiltered.nocontext', split=type_path, cache_dir='/mnt/joel/datasets')
            elif self.name == 'naturalQA':
                if type_path == 'train':
                    self.dataset = self.get_dataset('NQ/nq_train.json')
                else:
                    self.dataset = self.get_dataset('NQ/nq_dev.json')
            else:
                raise NameError('Name a correct dataset!')
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

    def get_dataset(self, file_path):
        data=[]
        with open(file_path) as f:
            for fileobj in f:
                line = json.loads(fileobj)
                data.append(line)
        return data

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
        if self.args.dataset == 'recentQA_context':
            self.dataset = self.split_into_segment(pd.read_csv("recentQA/recentqa_context.csv", delimiter='\t'),input_length)
        elif self.args.dataset == 'triviaQA_context':
            self.dataset = pd.read_csv("/triviaQA/context_preprocessed.csv", delimiter=',')
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
                try:
                    segment1, seg2_a = (' '.join(seg1)).rsplit('.',1)
                except ValueError as e:
                    seg2_a = ''
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


class Probe(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, args):
        self.args = args
        #self.nlp = pipeline("ner")
        if self.args.dataset == 'trex':
            parameters = self.get_TREx_parameters()
        elif self.args.dataset == 'googlere':
            parameters = self.get_GoogleRE_parameters()
        elif self.args.dataset == 'conceptnet':
            parameters = self.get_ConceptNet_parameters()
        elif self.args.dataset == 'squad':
            parameters = self.get_Squad_parameters()
        else:
            raise NameError('Select the correct Dataset!')
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.dataset, self.entity_relation = self.getdataset(*parameters) 
        
    def __len__(self):
        return len(self.dataset)
    
    def getdataset(self, relations, data_path_pre, data_path_post):
        full_dataset = []
        total_cnt=0
        string_match_cnt=0
        non_entity_cnt=0
        entity_lst = pd.read_csv('lama/entity_list.csv')
        entity_lst = list(entity_lst['entity'])
        entity_relation = {}
        for relation in relations:
            dataset_filename = "{}{}{}".format(data_path_pre, relation["relation"], data_path_post)
            try:
                all_samples = self.load_file(dataset_filename)
            except Exception as e:
                continue
            
            if relation['template'] and relation['template'] != "":
                facts = []
                for sample in all_samples:
                    total_cnt+=1
                    sub = sample["sub_label"]
                    obj = sample["obj_label"]
                    # Excluding samples with string match filter just like E-BERT
                    if obj in sub:
                        string_match_cnt+=1
                        continue 
                    # Exclude objects that are non-entity
                    if not (obj in entity_lst):
                        non_entity_cnt+=1
                        continue 
                    #Saving information for n to m relations
                    if sub not in entity_relation:
                        entity_relation[sub] = [obj]
                    elif obj not in entity_relation[sub]:
                        entity_relation[sub].append(obj)
                    if (sub, obj) not in facts:
                        facts.append((sub, obj))
                all_samples = []
                for fact in facts:
                    (sub, obj) = fact
                    sample = {}
                    sample["sub_label"] = sub
                    sample["obj_label"] = obj
                    # sobstitute all sentences with a standard template
                    sample["masked_sentences"] = self.parse_template(
                        relation['template'].strip(), sample["sub_label"].strip(), "[MASK]"
                    )
                    all_samples.append(sample)
            
            full_dataset = full_dataset + all_samples

        with open('entity_relation.json', 'w', encoding='utf-16') as fp:
            json.dump(entity_relation, fp)

        print('Number of object string match to subject: ', string_match_cnt)
        print('Number of object that are non entities: ', non_entity_cnt)
        print('Length of the modified dataset is :',len(full_dataset))
        random.shuffle(full_dataset)
        
        return full_dataset, entity_relation

    def parse_template(self, template, subject_label, object_label):
        SUBJ_SYMBOL = "[X]"
        OBJ_SYMBOL = "[Y]"
        template = template.replace(SUBJ_SYMBOL, subject_label)
        template = template.replace(OBJ_SYMBOL, object_label)
        return [template]

    def load_file(self, filename):
        data = []
        with open(filename, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def get_TREx_parameters(self, data_path_pre="lama/"):
        relations = self.load_file("{}relations.jsonl".format(data_path_pre))
        data_path_pre += "TREx/"
        data_path_post = ".jsonl"
        return relations, data_path_pre, data_path_post

    def get_GoogleRE_parameters(self):
        relations = [
            {
                "relation": "place_of_birth",
                "template": "[X] was born in [Y] .",
                "template_negated": "[X] was not born in [Y] .",
            },
            {
                "relation": "date_of_birth",
                "template": "[X] (born [Y]).",
                "template_negated": "[X] (not born [Y]).",
            },
            {
                "relation": "place_of_death",
                "template": "[X] died in [Y] .",
                "template_negated": "[X] did not die in [Y] .",
            },
        ]
        data_path_pre = "lama/Google_RE/"
        data_path_post = "_test.jsonl"
        return relations, data_path_pre, data_path_post


    def get_ConceptNet_parameters(self,data_path_pre="lama/"):
        relations = [{"relation": "test"}]
        data_path_pre += "ConceptNet/"
        data_path_post = ".jsonl"
        return relations, data_path_pre, data_path_post


    def get_Squad_parameters(self,data_path_pre="lama/"):
        relations = [{"relation": "test"}]
        data_path_pre += "Squad/"
        data_path_post = ".jsonl"
        return relations, data_path_pre, data_path_post

    def clean_text(self, text):
        if '[MASK]' not in text:
            raise NameError('No [MASK] in input sentence!')
        text = text.replace('[MASK]', '<extra_id_0>')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        text = re.sub("\\[.*\\]",'',text)
        return text

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        input_ = self.clean_text(example_batch['masked_sentences'][0])
        target_ = example_batch['obj_label']
        subject_ = example_batch['sub_label']
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        subjects = self.tokenizer.batch_encode_plus([subject_], max_length=30, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        return source, targets, subjects
  
    def __getitem__(self, index):
        source, targets, subjects = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        subject_ids = subjects["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        subject_mask = subjects["input_ids"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "subject_ids": subject_ids, "subject_mask": subject_mask}