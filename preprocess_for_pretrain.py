import pandas as pd

def split_into_segment(ds, input_length):
        new_rows = []
        input_length = int(input_length * 0.7)
        for index, row in ds.iterrows():
            print(index)
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
        print(f'Number of rows for preprocessed dataset : {len(ds)}')
        ds.to_csv(f'triviaQA/context_preprocessed_{input_length}.csv')
        return ds

INPUT_LENGTH = 512
dataset = pd.read_csv("triviaQA/context.csv", delimiter=',')
split_into_segment(dataset, INPUT_LENGTH)