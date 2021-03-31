from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

tokenizer = AutoTokenizer.from_pretrained("deep-learning-analytics/triviaqa-t5-base")
model = AutoModelWithLMHead.from_pretrained("deep-learning-analytics/triviaqa-t5-base")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

text = "Who is the best soccer player in the world?"

preprocess_text = text.strip().replace("\n","")
tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device)

outs = model.generate(
            tokenized_text,
            max_length=10,
            num_beams=2,
            early_stopping=True
           )

dec = [tokenizer.decode(ids) for ids in outs]
print("Predicted Answer: ", dec)