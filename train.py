import argparse
from argparse import ArgumentParser
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from transformers import (
    AdamW,
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)

import textwrap
from tqdm.auto import tqdm

#from models import T5FineTuner
from models_module_encoder import T5FineTuner
#from models_module_layerwise2 import T5FineTuner
#from models_module_encoder_small import T5FineTuner
#from models_adapter import T5FineTuner
#from models_ewc import T5FineTuner
from datasets import Pretrain, Finetune, Probe
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Input a config file dude!")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting gpus to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Logging into WANDB
    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name)
    else:
        wandb_logger = None

    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # path to save the checkpoints
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        freeze_encoder=False,
        freeze_embeds=False,
        learning_rate=hparam.learning_rate,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 1.0,
        n_val=-1,
        n_train=-1,
        n_test=-1,
        early_stop_callback=False,
        fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        check_validation=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator
    )
    args = argparse.Namespace(**args_dict)

    ## Define Checkpoint function
    if args.mode == 'pretrain':
        #checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #    dirpath = args.output_dir, monitor="bleu_score", mode="max", save_top_k=1, save_last=True
        #)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = args.output_dir, save_top_k=0, save_last=True
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath = args.output_dir, monitor="em_score", mode="max", save_top_k=1
        )

    if args.output_dir=="":
        checkpoint_callback = False

    ##Logging Learning Rate Scheduling
    if args.use_lr_scheduling:  
        lr_monitor = [pl.callbacks.LearningRateMonitor()]
    else:
        lr_monitor = []
    ## If resuming from checkpoint, add an arg resume_from_checkpoint
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision= 16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        callbacks = lr_monitor,
        accelerator=args.accelerator,
        #limit_train_batches = 0.05,
        #plugins='ddp_sharded'
    )
    
    if args.check_validation:
        if  args.mode == 'probe':
            set_seed(40)
            model = T5FineTuner(args)
            model = T5FineTuner.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args)
            #trainer = pl.Trainer(**train_params)
            #trainer.fit(model)
            model.eval()
            model.to('cuda')
        else:
            model = T5FineTuner(args)
            #model = T5FineTuner.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args)
            model.eval()
            model.to('cuda')

        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        if args.mode=='probe':
            dataset = Probe(tokenizer, 'validation', None, input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
            print('Length of validation data: ',len(dataset))
            entity_relation = dataset.entity_relation
        elif args.mode=='pretrain':
            dataset = Pretrain(tokenizer, 'validation', None, input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
        else:
            dataset = Finetune(tokenizer, 'validation', None, input_length=args.max_input_length, 
                            output_length=args.max_output_length, args=args)
        loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
        
        total_cnt = 0
        em_correct_num = 0
        subset_correct_num = 0

        def clean_up(text):
            text =text.replace('<pad>', '')
            text = text.replace('</s>', '')
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            text = text.replace("<extra_id_2>", "")
            text = text.replace("<extra_id_3>", "")
            text = text.replace(".", '')
            text = text.replace(',', '')
            text = text.replace("'", '')
            text = text.replace('"', '')
            return text     

        def clean_up2(text):
            text =text.replace('<pad>', '')
            text = text.replace('</s>', '')
            text = text.replace(".", '')
            text = text.replace(',', '')
            text = text.replace("'", '')
            text = text.replace('"', '')
            return text     

        preds=[]

        for batch in iter(loader):
            if args.mode=='probe':
                outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=args.max_output_length,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=1
                )
            else:
                outs = model.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_length=args.max_output_length,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=1
                )
            dec = [tokenizer.decode(ids) for ids in outs]
            texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = [tokenizer.decode(ids) for ids in batch['target_ids']]
  
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                #print(total_cnt)
                lines = textwrap.wrap("\n%s\n" % texts[i], width=200)
                ground_truth = clean_up(targets[i])
                predicted = clean_up(dec[i])
                em = model.exact_match_score(predicted, ground_truth)
                subset = model.approx_match_score(predicted, ground_truth)         
                #if em == 0 and subset == 1:
                print(f'{total_cnt} QUESTION : {lines[0]}')
                print(f'GROUD TRUTH: {ground_truth}, PREDICTED: {predicted}')
                if "<extra_id_0>" not in predicted:
                    predicted = predicted+" <extra_id_0>."
                #preds.append([clean_up(lines[0]), predicted, ground_truth])
                if em == 1:
                    em_correct_num+=1
                    #preds.append([clean_up2(lines[0]), predicted, ground_truth])
                if subset == 1:
                    subset_correct_num+=1

        print(f'Number of total validation data: {total_cnt}')
        print(f'Number of correct predictions out of {total_cnt} : {em_correct_num, subset_correct_num}. Percentage : {em_correct_num / total_cnt, subset_correct_num / total_cnt}')
        #pd.DataFrame(np.asarray(preds), columns=['id','cloze', 'predicted', 'answer']).to_csv('mini_recentnews_correct.csv')
    else:
        set_seed(40)
        model = T5FineTuner(args)
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)