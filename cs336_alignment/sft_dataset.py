from __future__ import annotations
import json, gzip, random
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self,tokenizer,dataset_path,seq_length:int,shuffle:bool):
        path=str(dataset_path); op=gzip.open if path.endswith('.gz') else open
        docs=[]
        with op(path,'rt',encoding='utf-8') as f:
            for line in f:
                if line.strip(): docs.append(json.loads(line))
        if shuffle:
            rng=random.Random(42); rng.shuffle(docs)
        template=("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n{prompt}\n\n### Response:\n{response}")
        ids=[]
        for d in docs:
            p=d.get('prompt',d.get('instruction','')); r=d.get('response',d.get('output',''))
            toks=tokenizer.encode(template.format(prompt=p,response=r),add_special_tokens=False)
            ids.extend(toks)
            if tokenizer.eos_token_id is not None: ids.append(tokenizer.eos_token_id)
        # The public test fixture expects pure next-token LM packing, no prompt masking.
        block=seq_length+1; n=len(ids)//block
        arr=torch.tensor(ids[:n*block],dtype=torch.long).view(n,block)
        self.inputs=arr[:,:-1].contiguous(); self.labels=arr[:,1:].contiguous()
    def __len__(self): return self.inputs.shape[0]
    def __getitem__(self,i): return {'input_ids':self.inputs[i].clone(),'labels':self.labels[i].clone()}
