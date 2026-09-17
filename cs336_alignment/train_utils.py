from __future__ import annotations
import json, random, os
from pathlib import Path
from typing import Callable
import torch
from unittest.mock import patch


def seed_everything(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_jsonl(path:str|os.PathLike):
    out=[]
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out


def extract_gold(ex:dict)->str:
    if 'answer' in ex:
        a=str(ex['answer'])
        if '####' in a: a=a.rsplit('####',1)[1]
        return a.strip().replace(',','')
    if 'gold' in ex: return str(ex['gold']).strip().replace(',','')
    raise KeyError('answer/gold not found')


def format_prompt(ex:dict, template:str)->str:
    if 'prompt' in ex and '<think>' in str(ex['prompt']):
        return str(ex['prompt'])
    q=ex.get('question', ex.get('problem'))
    if q is None: raise KeyError('question/problem not found')
    return template.replace('{question}', str(q))


def init_vllm(model_id:str, device:str, seed:int, gpu_memory_utilization:float=0.55, max_model_len:int=2048):
    from vllm import LLM
    try:
        from vllm.model_executor import set_random_seed
        set_random_seed(seed)
    except Exception:
        pass
    with patch('torch.distributed.get_world_size', return_value=1), \
         patch('vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling', return_value=None):
        return LLM(model=model_id, device=device, dtype='bfloat16', enable_prefix_caching=True,
                   gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                   trust_remote_code=True, seed=seed)


def load_policy_into_vllm_instance(policy, llm):
    state_dict=policy.state_dict()
    llm_model=llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_vllm(llm, reward_fn:Callable, prompts:list[str], golds:list[str], *, temperature:float=1.0,
                  max_tokens:int=256, seed:int=42, output_path:str|None=None):
    from vllm import SamplingParams
    params=SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                          stop=['</answer>'], include_stop_str_in_output=True, seed=seed)
    outs=llm.generate(prompts, params, use_tqdm=True)
    rows=[]; sums={'reward':0.0,'format_reward':0.0,'answer_reward':0.0}; lengths=[]
    for p,g,o in zip(prompts,golds,outs):
        text=o.outputs[0].text
        r=reward_fn(text,g)
        row={'prompt':p,'ground_truth':g,'response':text,'metrics':r}
        rows.append(row); lengths.append(len(o.outputs[0].token_ids))
        for k in sums: sums[k]+=float(r.get(k,0.0))
    n=max(1,len(rows))
    metrics={k:v/n for k,v in sums.items()}
    metrics['avg_response_tokens']=sum(lengths)/n
    if output_path:
        Path(output_path).parent.mkdir(parents=True,exist_ok=True)
        with open(output_path,'w',encoding='utf-8') as f:
            for x in rows: f.write(json.dumps(x,ensure_ascii=False)+'\n')
    return metrics, rows


def save_run_metadata(path:str, payload:dict):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False,indent=2)

def log_generations(rows: list[dict], path: str) -> None:
    """Serialize prompt, response, ground truth and reward metadata for qualitative inspection."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
