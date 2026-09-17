from __future__ import annotations
import re

def parse_mmlu_response(model_output:str)->str|None:
    if not model_output: return None
    m=re.search(r"(?:[Tt]he\s+)?correct\s+answer\s+is\s*[:\-]?\s*([A-D])\b",model_output)
    if m: return m.group(1).upper()
    return None

def parse_gsm8k_response(model_output:str)->str|None:
    if not model_output: return None
    text=re.sub(r'(?<=\d),(?=\d)','',model_output)
    nums=re.findall(r'[-+]?\d+(?:\.\d+)?',text)
    return nums[-1] if nums else None
