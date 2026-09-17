from __future__ import annotations
import json, gzip, random
import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length: int, shuffle: bool):
        path = str(dataset_path)
        opener = gzip.open if path.endswith('.gz') else open
        docs = []
        with opener(path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        if shuffle:
            rng = random.Random(42)
            rng.shuffle(docs)

        template = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{prompt}\n\n### Response:\n{response}"
        )
        stream: list[int] = []
        for d in docs:
            prompt = d.get('prompt', d.get('instruction', ''))
            response = d.get('response', d.get('output', ''))
            # Fixture semantics: each document begins with tokenizer BOS and ends with EOS.
            stream.extend(tokenizer.encode(template.format(prompt=prompt, response=response), add_special_tokens=True))
            if tokenizer.eos_token_id is not None:
                stream.append(tokenizer.eos_token_id)

        if len(stream) < seq_length + 1:
            raise ValueError(f'dataset has only {len(stream)} tokens, need at least {seq_length + 1}')

        # Global causal-LM shift first, then pack fixed-length examples. This preserves every token
        # across chunk boundaries rather than discarding one token per chunk.
        ids = torch.tensor(stream, dtype=torch.long)
        usable = ((ids.numel() - 1) // seq_length) * seq_length
        self.inputs = ids[:usable].view(-1, seq_length).contiguous()
        self.labels = ids[1:usable + 1].view(-1, seq_length).contiguous()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, i):
        return {'input_ids': self.inputs[i].clone(), 'labels': self.labels[i].clone()}
