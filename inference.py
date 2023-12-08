import copy
import logging
import os.path as osp
from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Sequence
from src.utils.io import set_logging

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def inference():
    print("Starting inference process")
    parser = transformers.HfArgumentParser((ModelArguments,))
    model_args = parser.parse_args_into_dataclasses()[0]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side='left',
        use_fast=False,
    )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    model = PeftModel.from_pretrained(model,"./output")
    model = model.half().cuda()
    model.eval()

    with open('test.txt', "r") as f:
        with open('test_output.txt','w') as fw:
            for line in f:
                inputs = tokenizer(line, return_tensors="pt")
                outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
                ret = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
                fw.write(ret + '\n')


if __name__ == "__main__":
    inference()