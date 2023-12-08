# LoRA Alpaca: Stanford Alpaca following Low-Rank Adaptation

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Weight Diff License](https://img.shields.io/badge/Weight%20Diff%20License-CC%20By%20NC%204.0-yellow)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/WEIGHT_DIFF_LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is the repo for reproducing the Stanford Alpaca project using Low-Rank Adaptation of Large Language Models (LoRA). The Lora weights for fine-tuning LLaMA model on Alpaca dataset is in [output](./output/) and the log for parameters and training loss are in [log](./log/record.log).

What's new compared to Stanford Alpaca:

- The code for [running inference on the adapted model](#inference).

**Usage and License Notices**: Alpaca is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. 
The weight diff is also CC BY NC 4.0 (allowing only non-commercial use).

## Inference

To load the Lora weights for LLaMA, first install the requirements

```bash
pip install -r requirements.txt
```

The inference function will take the input from [test.txt](test.txt) line by line, run inference on each line from the input, and output to [test_output.txt](test_output.txt). Use the following command to run the inference and replace `<your_path_to_hf_converted_llama_ckpt_and_tokenizer>` by the path to the Huggingface converted LLaMA checkpoints and tokenizer:

```bash
python3 inference.py --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer>
```

The LoRA fine-tuning weights are for LLaMA model instead of LLaMA2 because LLaMA2 is pre-trained using bf16 which is not accessible on V100 GPU and disabling bf16 will make the optimizing process unstable.

Here's an example on how to load the LoRA weights, more detailed usage can be found in [inference](inference.py):

```python
# load the LLaMA weights
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=torch.float16,
)

# load the LLaMA tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    padding_side='left',
    use_fast=False,
)

# load the LoRA weights
model = PeftModel.from_pretrained(model,"./output")
```
