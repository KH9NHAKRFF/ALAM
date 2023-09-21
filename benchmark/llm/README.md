
# ALAM large lange models ️

In this repository, we applied our ALAM to [lit-llama](https://github.com/Lightning-AI/lit-llama) Before running our code, we recommend first executing this repository. Thanks the authors for open-source code.

## Requirements
We conducted experiments in the torch 2.0.1 environment. This environment with requirements is available at: 
```bash
conda env create -f alam_llm.yaml
```
And then, install ALAM at [alam](https://github.com/KH9NHAKRFF/ALAM).

## Prepared the pretrained model

Check the guideline [guide](howto/download_weights.md).

## Prepair data

Prepair [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) datasets by
```bash
   python scripts/prepare_alpaca.py
   ```


## Finetune the model by ALAM

Finetune the model by ALAM as below:

   ```bash
   python finetune/lora.py --pretrained_path PRETRAINED_PATH --data_dir DATA_DIR --tokenizer TOKENIZER_PATH --do_alam True --bit 2
   ```
   or 
   ```bash
   python finetune/adapter.py --pretrained_path PRETRAINED_PATH --data_dir DATA_DIR --tokenizer TOKENIZER_PATH --do_alam True --bit 2
   ```

In experiments, we apply parameter efficient fine-tuning (PEFT) such as [LoRA](https://arxiv.org/abs/2106.09685) and [Adapter](https://arxiv.org/abs/2303.16199).

To apply apla, add ```--do_alam``` and ```--bit BIT```, choosing a average bit of activations. In the paper, we experimented with (4, 3, 2, 1.5, 1).
