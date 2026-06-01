## Getting Started

### Installation

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/lurisy/SEHPM
cd SEHPM
conda env create -f env_SEHPM.yaml
conda activate SEHPM
pip install git+https://github.com/openai/CLIP.git
```

### Vicuna
You can get the LLM Vicuna in [huggingface-vicuna-7b](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main)

Then set the downloaded vicuna folder path [here](minigpt4/configs/models/minigpt4_vicuna0.yaml) and the initial checkpoint [here](train_configs/minigpt4_stage2_finetune.yaml#L9)

### MiniGPT-4 Checkpoint
You can get the pretrained MiniGPT-4 checkpoint `pretrained_minigpt4_7b.pth`.

Then place it under `models/` and set the checkpoint path [here](train_configs/minigpt4_stage2_finetune.yaml).

### EVA_VIT_G
The code will automatically downloading the eva_vit_g.pth, we alse put it [huggingface](https://huggingface.co/lainxx/eva_vit_g/blob/main/eva_vit_g.pth), you can manually download it and put it in the cache dir: `.cache/torch/hub/checkpoints`

### bert-base-uncased
The code will automatically downloading this, but in case you don't have access to [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main), we also put it [here](https://pan.baidu.com/s/1XzAidcFinjsNxdz58M465w?pwd=b98f) , you can manually download it and alse put it in cache dir: `./SEHPM/tokenizers/bert-base-uncased`

## Dataset Processing

Before training, prepare the datasets using the scripts in `dataset_processing/`.  
For example:

```bash
python dataset_processing/cifar100_annotations.py
python dataset_processing/imagenet_r_annotations.py
python dataset_processing/imagenet_a_annotations.py
```

## Training

After setting all model and dataset config, you can run the following command to start fine-tuning.

```bash
python train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

## Testing
After training, you will get a model checkpoint of the last continual learning stage. put the path to scipts in eval_all.sh and specify a results directory.
Run the script:

```bash 
bash eval_all.sh
```

