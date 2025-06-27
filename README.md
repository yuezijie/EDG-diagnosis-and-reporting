# EDG-diagnosis-and-reporting

## Initialization

Create an environment with conda:
```
conda env create -f conda.yaml
conda activate mllm-edr-env
```
## Utilization

### Training
```
CUDA_VISIBLE_DEVICES=2 python main.py --save_dir results
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=2 python inference.py
```
Please note that some code is currently being updated and will be available shortly.
