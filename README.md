## 上传新版本的文件


```
- _src
  - datasets
    - data_field.py
    - depth_warp_dataloader.py
    - radym.py
    - ...
  - models
    - lyra2_model.py
    - ...
  - train
    - lyra_samples/DL3DV-ALL-480P-lyra-sample8
    - train_distill_SF_dmd_lora_v2.py
    - run_8gpus_lora_v2.sh
    - ...

- _ext
  - imaginaire
    - checkpoiter
      - dcp.py
```

## 运行命令
```
SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_INPUT_FRAMES=48 
source /public/bin/network_accelerate
./run_8gpus_lora_v2
```
