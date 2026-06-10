# Gemma TriviaQA Comparison

This project evaluates `google/gemma-2-2b` on the TriviaQA dataset and compares two deployment settings:

1. Cloud/API inference using FastAPI on RunPod GPU
2. Local/on-device inference using a PC GPU

Dataset version: `mandarjoshi/trivia_qa`, `rc.nocontext`.

The goal is to compare answer accuracy and resource usage during inference.

## Metrics

The project records:

- Accuracy
- Total inference time
- Average time per question
- GPU memory usage
- GPU power usage
- Estimated energy consumption

GPU memory and power are collected using `nvidia-smi`.

Energy is estimated as: `energy = inference time × GPU power`