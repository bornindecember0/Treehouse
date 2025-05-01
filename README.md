Vision-Language Bird Classification & QA Project
This project combines a Vision Transformer (ViT) and a pretrained language model to perform fine-grained bird classification and visual question answering (VQA) using the CUB-200-2011 dataset.

Project Structure
├── QnA_datapreprocess/         # Scripts to convert bird attributes into question-answer pairs
├── vit_logs/                   # Logs for ViT training and evaluation
├── vit_model/                  # ViT architecture and utilities (self-built)
├── attributes.txt              # List of attributes (used in QA generation)
├── eval-output.json            # Evaluation results from VLM
├── preprocess.py              # Script for organizing and preprocessing CUB dataset
├── tinyvit_cub200.pth         # Checkpoint of TinyViT trained on CUB-200-2011
├── train.launcher             # Launcher script (possibly outdated)
├── train.py                   # Main ViT training script
├── train_flxible.py           # Possibly duplicate ViT script (see below)
├── train_mac.py               # Mac-compatible ViT training script
├── train_vlm.py               # VLM training script (ViT + LLM)
├── train_vlmv2.py             # Possibly updated or refactored version of `train_vlm.py`
├── vlmv2.py                   # Vision-Language model definition

Codes Usage
1. Preprocess the Dataset
Use preprocess.py to organize images from CUB-200-2011 into train/val folders.

2. Train Vision Transformer (ViT)
Use train.py or train_mac.py depending on your system. These scripts train a TinyViT on bird classification.

python train.py           # For general GPU training
python train_mac.py       # For Mac compatibility (no CUDA)

3. Convert Attributes into QA Format
Use scripts in QnA_datapreprocess to generate natural language question-answer pairs for VQA training.

4. Train the Vision-Language Model (VLM)
Use train_vlm.py or train_vlmv2.py to connect frozen ViT embeddings with a pretrained LLM (Qwen2.5-0.5B-Instruct).

5. Run Evaluation
Evaluation results of VLM (keyword match, partial overlap, exact match) are saved in eval-output.json.

