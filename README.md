## Vision-Language Bird Reasoning with ViT 

This project explores fine-grained visual question answering (VQA) on bird species using a lightweight vision-language model (VLM). It leverages a TinyViT encoder and a Phi-3-mini LLM, trained on CUB-200-2011 and custom question-answer annotations.

### Project Structure
- `QnA_datapreprocess/`: Scripts to convert bird attribute data into natural language QA pairs.
- `vit_model/`: Custom ViT-Tiny architecture code.
- `train_flxible.py`: Full-featured ViT training script with tuning options.
- `train_mac.py`: Lightweight ViT training for macOS.
- `train_vlmv2.py`: Vision-language model training script (ViT + Phi-3).
- `vlmv2.py`: VLM architecture and dataloader with combined loss support.
- `eval-output.json`: QA evaluation results (exact match, keyword match, partial overlap).
- `attributes.txt`: Bird attributes used for QA generation.
- `preprocess.py`: Organizes the CUB dataset into train/val format.
- `tinyvit_cub200.pth`: Trained ViT weights on CUB.



### Model Training
# Train ViT
python train_flxible.py --exp_name scratch_vit_test

# Train VLM
python train_vlmv2.py

### Datasets
- **CUB-200-2011**: 11,788 images across 200 species (used for training the visual encoder).
- **Bird VQA (Custom)**: Natural language QA pairs per image generated from CUB attributes.
