# Real_CQA_ICDAR_2023

## Input image and annotations: 
- test 2022: https://drive.google.com/file/d/1DwTiFrjObCBdxnKCPe8lgl9r3meHxkYc/view?usp=share_link
- train 2022: https://drive.google.com/file/d/1UxWY4tD3uwV23MXj_5xzSMc4wkVopc9J/view?usp=share_link
- train 2023: https://drive.google.com/file/d/1ELfLNrOvqB01ho1qTXbSb63ttrl5fEJT/view?usp=share_link

### Input data: https://drive.google.com/file/d/1W4lYUJAv8OtoSbgaHr7GGYdXVPqWUh01/view?usp=share_link
### images with annotations only till task 6: saleem_qid_formatted_list.pkl
### images and corresponding QAs used for VLT5 and CRCT: saleem_qid_formatted_list.pkl

## changing data in order to make input for ChartQA: making_data_for_chartqa.ipynb

## Predictions
- Predictions for original VLT5 model on PMC dataset: https://drive.google.com/file/d/1jaRdBlRNXm2iA77UZPJZ2QM5EExTj85U/view?usp=share_link
- Predictions for Finetuned VLT5 model on PMC dataset: https://drive.google.com/file/d/1hmkFmMMjPBVeG3IFiN6MCP1p0L4V6Epv/view?usp=sharing

## Training and Inference using shell files:
- Training: ChartQA/Models/VL-T5/scripts/VQA_VLT5.sh
- Inference for original VLT5 model: ChartQA/Models/VL-T5/scripts/VQA_VLT5_inference_origvlt5.sh
- Inference for Finetuned VLT5 model: ChartQA/Models/VL-T5/scripts/VQA_VLT5_inference_ftvlt5_pmc.sh

## source code for VLT5: Real_CQA/ChartQA/Models/VL-T5/src/

## evaluation scripts: Real_CQA/evaluation_PMC/

## environment files:
- env for VLT5: /home/spandey8/Real_CQA/ChartQA/vlt5.yaml
- env for creating MaskRCNN features: /home/spandey8/detectron.yaml
- llm evaluation env: /home/spandey8/Real_CQA/llm_evaluation_PMC/env.yaml

## changing format of the predictions from VLT5 model, to format that can be used for evaluation:
- Finetuned model predictions: changing formats/changing_format_to_evaluation_format_from_ft_pred.py
- Original model predictions: changing formats/changing_format_to_evaluation_format_from_pt_pred.py

