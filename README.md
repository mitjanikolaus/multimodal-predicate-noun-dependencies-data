# Multimodal Evaluation

## Data

Download conceptual captions training data from
https://ai.google.com/research/ConceptualCaptions/download
and save to `data/conceptual_captions/`.

## Annotate

```
python annotate.py --input-file filtered_eval_sets/eval_set_shuffled.json --start-idx <start_idx> --end-idx <end_idx> --images-path <path_to_images>
```
(maximum end_idx: 2585)

## Acknowledgements

Image features for Visual BERT using detectron2 (for 1024 dimensional features): https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI

Image features for Visual BERT using detectron: https://github.com/huggingface/transformers/tree/master/examples/research_projects/visual_bert

Image features extraction using [bottom-up](https://github.com/peteanderson80/bottom-up-attention) (for 2048 dimensional features used by LXMert): https://github.com/airsplay/lxmert#faster-r-cnn-feature-extraction
