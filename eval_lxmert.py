import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from transformers import LxmertTokenizer, LxmertForPreTraining

from eval_model import eval_2afc, load_bottom_up_image_features


def get_classification_score(model, tokenizer, test_sentence, visual_features, correct_sentence=None):
    inputs = tokenizer(test_sentence, return_tensors="pt")

    decoded_sequence = tokenizer.decode(inputs["input_ids"][0])
    print(decoded_sequence)

    visual_embeds = torch.tensor(visual_features['features']).unsqueeze(0)

    # Normalize the boxes (to 0 ~ 1)
    image_height = visual_features["img_h"]
    image_width = visual_features["img_w"]
    boxes = torch.tensor(visual_features['boxes'])

    boxes[:, (0, 2)] /= image_width
    boxes[:, (1, 3)] /= image_height
    np.testing.assert_array_less(boxes, 1 + 1e-5)
    np.testing.assert_array_less(-boxes, 0 + 1e-5)

    boxes = boxes.unsqueeze(0)

    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

    inputs.update({
        "visual_feats": visual_embeds,
        "visual_pos": boxes,
        "visual_attention_mask": visual_attention_mask
    })

    outputs = model(**inputs)

    cross_relationship_score = outputs.cross_relationship_score

    # Apply softmax
    softmaxed = F.softmax(cross_relationship_score[0], dim=0)
    # return the probability for a match:
    return softmaxed[1].data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument("--img-features-path", type=str, required=True)

    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    model_name = "unc-nlp/lxmert-base-uncased"
    if args.offline:
        tokenizer = LxmertTokenizer.from_pretrained(os.path.expanduser("~/data/transformers/"+model_name))
        model = LxmertForPreTraining.from_pretrained(os.path.expanduser("~/data/transformers/"+model_name))
    else:
        tokenizer = LxmertTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(os.path.expanduser("~/data/transformers/"+model_name))
        model = LxmertForPreTraining.from_pretrained(model_name)
        model.save_pretrained(os.path.expanduser("~/data/transformers/"+model_name))

    img_features = load_bottom_up_image_features(args.img_features_path)

    eval_2afc(model, tokenizer, img_features, get_classification_score, args)
