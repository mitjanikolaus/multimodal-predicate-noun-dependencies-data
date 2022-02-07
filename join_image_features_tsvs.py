"""Script for joining multiple image feature TSV files."""
import argparse
import pandas as pd

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+")
    parser.add_argument("--out", type=str, default="image_features_joined.tsv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    joined = None
    for file in args.files:
        print(file)
        data = pd.read_csv(file, delimiter='\t', header=None, names=FIELDNAMES, index_col="img_id")

        if joined is None:
            joined = data
        else:
            data = data[~data.index.isin(joined.index)]
            joined = joined.append(data)

    print("Writing file to ", args.out)
    joined.to_csv(args.out, header=None, index_label="img_id", sep='\t')
