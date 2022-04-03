import argparse

from fiftyone import ViewField as F
import fiftyone
import fiftyone.zoo as foz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", type=str, nargs="+", default=[])
    parser.add_argument("--image-paths", type=str, nargs="+", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    arg_values = get_args()

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split="test",
        label_types=["relationships"],
    )

    # Include only samples with the given IDs in the view
    selected_view = dataset.match(
        F("open_images_id").is_in(arg_values.image_ids) |
        F("filepath").is_in(arg_values.image_paths)
    )

    session = fiftyone.launch_app(dataset)
    session.view = selected_view
    session.wait()
