import argparse
import os
import pickle

from PyQt5 import QtGui, QtCore
import sys

import fiftyone

from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QMessageBox,
    QPushButton,
    QGridLayout,
    QWidget,
    QLabel,
    QShortcut,
)

from utils import SYNONYMS, SUBJECT, OBJECT, get_target_and_distractor_sentence, BOUNDING_BOX, REL

EXCLUDED_OBJECTS = ["Smile", "Talk", "Table", "Coffee table", "Desk", "Chair", "Bench", "Car", "High heels", "Man", "Woman", "Girl", "Boy"]


class EvalSetFilter(QWidget):
    def __init__(self, args):
        super().__init__()

        self.input_file = args.input_file
        print("Loading: ", self.input_file)
        self.eval_sets = pickle.load(open(self.input_file, "rb"))
        print("Done.")

        total = 0
        for key, samples in self.eval_sets.items():
            if len(samples) > 0:
                samples = [sample for sample in samples if
                           sample["relationship_target"][OBJECT] not in EXCLUDED_OBJECTS]

                if "subject" in self.input_file:
                    # Sort values by object
                    self.eval_sets[key] = sorted(samples, key=lambda x: x["relationship_target"][OBJECT])

                elif "object" in self.input_file:
                    # Sort values by subject
                    self.eval_sets[key] = sorted(samples, key=lambda x: x["relationship_target"][SUBJECT])

                print(f"{key}: {len(samples)})")
                total += len(samples)

        print("Total: ", total)

        if args.continue_from:
            print("Continuing from: ", args.continue_from)
            self.eval_sets_filtered = pickle.load(open(args.continue_from, "rb"))
            self.sample_index = int(args.continue_from.split("sample_")[1].split(".p")[0]) - 1
            self.eval_set_index = int(args.continue_from.split("eval_set_")[1].split("_sample")[0]) - 1

            filename_rejected_examples = args.continue_from.replace("_filtered_", "_rejected_examples_")
            print(f"Loading rejected examples from {filename_rejected_examples}")
            self.eval_sets_rejected_examples = pickle.load(open(filename_rejected_examples, "rb"))

        else:
            self.eval_sets_filtered = {key: [] for key in self.eval_sets.keys()}
            self.eval_set_index = 0
            self.sample_index = 0

            self.eval_sets_rejected_examples = {key: [] for key in self.eval_sets.keys()}

        self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]

        while len(self.eval_sets[self.eval_set_key]) < 1:
            self.eval_set_index += 1
            self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]

        self.eval_set = self.eval_sets[self.eval_set_key]
        self.sample = self.eval_set[self.sample_index]

        grid = QGridLayout()
        self.setLayout(grid)

        self.text_title = QLabel(self)
        self.text_title.setFixedHeight(30)
        self.text_title.setFont(QFont("Arial", 20))
        self.text_title.setAlignment(QtCore.Qt.AlignCenter)
        grid.addWidget(self.text_title, 0, 0, 1, 2)

        self.pic_example = QLabel(self)
        grid.addWidget(self.pic_example, 1, 0)
        self.pic_counterexample = QLabel(self)
        grid.addWidget(self.pic_counterexample, 1, 1)

        self.text_example_target = QLabel(self)
        self.text_example_target.setFixedHeight(15)
        grid.addWidget(self.text_example_target, 2, 0)
        self.text_example_distractor = QLabel(self)
        self.text_example_distractor.setFixedHeight(15)
        grid.addWidget(self.text_example_distractor, 3, 0)
        self.text_example_filepath = QLabel(self)
        self.text_example_filepath.setFixedHeight(15)
        self.text_example_filepath.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        grid.addWidget(self.text_example_filepath, 4, 0)

        self.text_counterexample_target = QLabel(self)
        self.text_counterexample_target.setFixedHeight(15)
        grid.addWidget(self.text_counterexample_target, 2, 1)
        self.text_counterexample_distractor = QLabel(self)
        self.text_counterexample_distractor.setFixedHeight(15)
        grid.addWidget(self.text_counterexample_distractor, 3, 1)
        self.text_counterexample_filepath = QLabel(self)
        self.text_counterexample_filepath.setFixedHeight(15)
        self.text_counterexample_filepath.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        grid.addWidget(self.text_counterexample_filepath, 4, 1)

        self.pic_example.show()
        self.pic_counterexample.show()
        self.text_example_target.show()
        self.text_example_distractor.show()
        self.text_example_filepath.show()
        self.text_counterexample_target.show()
        self.text_counterexample_distractor.show()
        self.text_counterexample_filepath.show()

        button_accept = QPushButton("accept (ctrl+a)")
        button_accept.clicked.connect(self.accept)
        grid.addWidget(button_accept, 5, 0, 1, 2)
        shortcut_accept = QShortcut(QKeySequence("Ctrl+A"), self)
        shortcut_accept.activated.connect(self.accept)

        button_reject = QPushButton("reject example (ctrl+e)")
        button_reject.clicked.connect(self.reject_example)
        grid.addWidget(button_reject, 6, 0)
        shortcut_reject = QShortcut(QKeySequence("Ctrl+E"), self)
        shortcut_reject.activated.connect(self.reject_example)

        button_reject_counterexample = QPushButton("reject counterexample (ctrl+r)")
        button_reject_counterexample.clicked.connect(self.reject_counterexample)
        grid.addWidget(button_reject_counterexample, 6, 1)
        shortcut_reject = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut_reject.activated.connect(self.reject_counterexample)

        button_next_eval_set = QPushButton("Go to next eval set")
        button_next_eval_set.clicked.connect(self.goto_next_eval_set_and_load_sample)
        grid.addWidget(button_next_eval_set, 7, 0, 1, 2)

        button_save = QPushButton("Save filtered results")
        button_save.clicked.connect(self.save)
        grid.addWidget(button_save, 8, 0, 1, 2)

        self.setWindowTitle("Filter eval set")

        self.plot_sample()

        self.show()

    def plot_sample(self):
        title_text = f"{self.eval_set_key} (sample {self.sample_index+1}/{len(self.eval_set)}) (eval set {self.eval_set_index+1}/{len(self.eval_sets)})"
        self.text_title.setText(title_text)

        # set rectangle colors and thicknesses
        penRectangleGreen = QtGui.QPen(QtCore.Qt.green)
        penRectangleGreen.setWidth(3)
        penRectangleRed = QtGui.QPen(QtCore.Qt.red)
        penRectangleRed.setWidth(3)
        penWhite = QtGui.QPen(QtCore.Qt.white)
        penWhite.setWidth(3)

        text_target, text_distractor = get_target_and_distractor_sentence(self.sample)

        self.text_example_target.setText("Target: " + text_target)
        self.text_example_distractor.setText("Distractor: " + text_distractor)
        img_path_example = get_local_fiftyone_image_path(self.sample["img_example"])
        self.text_example_filepath.setText(img_path_example)

        self.text_counterexample_target.setText("Target: " + text_distractor)
        self.text_counterexample_distractor.setText("Distractor: " + text_target)
        img_path_counterexample = get_local_fiftyone_image_path(self.sample["img_counterexample"])
        self.text_counterexample_filepath.setText(img_path_counterexample)

        for img_path in [img_path_example, img_path_counterexample]:
            pixmap = QtGui.QPixmap(img_path)
            pixmap = pixmap.scaledToWidth(500)

            # create painter instance with pixmap
            self.painterInstance = QtGui.QPainter(pixmap)

            relationships = [
                self.sample["relationship_target"],
                self.sample["relationship_visual_distractor"],
            ]
            if img_path == img_path_counterexample:
                relationships = [
                    self.sample["counterexample_relationship_target"],
                    self.sample["counterexample_relationship_visual_distractor"],
                ]

            for relationship, pen in zip(
                relationships, [penRectangleGreen, penRectangleRed]
            ):
                bb = relationship[BOUNDING_BOX]

                # draw rectangle on painter
                self.painterInstance.setPen(pen)
                self.painterInstance.drawRect(
                    round(bb[0] * pixmap.width()),
                    round(bb[1] * pixmap.height()),
                    round(bb[2] * pixmap.width()),
                    round(bb[3] * pixmap.height()),
                )
                label = (
                    f"{relationship[SUBJECT]} {relationship[REL]} {relationship[OBJECT]}"
                )
                self.painterInstance.setPen(penWhite)
                self.painterInstance.drawText(
                    round(bb[0] * pixmap.width()), round(bb[1] * pixmap.height()), label
                )

            self.painterInstance.end()

            if img_path == img_path_counterexample:
                self.pic_counterexample.setPixmap(pixmap)
            else:
                self.pic_example.setPixmap(pixmap)

    def next_sample(self):
        self.sample_index += 1
        while self.sample_index >= len(self.eval_set):
            self.goto_next_eval_set()

        return self.eval_set[self.sample_index]

    def sample_already_processed(self, sample):
        """Return true if either example or counterexample already exist in the filtered or rejected samples"""
        rel_label = sample["rel_label"]

        def relationships_are_equal(s1, s2):
            return (
                s1["relationship_target"][SUBJECT]
                in SYNONYMS[s2["relationship_target"][SUBJECT]]
                and s1["relationship_target"][rel_label]
                in SYNONYMS[s2["relationship_target"][rel_label]]
                and s1["counterexample_relationship_target"][SUBJECT]
                in SYNONYMS[s2["counterexample_relationship_target"][SUBJECT]]
                and s1["counterexample_relationship_target"][rel_label]
                in SYNONYMS[s2["counterexample_relationship_target"][rel_label]]
            )

        for s in self.eval_sets_filtered[self.eval_set_key]:
            if (
                s["img_example"] == sample["img_example"]
                or s["img_counterexample"] == sample["img_counterexample"]
                or s["img_example"] == sample["img_counterexample"]
            ):
                if relationships_are_equal(s, sample):
                    return True

        for s in self.eval_sets_rejected_examples[self.eval_set_key]:
            if (
                s["rejected_image"] == sample["img_example"]
                or s["rejected_image"] == sample["img_counterexample"]
            ):
                if relationships_are_equal(s, sample):
                    return True

        return False

    def get_next_sample(self):
        sample = self.next_sample()
        example_in_filtered_eval_sets = self.sample_already_processed(sample)
        while example_in_filtered_eval_sets:
            sample = self.next_sample()
            example_in_filtered_eval_sets = self.sample_already_processed(sample)

        return sample

    def goto_next_eval_set_and_load_sample(self):
        self.goto_next_eval_set()
        self.sample = self.eval_set[self.sample_index]
        self.plot_sample()

    def goto_next_eval_set(self):
        self.sample_index = 0

        self.eval_set_index += 1
        if self.eval_set_index >= len(self.eval_sets.keys()):
            self.save()
            QMessageBox.information(
                self,
                "QMessageBox.information()",
                "End of dataset, filtered everything!",
            )
            self.close()
            sys.exit()
        self.eval_set_key = list(self.eval_sets.keys())[self.eval_set_index]
        self.eval_set = self.eval_sets[self.eval_set_key]

    def accept(self):
        self.eval_sets_filtered[self.eval_set_key].append(self.sample)
        self.sample = self.get_next_sample()
        self.plot_sample()

    def reject_example(self):
        rejected_example = self.sample
        rejected_example["rejected_image"] = rejected_example["img_example"]
        self.eval_sets_rejected_examples[self.eval_set_key].append(rejected_example)
        self.sample = self.get_next_sample()
        self.plot_sample()

    def reject_counterexample(self):
        rejected_example = self.sample
        rejected_example["rejected_image"] = rejected_example["img_counterexample"]
        self.eval_sets_rejected_examples[self.eval_set_key].append(rejected_example)
        self.sample = self.get_next_sample()
        self.plot_sample()

    def save(self):
        file_name = os.path.basename(self.input_file).replace(".p", f"_filtered_eval_set_{self.eval_set_index + 1}_sample_{self.sample_index + 1}.p")
        pickle.dump(
            self.eval_sets_filtered, open(os.path.join("filtered_eval_sets", file_name), "wb",),
        )
        file_name = os.path.basename(self.input_file).replace(".p", f"_rejected_examples_eval_set_{self.eval_set_index + 1}_sample_{self.sample_index + 1}.p")
        pickle.dump(
            self.eval_sets_rejected_examples, open(os.path.join("filtered_eval_sets", file_name), "wb",)
        )
        summary = ""
        for key, values in self.eval_sets_filtered.items():
            if len(values) > 0:
                summary += f"\n{key}: {len(values)}"
        QMessageBox.information(
            self,
            "QMessageBox.information()",
            f"Saved filtered results.\nSummary:{summary}",
        )


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input-file", type=str, required=True,
    )
    argparser.add_argument(
        "--continue-from", type=str, required=False,
    )

    args = argparser.parse_args()

    return args


def get_local_fiftyone_image_path(img_path):
    return os.path.join(*[fiftyone.config.dataset_zoo_dir, "open-images-v6", img_path.split("open-images-v6/")[1]])


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    ex = EvalSetFilter(args)
    sys.exit(app.exec_())
