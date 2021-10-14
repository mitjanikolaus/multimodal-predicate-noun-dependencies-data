import itertools

import pandas as pd

from time import time

SUBJECT = "Label1"
REL = "label"
OBJECT = "Label2"

IMAGE_RELATIONSHIPS = "relationships.detections"
BOUNDING_BOX = "bounding_box"

def get_tuples_no_duplicates(names):
    all_tuples = [
        (a1, a2) for a1, a2 in list(itertools.product(names, names)) if a1 != a2
    ]
    tuples = []
    for (a1, a2) in all_tuples:
        if not (a2, a1) in tuples:
            tuples.append((a1, a2))
    return tuples


# Objects (Label2)

OBJECTS_TEXTURES = [
    "Wooden",
    "Plastic",
    "Transparent",
    "(made of)Leather",
    "(made of)Textile",
]
OBJECTS_TEXTURES_TUPLES = get_tuples_no_duplicates(OBJECTS_TEXTURES)

OBJECTS_INSTRUMENTS = [
    "French horn",
    "Piano",
    "Saxophone",
    "Guitar",
    "Violin",
    "Trumpet",
    "Accordion",
    "Microphone",
    "Cello",
    "Trombone",
    "Flute",
    "Drum",
    "Musical keyboard",
    "Banjo"
]

OBJECTS_VEHICLES = [
    "Car",
    "Motorcycle",
    "Bicycle",
    "Horse",
    "Roller skates",
    "Skateboard",
    "Cart",
    "Bus",
    "Wheelchair",
    "Boat",
    "Canoe",
    "Truck",
    "Train",
    "Tank",
    "Airplane",
    "Van"
]

OBJECTS_ANIMALS = ["Dog", "Cat", "Horse", "Elephant"]

OBJECTS_VERBS = [
    "Smile",
    "Cry",
    "Talk",
    "Sing",
    "Sit",
    "Walk",
    "Lay",
    "Jump",
    "Run",
    "Stand",
]

OBJECTS_FURNITURE = ["Table", "Chair", "Bench", "Bed", "Sofa bed", "Billiard table"]

OBJECTS_OTHERS = [
    "Glasses",
    "Bottle",
    "Wine glass",
    "Coffee cup",
    "Sun hat",
    "Bicycle helmet",
    "High heels",
    "Necklace",
    "Scarf",
    "Belt",
    "Swim cap",
    "Handbag",
    "Crown",
    "Football",
    "Baseball glove",
    "Baseball bat",
    "Racket",
    "Surfboard",
    "Paddle",
    "Camera",
    "Mobile phone",
    "Houseplant",
    "Coffee",
    "Tea",
    "Cocktail",
    "Juice",
    "Cake",
    "Strawberry",
    "Wine",
    "Beer",
    "Woman",
    "Man",
    "Tent",
    "Tree",
    "Girl",
    "Boy",
    "Balloon",
    "Rifle",
    "Earrings",
    "Teddy bear",
    "Doll",
    "Bicycle wheel",
    "Ski",
    "Backpack",
    "Ice cream",
    "Book",
    "Cutting board",
    "Watch",
    "Tripod",
    "Rose"
]

OBJECTS_OTHERS += (
    OBJECTS_INSTRUMENTS
    + OBJECTS_VEHICLES
    + OBJECTS_ANIMALS
    + OBJECTS_VERBS
    + OBJECTS_FURNITURE
)
OBJECTS_OTHERS_TUPLES = get_tuples_no_duplicates(OBJECTS_OTHERS)

OBJECTS_TUPLES = OBJECTS_OTHERS_TUPLES + OBJECTS_TEXTURES_TUPLES

# Nouns (Label1)
SUBJECTS_FRUITS = [
    "Orange",
    "Strawberry",
    "Lemon",
    "Apple",
    "Coconut",
]

SUBJECTS_ACCESSORIES = ["Handbag", "Backpack", "Suitcase"]

SUBJECTS_FURNITURE = ["Chair", "Table", "Sofa bed", "Bed", "Bench"]

SUBJECTS_INSTRUMENTS = ["Piano", "Guitar", "Drum", "Violin"]

SUBJECTS_ANIMALS = ["Dog", "Cat"]

SUBJECTS_OTHERS = [
    "Wine glass",
    "Cake",
    "Beer",
    "Mug",
    "Bottle",
    "Bowl",
    "Flowerpot",
    "Chopsticks",
    "Platter",
    "Ski",
    "Candle",
    "Fork",
    "Spoon",
]

SUBJECTS = (
    SUBJECTS_OTHERS
    + SUBJECTS_FURNITURE
    + SUBJECTS_FRUITS
    + SUBJECTS_ACCESSORIES
    + SUBJECTS_INSTRUMENTS
    + SUBJECTS_ANIMALS
)

SUBJECTS_GENERAL_TUPLES = get_tuples_no_duplicates(SUBJECTS)

SUBJECTS_OTHERS_TUPLES = [
    ("Man", "Woman"),
    ("Man", "Girl"),
    ("Woman", "Boy"),
    ("Girl", "Boy"),
]

SUBJECT_TUPLES = SUBJECTS_GENERAL_TUPLES + SUBJECTS_OTHERS_TUPLES

# Relationships (.label)
RELATIONSHIPS_SPATIAL = ["at", "contain", "holds", "on", "hang", "inside_of", "under"]
RELATIONSHIPS_SPATIAL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_SPATIAL)

RELATIONSHIPS_BALL = ["throw", "catch", "kick", "holds", "hits"]
RELATIONSHIPS_BALL_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_BALL)

RELATIONSHIPS_OTHERS = [
    "eat",
    "drink",
    "read",
    "dance",
    "kiss",
    "skateboard",
    "surf",
    "ride",
    "hug",
    "plays",
]
RELATIONSHIPS_OTHERS_TUPLES = get_tuples_no_duplicates(RELATIONSHIPS_OTHERS)

RELATIONSHIPS = RELATIONSHIPS_SPATIAL + RELATIONSHIPS_BALL + RELATIONSHIPS_OTHERS
RELATIONSHIPS_TUPLES = (
    RELATIONSHIPS_SPATIAL_TUPLES
    + RELATIONSHIPS_BALL_TUPLES
    + RELATIONSHIPS_OTHERS_TUPLES
)

subjects_counter = pd.read_csv(
    "data/subject_occurrences.csv", index_col=None, header=None, names=["subject", "count"]
)
SUBJECT_NAMES = list(subjects_counter["subject"].values)

relationships_counter = pd.read_csv(
    "data/rel_occurrences.csv", index_col=None, header=None, names=["rel", "count"]
)
RELATIONSHIP_NAMES = list(relationships_counter["rel"].values)

objects_counter = pd.read_csv(
    "data/obj_occurrences.csv", index_col=None, header=None, names=["obj", "count"]
)
OBJECT_NAMES = list(objects_counter["obj"].values)

for obj1, obj2 in OBJECTS_TUPLES:
    assert obj1 in OBJECT_NAMES, f"{obj1} is misspelled"
    assert obj2 in OBJECT_NAMES, f"{obj2} is misspelled"

for subj1, subj2 in SUBJECT_TUPLES:
    assert subj1 in SUBJECT_NAMES, f"{subj1} is misspelled"
    assert subj2 in SUBJECT_NAMES, f"{subj2} is misspelled"

SYNONYMS_LIST = [
    ["Table", "Desk", "Coffee table"],
    ["Mug", "Coffee cup"],
    ["Glasses", "Sunglasses", "Goggles"],
    ["Sun hat", "Fedora", "Cowboy hat", "Sombrero"],
    ["Bicycle helmet", "Football helmet"],
    ["High heels", "Sandal", "Boot"],
    ["Racket", "Tennis racket", "Table tennis racket"],
    ["Crown", "Tiara"],
    ["Handbag", "Briefcase"],
    ["Cart", "Golf cart"],
    ["Football", "Volleyball (Ball)", "Rugby ball", "Cricket ball", "Tennis ball"],
    ["Tree", "Palm tree"]
]

SYNONYMS = {name: [name] for name in SUBJECT_NAMES + OBJECT_NAMES + RELATIONSHIP_NAMES}
for synonyms in SYNONYMS_LIST:
    SYNONYMS.update({item: synonyms for item in synonyms})

VALID_NAMES = {"label": RELATIONSHIPS, "Label2": OBJECT_NAMES}

REFERENCE_TIME = time()
TIME_LOG_THRESHOLD = 0.1


def log_time(message, reference_time=None):
    if reference_time:
        ref = reference_time
    else:
        global REFERENCE_TIME
        ref = REFERENCE_TIME

    current_time = time()
    timedelta = current_time - ref

    if timedelta > TIME_LOG_THRESHOLD:
        line = "="*40
        print(line)
        print(f"{message} | Time passed: {timedelta:.1f}s")
        print(line)
        print()

    if not reference_time:
        REFERENCE_TIME = current_time
