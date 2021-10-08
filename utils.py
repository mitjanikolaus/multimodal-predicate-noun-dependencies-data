import itertools


def get_tuples_no_duplicates(names):
    all_tuples = [(a1, a2) for a1, a2 in list(itertools.product(names, names)) if a1 != a2]
    tuples = []
    for (a1, a2) in all_tuples:
        if not (a2, a1) in tuples:
            tuples.append((a1, a2))
    return tuples


####ATTRIBUTES (Label2)
OBJECTS = ["Houseplant", "Coffee", "Tea", "Cake"]
OBJECTS_TUPLES = get_tuples_no_duplicates(OBJECTS)

TEXTURES = ["Wooden", "Plastic", "Transparent", "(made of)Leather", "(made of)Textile"]
TEXTURES_TUPLES = get_tuples_no_duplicates(TEXTURES)

INSTRUMENTS = [
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
    "Harp",
    "Flute",
    "Drum",
    "Musical keyboard",
]

VEHICLES = [
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
]

ANIMALS = ["Dog", "Cat", "Horse", "Elephant"]

VERBS = ["Smile", "Cry", "Talk", "Sing", "Sit", "Walk", "Lay", "Jump", "Run", "Stand"]

BALLS = ["Football", "Volleyball (Ball)", "Rugby ball", "Cricket ball", "Tennis ball"]

FURNITURE = ["Table", "Chair", "Wheelchair"]

ATTRIBUTES_PERSONS = [
    "Glasses",
    "Sun hat",
    "Bicycle helmet",
    "High heels",
    "Necklace",
    "Scarf",
    "Swim cap",
    "Handbag",
    "Crown",
    "Roller skates",
    "Skateboard",
    "Baseball glove",
    "Baseball bat",
    "Racket",
    "Surfboard",
    "Paddle",
    "Dog",
    "Table",
    "Chair",
    "Camera",
]
ATTRIBUTES_PERSONS += INSTRUMENTS + VEHICLES + BALLS + ANIMALS + VERBS + FURNITURE
ATTRIBUTES_PERSONS_TUPLES = get_tuples_no_duplicates(ATTRIBUTES_PERSONS)

ATTRIBUTE_TUPLES = ATTRIBUTES_PERSONS_TUPLES + TEXTURES_TUPLES + OBJECTS_TUPLES

### Nouns (Label1)
NOUNS_FRUITS = [
    "Orange",
    "Grapefruit",
    "Strawberry",
    "Lemon",
    "Grape",
    "Peach",
    "Apple",
    "Pear",
    "Coconut",
]

NOUNS_ACCESSORIES = ["Handbag", "Backpack", "Suitcase"]

NOUNS_FURNITURE = ["Chair", "Table", "Sofa bed", "Bed"]

NOUNS_INSTRUMENTS = ["Piano", "Guitar", "Drum", "Violin"]

NOUNS_ANIMALS = ["Dog"]

NOUNS_OBJECTS = [
    "Wine glass",
    "Mug",
    "Bottle",
    "Bowl",
    "Flowerpot",
    "Chopsticks",
    "Platter",
    "Ski",
]

NOUNS_OBJECTS += (
    NOUNS_FURNITURE
    + NOUNS_FRUITS
    + NOUNS_ACCESSORIES
    + NOUNS_INSTRUMENTS
    + NOUNS_ANIMALS
)

NOUNS_OBJECTS_TUPLES = get_tuples_no_duplicates(NOUNS_OBJECTS)

NOUNS_TUPLES_OTHER = [
    ("Man", "Woman"),
    ("Man", "Girl"),
    ("Woman", "Boy"),
    ("Girl", "Boy"),
]

NOUN_TUPLES = NOUNS_OBJECTS_TUPLES + NOUNS_TUPLES_OTHER

nouns_counter = [
    ("Man", 31890),
    ("Woman", 19579),
    ("Girl", 15104),
    ("Boy", 4375),
    ("Chair", 2665),
    ("Table", 2526),
    ("Bottle", 1139),
    ("Coffee cup", 677),
    ("Desk", 472),
    ("Flowerpot", 455),
    ("Piano", 323),
    ("Wine glass", 319),
    ("Guitar", 316),
    ("Handbag", 287),
    ("Orange", 244),
    ("Sofa bed", 207),
    ("Grapefruit", 175),
    ("Mug", 164),
    ("Bed", 164),
    ("Drum", 142),
    ("Backpack", 136),
    ("Suitcase", 134),
    ("Strawberry", 124),
    ("Coffee table", 121),
    ("Platter", 117),
    ("Violin", 96),
    ("Lemon", 86),
    ("Ski", 76),
    ("Grape", 62),
    ("Chopsticks", 54),
    ("Peach", 52),
    ("Apple", 50),
    ("Pear", 49),
    ("Dog", 42),
    ("Coconut", 40),
    ("Bowl", 37),
    ("Common fig", 34),
    ("Briefcase", 30),
    ("Fork", 28),
    ("Spoon", 25),
    ("Cat", 24),
    ("Knife", 16),
    ("Bench", 15),
    ("Beer", 12),
    ("Pomegranate", 10),
    ("Mango", 9),
    ("Tomato", 7),
    ("Kitchen knife", 6),
    ("Candle", 6),
    ("Cake", 5),
    ("Picnic basket", 5),
    ("Artichoke", 4),
    ("Teapot", 3),
    ("Pineapple", 2),
    ("Carrot", 2),
    ("Cheese", 1),
    ("Pizza", 1),
    ("Cucumber", 1),
    ("Bread", 1),
]
NOUN_NAMES = [name for name, _ in nouns_counter]

relationships_counter = [
    ("is", 68343),
    ("wears", 4668),
    ("at", 2095),
    ("contain", 1618),
    ("holds", 1587),
    ("ride", 1071),
    ("on", 985),
    ("hang", 835),
    ("plays", 460),
    ("interacts_with", 339),
    ("inside_of", 281),
    ("skateboard", 119),
    ("surf", 86),
    ("hits", 66),
    ("kick", 54),
    ("throw", 49),
    ("catch", 25),
    ("drink", 19),
    ("read", 15),
    ("eat", 14),
    ("under", 10),
    ("ski", 7),
]

attributes_counter = [
    ("Stand", 29768),
    ("Smile", 12492),
    ("Sit", 11017),
    ("Wooden", 4186),
    ("Walk", 2723),
    ("Table", 1850),
    ("Run", 1724),
    ("Plastic", 1684),
    ("Glasses", 931),
    ("Talk", 922),
    ("Transparent", 917),
    ("Lay", 822),
    ("Tree", 807),
    ("(made of)Leather", 707),
    ("Jump", 608),
    ("Sunglasses", 544),
    ("Roller skates", 489),
    ("(made of)Textile", 481),
    ("Houseplant", 432),
    ("Horse", 414),
    ("Coffee", 390),
    ("Car", 376),
    ("High heels", 315),
    ("Sandal", 314),
    ("Bicycle", 305),
    ("Tea", 289),
    ("Goggles", 288),
    ("Desk", 278),
    ("Sun hat", 264),
    ("Sing", 237),
    ("Fedora", 228),
    ("Dog", 222),
    ("Wine", 210),
    ("Boat", 209),
    ("Bicycle helmet", 183),
    ("Cowboy hat", 180),
    ("Canoe", 158),
    ("Football helmet", 157),
    ("Wheelchair", 152),
    ("Chair", 151),
    ("Boot", 149),
    ("Guitar", 144),
    ("Skateboard", 142),
    ("Football", 136),
    ("Necklace", 120),
    ("Baseball glove", 114),
    ("Surfboard", 110),
    ("Cake", 108),
    ("Tennis racket", 105),
    ("Paddle", 103),
    ("Motorcycle", 93),
    ("Scarf", 89),
    ("Trumpet", 87),
    ("Table tennis racket", 86),
    ("Swim cap", 82),
    ("Cart", 81),
    ("Racket", 81),
    ("French horn", 80),
    ("Violin", 80),
    ("Camera", 69),
    ("Coffee table", 66),
    ("Bed", 62),
    ("Beer", 61),
    ("Bus", 60),
    ("Volleyball (Ball)", 56),
    ("Cry", 55),
    ("Sushi", 55),
    ("Handbag", 53),
    ("Tiara", 53),
    ("Microphone", 52),
    ("Cello", 52),
    ("Trombone", 51),
    ("Baseball bat", 51),
    ("Crown", 48),
    ("Cocktail", 48),
    ("Juice", 46),
    ("Accordion", 43),
    ("Segway", 43),
    ("Saxophone", 43),
    ("Balance beam", 43),
    ("Billiard table", 42),
    ("Sombrero", 41),
    ("Piano", 41),
    ("Rifle", 40),
    ("Harp", 38),
    ("Flute", 38),
    ("Drum", 35),
    ("Musical keyboard", 33),
    ("Bicycle wheel", 32),
    ("Bottle", 31),
    ("Hiking equipment", 30),
    ("Earrings", 29),
    ("Rugby ball", 28),
    ("Bow and arrow", 25),
    ("Palm tree", 24),
    ("Rose", 24),
    ("Wine glass", 23),
    ("Golf cart", 23),
    ("Elephant", 22),
    ("Balloon", 22),
    ("Belt", 21),
    ("Oyster", 20),
    ("Infant bed", 19),
    ("Watermelon", 19),
    ("Ski", 17),
    ("Cutting board", 17),
    ("Book", 15),
    ("Salad", 15),
    ("Tripod", 15),
    ("Orange", 15),
    ("Shotgun", 14),
    ("Dumbbell", 14),
    ("Grape", 13),
    ("Strawberry", 13),
    ("Dog bed", 12),
    ("Coffee cup", 11),
    ("Harpsichord", 11),
    ("Mobile phone", 11),
    ("Gondola", 10),
    ("Cat", 10),
    ("Apple", 10),
    ("Truck", 10),
    ("Grapefruit", 10),
    ("Bowling equipment", 9),
    ("Horizontal bar", 9),
    ("Tennis ball", 9),
    ("Stationary bicycle", 9),
    ("Tent", 8),
    ("Airplane", 8),
    ("Watch", 8),
    ("Organ (Musical Instrument)", 8),
    ("Ladder", 8),
    ("Sofa bed", 7),
    ("Muffin", 7),
    ("Suitcase", 7),
    ("Unicycle", 6),
    ("Backpack", 6),
    ("Binoculars", 6),
    ("Van", 6),
    ("Doll", 6),
    ("Handgun", 5),
    ("Limousine", 5),
    ("Cake stand", 5),
    ("Lobster", 5),
    ("Ambulance", 5),
    ("Panda", 5),
    ("Cricket ball", 5),
    ("Countertop", 5),
    ("Stool", 4),
    ("Banjo", 4),
    ("Cat furniture", 4),
    ("Indoor rower", 4),
    ("Monkey", 4),
    ("Bench", 4),
    ("Cannon", 3),
    ("Helicopter", 3),
    ("Oboe", 3),
    ("Sword", 3),
    ("Tank", 3),
    ("Flying disc", 3),
    ("Teddy bear", 3),
    ("Washing machine", 3),
    ("Treadmill", 2),
    ("Box", 2),
    ("Ice cream", 2),
    ("Taxi", 2),
    ("Studio couch", 2),
    ("Train", 2),
    ("Carrot", 2),
    ("Candy", 2),
    ("Wok", 2),
    ("Common sunflower", 2),
    ("Sea lion", 2),
    ("Stethoscope", 2),
    ("Jet ski", 2),
    ("Pen", 2),
    ("Kitchen & dining room table", 2),
    ("Pomegranate", 2),
    ("Milk", 1),
    ("Plate", 1),
    ("Honeycomb", 1),
    ("Egg (Food)", 1),
    ("Lizard", 1),
    ("Loveseat", 1),
    ("Dolphin", 1),
    ("Whale", 1),
    ("Brown bear", 1),
    ("Picnic basket", 1),
    ("Plastic bag", 1),
    ("Punching bag", 1),
    ("Lemon", 1),
    ("Cheese", 1),
    ("Cupboard", 1),
    ("Personal flotation device", 1),
    ("Snowmobile", 1),
    ("Flowerpot", 1),
    ("Broccoli", 1),
    ("Cucumber", 1),
    ("Christmas tree", 1),
    ("Hamster", 1),
    ("Pasta", 1),
    ("Shark", 1),
    ("Kite", 1),
    ("Tart", 1),
    ("Pumpkin", 1),
    ("Crab", 1),
    ("Mug", 1),
    ("Dinosaur", 1),
    ("Tablet computer", 1),
    ("Bowl", 1),
]
ATTRIBUTES_NAMES = [name for name, _ in attributes_counter]

for attr1, attr2 in ATTRIBUTE_TUPLES:
    assert attr1 in ATTRIBUTES_NAMES, f"{attr1} is misspelled"
    assert attr2 in ATTRIBUTES_NAMES, f"{attr2} is misspelled"

for noun1, noun2 in NOUN_TUPLES:
    assert noun1 in NOUN_NAMES, f"{noun1} is misspelled"
    assert noun2 in NOUN_NAMES, f"{noun2} is misspelled"


SYNONYMS_LIST = [
    ["Table", "Desk", "Coffee table"],
    ["Mug", "Coffee cup"],
    ["Glasses", "Sunglasses", "Goggles"],
    ["Sun hat", "Fedora", "Cowboy hat", "Sombrero"],
    ["Bicycle helmet", "Football helmet"],
    ["High heels", "Sandal", "Boot"],
    ["Racket", "Tennis racket", "Table tennis racket"],
    ["Crown", "Tiara"],
]

SYNONYMS = {name: [name] for name in NOUN_NAMES + ATTRIBUTES_NAMES}
for synonyms in SYNONYMS_LIST:
    SYNONYMS.update({item: synonyms for item in synonyms})



