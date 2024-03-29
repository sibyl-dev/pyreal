from urllib.parse import urljoin

AWS_BASE_URL = "https://pyreal-data.s3.amazonaws.com/"

DEFAULT_DATASET_NAMES = [
    "kr-vs-kp",
    "letter",
    "balance-scale",
    "mfeat-factors",
    "mfeat-fourier",
    "breast-w",
    "mfeat-karhunen",
    "mfeat-morphological",
    "mfeat-zernike",
    "cmc",
    "optdigits",
    "credit-approval",
    "credit-g",
    "pendigits",
    "diabetes",
    "spambase",
    "splice",
    "tic-tac-toe",
    "vehicle",
    "electricity",
    "satimage",
    "eucalyptus",
    "sick",
    "vowel",
    "isolet",
    "analcatdata_authorship",
    "analcatdata_dmft",
    "mnist_784",
    "pc4",
    "pc3",
    "jm1",
    "kc2",
    "kc1",
    "pc1",
    "adult",
    "Bioresponse",
    "wdbc",
    "phoneme",
    "qsar-biodeg",
    "wall-robot-navigation",
    "semeion",
    "ilpd",
    "madelon",
    "nomao",
    "ozone-level-8hr",
    "cnae-9",
    "first-order-theorem-proving",
    "banknote-authentication",
    "blood-transfusion-service-center",
    "PhishingWebsites",
    "cylinder-bands",
    "bank-marketing",
    "GesturePhaseSegmentationProcessed",
    "har",
    "dresses-sales",
    "texture",
    "connect-4",
    "MiceProtein",
    "steel-plates-fault",
    "climate-model-simulation-crashes",
    "wilt",
    "car",
    "segment",
    "mfeat-pixel",
    "Fashion-MNIST",
    "jungle_chess_2pcs_raw_endgame_complete",
    "numerai28.6",
    "Devnagari-Script",
    "CIFAR_10",
    "Internet-Advertisements",
    "dna",
    "churn",
]


def get_dataset_url(name):
    if not name.endswith(".csv"):
        name = name + ".csv"

    return urljoin(AWS_BASE_URL, name)
