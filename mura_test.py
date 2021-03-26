import tensorflow as tf
from tensorflow_datasets import testing
from mura import MURA

class MURATest(testing.DatasetBuilderTestCase):
    DATATSET_CLASS = MURA
    SPLITS = {
        "train": 2,
        "valid": 2
    }

if __name__ == "__main__":
    testing.test_main()