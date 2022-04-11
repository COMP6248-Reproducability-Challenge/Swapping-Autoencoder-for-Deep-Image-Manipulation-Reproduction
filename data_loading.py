import os

from taesung_data_loading import ConfigurableDataLoader


class DataLoadOptions:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_church_data(image_crop_size, phase='train', batch_size=1, num_gpus=1):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "..\CwkData\lsun\church_outdoor_train_lmdb")
    options = DataLoadOptions(
        dataroot=filename,
        dataset_mode="lmdb",
        phase=phase,
        isTrain=phase == "train",
        batch_size=batch_size,
        num_gpus=num_gpus,
        # scales the image such that the short side is |load_size|, and crops a square window of |crop_size|.
        load_size=256,
        crop_size=image_crop_size,
        shuffle_dataset=None,
        preprocess="scale_shortside_and_crop",
        preprocess_crop_padding=None,
        no_flip=True
    )
    return ConfigurableDataLoader(options)
