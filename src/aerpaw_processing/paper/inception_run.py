from aerpaw_processing.paper.inception_model import InceptionTime, train
from aerpaw_processing.paper.preprocess_utils import DatasetConfig
from aerpaw_processing.paper.inception_dataloader import InceptionDataset

config = DatasetConfig()

dataset = InceptionDataset(config)

model = train(dataset, epochs=50, checkpoint_dir="checkpoints")
