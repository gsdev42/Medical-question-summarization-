!pip install -Uqq accelerate
!pip install -Uqq transformers
import transformers
transformers.__version__
!pip install datasets
!pip install torch torchvision
!pip install Pillow
!pip install requests
!pip install matplotlib

import os
import datasets
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import requests
import random
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode
from pdb import set_trace

import transformers
from transformers import (
    VisionTextDualEncoderProcessor,
    VisionTextDualEncoderModel,
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/contrastive-image-text/requirements.txt")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    image_processor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})
    image_column: Optional[str] = field(
        default="image_path",
        metadata={"help": "The name of the column in the datasets containing the full image file paths."},
    )
    caption_column: Optional[str] = field(
        default="caption",
        metadata={"help": "The name of the column in the datasets containing the image captions."},
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


##Loading Model

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")

import pandas as pd
import datasets
from datasets import Dataset, DatasetDict

df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')


train = Dataset.from_pandas(df_train)
test = Dataset.from_pandas(df_test)


dataset = DatasetDict()

dataset['train'] = train
dataset['test'] = test

dataset


args_dict = {'output_dir': './clip-roberta-finetuned',
 'model_name_or_path': './clip-roberta',
 'data_dir': './data',
 'dataset_name': dataset,
 'image_column': 'image_path',
 'caption_column': 'description',
 'remove_unused_columns': False,
 'per_device_train_batch_size': 64,
 'per_device_eval_batch_size': 64,
 'learning_rate': 5e-05,
 'warmup_steps': 0,
 'weight_decay': 0.1,
 'overwrite_output_dir': True,
 'push_to_hub': False}

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_dict(args_dict)

model_args, data_args


##Model Prep


tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
)


class Transform(torch.nn.Module):
    def __init__(self, image_size, mean, std):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(mean, std),
        )
    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }


image_processor = AutoImageProcessor.from_pretrained(
    model_args.image_processor_name or model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
config = model.config


set_seed(training_args.seed)
image_transformations = Transform(
    config.vision_config.image_size, image_processor.image_mean, image_processor.image_std
)
image_transformations = torch.jit.script(image_transformations)


def tokenize_captions(examples):
    captions = [example[0] for example in examples[data_args.caption_column]]
    text_inputs = tokenizer(captions, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
    examples["input_ids"] = text_inputs.input_ids
    examples["attention_mask"] = text_inputs.attention_mask
    return examples

import torchvision.transforms as T

# Define image transformations
image_transformations = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224 pixels
    T.CenterCrop(224),     # Center crop the images
    T.ConvertImageDtype(torch.float),  # Convert image to float tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

def transform_images(examples):
    images = []
    for image_path in examples[data_args.image_column]:
        # Open image using PIL
        image_path = '/content/drive/MyDrive/Multimodal_img/'+ image_path
        image = Image.open(image_path)
        # Convert image to numpy array
        image_np = np.array(image)
        # Convert numpy array to PyTorch tensor and permute dimensions
        image_tensor = torch.tensor(image_np).permute(2, 0, 1)
        images.append(image_tensor)
    examples["pixel_values"] = [image_transformations(image) for image in images]
    return examples

def filter_corrupt_images(examples):
    """remove problematic images"""
    valid_images = []
    for image_file in examples[data_args.image_column]:
        try:
            Image.open(image_file)
            valid_images.append(True)
        except Exception:
            valid_images.append(False)
    return valid_images


train_dataset


eval_dataset = dataset["test"]
eval_dataset = eval_dataset.map(
    function=tokenize_captions,
    batched=True,
    num_proc=data_args.preprocessing_num_workers,
    load_from_cache_file=not data_args.overwrite_cache,
    desc="Running tokenizer on validation dataset",
)
eval_dataset.set_transform(transform_images)


##how the dataset looks
train_dataset, eval_dataset

/*
(Dataset({
     features: ['Question', 'Question_summ', 'image_path', 'category', 'context', 'description', 'input_ids', 'attention_mask'],
     num_rows: 2412
 }),
 Dataset({
     features: ['Question', 'Question_summ', 'image_path', 'category', 'context', 'description', 'Unnamed: 6', 'input_ids', 'attention_mask'],
     num_rows: 603
 }))
*/

processor =  VisionTextDualEncoderProcessor(image_processor, tokenizer)

df= dataset['train']
def prep_img(image_path):
  for image_path in df['image_path']:
    image_path = '/content/drive/MyDrive/Multimodal_img/'+ image_path
    image = Image.open(image_path)

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# Define image transformations
image_transformations = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224 pixels
    T.CenterCrop(224),     # Center crop the images
    T.ToTensor(),          # Convert image to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

# Transformation function

image_transformations = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224 pixels
    T.CenterCrop(224),     # Center crop the images
    T.ToTensor(),          # Convert image to tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the images
])

def transform_images(examples):
    images = []
    for image_path in examples[data_args.image_column]:
        # Open image using PIL
        image_path = '/content/drive/MyDrive/Multimodal_img/'+ image_path
        try:
            # Open image using PIL
            image = Image.open(image_path)
            # Convert to RGB only if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Apply transformations
            image = image_transformations(image)
            images.append(image)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            images.append(torch.zeros((3, 224, 224)))  # Placeholder for problematic images
    examples["pixel_values"] = images
    return examples



# Assuming your image column is named 'image_path'
data_args = type('', (), {})()  # Create a dummy object to hold attributes
data_args.image_column = 'image_path'  # Set your image column name

# Preprocess the dataset
dataset = dataset.map(transform_images, batched=True)

# Define your training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # Your model here
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

# Train the model
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
metrics = trainer.evaluate()

