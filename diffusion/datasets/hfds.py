import logging
import random
from typing import Callable, Dict, List, Optional, Union

import datasets as ds
import torch
from datasets import DownloadConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers.tokenization_utils import PreTrainedTokenizer

from diffusion.datasets.laion.transforms import (
    RandomCropBucketedAspectRatioTransform,
    RandomCropSquare,
)
from diffusion.models.text_encoder import MultiTextEncoder, MultiTokenizer

logger = logging.getLogger(__name__)


def func(
    sample,
    *,
    crop: RandomCropSquare,
    transform: transforms.Compose,
    image_key: str,
    caption_key: str,
    tokenizer: Optional[Union[PreTrainedTokenizer, MultiTokenizer]] = None,
    sdxl_conditioning: bool = True,
    microcond_drop_prob: float = 0.1,
    caption_drop_prob: float = 0.1,
    zero_dropped_captions: bool = True,
    aspect_ratio_bucket_key: Optional[str] = None,
    caption_selection: str = "first",
):
    out = {}

    # Image
    img = sample[image_key]
    img = img.convert("RGB") if img.mode != "RGB" else img
    orig_w, orig_h = img.size

    assert isinstance(crop, RandomCropSquare)
    img, crop_top, crop_left = crop(img)

    img = transform(img)
    assert isinstance(img, torch.Tensor), type(img)
    out["image"] = img

    # SDXL microconditioning on image characteristics
    if sdxl_conditioning:
        # Get the new height and width
        if isinstance(img, torch.Tensor):
            img_h, img_w = img.shape[-2], img.shape[-1]
        elif isinstance(img, Image.Image):
            img_w, img_h = img.size
        else:
            raise ValueError(
                "Image after transformations must either be a PIL Image or Torch Tensor"
            )

        out["cond_crops_coords_top_left"] = torch.tensor([crop_top, crop_left])
        out["cond_original_size"] = torch.tensor([orig_w, orig_h])
        out["cond_target_size"] = torch.tensor([img_w, img_h])

        # Microconditioning dropout as in Stability repo
        # https://github.com/Stability-AI/generative-models/blob/477d8b9a7730d9b2e92b326a770c0420d00308c9/sgm/modules/encoders/modules.py#L151-L160
        if torch.rand(1) < microcond_drop_prob:
            out["cond_crops_coords_top_left"] = out["cond_crops_coords_top_left"] * 0
        if torch.rand(1) < microcond_drop_prob:
            out["cond_original_size"] = out["cond_original_size"] * 0
        if torch.rand(1) < microcond_drop_prob:
            out["cond_target_size"] = out["cond_target_size"] * 0

    # Caption
    if torch.rand(1) < caption_drop_prob:
        caption = ""
        if zero_dropped_captions:
            out["drop_caption_mask"] = 0.0
        else:
            out["drop_caption_mask"] = 1.0
    else:
        caption = sample[caption_key]
        if isinstance(caption, List) and caption_selection == "first":
            caption = caption[0]
        if isinstance(caption, List) and caption_selection == "random":
            caption = random.sample(caption, k=1)[0]
        out["drop_caption_mask"] = 1.0

    if tokenizer:
        tokenizer_out = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out["captions"] = tokenizer_out["input_ids"].squeeze()
        out["attention_mask"] = tokenizer_out["attention_mask"].squeeze()
    else:
        out["captions"] = caption

    # print(out)

    return out


def build_hfds_dataloader(
    dataset_path: str,
    batch_size: int,
    tokenizer: MultiTokenizer,
    resize_size: int,
    image_key: str,
    caption_key: str,
    transform: Optional[List[Callable]] = None,
    num_samples: Optional[int] = None,
    split: str = "train",
    dataloader_kwargs: Optional[Dict] = None,
) -> DataLoader:
    logger.info(dataset_path)

    dataset = ds.load_dataset(
        path=dataset_path,
        split=split,
        streaming=True,
        download_config=DownloadConfig(
            max_retries=5,
            use_etag=False,
        ),
    )
    assert isinstance(dataset, ds.IterableDataset)

    if num_samples is not None:
        dataset = dataset.take(num_samples)

    dataset = dataset.select_columns(
        column_names=[image_key, caption_key],
    )
    logger.info(dataset)

    transform = transform or [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataset = dataset.map(
        func,
        fn_kwargs={
            "crop": RandomCropSquare(size=resize_size),
            "transform": transforms.Compose(transform),
            "tokenizer": tokenizer,
            "image_key": image_key,
            "caption_key": caption_key,
        },
        remove_columns=[image_key, caption_key],
    )

    dataloader_kwargs = dataloader_kwargs or {}

    dataloader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )
    return dataloader
