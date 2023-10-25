from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import torch.nn.functional as F


@dataclass
class ADOutput(BaseOutput):
    images: list[Image.Image]
    init_images: list[Image.Image]


def mask_dilate2(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    input_tensor = to_tensor(image)
    dilation_kernel = torch.ones((1, 1, value, value), dtype=torch.float32)
    dilated_tensor = F.conv2d(input_tensor.unsqueeze(0), dilation_kernel, stride=1, padding=(value-1)//2)
    dilated_image = to_pil_image(dilated_tensor.squeeze())

    return dilated_image


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


def mask_box_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.BoxBlur(value)
    return image.filter(blur)


def bbox_padding(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int], value: int = 32
) -> tuple[int, int, int, int]:
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.flatten())


def composite(
    init: Image.Image,
    mask: Image.Image,
    gen: Image.Image,
    bbox_padded: tuple[int, int, int, int],
    composite_image_enhance: dict[str] | None = None,
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)

    enhance_keys = ["sharpness", "color", "contrast", "brightness"]

    if composite_image_enhance is not None and any(key in composite_image_enhance for key in enhance_keys):
        if "sharpness" in composite_image_enhance:
            output = ImageEnhance.Sharpness(output).enhance(composite_image_enhance["sharpness"])

        if "color" in composite_image_enhance:
            output = ImageEnhance.Color(output).enhance(composite_image_enhance["color"])

        if "contrast" in composite_image_enhance:
            output = ImageEnhance.Contrast(output).enhance(composite_image_enhance["contrast"])

        if "brightness" in composite_image_enhance:
            output = ImageEnhance.Brightness(output).enhance(composite_image_enhance["brightness"])
    else:
        output = ImageEnhance.Brightness(output).enhance(1.0875)
        output = ImageEnhance.Contrast(output).enhance(1.05)

    output.alpha_composite(img_masked)
    return output.convert("RGBA")


def _dilate(input_tensor, value: int) -> Image.Image:
    kernel_size = value
    dilated_tensor = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
    dilated_image = to_pil_image(dilated_tensor.squeeze())

    return dilated_image


def _erode(input_tensor, value: int) -> Image.Image:
    kernel_size = max(value, 1)  # Ensure kernel size is at least 1 for erosion
    eroded_tensor = F.max_pool2d(input_tensor, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
    eroded_image = to_pil_image(eroded_tensor.squeeze())

    return eroded_image


def dilate_erode(img: Image.Image, value: int) -> Image.Image:
    if value == 0:
        return img

    input_tensor = to_tensor(img).unsqueeze(0).unsqueeze(0)
    dilated_eroded_tensor = _dilate(input_tensor, value) if value > 0 else _erode(input_tensor, -value)
    dilated_eroded_image = to_pil_image(dilated_eroded_tensor.squeeze())

    return dilated_eroded_image


def _dilate2(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.dilate(arr, kernel, iterations=1)


def _erode2(arr: np.ndarray, value: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    return cv2.erode(arr, kernel, iterations=1)


def dilate_erode2(img: Image.Image, value: int) -> Image.Image:
    if value == 0:
        return img

    arr = np.array(img)
    arr = _dilate(arr, value) if value > 0 else _erode(arr, -value)

    return Image.fromarray(arr)

