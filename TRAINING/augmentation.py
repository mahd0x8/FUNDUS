from PIL import Image
from torchvision import transforms
import random

# ============================================================
# AUGMENTATIONS
# scale [0.9,1.1], H/V flips, rotations, grayscale
# no color jitter, no gaussian blur
# ============================================================

class RandomScalePadCrop:
    """
    Approximate scale augmentation in [0.9, 1.1].
    - scale < 1.0: shrink then pad
    - scale > 1.0: enlarge then center crop
    """
    def __init__(self, scale_range=(0.9, 1.1), size=224):
        self.scale_range = scale_range
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        scale = random.uniform(*self.scale_range)
        w, h = img.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        if scale < 1.0:
            canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
            x = (self.size - new_w) // 2
            y = (self.size - new_h) // 2
            canvas.paste(img, (x, y))
            return canvas

        left = max(0, (new_w - self.size) // 2)
        top = max(0, (new_h - self.size) // 2)
        right = left + self.size
        bottom = top + self.size
        return img.crop((left, top, right, bottom))
    

def get_train_transform(image_size=224):
    return transforms.Compose([
        RandomScalePadCrop(scale_range=(0.9, 1.1), size=image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])