import torch
import torchvision
from torchvision import transforms as pth_transforms

from PIL import Image

# Dataset
def get_dataset(path='../data/imagenet/val'):
  # transform = pth_transforms.Compose([
  #       pth_transforms.Resize((480, 480)),
  #       pth_transforms.ToTensor(),
  #       # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  #   ])
  # ds = torchvision.datasets.ImageFolder('data_softlink/imagenet_val', transform=transform)

  # transform = pth_transforms.Compose([
  #       pth_transforms.Resize((480, 480)),
  #       pth_transforms.ToTensor(),
  #       pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  # ])
  # nds = torchvision.datasets.ImageFolder('data_softlink/imagenet_val', transform=transform)

  # transform = None
  val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
  ])
  ds = torchvision.datasets.ImageFolder(path, transform=val_transform)
  return ds

# Plotting
def hconcat_pil_images(images):
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  return new_im

def vconcat_pil_images(images):
  widths, heights = zip(*(i.size for i in images))

  max_width = max(widths)
  total_height = sum(heights)

  new_im = Image.new('RGB', (max_width, total_height))

  y_offset = 0
  for im in images:
    new_im.paste(im, (0,y_offset))
    y_offset += im.size[1]

  return new_im

def array_to_pil(x):
  if isinstance(x, torch.Tensor):
    x = x.numpy()
  x = pth_transforms.ToTensor()(x)
  if x.shape[0] > 3:
    x = x.permute(1, 2, 0)
  x = torchvision.transforms.ToPILImage()(x)
  x = torchvision.transforms.Resize((448, 488))(x)
  return x

def plot_image_and_seg(im, mask):
  mask_im = array_to_pil(mask)
  plot_im = hconcat_pil_images([im, mask_im])
  return plot_im

def plot_images(images):
   images = [torchvision.transforms.Resize((448, 488))(x) if hasattr(x, 'width') else array_to_pil(x) for x in images]
   return hconcat_pil_images(images)


# Load model
import sys
sys.path.append('../dino')
import vision_transformer as vits

def load_model(pretrained_weights=None):
  patch_size = 16
  model = vits.__dict__['vit_small'](patch_size=patch_size, num_classes=0)
  model.apply(model._init_weights)
  model.patch_size = patch_size

  if pretrained_weights is None:
    state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth")
  elif pretrained_weights == 'random':
    state_dict = None
  else:
    state_dict = torch.load(pretrained_weights, map_location="cpu")['teacher']

  for p in model.parameters():
      p.requires_grad = False
  model.eval()

  if state_dict is not None:
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    # model.to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    # print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

  model.eval()
  return model

# DINO
DEFAULT_SMALLER_EDGE_SIZE = 448
DEFAULT_BACKGROUND_THRESHOLD = 0
DEFAULT_APPLY_OPENING = True
DEFAULT_APPLY_CLOSING = True

from PIL import Image
from typing import Tuple


def make_transform(smaller_edge_size: int) -> pth_transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = pth_transforms.InterpolationMode.BICUBIC

    return pth_transforms.Compose([
        pth_transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def prepare_image(image: Image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    return image_tensor, grid_size

def render_features_and_class_token(model, image: Image,
                     smaller_edge_size: float = 448,
                     patch_size: int = 14):
  image_tensor, grid_size = prepare_image(image, smaller_edge_size, patch_size)

  with torch.inference_mode():
      image_batch = image_tensor.unsqueeze(0)
      tokens = model.get_intermediate_layers(image_batch)[0].squeeze()
      class_token = tokens[0]
      tokens = tokens[1:]

  array = tokens #.numpy()
  array = array.reshape(*grid_size, 384)
  array = array.permute(2, 0, 1)
  # array = pth_transforms.Resize((image.width, image.height), interpolation=2)(array)
  array = array.permute(1, 2, 0)

  return class_token, array


def render_features(model, image: Image,
                     smaller_edge_size: float = 448,
                     patch_size: int = 14,
                     background_threshold: float = 0.05,
                     apply_opening: bool = False,
                     apply_closing: bool = False) -> Image:
    image_tensor, grid_size = prepare_image(image, smaller_edge_size, patch_size)

    with torch.inference_mode():
        image_batch = image_tensor.unsqueeze(0)
        tokens = model.get_intermediate_layers(image_batch)[0].squeeze()
        if len(tokens) % 2 == 1:
            tokens = tokens[1:]

    array = tokens #.numpy()
    array = array.reshape(*grid_size, 384)
    array = array.permute(2, 0, 1)
    # array = pth_transforms.Resize((image.width, image.height), interpolation=2)(array)
    array = array.permute(1, 2, 0)

    return array

def image_to_patches(x, smaller_edge_size: float = 448, patch_size: int = 14):
    transform =  pth_transforms.Compose([
        pth_transforms.Resize(size=smaller_edge_size, interpolation=pth_transforms.InterpolationMode.BICUBIC, antialias=True),
        # pth_transforms.ToTensor(),
    ])
    image_tensor = transform(x)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)

    # Get patches
    patches = image_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)

    return patches, grid_size