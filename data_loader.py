# import os
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
#
#
# class CustomImageDataset(Dataset):
#     def __init__(self, txt_file, root_dir, transform=None):
#         self.imgs = []
#         self.transform = transform
#         class_to_idx = {}
#         with open(txt_file, 'r') as file:
#             for line in file:
#                 img_name, class_name = line.strip().split()
#                 if class_name not in class_to_idx:
#                     class_to_idx[class_name] = len(class_to_idx)
#                 label = class_to_idx[class_name]
#                 self.imgs.append((os.path.join(root_dir, img_name), label))
#
#     def __len__(self):
#         return len(self.imgs)
#
#     def __getitem__(self, idx):
#         img_path, label = self.imgs[idx]
#         image = Image.open(img_path).convert('RGB')
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, label
#
#
# def get_data_loaders(data_dir, batch_size=32):
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ColorJitter(brightness=0.126, saturation=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#
#     image_datasets = {
#         x: CustomImageDataset(os.path.join(data_dir, f"{x}.txt"), os.path.join(data_dir, 'images'), data_transforms[x])
#         for x in ['train', 'val', 'test']}
#     dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4)
#                    for x in ['train', 'val', 'test']}
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
#     class_names = list(set([label for _, label in image_datasets['train'].imgs]))
#
#     return dataloaders, dataset_sizes, class_names


# 改进版：使用 ImageNet 风格目录结构 + ImageFolder
# import os
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
#
# def get_data_loaders(data_dir, batch_size=32, num_workers=4):
#     """
#     使用标准 ImageNet 目录格式加载数据集:
#         data_dir/
#             train/
#             val/
#             test/
#
#     每个子目录下按类别分文件夹存放图像。
#     """
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'test': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
#
#     # 使用 ImageFolder 自动构建数据集（基于目录结构）
#     image_datasets = {
#         x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
#         for x in ['train', 'val', 'test']
#     }
#
#     # 创建 DataLoader
#     dataloaders = {
#         x: DataLoader(
#             image_datasets[x],
#             batch_size=batch_size,
#             shuffle=(x == 'train'),
#             num_workers=num_workers,
#             pin_memory=True,
#             drop_last=(x == 'train')  # 可选：防止最后一个 batch 太小
#         )
#         for x in ['train', 'val', 'test']
#     }
#
#     # 统计信息
#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
#     class_names = image_datasets['train'].classes  # 按文件夹名称排序的类名列表
#
#     return dataloaders, dataset_sizes, class_names

from PIL import Image
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 🛡 在导入数据前设置最大像素限制（防止 DecompressionBombError）
# 同时保留安全性：不限为 None，而是设一个合理的高值
Image.MAX_IMAGE_PIXELS = 1000000000  # 允许最多 10 亿像素（足够处理大多数合法大图）

# 🔇 可选：忽略警告（避免终端刷屏）
import warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    使用标准 ImageNet 目录格式加载数据集:
        data_dir/
            train/
            val/
            test/

    每个子目录下按类别分文件夹存放图像。
    """

    # 💡 增强 transform：确保即使原图极大，在 Resize/Crop 阶段也会被缩小
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),   # ⚠️ 这一步会主动缩放和裁剪，是关键防护！
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),             # 缩小长边到 256
            transforms.CenterCrop(224),         # 再中心裁剪出 224×224
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 自定义安全图像加载器（进一步防御）
    def safe_pil_loader(path):
        try:
            img = Image.open(path)
            # 如果图片太大，先缩略图再返回（防止内存爆炸）
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024))  # 最大边不超过 1024
            return img.convert('RGB')
        except Exception as e:
            print(f"[ERROR] Unable to load image {path}: {e}")
            # 返回空白图像占位符（避免整个训练中断）
            return Image.new('RGB', (224, 224))

    # 使用 ImageFolder + 自定义 loader 构建数据集
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x],
            loader=safe_pil_loader  # ✅ 使用安全加载器
        )
        for x in ['train', 'val', 'test']
    }

    # 创建 DataLoader
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(x == 'train'),
            persistent_workers=True if num_workers > 0 else False  # 提高多 epoch 效率
        )
        for x in ['train', 'val', 'test']
    }

    # 统计信息
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names
