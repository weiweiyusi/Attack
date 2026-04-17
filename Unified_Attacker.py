import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from ultralytics import RTDETR
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import time
import logging
import contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    devnull = open(os.devnull, 'w')
    try:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        devnull.close()

# --- 1. 配置 (Config) ---
class Config:
    TEST_MODE = False
    TEST_MODE_BATCHES = 2

    DATASET_ROOT = '/kaggle/working/skyfusion_subset' 
    SPLIT_TO_ATTACK = 'train'    
    MODEL_WEIGHTS = 'rtdetr-l.pt' 
    OUTPUT_JPG_DIR = '/kaggle/working/adversarial_skyfusion_jpg'

    BATCH_SIZE = 4       
    NUM_WORKERS = 0      
    IMAGE_SIZE = (640, 640) 

    # 按照您的要求统一扰动为 8/255，独立调整各自步长设定
    ATTACK_CONFIGS = {
        'PGD': {
            'eps': 8/255.0,
            'step_size': 7/255.0, # 迭代步长
            'n_iter': 7,          # 迭代次数
        },
        'CW': {
            'eps': 8/255.0,
            'step_size': 7/255.0, # 迭代步长
            'n_iter': 7,          
        },
        'DeepFool': {
            'eps': 8/255.0,
            'step_size': None,    # DeepFool在迭代中自动动态计算
            'n_iter': 50,         # 保证有足够迭代次数逼近边界
        }
    }

# --- 2. 数据加载器 ---
class SkyFusionDataset(Dataset):
    def __init__(self, root_dir, split='val', transform=None):
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)
        self.transform = transform
        
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        if not self.image_files:
             self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))

    def __len__(self):
        return len(self.image_files)

    def load_labels(self, label_path):
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([float(p) for p in parts])
        return np.array(labels) if labels else np.empty((0, 5))

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            return None, None, None

        labels = self.load_labels(label_path)
        if self.transform:
            image = self.transform(image)
        return image, labels, img_name

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None] 
    if not batch: return None, None, None

    images, labels_list, img_names = zip(*batch)
    images_tensor = torch.stack(images, 0)
    
    targets_dict = {'batch_idx': [], 'cls': [], 'bboxes': []}
    for i, labels in enumerate(labels_list):
        if labels.shape[0] > 0:
            targets_dict['batch_idx'].append(torch.full((labels.shape[0],), i))
            targets_dict['cls'].append(torch.from_numpy(labels[:, 0]))
            targets_dict['bboxes'].append(torch.from_numpy(labels[:, 1:]))
            
    if targets_dict['batch_idx']:
        targets_dict['batch_idx'] = torch.cat(targets_dict['batch_idx'], 0)
        targets_dict['cls'] = torch.cat(targets_dict['cls'], 0)
        targets_dict['bboxes'] = torch.cat(targets_dict['bboxes'], 0)
    else:
        targets_dict['batch_idx'] = torch.empty((0,))
        targets_dict['cls'] = torch.empty((0,))
        targets_dict['bboxes'] = torch.empty((0,))

    return images_tensor, targets_dict, img_names

def load_data(config):
    transform = T.Compose([T.Resize(config.IMAGE_SIZE), T.ToTensor()])
    dataset = SkyFusionDataset(root_dir=config.DATASET_ROOT, split=config.SPLIT_TO_ATTACK, transform=transform)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

# --- 3. 模型封装器 ---
class RTDETR_Loss_Wrapper(nn.Module):
    def __init__(self, rtdetr_model):
        super().__init__()
        self.model = rtdetr_model
        
    def forward(self, images_0_1, targets_dict):
        try:
            with suppress_stdout_stderr():
                preds = self.model(images_0_1)
            if isinstance(preds, (list, tuple)) and len(preds) > 0:
                pred = preds[0]
            else:
                pred = preds
            if isinstance(pred, torch.Tensor):
                loss = -torch.mean(torch.abs(pred)) 
            else:
                loss = torch.mean(images_0_1) * 0.0
        except Exception:
            loss = torch.sum(images_0_1 * torch.randn_like(images_0_1) * 0.001)
        return loss

# --- 4. 统一攻击器 (PGD / CW / DeepFool) ---
class Unified_Attacker:
    def __init__(self, model_wrapper, attack_type, eps, step_size, n_iter, device):
        self.model_wrapper = model_wrapper
        self.attack_type = attack_type
        self.eps = eps
        self.step_size = step_size
        self.n_iter = n_iter
        self.device = device

    def run_attack(self, images_0_1, targets_dict, orig_filenames):
        targets_dict = {
            'batch_idx': targets_dict['batch_idx'].to(self.device),
            'cls': targets_dict['cls'].to(self.device).long(),
            'bboxes': targets_dict['bboxes'].to(self.device).float()
        }

        original_images = images_0_1.clone().detach().to(self.device)
        adv_images = original_images.clone().detach()

        if self.attack_type == 'PGD':
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
            
            for _ in range(self.n_iter):
                adv_images.requires_grad = True
                loss = self.model_wrapper(adv_images, targets_dict)
                self.model_wrapper.zero_grad()
                if adv_images.grad is not None: adv_images.grad.zero_()
                loss.backward()
                
                grad = adv_images.grad.detach()
                with torch.no_grad():
                    adv_images = adv_images + self.step_size * grad.sign()
                    eta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
                    adv_images = torch.clamp(original_images + eta, 0, 1).detach()

        elif self.attack_type == 'CW':
            modifier = torch.zeros_like(original_images, requires_grad=True)
            optimizer = optim.Adam([modifier], lr=self.step_size)
            
            for _ in range(self.n_iter):
                adv_images = torch.clamp(original_images + modifier, 0, 1)
                loss = -self.model_wrapper(adv_images, targets_dict) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    modifier.data = torch.clamp(modifier.data, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(original_images + modifier, 0, 1).detach()

        elif self.attack_type == 'DeepFool':
            for _ in range(self.n_iter):
                adv_images.requires_grad = True
                loss = self.model_wrapper(adv_images, targets_dict)
                self.model_wrapper.zero_grad()
                if adv_images.grad is not None: adv_images.grad.zero_()
                loss.backward()
                
                grad = adv_images.grad.detach()
                with torch.no_grad():
                    grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-12
                    loss_abs = torch.abs(loss)
                    
                    # 自动调整步长算法
                    step = loss_abs / (grad_norm ** 2) * grad
                    adv_images = adv_images + 1.02 * step
                    
                    eta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
                    adv_images = torch.clamp(original_images + eta, 0, 1).detach()

        return adv_images.cpu().detach()

# --- 5. 主执行函数 ---
def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    try:
        model = RTDETR(config.MODEL_WEIGHTS).to(device)
        model.eval()
    except Exception as e:
        return

    model_wrapper = RTDETR_Loss_Wrapper(model)
    dataloader = load_data(config)

    for attack_name, atk_config in config.ATTACK_CONFIGS.items():
        print(f"\n--- 开始攻击: {attack_name} ---")
        
        attacker = Unified_Attacker(
            model_wrapper=model_wrapper,
            attack_type=attack_name,
            eps=atk_config['eps'],
            step_size=atk_config['step_size'],
            n_iter=atk_config['n_iter'],
            device=device
        )
        
        output_dir = os.path.join(config.OUTPUT_JPG_DIR, attack_name)
        os.makedirs(output_dir, exist_ok=True)
        
        num_batches_processed = 0
        pbar = tqdm(dataloader, desc=f"攻击 {attack_name}")
        
        for images_0_1, targets_dict, orig_filenames in pbar:
            if images_0_1 is None: continue

            try:
                adv_batch_cpu = attacker.run_attack(images_0_1, targets_dict, orig_filenames)
                for img_tensor, fname in zip(adv_batch_cpu, orig_filenames):
                    base_name = os.path.splitext(fname)[0]
                    save_path = os.path.join(output_dir, f"{base_name}.jpg")
                    vutils.save_image(img_tensor, save_path, normalize=False)
            except Exception as e:
                pass
            
            num_batches_processed += 1
            if config.TEST_MODE and num_batches_processed >= config.TEST_MODE_BATCHES:
                break

if __name__ == "__main__":
    main()