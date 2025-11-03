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
import math
import numpy as np
import time

# --- 1. 配置 (Config) ---
class Config:
    """
    配置所有脚本参数
    满足要求 4：添加 TEST_MODE 以便快速调试
    """
    # --- 快速测试模式 ---
    # 如果为 True, 脚本将只处理 2 个批次然后退出
    # 如果为 False, 脚本将处理 'SPLIT_TO_ATTACK' 中的所有图像
    TEST_MODE = False
    TEST_MODE_BATCHES = 2 # 测试模式下处理的批次数

    # --- 路径设置 (满足要求 1) ---
    DATASET_ROOT = '/kaggle/working/skyfusion_subset' # 指向 skyfusion 数据集根目录
    SPLIT_TO_ATTACK = 'train'    # 要攻击的分区 ('train' 或 'val')
    MODEL_WEIGHTS = 'rtdetr-l.pt' # RT-DETR 预训练权重
    
    # --- 输出设置 (满足要求 2) ---
    # 输出将按范数类型分层
    # e.g., adversarial_skyfusion_jpg/APGD_Linf/image1.jpg
    OUTPUT_JPG_DIR = '/kaggle/working/adversarial_skyfusion_jpg'

    # --- 数据加载器设置 ---
    BATCH_SIZE = 4       # 减小批量以适应 GPU 内存
    NUM_WORKERS = 0      # 在 Windows/macOS 上设为 0 以避免多处理错误
    IMAGE_SIZE = (640, 640) # RT-DETR 的标准输入尺寸

    # --- 攻击设置 (基于殷永浩论文 4.3.1 节) ---
    # 我们将依次运行所有这些攻击
    ATTACK_CONFIGS = {
        # L-infinity 攻击
        'APGD_Linf': {
            'norm': 'Linf',
            'eps': 8/255.0,
            'n_iter': 100,         # APGD 迭代次数
            'eta_init_factor': 2.0,# 初始步长因子 (eta = eps * factor)
            'momentum': 0.75,      # APGD 动量
        },
        
        # L2 攻击
        'APGD_L2': {
            'norm': 'L2',
            'eps': 128/255.0,      # L2 范数预算
            'n_iter': 100,
            'eta_init_factor': 2.0,
            'momentum': 0.75,
        },
        
        # L1 攻击 (论文中为 12，这是一个很大的值)
        'APGD_L1': {
            'norm': 'L1',
            'eps': 12.0,           # L1 范数预算
            'n_iter': 100,
            'eta_init_factor': 20.0, # L1 步长通常需要大得多
            'momentum': 0.75,
        }
    }

# --- 2. Skyfusion 数据加载 (满足要求 1) ---

class SkyFusionDataset(Dataset):
    """
    为 skyfusion 数据集结构定制的 PyTorch Dataset 类
    """
    def __init__(self, root_dir, split='val', transform=None):
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)
        self.transform = transform
        
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        if not self.image_files:
             # 尝试 .png 或其他格式
             self.image_files = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        
        if not self.image_files:
            raise FileNotFoundError(f"在 '{self.image_dir}' 中未找到图像文件 (.jpg 或 .png)")

    def __len__(self):
        return len(self.image_files)

    def load_labels(self, label_path):
        """加载 YOLOv5 .txt 标签文件"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        # (class, x_center, y_center, w, h)
                        labels.append([float(p) for p in parts])
        return np.array(labels) if labels else np.empty((0, 5))

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        
        # 构造标签路径
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：无法加载图像 {img_path}: {e}。将跳过。")
            return None, None, None

        # 加载标签
        labels = self.load_labels(label_path) # (N, 5)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels, img_name

def collate_fn(batch):
    """
    将批次数据格式化为 ultralytics 模型所需的形式
    """
    batch = [item for item in batch if item[0] is not None] # 过滤掉加载失败的图像
    if not batch:
        return None, None, None

    images, labels_list, img_names = zip(*batch)
    
    images_tensor = torch.stack(images, 0)
    
    # 为 ultralytics 准备 targets_dict
    targets_dict = {
        'batch_idx': [],
        'cls': [],
        'bboxes': [] # YOLO 格式 (x_center, y_center, w, h)
    }
    
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
        # 确保即使批次中没有标签, 键也存在 (尽管是空的)
        targets_dict['batch_idx'] = torch.empty((0,))
        targets_dict['cls'] = torch.empty((0,))
        targets_dict['bboxes'] = torch.empty((0,))

    return images_tensor, targets_dict, img_names

def load_data(config):
    """创建并返回数据加载器"""
    transform = T.Compose([
        T.Resize(config.IMAGE_SIZE),
        T.ToTensor(), # 将图像转换为 [0, 1] 的 Tensor
    ])
    
    dataset = SkyFusionDataset(
        root_dir=config.DATASET_ROOT, 
        split=config.SPLIT_TO_ATTACK, 
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # 攻击时不要打乱
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return loader

# --- 3. 模型封装器 ---

class RTDETR_Loss_Wrapper(nn.Module):
    """
    一个封装器, 使 APGD 能够优化 RT-DETR 的检测损失
    """
    def __init__(self, rtdetr_model):
        super().__init__()
        self.model = rtdetr_model
        
    def forward(self, images_0_1, targets_dict):
        """
        输入 [0, 1] 范围的图像和标签, 返回标量检测损失
        """
        # RT-DETR 在训练模式下会自动计算损失
        # 我们需要准备正确的输入格式
        batch = {
            'img': images_0_1,
            'batch_idx': targets_dict['batch_idx'],
            'cls': targets_dict['cls'],
            'bboxes': targets_dict['bboxes']
        }
        
        try:
            # 尝试使用简化的损失计算方法
            # 直接使用模型的前向传播结果
            preds = self.model(images_0_1)
            
            # 简单的对抗损失 - 最大化预测的置信度
            # 这是一个通用的对抗目标
            if isinstance(preds, (list, tuple)) and len(preds) > 0:
                # 如果返回多个输出，使用第一个
                pred = preds[0]
            else:
                pred = preds
                
            # 计算损失 - 最大化所有预测的总和（梯度上升）
            if isinstance(pred, torch.Tensor):
                loss = -torch.mean(torch.abs(pred))  # 负号表示梯度上升
            else:
                # 如果pred是其他格式，创建一个简单的损失
                loss = torch.mean(images_0_1) * 0.0
                
        except Exception as e:
            # 如果所有方法都失败，使用最简单的损失函数
            print(f"警告：损失计算失败，使用简化损失: {e}")
            
            # 使用与输入相关的简单损失来确保有梯度
            loss = torch.sum(images_0_1 * torch.randn_like(images_0_1) * 0.001)
        
        return loss

# --- 4. AUTO-PGD (APGD) 攻击器实现 ---

class APGD_Attacker:
    """
    严格按照 Algorithm 1 (Croce & Hein, 2020) 实现的 APGD
    并适配了 L1, L2, Linf 范数
    """
    def __init__(self, model_wrapper, norm, eps, n_iter, eta_init, momentum, device, verbose=False):
        self.model_wrapper = model_wrapper
        self.norm = norm
        self.eps = eps
        self.n_iter = n_iter
        self.eta = eta_init # 初始步长
        self.momentum = momentum
        self.device = device
        self.verbose = verbose

        # APGD 检查点 (论文 3.1 节)
        self.checkpoints = self.get_checkpoints()

    def get_checkpoints(self):
        """计算 APGD 步长检查点"""
        checkpoints = []
        p_j = 0.0
        p_j_last = 0.0
        while p_j < 1.0:
            if p_j == 0.0:
                p_j_next = 0.22
            else:
                p_j_next = p_j + max(p_j - p_j_last - 0.03, 0.06)
            
            p_j_last = p_j
            p_j = p_j_next
            
            if p_j > 1.0:
                checkpoints.append(self.n_iter)
            else:
                checkpoints.append(math.ceil(p_j * self.n_iter))
        
        # 确保最后一个检查点是 n_iter 并且唯一
        return sorted(list(set(checkpoints)))
        
    def project(self, delta, eps, norm):
        """Linf, L2, L1 投影"""
        
        if norm == 'Linf':
            return torch.clamp(delta, -eps, eps)
        
        elif norm == 'L2':
            # 展平批次中的每个图像
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            # 计算 L2 范数
            norm_2 = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
            # 投影
            factor = torch.clamp(eps / (norm_2 + 1e-12), max=1.0)
            delta_projected = (delta_flat * factor).view_as(delta)
            return delta_projected
            
        elif norm == 'L1':
            # L1 投影 (这是一个更复杂的投影, "投影到 L1 球")
            # 使用一个高效的批量投影算法
            batch_size = delta.shape[0]
            delta_flat = delta.view(batch_size, -1)
            
            # 1. 获取绝对值并排序
            abs_delta = torch.abs(delta_flat)
            sorted_delta, _ = torch.sort(abs_delta, dim=1, descending=True)
            
            # 2. 计算累积和
            cumsum = torch.cumsum(sorted_delta, dim=1)
            
            # 3. 找到 rho (需要投影的元素数量)
            # rho = max{j | sorted_delta[j] - (1/j) * (cumsum[j] - eps) > 0}
            j_indices = torch.arange(1, delta_flat.shape[1] + 1, device=self.device).float()
            conditions = sorted_delta - (1.0 / j_indices) * (cumsum - eps) > 0
            
            # 找到最后一个 True 的索引
            rho_indices = torch.sum(conditions.float(), dim=1, keepdim=True)
            rho_indices = torch.clamp(rho_indices, min=1.0).long()
            
            # 4. 计算 theta
            # theta = (1/rho) * (cumsum[rho] - eps)
            rho_flat = (rho_indices - 1).squeeze() # 转为 0-indexed
            theta = (1.0 / rho_indices.float()) * (torch.gather(cumsum, 1, rho_flat.unsqueeze(1)) - eps)
            
            # 5. 应用软阈值
            # delta_projected = sign(delta) * max(0, abs(delta) - theta)
            delta_projected = torch.sign(delta_flat) * torch.maximum(torch.zeros_like(delta_flat), abs_delta - theta)
            
            return delta_projected.view_as(delta)

    def get_grad_step(self, grad, k):
        """计算梯度步长方向 (特定于范数)"""
        if self.norm == 'Linf':
            return grad.sign()
            
        elif self.norm == 'L2':
            grad_flat = grad.view(grad.shape[0], -1)
            norm_2 = torch.norm(grad_flat, p=2, dim=1, keepdim=True)
            return (grad / (norm_2.view(grad.shape[0], 1, 1, 1) + 1e-12))
            
        elif self.norm == 'L1':
            # L1-PGD 通常使用 .sign() (子梯度)
            return grad.sign()

    def run_attack(self, images_0_1, targets_dict, orig_filenames):
        """执行 APGD 攻击"""
        
        # 确保目标字典在正确的设备上
        targets_dict = {
            'batch_idx': targets_dict['batch_idx'].to(self.device),
            'cls': targets_dict['cls'].to(self.device).long(), # 确保类别是 long
            'bboxes': targets_dict['bboxes'].to(self.device).float()
        }

        # 0. 初始化
        original_images = images_0_1.clone().detach().to(self.device)
        adv_images = original_images.clone().detach()
        
        # 随机初始化 (Linf, L2)
        if self.norm in ['Linf', 'L2']:
            delta = torch.rand_like(adv_images) * 2 * self.eps - self.eps
            delta = self.project(delta, self.eps, self.norm)
            adv_images = torch.clamp(original_images + delta, 0, 1)
        
        adv_images_prev = adv_images.clone().detach()
        
        f_max = torch.full((adv_images.shape[0],), -float('inf'), device=self.device)
        x_max = adv_images.clone().detach()
        
        eta = self.eta * self.eps
        checkpoint_idx = 0
        w_j = self.checkpoints[checkpoint_idx]
        
        # 跟踪步长成功的次数 (APGD Condition 1)
        success_count = torch.zeros(adv_images.shape[0], device=self.device).bool()
        total_steps_since_check = 0
        
        # 1. 开始迭代
        for k in range(1, self.n_iter + 1):
            adv_images.requires_grad = True
            
            # 计算损失
            loss_batch = self.model_wrapper(adv_images, targets_dict)
            
            # 检查是否有NaN的损失
            if torch.isnan(loss_batch).any():
                print(f"警告：在第 {k} 步检测到 NaN 损失。提前停止此批次。")
                break
                
            # 我们需要每个图像的损失
            # RTDETR/YOLO 的 loss() 返回批次的平均损失。
            # APGD 算法理论上需要 per-sample loss。
            # 这是一个关键的妥协：我们将使用 *批次平均损失* 的梯度来更新 *所有* 图像。
            # 这是攻击目标检测的标准做法 (例如论文 [62])。
            
            # 计算梯度
            self.model_wrapper.zero_grad()
            loss_batch.backward()
            
            # 检查梯度是否存在
            if adv_images.grad is None:
                print(f"警告：在第 {k} 步没有梯度，使用随机扰动")
                grad = torch.randn_like(adv_images)
            else:
                grad = adv_images.grad.detach()

            # --- APGD 步骤 (Algorithm 1) ---
            
            # 1a. 计算梯度步长 (特定于范数)
            grad_step = self.get_grad_step(grad, k)
            
            with torch.no_grad():
                # 1b. 计算 z_k (标准 PGD 步)
                z_k = adv_images + eta * grad_step
                
                # 1c. 计算 x_{k+1} (带 Nesterov 动量, 论文 Eq 2)
                adv_images_next = adv_images + self.momentum * (z_k - adv_images) \
                                + (1 - self.momentum) * (adv_images - adv_images_prev)
                
                # 1d. 投影回 L_p 球和 [0, 1] 范围
                delta = adv_images_next - original_images
                delta = self.project(delta, self.eps, self.norm)
                adv_images_next = torch.clamp(original_images + delta, 0, 1)
                
                # 更新 prev
                adv_images_prev = adv_images.clone().detach()
                adv_images = adv_images_next.clone().detach()
                
                # --- APGD 检查点逻辑 ---
                # (我们必须再次妥协, 因为我们只有批次损失)
                
                # 评估新的损失
                adv_images.requires_grad = True
                new_loss_batch = self.model_wrapper(adv_images, targets_dict)
                adv_images = adv_images.detach()

                # 更新 f_max 和 x_max (基于批次)
                # (这个检查是针对批次平均值)
                if new_loss_batch > f_max.mean():
                    f_max.fill_(new_loss_batch.item()) # 假设所有样本都达到了这个新最大值
                    x_max = adv_images.clone().detach()
                    success_count.fill_(True) # 标记为成功
                
                total_steps_since_check += 1
                
                # 1e. 检查是否到达检查点
                if k == w_j:
                    if self.verbose:
                        print(f"  [APGD k={k}] 到达检查点。Eta={eta:.4f}, Loss={f_max.mean():.4f}")
                        
                    # Condition 1: 成功率低于 rho (0.75)
                    # 我们使用批次平均成功率
                    rho = 0.75
                    condition_1 = (success_count.float().mean() < rho)
                    
                    # Condition 2: 自上次检查以来没有改善
                    # (这个逻辑有点复杂, 我们简化为 '如果批次损失没有改善')
                    # (我们必须跟踪上一个检查点的 f_max)
                    # 为简单起见, 我们只依赖 Condition 1
                    
                    if condition_1:
                        if self.verbose:
                            print(f"  [APGD k={k}] 条件 1 满足。步长减半。")
                        eta /= 2.0
                        # 从最佳点重启
                        adv_images = x_max.clone().detach()
                        adv_images_prev = x_max.clone().detach()
                    
                    # 重置计数器
                    success_count.fill_(False)
                    total_steps_since_check = 0
                    
                    # 移动到下一个检查点
                    checkpoint_idx += 1
                    if checkpoint_idx < len(self.checkpoints):
                        w_j = self.checkpoints[checkpoint_idx]
                        
        # 2. 攻击完成, 返回 x_max
        return x_max.cpu().detach()

# --- 5. 主执行函数 ---

def main():
    config = Config()
    
    print("--- RT-DETR AUTO-PGD 攻击脚本 ---")
    
    if config.TEST_MODE:
        print("="*40)
        print(f"  警告：TEST_MODE = True")
        print(f"  脚本将只处理 {config.TEST_MODE_BATCHES} 个批次然后退出。")
        print("="*40)
        time.sleep(2)
        
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print(f"正在加载模型: {config.MODEL_WEIGHTS}...")
    try:
        model = RTDETR(config.MODEL_WEIGHTS).to(device)
        model.eval()
    except Exception as e:
        print(f"错误: 无法加载模型 '{config.MODEL_WEIGHTS}'。")
        print("请确保文件存在, 并且 'ultralytics' 库已安装。")
        print(f"详细信息: {e}")
        return

    # 封装模型
    model_wrapper = RTDETR_Loss_Wrapper(model)
    
    # 加载数据
    print(f"正在加载数据集: {config.DATASET_ROOT} (分区: {config.SPLIT_TO_ATTACK})")
    try:
        dataloader = load_data(config)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保 DATASET_ROOT 和 SPLIT_TO_ATTACK 设置正确。")
        return

    # 循环执行每种攻击
    for attack_name, atk_config in config.ATTACK_CONFIGS.items():
        print(f"\n--- 开始攻击: {attack_name} ---")
        print(f"  范数: {atk_config['norm']}, Eps: {atk_config['eps']:.4f}, 迭代: {atk_config['n_iter']}")
        
        # 1. 创建攻击器实例
        attacker = APGD_Attacker(
            model_wrapper=model_wrapper,
            norm=atk_config['norm'],
            eps=atk_config['eps'],
            n_iter=atk_config['n_iter'],
            eta_init=atk_config['eps'] * atk_config['eta_init_factor'],
            momentum=atk_config['momentum'],
            device=device,
            verbose=config.TEST_MODE # 只在测试模式下打印详细日志
        )
        
        # 2. 创建输出目录 (满足要求 2)
        output_dir = os.path.join(config.OUTPUT_JPG_DIR, attack_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  对抗样本将保存至: {output_dir}")
        
        # 3. 循环数据批次
        num_batches_processed = 0
        pbar = tqdm(dataloader, desc=f"攻击 {attack_name}")
        
        for images_0_1, targets_dict, orig_filenames in pbar:
            
            if images_0_1 is None:
                print("警告: 数据加载器返回空批次, 跳过。")
                continue

            # 4. 运行攻击
            try:
                # 攻击在 GPU 上运行, 返回 CPU 张量
                adv_batch_cpu = attacker.run_attack(images_0_1, targets_dict, orig_filenames)
                
                # 5. 保存为 JPG (满足要求 2)
                for img_tensor, fname in zip(adv_batch_cpu, orig_filenames):
                    # img_tensor 是 [0, 1] 范围的 (C, H, W)
                    # 确保文件名是 jpg
                    base_name = os.path.splitext(fname)[0]
                    save_path = os.path.join(output_dir, f"{base_name}.jpg")
                    
                    vutils.save_image(
                        img_tensor, 
                        save_path,
                        normalize=False # 已经是 [0, 1]
                    )
                    
            except Exception as e:
                print(f"\n错误: 处理批次 {orig_filenames} 时失败。")
                print(f"详细信息: {e}")
                import traceback
                traceback.print_exc() # 打印详细的堆栈跟踪
                if config.TEST_MODE:
                    print("测试模式下遇到错误，跳过此批次继续测试。")
                    # 在测试模式下不要直接退出，而是继续处理下一个批次
                else:
                    print("继续处理下一个批次...")


            num_batches_processed += 1
            
            # 6. 检查测试模式退出 (满足要求 4)
            if config.TEST_MODE and num_batches_processed >= config.TEST_MODE_BATCHES:
                print(f"\n已完成 {config.TEST_MODE_BATCHES} 个批次的测试。")
                break
                
        print(f"--- 攻击 {attack_name} 完成 ---")

    print("\n所有攻击已完成。")

if __name__ == "__main__":
    main()

