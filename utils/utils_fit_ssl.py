import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
from torch.nn.functional import mse_loss

from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor
from PIL import Image

from torch.utils.data import DataLoader

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def get_transforms():
    base_transform = Compose([

    ])
    transform_weak = Compose([
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2),
        base_transform,
    ])
    transform_strong = Compose([
        RandomHorizontalFlip(),
        RandomRotation(degrees=20),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        base_transform,
    ])
    return base_transform, transform_weak, transform_strong

# unlabeled_preprocess = preprocess()  # 初始化预处理函数

def preprocess():
    return transforms.Compose([
        transforms.Resize((512, 512)),        # Resize 图片到 512x512
        transforms.ToTensor(),                # 转换图片为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理，使用ImageNet的均值与标准差
                             std=[0.229, 0.224, 0.225])
    ])

class UnlabeledDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# 创建 DataLoader
unlabeled_image_paths = "VOCdevkit/VOC2007/unlabeled"
unlabeled_dataset = UnlabeledDataset(unlabeled_image_paths)
unlabeled_gen = DataLoader(unlabeled_dataset, batch_size=1, shuffle=True)

def consistency_loss(output1, output2):
    return mse_loss(output1, output2)

def get_image_paths(image_dir):
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
    return image_paths

def get_unlabeled_data_generator():
    image_dir = 'VOCdevkit/VOC2007/unlabeled'  # 无标签图片存放路径
    image_paths = get_image_paths(image_dir)
    dataset = UnlabeledDataset(image_paths, transform=preprocess())
    batch_size = 1  # 根据需求和硬件条件调整
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

def triple_contrastive_loss(output_no_aug, output_weak_aug, output_strong_aug, alpha=0.5, beta=1.0, margin=1.0):
    # 相似性损失：无增强与弱增强之间应该相似
    similarity_loss = mse_loss(output_no_aug, output_weak_aug)

    # 差异性损失：无增强与强增强之间应该有较大差异
    dissimilarity_loss_no_strong = torch.clamp(margin - mse_loss(output_no_aug, output_strong_aug), min=0)

    # 差异性损失：弱增强与强增强之间也应该有较大差异
    dissimilarity_loss_weak_strong = torch.clamp(margin - mse_loss(output_weak_aug, output_strong_aug), min=0)

    # 组合损失，其中 alpha 和 beta 是用来平衡三个损失组件的权重系数
    return similarity_loss + alpha * dissimilarity_loss_no_strong + beta * dissimilarity_loss_weak_strong


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    base_transform, transform_weak, transform_strong = get_transforms()
    unlabeled_gen = get_unlabeled_data_generator()  # 需要定义这个生成器

    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, (batch, unlabeled_imgs) in enumerate(zip(gen, unlabeled_gen)):

        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        # print(unlabeled_imgs.shape)
        # 应用变换
        imgs_no_aug = base_transform(unlabeled_imgs)
        imgs_weak_aug = transform_weak(unlabeled_imgs)
        imgs_strong_aug = transform_strong(unlabeled_imgs)

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs_no_aug = imgs_no_aug.cuda(local_rank)
                imgs_weak_aug = imgs_weak_aug.cuda(local_rank)
                imgs_strong_aug = imgs_strong_aug.cuda(local_rank)

                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()

        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)

            outputs_no_aug = model_train(imgs_no_aug)
            outputs_weak_aug = model_train(imgs_weak_aug)
            outputs_strong_aug = model_train(imgs_strong_aug)

            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            # 计算三重对比损失
            triple_loss = triple_contrastive_loss(outputs_no_aug, outputs_weak_aug, outputs_strong_aug, alpha=0.5,
                                           beta=1.0, margin=1.0)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            loss = loss + triple_loss

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, (batch, unlabeled_batch) in enumerate(zip(gen, unlabeled_gen)):
    # for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        unlabeled_imgs = unlabeled_batch
        imgs1 = transform1(unlabeled_imgs)
        imgs2 = transform2(unlabeled_imgs)

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

                imgs1 = imgs1.cuda(local_rank)
                imgs2 = imgs2.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            outputs1 = model_train(imgs1)
            outputs2 = model_train(imgs2)
            #----------------------#
            #   损失计算
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice
            unsupervised_loss = consistency_loss(outputs1, outputs2)
            loss = loss + unsupervised_loss
            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   损失计算
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))