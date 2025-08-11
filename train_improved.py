#!/usr/bin/env python3
"""
改进版仇恨言论检测模型
基于第一性原理重新设计，专注于核心问题解决
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import re
import warnings
from typing import List
import logging
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class HateSpeechDataProcessor:
    """简化的数据处理器"""

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据集"""
        try:
            df = pd.read_csv(file_path)
            df['text'] = df['text'].apply(self.clean_text)
            df = df.dropna(subset=['HS'])
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            # 创建示例数据
            sample_data = {
                'text': [
                    'i hate you so much you stupid idiot',
                    'women are inferior and should stay home',
                    'all immigrants should go back to their countries',
                    'this is a normal conversation about weather',
                    'i love spending time with my family',
                    'great job on your presentation today',
                    'you are such a loser and worthless person',
                    'muslims are terrorists and dangerous people',
                    'what a beautiful day for a walk',
                    'congratulations on your achievement'
                ],
                'HS': [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]
            }
            return pd.DataFrame(sample_data)

class HateSpeechDataset(Dataset):
    """数据集类"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class HardSampleMiner:
    """困难样本挖掘器 - 基于预测不确定性"""

    def __init__(self, model, uncertainty_threshold=0.7):
        self.model = model
        self.threshold = uncertainty_threshold

    def mine_hard_samples(self, dataloader, device):
        """挖掘预测不确定的样本"""
        hard_samples = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = self.model(input_ids, attention_mask)
                probs = F.softmax(outputs['logits'], dim=-1)

                # 计算预测熵作为不确定性度量
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                max_entropy = torch.log(torch.tensor(probs.size(-1), dtype=torch.float))
                normalized_entropy = entropy / max_entropy

                # 选择高不确定性样本
                hard_mask = normalized_entropy > self.threshold
                if hard_mask.any():
                    for i, is_hard in enumerate(hard_mask):
                        if is_hard:
                            hard_samples.append({
                                'text': batch['text'][i],
                                'label': batch['label'][i].item(),
                                'uncertainty': normalized_entropy[i].item()
                            })

        return hard_samples

class AdaptiveFocalLoss(nn.Module):
    """自适应Focal Loss - 可学习的gamma参数"""

    def __init__(self, alpha=1.0, gamma_init=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)

        # 自适应gamma：训练过程中自动调整
        adaptive_gamma = torch.clamp(self.gamma, min=0.5, max=5.0)

        focal_loss = self.alpha * (1 - pt) ** adaptive_gamma * ce_loss
        return focal_loss.mean()

class FocusedContrastiveLoss(nn.Module):
    """专注的对比学习损失 - 简化但有效"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """只在同类样本间进行对比学习"""
        batch_size = embeddings.size(0)

        # 归一化embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 创建同类样本mask
        labels = labels.view(-1, 1)
        same_class_mask = torch.eq(labels, labels.T).float()

        # 排除自身
        same_class_mask.fill_diagonal_(0)

        # 只对同类样本计算对比损失
        if same_class_mask.sum() > 0:
            # 正样本：同类样本
            pos_mask = same_class_mask
            # 负样本：不同类样本
            neg_mask = 1 - same_class_mask
            neg_mask.fill_diagonal_(0)  # 排除自身

            # 计算对比损失
            pos_exp = torch.exp(sim_matrix) * pos_mask
            neg_exp = torch.exp(sim_matrix) * neg_mask

            # 避免除零
            pos_sum = pos_exp.sum(dim=1, keepdim=True)
            neg_sum = neg_exp.sum(dim=1, keepdim=True)

            # 只对有正样本的行计算损失
            valid_rows = (pos_sum.squeeze() > 0)
            if valid_rows.any():
                pos_sum = pos_sum[valid_rows]
                neg_sum = neg_sum[valid_rows]

                loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
                return loss.mean()

        return torch.tensor(0.0, device=embeddings.device)

class SemanticConsistencyRegularizer(nn.Module):
    """语义一致性正则化"""

    def __init__(self, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, embeddings, texts):
        """基于文本相似度的一致性约束"""
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        consistency_loss = 0.0
        count = 0

        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # 计算文本相似度（简单的词汇重叠）
                text_sim = self._compute_text_similarity(texts[i], texts[j])

                # 计算embedding相似度
                emb_sim = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                )

                # 一致性损失：文本相似度和embedding相似度应该一致
                consistency_loss += torch.abs(text_sim - emb_sim)
                count += 1

        if count > 0:
            return self.lambda_reg * consistency_loss / count
        else:
            return torch.tensor(0.0, device=embeddings.device)

    def _compute_text_similarity(self, text1, text2):
        """计算文本相似度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if len(words1) == 0 or len(words2) == 0:
            return torch.tensor(0.0)

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard_sim = intersection / union if union > 0 else 0.0
        return torch.tensor(jaccard_sim, dtype=torch.float)

class ImprovedHateSpeechClassifier(nn.Module):
    """改进的仇恨言论分类器 - 基于第一性原理"""

    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 核心组件：只保留真正有效的
        self.focal_loss = AdaptiveFocalLoss()
        self.contrastive_loss = FocusedContrastiveLoss()
        self.consistency_reg = SemanticConsistencyRegularizer()

        # 损失权重（可学习）
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.3, 0.1]))

    def forward(self, input_ids, attention_mask, labels=None, texts=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.pooler_output
        embeddings = self.dropout(embeddings)
        logits = self.classifier(embeddings)

        result = {'logits': logits}

        if labels is not None:
            # 主要损失：自适应Focal Loss
            focal_loss = self.focal_loss(logits, labels)

            # 辅助损失：对比学习
            contrastive_loss = self.contrastive_loss(embeddings, labels)

            # 正则化：语义一致性
            consistency_loss = self.consistency_reg(embeddings, texts) if texts else torch.tensor(0.0, device=embeddings.device)

            # 加权总损失
            total_loss = (torch.abs(self.loss_weights[0]) * focal_loss +
                         torch.abs(self.loss_weights[1]) * contrastive_loss +
                         torch.abs(self.loss_weights[2]) * consistency_loss)

            result.update({
                'loss': total_loss,
                'loss_components': {
                    'focal': focal_loss.item(),
                    'contrastive': contrastive_loss.item(),
                    'consistency': consistency_loss.item(),
                    'weights': self.loss_weights.detach().cpu().numpy().tolist()
                }
            })

        return result

class ImprovedTrainingPipeline:
    """改进的训练管道"""

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.hard_miner = HardSampleMiner(model)

    def create_data_loaders(self, train_texts, train_labels, val_texts, val_labels, batch_size=16):
        """创建数据加载器"""
        train_dataset = HateSpeechDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train_with_hard_mining(self, train_texts, train_labels, val_texts, val_labels, epochs=15):
        """使用困难样本挖掘的训练"""
        logger.info("=== 开始改进训练（困难样本挖掘 + 自适应损失） ===")

        # 创建初始数据加载器
        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels
        )

        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader),
            num_training_steps=len(train_loader) * epochs
        )

        best_f1 = 0.0
        hard_mining_interval = 3  # 每3个epoch进行一次困难样本挖掘

        for epoch in range(epochs):
            # 困难样本挖掘
            if epoch > 0 and epoch % hard_mining_interval == 0:
                logger.info(f"第{epoch}轮：进行困难样本挖掘...")
                hard_samples = self.hard_miner.mine_hard_samples(train_loader, self.device)
                logger.info(f"发现 {len(hard_samples)} 个困难样本")

                # 将困难样本加入训练集（重复训练）
                if hard_samples:
                    hard_texts = [sample['text'] for sample in hard_samples]
                    hard_labels = [sample['label'] for sample in hard_samples]

                    # 创建包含困难样本的新训练集
                    enhanced_texts = train_texts + hard_texts * 2  # 困难样本重复2次
                    enhanced_labels = train_labels + hard_labels * 2

                    train_loader, _ = self.create_data_loaders(
                        enhanced_texts, enhanced_labels, val_texts, val_labels
                    )

            # 训练一个epoch
            self.model.train()
            total_loss = 0
            loss_components = {'focal': 0, 'contrastive': 0, 'consistency': 0}

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch['text']

                outputs = self.model(input_ids, attention_mask, labels, texts)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                components = outputs['loss_components']
                for key in loss_components:
                    loss_components[key] += components[key]

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'focal': f'{components["focal"]:.4f}',
                    'contr': f'{components["contrastive"]:.4f}',
                    'consist': f'{components["consistency"]:.4f}'
                })

            # 验证
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics['f1_macro']

            logger.info(f'Epoch {epoch + 1}: Val F1 = {val_f1:.4f}, '
                       f'Focal = {loss_components["focal"]/len(train_loader):.4f}, '
                       f'Contrastive = {loss_components["contrastive"]/len(train_loader):.4f}, '
                       f'Consistency = {loss_components["consistency"]/len(train_loader):.4f}')

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'outputs/improved_best_model.pt')
                logger.info(f'新的最佳模型已保存，F1: {best_f1:.4f}')

        logger.info(f'训练完成，最佳F1分数: {best_f1:.4f}')
        return best_f1

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs['logits'], dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算各种指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }

def ablation_study(train_texts, train_labels, val_texts, val_labels, tokenizer, device):
    """消融实验：验证每个组件的贡献"""
    logger.info("=== 开始消融实验 ===")

    configs = {
        'baseline': {
            'use_focal': False,
            'use_contrastive': False,
            'use_consistency': False
        },
        'focal_only': {
            'use_focal': True,
            'use_contrastive': False,
            'use_consistency': False
        },
        'focal_contrastive': {
            'use_focal': True,
            'use_contrastive': True,
            'use_consistency': False
        },
        'full_model': {
            'use_focal': True,
            'use_contrastive': True,
            'use_consistency': True
        }
    }

    results = {}

    for name, config in configs.items():
        logger.info(f"训练配置: {name}")

        # 创建模型
        model = ImprovedHateSpeechClassifier()

        # 根据配置禁用某些损失
        if not config['use_focal']:
            model.loss_weights.data[0] = 0.0
        if not config['use_contrastive']:
            model.loss_weights.data[1] = 0.0
        if not config['use_consistency']:
            model.loss_weights.data[2] = 0.0

        # 训练
        trainer = ImprovedTrainingPipeline(model, tokenizer, device)
        train_loader, val_loader = trainer.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels
        )

        # 简化训练（5个epoch用于快速验证）
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        for epoch in range(5):
            progress_bar = tqdm(train_loader, desc=f'{name} Epoch {epoch + 1}/5')
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                texts = batch['text']

                outputs = model(input_ids, attention_mask, labels, texts)
                loss = outputs['loss']

                loss.backward()
                optimizer.step()

        # 评估
        metrics = trainer.evaluate(val_loader)
        results[name] = metrics

        logger.info(f"{name} - F1: {metrics['f1_macro']:.4f}, Acc: {metrics['accuracy']:.4f}")

    return results

def prepare_data(data_dir='dataset/'):
    """准备训练数据"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = HateSpeechDataProcessor(tokenizer)

    # 加载数据
    train_df = processor.load_data(f'{data_dir}hateval2019_en_train.csv')
    val_df = processor.load_data(f'{data_dir}hateval2019_en_dev.csv')
    test_df = processor.load_data(f'{data_dir}hateval2019_en_test.csv')

    train_texts, train_labels = train_df['text'].tolist(), train_df['HS'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['HS'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['HS'].tolist()

    logger.info(f"训练集大小: {len(train_texts)}")
    logger.info(f"验证集大小: {len(val_texts)}")
    logger.info(f"测试集大小: {len(test_texts)}")

    return (train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, tokenizer)

def main():
    """主函数"""
    logger.info("开始改进版仇恨言论检测模型实验")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 准备数据
    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels, tokenizer) = prepare_data()

    # 1. 消融实验
    logger.info("步骤1: 消融实验")
    ablation_results = ablation_study(train_texts, train_labels, val_texts, val_labels, tokenizer, device)

    # 2. 完整模型训练
    logger.info("步骤2: 完整模型训练")
    model = ImprovedHateSpeechClassifier()
    trainer = ImprovedTrainingPipeline(model, tokenizer, device)

    best_f1 = trainer.train_with_hard_mining(
        train_texts, train_labels, val_texts, val_labels
    )

    # 3. 测试集评估
    logger.info("步骤3: 测试集最终评估")
    _, test_loader = trainer.create_data_loaders(
        test_texts, test_labels, test_texts, test_labels
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('outputs/improved_best_model.pt'))
    trainer.model = model.to(device)

    final_metrics = trainer.evaluate(test_loader)

    # 输出结果
    logger.info("=== 消融实验结果 ===")
    for config, metrics in ablation_results.items():
        logger.info(f"{config}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

    logger.info("=== 最终测试结果 ===")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'outputs/improved_hate_speech_classifier_final.pt')
    logger.info("改进模型训练完成！")

if __name__ == "__main__":
    main()

