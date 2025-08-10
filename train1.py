import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import re
import warnings
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


class HateSpeechDataProcessor:
    """仇恨言论数据处理器"""

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hate_keywords = self._load_hate_keywords()

    def _load_hate_keywords(self) -> set:
        """加载仇恨关键词词典"""
        hate_words = {
            'hate', 'stupid', 'idiot', 'moron', 'dumb', 'ugly', 'loser', 'pathetic',
            'disgusting', 'worthless', 'useless', 'trash', 'garbage', 'scum', 'pig',
            'bitch', 'bastard', 'damn', 'hell', 'shit', 'fuck', 'ass', 'cunt',
            'nigger', 'faggot', 'retard', 'whore', 'slut', 'terrorist', 'nazi',
            'kill', 'die', 'murder', 'death', 'destroy', 'eliminate', 'attack',
            'violence', 'punch', 'beat', 'shoot', 'bomb', 'burn', 'crush'
        }
        return hate_words

    def clean_text(self, text: str) -> str:
        """文本清洗"""
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 移除用户名
        text = re.sub(r'@\w+', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据集"""
        try:
            df = pd.read_csv(file_path)
            df['text'] = df['text'].apply(self.clean_text)
            return df
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            # 创建示例数据用于演示
            sample_data = {
                'text': [
                    'i hate you so much you stupid idiot',
                    'women are inferior and should stay home',
                    'all immigrants should go back to their countries',
                    'this is a normal conversation about weather',
                    'i love spending time with my family',
                    'great job on your presentation today'
                ],
                'HS': [1, 1, 1, 0, 0, 0]
            }
            return pd.DataFrame(sample_data)


class HateSpeechDataset(Dataset):
    """仇恨言论数据集类"""

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


class HateKeywordIdentifier:
    """仇恨关键词识别模块"""

    def __init__(self, hate_keywords: set):
        self.hate_keywords = hate_keywords

    def identify_importance(self, text: str, tokenizer) -> List[float]:
        """识别文本中每个token的仇恨重要性分数"""
        tokens = tokenizer.tokenize(text.lower())
        importance_scores = []

        for token in tokens:
            # 移除BERT的特殊标记前缀
            clean_token = token.replace('##', '')

            # 计算仇恨重要性分数
            base_score = 0.0

            # 词典匹配得分
            if clean_token in self.hate_keywords:
                base_score += 0.8

            # 部分匹配得分
            for hate_word in self.hate_keywords:
                if clean_token in hate_word or hate_word in clean_token:
                    base_score += 0.3
                    break

            # 长度惩罚（避免短词获得过高分数）
            if len(clean_token) <= 2:
                base_score *= 0.5

            importance_scores.append(min(base_score, 1.0))

        return importance_scores


class SelectiveAugmenter:
    """选择性数据增强模块"""

    def __init__(self, tokenizer, keyword_identifier):
        self.tokenizer = tokenizer
        self.keyword_identifier = keyword_identifier
        # 简单的同义词字典
        self.synonyms = {
            'good': ['great', 'nice', 'fine', 'ok'],
            'bad': ['terrible', 'awful', 'poor'],
            'big': ['large', 'huge', 'massive'],
            'small': ['tiny', 'little', 'mini'],
            'very': ['extremely', 'really', 'quite'],
            'people': ['folks', 'individuals', 'persons'],
            'think': ['believe', 'feel', 'consider'],
            'like': ['love', 'enjoy', 'appreciate'],
        }

    def selective_perturbation(self, text: str, preserve_threshold: float = 0.3) -> str:
        """选择性扰动：保护重要词汇，扰动非重要词汇"""
        # 获取重要性分数
        importance_scores = self.keyword_identifier.identify_importance(text, self.tokenizer)
        tokens = text.split()

        # 确保长度一致
        if len(importance_scores) != len(tokens):
            # 如果长度不匹配，使用简单策略
            importance_scores = [0.5] * len(tokens)

        perturbed_tokens = []
        for i, (token, importance) in enumerate(zip(tokens, importance_scores)):
            if importance > preserve_threshold:
                # 保护重要词汇
                perturbed_tokens.append(token)
            else:
                # 扰动非重要词汇
                if token.lower() in self.synonyms and random.random() < 0.3:
                    synonym = random.choice(self.synonyms[token.lower()])
                    perturbed_tokens.append(synonym)
                else:
                    perturbed_tokens.append(token)

        return ' '.join(perturbed_tokens)

    def generate_augmented_data(self, texts: List[str], labels: List[int],
                                num_augmentations: int = 2) -> Tuple[List[str], List[int]]:
        """生成增强数据"""
        augmented_texts = []
        augmented_labels = []

        for text, label in zip(texts, labels):
            # 保留原始数据
            augmented_texts.append(text)
            augmented_labels.append(label)

            # 生成扰动数据
            for _ in range(num_augmentations):
                perturbed_text = self.selective_perturbation(text)
                augmented_texts.append(perturbed_text)
                augmented_labels.append(label)

        return augmented_texts, augmented_labels


class DualContrastiveLearner(nn.Module):
    """双重对比学习模块"""

    def __init__(self, hidden_size=768, temperature=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        )

    def forward(self, embeddings, labels=None, augmented_embeddings=None):
        """计算双重对比学习损失"""
        projected_embeddings = self.projection(embeddings)
        projected_embeddings = F.normalize(projected_embeddings, dim=-1)

        total_loss = 0.0
        loss_count = 0

        # 自监督对比学习损失
        if augmented_embeddings is not None:
            augmented_projected = self.projection(augmented_embeddings)
            augmented_projected = F.normalize(augmented_projected, dim=-1)

            self_supervised_loss = self._compute_self_supervised_loss(
                projected_embeddings, augmented_projected
            )
            total_loss += self_supervised_loss
            loss_count += 1

        # 监督对比学习损失
        if labels is not None:
            supervised_loss = self._compute_supervised_loss(projected_embeddings, labels)
            total_loss += supervised_loss
            loss_count += 1

        return total_loss / max(loss_count, 1)

    def _compute_self_supervised_loss(self, embeddings, augmented_embeddings):
        """计算自监督对比学习损失"""
        batch_size = embeddings.size(0)

        # 计算正样本相似度
        pos_similarity = torch.sum(embeddings * augmented_embeddings, dim=-1) / self.temperature

        # 计算所有样本相似度
        all_embeddings = torch.cat([embeddings, augmented_embeddings], dim=0)
        similarity_matrix = torch.matmul(embeddings, all_embeddings.T) / self.temperature

        # 创建掩码，排除自身
        mask = torch.eye(batch_size).bool().to(embeddings.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = pos_similarity - torch.log(torch.sum(exp_sim, dim=-1))

        return -log_prob.mean()

    def _compute_supervised_loss(self, embeddings, labels):
        """计算监督对比学习损失"""
        batch_size = embeddings.size(0)
        labels = labels.contiguous().view(-1, 1)

        # 创建标签掩码
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 排除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(embeddings.device),
            0
        )
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        return -mean_log_prob_pos.mean()


class AdaptiveLossIntegrator(nn.Module):
    """自适应损失融合模块"""

    def __init__(self):
        super().__init__()
        # 可学习的损失权重
        self.classification_weight = nn.Parameter(torch.tensor(1.0))
        self.contrastive_weight = nn.Parameter(torch.tensor(0.5))
        self.focal_weight = nn.Parameter(torch.tensor(0.3))

    def focal_loss(self, predictions, labels, alpha=0.25, gamma=2.0):
        """计算Focal Loss"""
        ce_loss = F.cross_entropy(predictions, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def forward(self, classification_logits, labels, contrastive_loss=None):
        """计算融合损失"""
        # 分类损失
        classification_loss = F.cross_entropy(classification_logits, labels)

        # Focal损失
        focal_loss = self.focal_loss(classification_logits, labels)

        # 总损失
        total_loss = (self.classification_weight * classification_loss +
                      self.focal_weight * focal_loss)

        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss

        return total_loss, {
            'classification_loss': classification_loss.item(),
            'focal_loss': focal_loss.item(),
            'contrastive_loss': contrastive_loss.item() if contrastive_loss is not None else 0.0
        }


class HateSpeechClassifier(nn.Module):
    """仇恨言论分类器主模型"""

    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.contrastive_learner = DualContrastiveLearner(self.bert.config.hidden_size)
        self.loss_integrator = AdaptiveLossIntegrator()

    def forward(self, input_ids, attention_mask, labels=None,
                augmented_input_ids=None, augmented_attention_mask=None):
        """前向传播"""
        # 获取BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # 分类预测
        logits = self.classifier(pooled_output)

        result = {'logits': logits}

        if labels is not None:
            # 计算对比学习损失
            contrastive_loss = None
            if augmented_input_ids is not None:
                aug_outputs = self.bert(input_ids=augmented_input_ids,
                                        attention_mask=augmented_attention_mask)
                aug_pooled_output = aug_outputs.pooler_output

                contrastive_loss = self.contrastive_learner(
                    pooled_output, labels, aug_pooled_output
                )
            else:
                contrastive_loss = self.contrastive_learner(pooled_output, labels)

            # 计算总损失
            total_loss, loss_dict = self.loss_integrator(logits, labels, contrastive_loss)

            result.update({
                'loss': total_loss,
                'loss_dict': loss_dict
            })

        return result


class StagedTrainingPipeline:
    """阶段性训练管道"""

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.current_stage = 1

    def create_data_loaders(self, train_texts, train_labels, val_texts, val_labels,
                            batch_size=16, augment_data=True):
        """创建数据加载器"""
        # 数据增强
        if augment_data:
            processor = HateSpeechDataProcessor(self.tokenizer)
            keyword_identifier = HateKeywordIdentifier(processor.hate_keywords)
            augmenter = SelectiveAugmenter(self.tokenizer, keyword_identifier)

            train_texts, train_labels = augmenter.generate_augmented_data(
                train_texts, train_labels, num_augmentations=1
            )
            logger.info(f"数据增强后训练集大小: {len(train_texts)}")

        # 创建数据集
        train_dataset = HateSpeechDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = HateSpeechDataset(val_texts, val_labels, self.tokenizer)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def stage1_basic_training(self, train_loader, val_loader, epochs=5):
        """阶段1：基础分类器训练"""
        logger.info("=== 阶段1：基础分类器预训练 ===")

        # 冻结对比学习模块
        for param in self.model.contrastive_learner.parameters():
            param.requires_grad = False

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=2e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )

        best_f1 = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f'阶段1 Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # 只计算分类损失，不使用对比学习
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
                })

            # 验证
            val_f1 = self.evaluate(val_loader)
            logger.info(f'阶段1 Epoch {epoch + 1}: Validation F1 = {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'stage1_best_model.pt')

        logger.info(f'阶段1完成，最佳F1分数: {best_f1:.4f}')

    def stage2_contrastive_training(self, train_loader, val_loader, epochs=8):
        """阶段2：对比学习模块训练"""
        logger.info("=== 阶段2：对比学习模块训练 ===")

        # 冻结分类器，解冻对比学习模块
        for param in self.model.bert.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = False
        for param in self.model.contrastive_learner.parameters():
            param.requires_grad = True

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                          lr=1e-5, eps=1e-8)

        best_f1 = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f'阶段2 Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (progress_bar.n + 1):.4f}'
                })

            # 验证
            val_f1 = self.evaluate(val_loader)
            logger.info(f'阶段2 Epoch {epoch + 1}: Validation F1 = {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'stage2_best_model.pt')

        logger.info(f'阶段2完成，最佳F1分数: {best_f1:.4f}')

    def stage3_joint_training(self, train_loader, val_loader, epochs=12):
        """阶段3：联合微调"""
        logger.info("=== 阶段3：联合微调优化 ===")

        # 解冻所有参数
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )

        best_f1 = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f'阶段3 Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                loss_dict = outputs.get('loss_dict', {})
                progress_bar.set_postfix({
                    'total_loss': f'{loss.item():.4f}',
                    'cls_loss': f'{loss_dict.get("classification_loss", 0):.4f}',
                    'cont_loss': f'{loss_dict.get("contrastive_loss", 0):.4f}'
                })

            # 验证
            val_f1 = self.evaluate(val_loader)
            logger.info(f'阶段3 Epoch {epoch + 1}: Validation F1 = {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'stage3_best_model.pt')

        logger.info(f'阶段3完成，最佳F1分数: {best_f1:.4f}')
        return best_f1

    def train_staged(self, train_texts, train_labels, val_texts, val_labels):
        """执行完整的阶段性训练"""
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels
        )

        # 执行三个训练阶段
        self.stage1_basic_training(train_loader, val_loader)
        self.stage2_contrastive_training(train_loader, val_loader)
        best_f1 = self.stage3_joint_training(train_loader, val_loader)

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

        # 计算指标
        f1 = f1_score(all_labels, all_predictions, average='macro')
        return f1

    def detailed_evaluate(self, data_loader):
        """详细评估"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
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

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }

        return metrics


def prepare_data(data_path='dataset/hateval2019_en_train.csv'):
    """准备训练数据"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = HateSpeechDataProcessor(tokenizer)

    # 加载数据
    df = processor.load_data(data_path)

    # 数据分割
    from sklearn.model_selection import train_test_split

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(), df['HS'].tolist(), test_size=0.3, random_state=42
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )

    logger.info(f"训练集大小: {len(train_texts)}")
    logger.info(f"验证集大小: {len(val_texts)}")
    logger.info(f"测试集大小: {len(test_texts)}")

    return (train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, tokenizer)


def main():
    """主函数"""
    logger.info("开始仇恨言论检测模型训练")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 准备数据
    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels, tokenizer) = prepare_data()

    # 创建模型
    model = HateSpeechClassifier()

    # 创建训练管道
    trainer = StagedTrainingPipeline(model, tokenizer, device)

    # 执行阶段性训练
    logger.info("开始阶段性训练...")
    best_f1 = trainer.train_staged(train_texts, train_labels, val_texts, val_labels)

    # 在测试集上评估
    logger.info("在测试集上进行最终评估...")
    _, test_loader = trainer.create_data_loaders(
        test_texts, test_labels, test_texts, test_labels, augment_data=False
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('stage3_best_model.pt'))
    trainer.model = model.to(device)

    final_metrics = trainer.detailed_evaluate(test_loader)

    logger.info("=== 最终测试结果 ===")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'hate_speech_classifier_final.pt')
    logger.info("模型训练完成，已保存到 hate_speech_classifier_final.pt")


if __name__ == "__main__":
    main()