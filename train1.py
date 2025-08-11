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
from typing import List, Tuple
import logging
from tqdm import tqdm

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
            # 处理标签列的缺失值
            df = df.dropna(subset=['HS'])
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
    """仇恨关键词识别模块 - 增强版"""

    def __init__(self, hate_keywords: set):
        self.hate_keywords = hate_keywords
        # 添加上下文敏感的重要性评估
        self.context_patterns = {
            'target_groups': {'women', 'immigrants', 'muslims', 'jews', 'blacks', 'gays', 'people'},
            'intensity_words': {'very', 'extremely', 'totally', 'completely', 'absolutely'},
            'negation_words': {'not', 'never', 'no', 'none', 'nothing'}
        }

    def identify_importance(self, text: str, tokenizer) -> List[float]:
        """识别文本中每个token的仇恨重要性分数 - 增强版"""
        tokens = tokenizer.tokenize(text.lower())
        words = text.lower().split()  # 用于上下文分析
        importance_scores = []

        for i, token in enumerate(tokens):
            # 移除BERT的特殊标记前缀
            clean_token = token.replace('##', '')

            # 基础仇恨词典得分
            base_score = self._get_hate_dictionary_score(clean_token)

            # 上下文增强得分
            context_score = self._get_context_enhanced_score(clean_token, words, i)

            # 位置权重（句首句尾的重要词汇权重更高）
            position_weight = self._get_position_weight(i, len(tokens))

            # 综合得分
            final_score = min((base_score + context_score) * position_weight, 1.0)
            importance_scores.append(final_score)

        return importance_scores

    def _get_hate_dictionary_score(self, token: str) -> float:
        """基于仇恨词典的基础得分"""
        score = 0.0

        # 精确匹配
        if token in self.hate_keywords:
            score += 0.8

        # 部分匹配（更精确的匹配策略）
        for hate_word in self.hate_keywords:
            if len(token) > 3:  # 避免短词误匹配
                if token in hate_word and len(token) >= len(hate_word) * 0.6:
                    score += 0.4
                elif hate_word in token and len(hate_word) >= len(token) * 0.6:
                    score += 0.4

        return min(score, 0.8)

    def _get_context_enhanced_score(self, token: str, words: List[str], token_idx: int) -> float:
        """基于上下文的增强得分"""
        context_score = 0.0

        # 寻找token在words中的大致位置
        word_idx = min(token_idx // 2, len(words) - 1)  # 粗略映射

        # 检查周围上下文（前后2个词）
        context_window = []
        for j in range(max(0, word_idx - 2), min(len(words), word_idx + 3)):
            if j < len(words):
                context_window.append(words[j])

        # 目标群体上下文增强
        if any(group in context_window for group in self.context_patterns['target_groups']):
            if token in self.hate_keywords:
                context_score += 0.3  # 针对特定群体的仇恨词汇权重更高

        # 强度词增强
        if any(intensity in context_window for intensity in self.context_patterns['intensity_words']):
            if token in self.hate_keywords:
                context_score += 0.2

        # 否定词降权
        if any(neg in context_window for neg in self.context_patterns['negation_words']):
            context_score -= 0.3  # 否定语境下降低仇恨词权重

        return max(context_score, 0.0)

    def _get_position_weight(self, position: int, total_length: int) -> float:
        """位置权重：句首句尾的重要词汇权重更高"""
        if total_length <= 3:
            return 1.0

        # 句首和句尾权重较高
        if position < total_length * 0.2 or position > total_length * 0.8:
            return 1.1

        return 1.0


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
        """选择性扰动：保护重要词汇，扰动非重要词汇 - 增强版"""
        # 获取重要性分数
        importance_scores = self.keyword_identifier.identify_importance(text, self.tokenizer)
        tokens = text.split()

        # 处理tokenizer和word-level的长度不匹配问题
        if len(importance_scores) != len(tokens):
            # 使用滑动窗口平均来对齐分数
            importance_scores = self._align_importance_scores(importance_scores, len(tokens))

        perturbed_tokens = []
        for i, (token, importance) in enumerate(zip(tokens, importance_scores)):
            if importance > preserve_threshold:
                # 保护重要词汇（仇恨关键词等）
                perturbed_tokens.append(token)
            else:
                # 对非重要词汇进行多样化扰动
                perturbed_token = self._apply_perturbation(token, importance)
                perturbed_tokens.append(perturbed_token)

        return ' '.join(perturbed_tokens)

    def _align_importance_scores(self, token_scores: List[float], word_count: int) -> List[float]:
        """对齐token级分数到word级分数"""
        if len(token_scores) == word_count:
            return token_scores

        # 使用平均池化来对齐
        aligned_scores = []
        tokens_per_word = len(token_scores) / word_count

        for i in range(word_count):
            start_idx = int(i * tokens_per_word)
            end_idx = int((i + 1) * tokens_per_word)
            word_score = sum(token_scores[start_idx:end_idx]) / max(1, end_idx - start_idx)
            aligned_scores.append(word_score)

        return aligned_scores

    def _apply_perturbation(self, token: str, importance: float) -> str:
        """对token应用多样化扰动策略"""
        # 根据重要性分数调整扰动概率
        perturbation_prob = 0.4 * (1 - importance)  # 重要性越低，扰动概率越高

        if random.random() > perturbation_prob:
            return token

        # 多种扰动策略
        strategies = ['synonym', 'case_change', 'char_repeat']
        strategy = random.choice(strategies)

        if strategy == 'synonym' and token.lower() in self.synonyms:
            return random.choice(self.synonyms[token.lower()])
        elif strategy == 'case_change' and len(token) > 2:
            # 随机改变大小写
            return ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in token)
        elif strategy == 'char_repeat' and len(token) > 1:
            # 随机重复字符（模拟网络语言）
            idx = random.randint(0, len(token) - 1)
            return token[:idx] + token[idx] + token[idx:]

        return token

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
    """双重对比学习模块 - 增强版"""

    def __init__(self, hidden_size=768, temperature=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256)
        )

    def forward(self, embeddings, labels=None, augmented_embeddings=None, importance_weights=None):
        """计算双重对比学习损失 - 支持token级加权"""
        projected_embeddings = self.projection(embeddings)
        projected_embeddings = F.normalize(projected_embeddings, dim=-1)

        total_loss = 0.0
        loss_count = 0

        # 自监督对比学习损失
        if augmented_embeddings is not None:
            augmented_projected = self.projection(augmented_embeddings)
            augmented_projected = F.normalize(augmented_projected, dim=-1)

            self_supervised_loss = self._compute_self_supervised_loss(
                projected_embeddings, augmented_projected, importance_weights
            )
            total_loss += self_supervised_loss
            loss_count += 1

        # 监督对比学习损失
        if labels is not None:
            supervised_loss = self._compute_supervised_loss(
                projected_embeddings, labels, importance_weights
            )
            total_loss += supervised_loss
            loss_count += 1

        return total_loss / max(loss_count, 1)

    def _compute_self_supervised_loss(self, embeddings, augmented_embeddings, importance_weights=None):
        """计算自监督对比学习损失"""
        batch_size = embeddings.size(0)

        # 计算正样本相似度
        pos_similarity = torch.sum(embeddings * augmented_embeddings, dim=-1) / self.temperature

        # 计算所有样本相似度
        all_embeddings = torch.cat([embeddings, augmented_embeddings], dim=0)
        similarity_matrix = torch.matmul(embeddings, all_embeddings.T) / self.temperature

        # 创建掩码，排除自身（注意：相似度矩阵是 batch_size x (2*batch_size)）
        # 我们需要排除的是与自身的相似度，即前batch_size个位置
        mask = torch.zeros(batch_size, 2 * batch_size).bool().to(embeddings.device)
        mask[torch.arange(batch_size), torch.arange(batch_size)] = True
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = pos_similarity - torch.log(torch.sum(exp_sim, dim=-1))

        # 应用重要性权重
        if importance_weights is not None:
            # 确保权重维度匹配
            if importance_weights.size(0) == batch_size:
                weighted_loss = -log_prob * importance_weights
                return weighted_loss.mean()

        return -log_prob.mean()

    def _compute_supervised_loss(self, embeddings, labels, importance_weights=None):
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

        # 应用重要性权重
        if importance_weights is not None:
            # 确保权重维度匹配
            if importance_weights.size(0) == batch_size:
                weighted_loss = -mean_log_prob_pos * importance_weights
                return weighted_loss.mean()

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

    def stage1_warmup_training(self, train_loader, val_loader, epochs=8):
        """阶段1：预热训练 - 分类器和对比学习联合训练"""
        logger.info("=== 阶段1：预热训练（分类+对比学习） ===")

        # 使用较小的学习率进行预热
        optimizer = AdamW(self.model.parameters(), lr=1e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader),
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
            logger.info(f'阶段1 Epoch {epoch + 1}: Validation F1 = {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'outputs/stage1_best_model.pt')

        logger.info(f'阶段1完成，最佳F1分数: {best_f1:.4f}')

    def stage2_fine_tuning(self, train_loader, val_loader, epochs=12):
        """阶段2：精细调优 - 使用更高学习率进行最终优化"""
        logger.info("=== 阶段2：精细调优 ===")

        # 使用稍高的学习率进行精细调优
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )

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
            logger.info(f'阶段2 Epoch {epoch + 1}: Validation F1 = {val_f1:.4f}')

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 'outputs/stage2_best_model.pt')

        logger.info(f'阶段2完成，最佳F1分数: {best_f1:.4f}')
        return best_f1

    def train_staged(self, train_texts, train_labels, val_texts, val_labels):
        """执行两阶段训练策略"""
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            train_texts, train_labels, val_texts, val_labels
        )

        # 执行两个训练阶段
        self.stage1_warmup_training(train_loader, val_loader)
        best_f1 = self.stage2_fine_tuning(train_loader, val_loader)

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


def prepare_data(data_dir='dataset/'):
    """准备训练数据"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = HateSpeechDataProcessor(tokenizer)

    # 加载三个数据文件
    train_df = processor.load_data(f'{data_dir}hateval2019_en_train.csv')
    val_df = processor.load_data(f'{data_dir}hateval2019_en_dev.csv')
    test_df = processor.load_data(f'{data_dir}hateval2019_en_test.csv')

    # 提取文本和标签
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
    model.load_state_dict(torch.load('outputs/stage2_best_model.pt'))
    trainer.model = model.to(device)

    final_metrics = trainer.detailed_evaluate(test_loader)

    logger.info("=== 最终测试结果 ===")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'outputs/hate_speech_classifier_final.pt')
    logger.info("模型训练完成，已保存到 hate_speech_classifier_final.pt")


if __name__ == "__main__":
    main()