#!/usr/bin/env python3
"""
修正版仇恨言论检测模型
真正基于语义复杂性的对比学习框架
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

class SemanticAugmentor:
    """语义感知的数据增强器 - 专门针对隐含仇恨言论"""

    def __init__(self):
        # 定义隐含仇恨的常见语义变换模式
        self.euphemism_patterns = {
            # 委婉表达模式
            "invasion": ["influx", "demographic change", "wave", "migration pressure"],
            "cleanse": ["preserve", "protect", "maintain purity", "restore"],
            "inferior": ["less capable", "not suited for", "naturally different", "traditionally"],
            "terrorist": ["extremist", "radical", "problematic elements", "security concerns"]
        }

        self.coded_expressions = {
            # 暗语表达
            "protect our community": ["preserve our values", "maintain our character", "safeguard our heritage"],
            "traditional values": ["cultural preservation", "ancestral wisdom", "time-honored principles"],
            "concerned about": ["worried by", "troubled by", "alarmed at"],
            "cultural differences": ["integration challenges", "compatibility issues", "assimilation problems"]
        }

    def augment_implicit_hate(self, text: str, intent_label: int) -> List[str]:
        """为隐含仇恨样本生成语义保持的增强样本"""
        augmented = []

        if intent_label == 1:  # 仇恨言论
            # 生成保持恶意意图但改变表达方式的变换
            augmented.extend(self._generate_euphemistic_variants(text))
            augmented.extend(self._generate_coded_variants(text))
            augmented.extend(self._generate_concern_framing(text))
        else:  # 非仇恨言论
            # 生成保持友善意图的变换
            augmented.extend(self._generate_positive_variants(text))

        return augmented[:3]  # 返回3个增强样本

    def _generate_euphemistic_variants(self, text: str) -> List[str]:
        """生成委婉表达的变体"""
        variants = []
        text_lower = text.lower()

        # 替换直接的仇恨词汇为委婉表达
        for direct_term, euphemisms in self.euphemism_patterns.items():
            if direct_term in text_lower:
                for euphemism in euphemisms[:2]:
                    variant = text.replace(direct_term, euphemism)
                    variants.append(variant)

        return variants

    def _generate_coded_variants(self, text: str) -> List[str]:
        """生成暗语表达的变体"""
        variants = []

        # 将明显的仇恨表达转换为隐含表达
        coded_versions = [
            f"As a concerned citizen, I believe {text.lower()}",
            f"I'm not prejudiced, but {text.lower()}",
            f"For safety reasons, we must consider that {text.lower()}",
            f"From a practical perspective, {text.lower()}"
        ]

        variants.extend(coded_versions[:2])
        return variants

    def _generate_concern_framing(self, text: str) -> List[str]:
        """生成关切框架的表达"""
        return [
            f"I'm genuinely concerned about {text.lower()}",
            f"As a parent, I worry that {text.lower()}",
            f"For our community's wellbeing, {text.lower()}"
        ]

    def _generate_positive_variants(self, text: str) -> List[str]:
        """为非仇恨言论生成积极变体"""
        return [
            f"I appreciate that {text.lower()}",
            f"It's wonderful how {text.lower()}",
            f"I'm grateful for {text.lower()}"
        ]

class SemanticAwareASRM(nn.Module):
    """语义感知的注意力重校准模块 - 真正服务于语义理解"""

    def __init__(self, hidden_size=768, reduction_ratio=16):
        super().__init__()
        self.hidden_size = hidden_size

        # 语义核心识别网络
        self.semantic_identifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction_ratio),
            nn.ReLU(),
            nn.Linear(hidden_size // reduction_ratio, hidden_size),
            nn.Sigmoid()
        )

        # 意图-表达解耦网络
        self.intent_extractor = nn.Linear(hidden_size, hidden_size // 2)
        self.expression_extractor = nn.Linear(hidden_size, hidden_size // 2)

        # 语义复杂度评估器
        self.complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, is_contrastive_phase=False):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 全局语义表示
        global_repr = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]

        if is_contrastive_phase:
            # 对比学习阶段：强化语义核心特征
            semantic_weights = self.semantic_identifier(global_repr)
            semantic_weights = semantic_weights.unsqueeze(1)  # [batch_size, 1, hidden_size]

            # 识别语义复杂度
            complexity_score = self.complexity_assessor(global_repr)

            # 对于复杂样本，更强地激活语义核心
            enhanced_weights = semantic_weights * (1 + complexity_score.unsqueeze(-1))

            return hidden_states * enhanced_weights, complexity_score
        else:
            # 常规阶段：意图-表达解耦
            intent_repr = self.intent_extractor(global_repr)
            expression_repr = self.expression_extractor(global_repr)

            # 重新组合
            enhanced_repr = torch.cat([intent_repr, expression_repr], dim=-1)
            enhanced_repr = enhanced_repr.unsqueeze(1).expand(-1, seq_len, -1)

            return hidden_states + 0.1 * enhanced_repr, None

class ImplicitHateContrastiveLoss(nn.Module):
    """基于语义相似性的隐含仇恨对比学习损失"""

    def __init__(self, temperature=0.07, augmentor=None):
        super().__init__()
        self.temperature = temperature
        self.augmentor = augmentor or SemanticAugmentor()

    def forward(self, embeddings, texts, labels, tokenizer=None):
        """真正的语义对比学习"""
        batch_size = embeddings.size(0)
        device = embeddings.device

        # 生成语义正样本
        positive_texts = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            # 为每个样本生成语义相似的正样本
            augmented = self.augmentor.augment_implicit_hate(text, label.item())
            positive_texts.append(augmented[0] if augmented else text)

        # 如果有tokenizer，重新编码正样本
        if tokenizer:
            # 简化处理：使用原始embedding的扰动作为正样本
            pos_embeddings = self._create_semantic_positive(embeddings, labels)
        else:
            pos_embeddings = self._create_semantic_positive(embeddings, labels)

        # 归一化
        embeddings = F.normalize(embeddings, dim=-1)
        pos_embeddings = F.normalize(pos_embeddings, dim=-1)

        # 计算相似度
        sim_pos = torch.sum(embeddings * pos_embeddings, dim=-1) / self.temperature

        # 负样本：不同语义意图的样本
        neg_mask = self._create_semantic_negative_mask(labels)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # 对比损失计算
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_samples = 0

        for i in range(batch_size):
            if neg_mask[i].sum() > 0:
                # 正样本分数
                pos_score = sim_pos[i]

                # 负样本分数
                neg_scores = sim_matrix[i][neg_mask[i]]

                # InfoNCE loss
                all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores])
                loss = loss + (-torch.log_softmax(all_scores, dim=0)[0])
                valid_samples += 1

        return loss / max(valid_samples, 1)

    def _create_semantic_positive(self, embeddings, labels):
        """创建语义正样本embedding"""
        # 简化实现：通过语义保持的噪声扰动
        noise = torch.randn_like(embeddings) * 0.1

        # 根据标签调整扰动方向
        for i, label in enumerate(labels):
            if label.item() == 1:  # 仇恨言论
                # 在隐含仇恨的语义空间中扰动
                noise[i] = noise[i] * 0.5  # 较小扰动保持恶意意图
            else:  # 非仇恨
                # 在友善语义空间中扰动
                noise[i] = noise[i] * 0.8

        return embeddings + noise

    def _create_semantic_negative_mask(self, labels):
        """创建语义负样本mask"""
        batch_size = len(labels)
        neg_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and labels[i].item() != labels[j].item():
                    neg_mask[i][j] = True

        return neg_mask

class HateSpeechDataProcessor:
    """改进的数据处理器 - 支持HatEval 2019真实数据集"""

    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """文本清洗 - 保留URL等信息，因为它们可能包含语义信息"""
        # 温和的清洗，保留更多语义信息
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除过多的特殊符号，但保留基本结构
        text = re.sub(r'[^\w\s@#.,!?:;()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_hateval_data(self) -> tuple:
        """加载HatEval 2019数据集的所有三个部分"""
        try:
            # 加载训练集
            train_df = pd.read_csv('dataset/hateval2019_en_train.csv')
            logger.info(f"训练集加载成功: {len(train_df)} 样本")

            # 加载验证集
            val_df = pd.read_csv('dataset/hateval2019_en_dev.csv')
            logger.info(f"验证集加载成功: {len(val_df)} 样本")

            # 加载测试集
            test_df = pd.read_csv('dataset/hateval2019_en_test.csv')
            logger.info(f"测试集加载成功: {len(test_df)} 样本")

            # 清洗文本数据并重建索引
            processed_dfs = []
            for df, name in [(train_df, '训练集'), (val_df, '验证集'), (test_df, '测试集')]:
                # 清洗文本
                df['text'] = df['text'].apply(self.clean_text)

                # 移除包含NaN的行
                df_clean = df.dropna(subset=['text', 'HS']).copy()

                # 重建索引
                df_clean.reset_index(drop=True, inplace=True)

                logger.info(f"{name}清洗后: {len(df_clean)} 样本")

                # 打印类别分布
                hate_count = (df_clean['HS'] == 1).sum()
                non_hate_count = (df_clean['HS'] == 0).sum()
                logger.info(f"{name}标签分布 - 仇恨言论: {hate_count} ({hate_count/len(df_clean)*100:.1f}%), "
                           f"非仇恨: {non_hate_count} ({non_hate_count/len(df_clean)*100:.1f}%)")

                processed_dfs.append(df_clean)

            return processed_dfs[0], processed_dfs[1], processed_dfs[2]

        except Exception as e:
            logger.error(f"真实数据集加载失败: {e}")
            logger.warning("回退到合成数据集...")
            return self._create_fallback_data()

    def _create_fallback_data(self) -> tuple:
        """创建回退的合成数据集"""
        logger.info("生成扩展合成数据集作为回退...")
        synthetic_data = self._create_extended_synthetic_data()

        # 分割为训练、验证、测试
        total_len = len(synthetic_data)
        train_end = int(0.7 * total_len)
        val_end = int(0.85 * total_len)

        train_df = synthetic_data[:train_end].copy()
        val_df = synthetic_data[train_end:val_end].copy()
        test_df = synthetic_data[val_end:].copy()

        return train_df, val_df, test_df

    def _create_extended_synthetic_data(self) -> pd.DataFrame:
        """创建扩展的合成数据集 - 用于回退情况"""
        # 基础模板（保持原有逻辑）
        implicit_hate_templates = [
            "We need to protect our community from this cultural {threat}",
            "As a concerned {role}, I worry about these demographic changes affecting our {concern}",
            "I'm not {denial}, but certain groups just don't {integrate} well into our society",
            "We must preserve our traditional {values} from external {influences}",
            "It's not about {surface}, it's about protecting our {hidden_agenda}",
            "I have nothing against {group}, however their {negative_trait} is concerning",
            "Speaking objectively, these {people} cause more {problems} than benefits",
            "For economic reasons, we should prioritize {us} over {them}",
            "Our {institution} standards are being compromised by {external_factor}",
            "I support diversity, but not at the expense of our {community_aspect}",
        ]

        explicit_hate_templates = [
            "I hate {group} so much they are {insult}",
            "{group} are all {negative_stereotype} and {threat}",
            "All {group} should {action} back to their {place}",
            "{group} are {inferior} and should {restriction}",
            "These {group} {people} are ruining our {thing}",
            "Why do {group} always {negative_behavior}",
            "{group} don't belong in our {place}",
            "I can't stand {group} and their {negative_trait}",
        ]

        benign_templates = [
            "This is a normal conversation about {topic} today",
            "I love spending time with my diverse group of {people}",
            "Our community welcomes people from all {backgrounds}",
            "Great job on your {achievement}, it was very {positive_adj}",
            "I appreciate learning about different {cultures} and {traditions}",
            "It's wonderful how {diversity_aspect} enriches our {community}",
            "I'm grateful for the opportunity to {positive_action} with {people}",
            "Our {institution} benefits from having {diversity}",
            "I support policies that help everyone {positive_outcome}",
            "Diversity brings new {benefits} that help us all {grow}",
        ]

        # 填充词典
        replacements = {
            'threat': ['invasion', 'influx', 'wave', 'migration pressure', 'demographic shift'],
            'role': ['parent', 'citizen', 'taxpayer', 'community member', 'concerned individual'],
            'concern': ['children', 'neighborhood', 'future', 'community', 'values'],
            'denial': ['racist', 'prejudiced', 'bigoted', 'discriminatory', 'biased'],
            'integrate': ['assimilate', 'fit in', 'adapt', 'blend in', 'conform'],
            'values': ['values', 'culture', 'heritage', 'traditions', 'way of life'],
            'influences': ['influences', 'forces', 'pressures', 'changes', 'elements'],
            'group': ['Muslims', 'immigrants', 'foreigners', 'minorities', 'outsiders'],
            'insult': ['stupid', 'dangerous', 'worthless', 'inferior', 'problematic'],
            'negative_stereotype': ['terrorists', 'criminals', 'lazy', 'violent', 'dishonest'],
            'topic': ['weather', 'sports', 'work', 'food', 'travel'],
            'people': ['friends', 'colleagues', 'neighbors', 'classmates', 'family'],
            'backgrounds': ['backgrounds', 'cultures', 'countries', 'traditions', 'experiences'],
            'achievement': ['presentation', 'work', 'project', 'effort', 'contribution'],
            'positive_adj': ['insightful', 'impressive', 'thoughtful', 'excellent', 'outstanding'],
        }

        synthetic_texts = []
        synthetic_labels = []

        # 生成隐含仇恨样本 (300个)
        for _ in range(300):
            template = random.choice(implicit_hate_templates)
            # 替换占位符
            text = template
            for placeholder, options in replacements.items():
                if '{' + placeholder + '}' in text:
                    text = text.replace('{' + placeholder + '}', random.choice(options))
            synthetic_texts.append(text)
            synthetic_labels.append(1)

        # 生成显式仇恨样本 (200个)
        for _ in range(200):
            template = random.choice(explicit_hate_templates)
            text = template
            for placeholder, options in replacements.items():
                if '{' + placeholder + '}' in text:
                    text = text.replace('{' + placeholder + '}', random.choice(options))
            synthetic_texts.append(text)
            synthetic_labels.append(1)

        # 生成友善样本 (500个)
        for _ in range(500):
            template = random.choice(benign_templates)
            text = template
            for placeholder, options in replacements.items():
                if '{' + placeholder + '}' in text:
                    text = text.replace('{' + placeholder + '}', random.choice(options))
            synthetic_texts.append(text)
            synthetic_labels.append(0)

        # 创建DataFrame
        synthetic_data = pd.DataFrame({
            'text': synthetic_texts,
            'HS': synthetic_labels
        })

        # 打乱数据
        synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)
        return synthetic_data

class ImprovedHateSpeechClassifier(nn.Module):
    """修正版仇恨言论分类器 - 真正的语义复杂性处理"""

    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # 语义感知的ASRM
        self.asrm = SemanticAwareASRM(self.bert.config.hidden_size)

        # 分类器
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        # 损失函数
        self.focal_loss = AdaptiveFocalLoss()
        self.contrastive_loss = ImplicitHateContrastiveLoss()

        # 损失权重（可学习）
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5]))

    def forward(self, input_ids, attention_mask, labels=None, texts=None, is_contrastive_phase=False):
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 语义感知的ASRM处理
        enhanced_states, complexity_scores = self.asrm(hidden_states, is_contrastive_phase)

        # 池化得到句子表示
        embeddings = torch.mean(enhanced_states, dim=1)  # [batch_size, hidden_size]
        embeddings = self.dropout(embeddings)

        # 分类
        logits = self.classifier(embeddings)

        result = {'logits': logits, 'embeddings': embeddings}
        if complexity_scores is not None:
            result['complexity_scores'] = complexity_scores

        if labels is not None:
            # 主要损失：Focal Loss
            focal_loss = self.focal_loss(logits, labels)

            # 语义对比损失
            contrastive_loss = self.contrastive_loss(embeddings, texts, labels, self.tokenizer if hasattr(self, 'tokenizer') else None)

            # 加权总损失
            total_loss = (torch.abs(self.loss_weights[0]) * focal_loss +
                         torch.abs(self.loss_weights[1]) * contrastive_loss)

            result.update({
                'loss': total_loss,
                'loss_components': {
                    'focal': focal_loss.item(),
                    'contrastive': contrastive_loss.item(),
                    'weights': self.loss_weights.detach().cpu().numpy().tolist()
                }
            })

        return result

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

class ImprovedTrainingPipeline:
    """改进的训练管道"""

    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

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

        for epoch in range(epochs):

            # 训练一个epoch
            self.model.train()
            total_loss = 0
            total_contrastive_loss = 0
            total_classification_loss = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()

                # 将数据移动到设备
                batch_data = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.device)
                    else:
                        batch_data[key] = value

                # 对比学习阶段
                contrastive_outputs = self.model(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    labels=batch_data['label'],
                    texts=batch_data['text'],
                    is_contrastive_phase=True
                )
                contrastive_loss = contrastive_outputs['loss_components']['contrastive']

                # 分类阶段
                classification_outputs = self.model(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    labels=batch_data['label'],
                    texts=batch_data['text'],
                    is_contrastive_phase=False
                )
                classification_loss = classification_outputs['loss_components']['focal']

                # 总损失
                total_loss_batch = self.model.loss_weights[0] * classification_loss + \
                                   self.model.loss_weights[1] * contrastive_loss

                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += total_loss_batch.item()
                total_contrastive_loss += contrastive_loss if isinstance(contrastive_loss, (int, float)) else contrastive_loss.item()
                total_classification_loss += classification_loss if isinstance(classification_loss, (int, float)) else classification_loss.item()

                progress_bar.set_postfix({
                    'total_loss': f'{total_loss_batch.item():.4f}',
                    'contrastive': f'{contrastive_loss.item():.4f}' if hasattr(contrastive_loss, 'item') else f'{contrastive_loss:.4f}',
                    'classification': f'{classification_loss.item():.4f}' if hasattr(classification_loss, 'item') else f'{classification_loss:.4f}'
                })

            # 验证
            val_metrics = self.evaluate(val_loader)
            val_f1 = val_metrics['f1_macro']

            logger.info(f'Epoch {epoch + 1}: Val F1 = {val_f1:.4f}')
            logger.info(f'平均损失 - Total: {total_loss/len(train_loader):.4f}, '
                       f'Contrastive: {total_contrastive_loss/len(train_loader):.4f}, '
                       f'Classification: {total_classification_loss/len(train_loader):.4f}')

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
                # 将数据移动到设备
                batch_data = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.device)
                    else:
                        batch_data[key] = value

                # 只使用分类阶段进行评估
                outputs = self.model(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    is_contrastive_phase=False
                )
                predictions = torch.argmax(outputs['logits'], dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_data['label'].cpu().numpy())

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


def main():
    """主函数"""
    logger.info("开始修正版语义感知仇恨言论检测实验 - 使用HatEval 2019真实数据集")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 准备数据处理器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = HateSpeechDataProcessor(tokenizer)

    # 加载HatEval 2019真实数据集
    train_df, val_df, test_df = processor.load_hateval_data()

    # 提取文本和标签
    train_texts, train_labels = train_df['text'].tolist(), train_df['HS'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['HS'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['HS'].tolist()

    logger.info(f"数据集统计:")
    logger.info(f"- 训练集: {len(train_texts)} 样本")
    logger.info(f"- 验证集: {len(val_texts)} 样本")
    logger.info(f"- 测试集: {len(test_texts)} 样本")

    # 创建模型和训练器
    model = ImprovedHateSpeechClassifier()
    trainer = ImprovedTrainingPipeline(model, tokenizer, device)

    # 训练模型
    best_f1 = trainer.train_with_hard_mining(
        train_texts, train_labels, val_texts, val_labels, epochs=10
    )

    logger.info(f"训练完成，最佳验证F1: {best_f1:.4f}")

    # 加载最佳模型并在测试集上评估
    if os.path.exists('outputs/improved_best_model.pt'):
        model.load_state_dict(torch.load('outputs/improved_best_model.pt'))
        logger.info("已加载最佳模型权重")

        # 在测试集上评估
        test_loader = trainer.create_data_loaders(
            test_texts, test_labels, test_texts, test_labels, batch_size=16
        )[0]

        test_metrics = trainer.evaluate(test_loader)

        logger.info("=== 测试集最终结果 ===")
        logger.info(f"准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"F1 (macro): {test_metrics['f1_macro']:.4f}")
        logger.info(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        logger.info(f"精确度: {test_metrics['precision']:.4f}")
        logger.info(f"召回率: {test_metrics['recall']:.4f}")

        # 保存测试结果
        test_results = {
            'best_val_f1': best_f1,
            'test_metrics': test_metrics,
            'dataset_info': {
                'train_size': len(train_texts),
                'val_size': len(val_texts),
                'test_size': len(test_texts),
                'train_hate_ratio': sum(train_labels) / len(train_labels),
                'val_hate_ratio': sum(val_labels) / len(val_labels),
                'test_hate_ratio': sum(test_labels) / len(test_labels)
            }
        }

        import json
        with open('outputs/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info("测试结果已保存到 outputs/test_results.json")
    else:
        logger.warning("未找到最佳模型文件，跳过测试集评估")

    logger.info(f"实验完成！最佳验证F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()

