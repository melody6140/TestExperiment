#!/usr/bin/env python3
"""
模型对比实验脚本
比较原始复杂模型 vs 改进简化模型的性能
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.optim import AdamW
import time
import logging
from train1 import HateSpeechClassifier as OriginalModel, StagedTrainingPipeline as OriginalTrainer
from train_improved import ImprovedHateSpeechClassifier, ImprovedTrainingPipeline, prepare_data
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    """模型对比器"""

    def __init__(self, device):
        self.device = device
        self.results = {}

    def compare_models(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer):
        """对比两个模型的性能"""
        logger.info("=== 开始模型对比实验 ===")

        # 1. 测试原始复杂模型
        logger.info("1. 训练原始复杂模型...")
        original_results = self._train_original_model(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer
        )

        # 2. 测试改进简化模型
        logger.info("2. 训练改进简化模型...")
        improved_results = self._train_improved_model(
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer
        )

        # 3. 对比结果
        self._compare_results(original_results, improved_results)

        return {
            'original': original_results,
            'improved': improved_results
        }

    def _train_original_model(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer):
        """训练原始模型"""
        start_time = time.time()

        # 创建原始模型
        model = OriginalModel()
        trainer = OriginalTrainer(model, tokenizer, self.device)

        # 简化训练（减少epoch以节省时间）
        train_loader, val_loader = trainer.create_data_loaders(
            train_texts[:100], train_labels[:100],  # 使用小数据集快速测试
            val_texts[:50], val_labels[:50],
            batch_size=8
        )

        # 简化的训练过程
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        best_f1 = 0.0
        for epoch in range(3):  # 只训练3个epoch
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = model(input_ids, attention_mask, labels)
                loss = outputs['loss']

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 验证
            val_f1 = trainer.evaluate(val_loader)
            if val_f1 > best_f1:
                best_f1 = val_f1

        # 测试集评估
        test_loader = trainer.create_data_loaders(
            test_texts[:50], test_labels[:50],
            test_texts[:50], test_labels[:50],
            augment_data=False
        )[1]

        final_metrics = trainer.detailed_evaluate(test_loader)

        training_time = time.time() - start_time

        return {
            'metrics': final_metrics,
            'training_time': training_time,
            'model_complexity': self._count_parameters(model),
            'best_val_f1': best_f1
        }

    def _train_improved_model(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, tokenizer):
        """训练改进模型"""
        start_time = time.time()

        # 创建改进模型
        model = ImprovedHateSpeechClassifier()
        trainer = ImprovedTrainingPipeline(model, tokenizer, self.device)

        # 简化训练
        train_loader, val_loader = trainer.create_data_loaders(
            train_texts[:100], train_labels[:100],  # 使用小数据集快速测试
            val_texts[:50], val_labels[:50],
            batch_size=8
        )

        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()

        best_f1 = 0.0
        for epoch in range(3):  # 只训练3个epoch
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                texts = batch['text']

                outputs = model(input_ids, attention_mask, labels, texts)
                loss = outputs['loss']

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 验证
            val_f1 = trainer.evaluate(val_loader)
            if val_f1 > best_f1:
                best_f1 = val_f1

        # 测试集评估
        test_loader = trainer.create_data_loaders(
            test_texts[:50], test_labels[:50],
            test_texts[:50], test_labels[:50]
        )[1]

        final_metrics = trainer.evaluate(test_loader)

        training_time = time.time() - start_time

        return {
            'metrics': final_metrics,
            'training_time': training_time,
            'model_complexity': self._count_parameters(model),
            'best_val_f1': best_f1
        }

    def _count_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _compare_results(self, original_results, improved_results):
        """对比结果"""
        logger.info("=== 模型对比结果 ===")

        # 性能对比
        logger.info("性能对比:")
        if 'f1_macro' in original_results['metrics']:
            orig_f1 = original_results['metrics']['f1_macro']
            impr_f1 = improved_results['metrics']['f1_macro']
            logger.info(f"F1 Score - 原始: {orig_f1:.4f}, 改进: {impr_f1:.4f}, 提升: {impr_f1-orig_f1:.4f}")

        if 'accuracy' in original_results['metrics']:
            orig_acc = original_results['metrics']['accuracy']
            impr_acc = improved_results['metrics']['accuracy']
            logger.info(f"Accuracy - 原始: {orig_acc:.4f}, 改进: {impr_acc:.4f}, 提升: {impr_acc-orig_acc:.4f}")

        # 效率对比
        logger.info("效率对比:")
        orig_time = original_results['training_time']
        impr_time = improved_results['training_time']
        logger.info(f"训练时间 - 原始: {orig_time:.2f}s, 改进: {impr_time:.2f}s, 节省: {orig_time-impr_time:.2f}s")

        # 复杂度对比
        logger.info("复杂度对比:")
        orig_params = original_results['model_complexity']
        impr_params = improved_results['model_complexity']
        logger.info(f"参数数量 - 原始: {orig_params:,}, 改进: {impr_params:,}")

        # 保存结果
        comparison_results = {
            'original_model': original_results,
            'improved_model': improved_results,
            'comparison': {
                'f1_improvement': impr_f1 - orig_f1 if 'f1_macro' in original_results['metrics'] else 0,
                'accuracy_improvement': impr_acc - orig_acc if 'accuracy' in original_results['metrics'] else 0,
                'time_saved': orig_time - impr_time,
                'parameter_reduction': orig_params - impr_params
            }
        }

        # 保存到文件
        with open('outputs/model_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)

        logger.info("对比结果已保存到 outputs/model_comparison.json")

def quick_test():
    """快速测试函数"""
    logger.info("开始快速模型对比测试...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs('outputs', exist_ok=True)

    # 准备数据（使用小数据集进行快速测试）
    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels, tokenizer) = prepare_data()

    # 创建对比器
    comparator = ModelComparator(device)

    # 执行对比
    results = comparator.compare_models(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer
    )

    return results

def ablation_analysis():
    """消融实验分析"""
    logger.info("=== 消融实验分析 ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (train_texts, train_labels, val_texts, val_labels,
     test_texts, test_labels, tokenizer) = prepare_data()

    from train_improved import ablation_study

    # 执行消融实验
    ablation_results = ablation_study(
        train_texts[:100], train_labels[:100],  # 小数据集快速测试
        val_texts[:50], val_labels[:50],
        tokenizer, device
    )

    # 分析结果
    logger.info("消融实验结果分析:")
    baseline_f1 = ablation_results['baseline']['f1_macro']

    for config, metrics in ablation_results.items():
        if config != 'baseline':
            improvement = metrics['f1_macro'] - baseline_f1
            logger.info(f"{config}: F1={metrics['f1_macro']:.4f} (提升: {improvement:.4f})")

    # 保存消融实验结果
    with open('outputs/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    return ablation_results

def main():
    """主函数"""
    logger.info("开始完整的模型对比和分析实验")

    # 1. 快速对比测试
    logger.info("步骤1: 快速模型对比")
    comparison_results = quick_test()

    # 2. 消融实验分析
    logger.info("步骤2: 消融实验分析")
    ablation_results = ablation_analysis()

    # 3. 生成总结报告
    logger.info("步骤3: 生成总结报告")

    summary = {
        'experiment_summary': {
            'purpose': '验证基于第一性原理的模型改进效果',
            'key_improvements': [
                '困难样本挖掘',
                '自适应Focal Loss',
                '专注的对比学习',
                '语义一致性正则化'
            ],
            'results': {
                'model_comparison': comparison_results,
                'ablation_study': ablation_results
            }
        }
    }

    with open('outputs/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=== 实验完成 ===")
    logger.info("所有结果已保存到 outputs/ 目录")
    logger.info("主要文件:")
    logger.info("- model_comparison.json: 模型对比结果")
    logger.info("- ablation_results.json: 消融实验结果")
    logger.info("- experiment_summary.json: 实验总结")

if __name__ == "__main__":
    main()

