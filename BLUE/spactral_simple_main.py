import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import logging
from simple_graph_dataset import SimpleGraphDataset
from HeteroGraphNetwork import FusionGNN
from FullHeteroGNN import FullHeteroGNN
from metrics import calculate_metrics
import datetime
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import torch.linalg
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 预测后处理模块
class PredictionPostProcessor:
    def __init__(self, detection_threshold=0.3, min_prediction=0.0):
        self.detection_threshold = detection_threshold
        self.min_prediction = min_prediction
        
    def apply_threshold_strategy(self, predictions, apply_detection=True):
        """
        应用阈值策略进行预测后处理
        
        Args:
            predictions: 原始预测值
            apply_detection: 是否应用检测阈值
            
        Returns:
            processed_predictions: 处理后的预测值
        """
        processed = predictions.copy()
        
        if apply_detection:
            # 低于阈值的预测设为0
            processed[processed < self.detection_threshold] = 0.0
            
        # 确保预测值非负
        processed = np.maximum(processed, self.min_prediction)
        
        return processed
    
    def get_detection_metrics(self, predictions, targets, threshold=None):
        """
        计算检测任务的指标（二元分类）
        
        Args:
            predictions: 预测值
            targets: 真实值  
            threshold: 检测阈值
            
        Returns:
            detection_metrics: 检测指标字典
        """
        if threshold is None:
            threshold = self.detection_threshold
            
        # 转换为二元标签
        pred_binary = (predictions > threshold).astype(int)
        true_binary = (targets > threshold).astype(int)
        
        # 计算混淆矩阵元素
        tp = np.sum((pred_binary == 1) & (true_binary == 1))
        tn = np.sum((pred_binary == 0) & (true_binary == 0))
        fp = np.sum((pred_binary == 1) & (true_binary == 0))
        fn = np.sum((pred_binary == 0) & (true_binary == 1))
        
        # 计算指标
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        
        return {
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'detection_accuracy': accuracy,
            'detection_specificity': specificity,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def get_regression_metrics(self, predictions, targets, mask=None):
        """
        计算回归任务的指标（仅对检测到感染的样本）
        
        Args:
            predictions: 预测值
            targets: 真实值
            mask: 掩码（True表示有感染的样本）
            
        Returns:
            regression_metrics: 回归指标字典
        """
        if mask is None:
            # 默认对所有非零真实值计算回归指标
            mask = targets > self.detection_threshold
            
        if mask.sum() == 0:
            return {
                'regression_mse': 0.0,
                'regression_mae': 0.0,
                'regression_rmse': 0.0,
                'regression_mape': 0.0,
                'num_regression_samples': 0
            }
        
        pred_masked = predictions[mask]
        true_masked = targets[mask]
        
        mse = np.mean((pred_masked - true_masked) ** 2)
        mae = np.mean(np.abs(pred_masked - true_masked))
        rmse = np.sqrt(mse)
        
        # MAPE (避免除零)
        mape_mask = true_masked > 1e-6
        if mape_mask.sum() > 0:
            mape = np.mean(np.abs((pred_masked[mape_mask] - true_masked[mape_mask]) / true_masked[mape_mask]))
        else:
            mape = 0.0
            
        return {
            'regression_mse': mse,
            'regression_mae': mae,
            'regression_rmse': rmse,
            'regression_mape': mape,
            'num_regression_samples': mask.sum()
        }


# 改进的评估函数
def enhanced_evaluate_predictions(predictions, targets, post_processor=None, dataset_name=''):
    """
    增强的预测评估函数，包含检测和回归两个任务的指标
    
    Args:
        predictions: 预测值数组
        targets: 真实值数组
        post_processor: 后处理器对象
        dataset_name: 数据集名称（用于日志）
        
    Returns:
        comprehensive_metrics: 综合指标字典
    """
    if post_processor is None:
        post_processor = PredictionPostProcessor()
    
    # 应用后处理
    processed_predictions = post_processor.apply_threshold_strategy(predictions)
    
    # 计算检测指标
    detection_metrics = post_processor.get_detection_metrics(predictions, targets)
    
    # 计算回归指标（对真实有感染的样本）
    infection_mask = targets > post_processor.detection_threshold
    regression_metrics = post_processor.get_regression_metrics(processed_predictions, targets, infection_mask)
    
    # 计算整体回归指标
    overall_mse = np.mean((processed_predictions - targets) ** 2)
    overall_mae = np.mean(np.abs(processed_predictions - targets))
    overall_rmse = np.sqrt(overall_mse)
    
    # Pearson相关系数
    try:
        correlation = np.corrcoef(processed_predictions.flatten(), targets.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # 组合所有指标
    comprehensive_metrics = {
        # 整体回归指标
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_correlation': correlation,
        
        # 检测指标
        **detection_metrics,
        
        # 感染样本回归指标
        **regression_metrics,
        
        # 数据统计
        'total_samples': len(targets),
        'infection_samples': infection_mask.sum(),
        'infection_rate': infection_mask.mean(),
        'mean_true_value': targets.mean(),
        'mean_pred_value': processed_predictions.mean(),
        'std_true_value': targets.std(),
        'std_pred_value': processed_predictions.std()
    }
    
    # 打印详细指标
    logger.info(f"=== Enhanced Evaluation Results for {dataset_name} ===")
    logger.info(f"Overall Metrics:")
    logger.info(f"  MSE: {overall_mse:.4f}, MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}")
    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info(f"Detection Metrics:")
    logger.info(f"  Precision: {detection_metrics['detection_precision']:.4f}, Recall: {detection_metrics['detection_recall']:.4f}")
    logger.info(f"  F1: {detection_metrics['detection_f1']:.4f}, Accuracy: {detection_metrics['detection_accuracy']:.4f}")
    logger.info(f"Regression Metrics (infection samples only):")
    logger.info(f"  MSE: {regression_metrics['regression_mse']:.4f}, MAE: {regression_metrics['regression_mae']:.4f}")
    logger.info(f"  Samples: {regression_metrics['num_regression_samples']}/{len(targets)}")
    logger.info(f"Data Distribution:")
    logger.info(f"  Infection rate: {infection_mask.mean():.2%}")
    logger.info(f"  Mean true: {targets.mean():.4f}, Mean pred: {processed_predictions.mean():.4f}")
    
    return comprehensive_metrics


# 改进的Focal Loss for 回归任务
class ImprovedFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-8
        
    def forward(self, input, target):
        # 计算基础MSE损失
        mse_loss = (input - target) ** 2
        
        # 计算focal权重：基于预测难度动态调整
        # 使用归一化的MSE作为难度指标
        difficulty = mse_loss / (mse_loss.mean() + self.eps)
        focal_weight = (1 + difficulty) ** self.gamma
        
        # 应用focal权重
        loss = self.alpha * focal_weight * mse_loss
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

# 智能加权MSE损失，针对感染数据特点优化
class InfectionWeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=1.0, low_weight=5.0, med_weight=10.0, high_weight=20.0,
                 low_threshold=0.5, med_threshold=5.0, high_threshold=20.0):
        super(InfectionWeightedMSELoss, self).__init__()
        self.zero_weight = zero_weight      # 零感染权重
        self.low_weight = low_weight        # 低感染权重  
        self.med_weight = med_weight        # 中等感染权重
        self.high_weight = high_weight      # 高感染权重
        self.low_threshold = low_threshold
        self.med_threshold = med_threshold
        self.high_threshold = high_threshold
        
    def forward(self, input, target):
        mse_loss = (input - target) ** 2
        
        # 根据真实感染数分配权重
        weights = torch.ones_like(target) * self.zero_weight
        
        # 低感染
        low_mask = (target > self.low_threshold) & (target <= self.med_threshold)
        weights[low_mask] = self.low_weight
        
        # 中等感染
        med_mask = (target > self.med_threshold) & (target <= self.high_threshold)
        weights[med_mask] = self.med_weight
        
        # 高感染
        high_mask = target > self.high_threshold
        weights[high_mask] = self.high_weight
        
        # 应用权重
        weighted_loss = weights * mse_loss
        
        return weighted_loss.mean()

# 分层损失：检测+回归
class HierarchicalLoss(nn.Module):
    def __init__(self, detection_weight=1.0, regression_weight=1.0, threshold=0.5):
        super(HierarchicalLoss, self).__init__()
        self.detection_weight = detection_weight
        self.regression_weight = regression_weight  
        self.threshold = threshold
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, input, target):
        # 检测任务：是否有感染
        target_binary = (target > self.threshold).float()
        input_binary = torch.sigmoid(input)  # 转换为概率
        detection_loss = self.bce_loss(input, target_binary)
        
        # 回归任务：感染数量预测（仅对有感染的样本）
        infection_mask = target > self.threshold
        if infection_mask.sum() > 0:
            regression_loss = self.mse_loss(input[infection_mask], target[infection_mask])
        else:
            regression_loss = torch.tensor(0.0, device=input.device, requires_grad=True)
        
        # 组合损失
        total_loss = (self.detection_weight * detection_loss + 
                     self.regression_weight * regression_loss)
        
        return total_loss

# Huber Loss for 鲁棒性
class AdaptiveHuberLoss(nn.Module):
    def __init__(self, delta=1.0, weight_factor=2.0):
        super(AdaptiveHuberLoss, self).__init__()
        self.delta = delta
        self.weight_factor = weight_factor
        
    def forward(self, input, target):
        diff = torch.abs(input - target)
        
        # 根据目标值大小自适应调整delta
        adaptive_delta = self.delta * (1 + target * self.weight_factor)
        
        # Huber loss with adaptive delta
        loss = torch.where(
            diff < adaptive_delta,
            0.5 * diff ** 2 / adaptive_delta,
            diff - 0.5 * adaptive_delta
        )
        
        return loss.mean()

# 组合损失函数：多种策略结合
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.4, focal_weight=0.3, weighted_weight=0.3,
                 focal_gamma=2.0, infection_weights=[1.0, 5.0, 10.0, 20.0]):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.focal_weight = focal_weight
        self.weighted_weight = weighted_weight
        
        self.mse_loss = nn.MSELoss()
        self.focal_loss = ImprovedFocalLoss(gamma=focal_gamma)
        self.weighted_loss = InfectionWeightedMSELoss(
            zero_weight=infection_weights[0],
            low_weight=infection_weights[1], 
            med_weight=infection_weights[2],
            high_weight=infection_weights[3]
        )
        
    def forward(self, input, target):
        mse = self.mse_loss(input, target)
        focal = self.focal_loss(input, target)
        weighted = self.weighted_loss(input, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.focal_weight * focal + 
                     self.weighted_weight * weighted)
        
        return total_loss


def compute_county_spatial_laplacian(x_dict, edge_index_dict, device):
    if 'county' not in x_dict or ('county', 'spatial', 'county') not in edge_index_dict:
        logger.warning("County nodes or spatial edges not found for Hetero Laplacian.")
        return None

    county_features = x_dict['county']
    num_counties = county_features.size(0)
    spatial_edges = edge_index_dict[('county', 'spatial', 'county')]

    if num_counties == 0 or spatial_edges.size(1) == 0:
        logger.warning("No counties or spatial edges for Hetero Laplacian computation.")
        return torch.eye(num_counties, device=device) if num_counties > 0 else None

    adj = torch.zeros((num_counties, num_counties), device=device)
    src, dst = spatial_edges
    valid_mask = (src < num_counties) & (dst < num_counties)
    src, dst = src[valid_mask], dst[valid_mask]
    adj[src, dst] = 1
    adj[dst, src] = 1

    deg = torch.diag(torch.sum(adj, dim=1))

    laplacian = deg - adj
    return laplacian


def compute_fusion_laplacian_learned(fusion_features, fusion_edge_index, device):
    num_fusion_nodes = fusion_features.size(0)
    if num_fusion_nodes == 0 or fusion_edge_index.size(1) == 0:
        logger.warning("No fusion nodes or learned edges for Fusion Laplacian.")
        return torch.eye(num_fusion_nodes, device=device) if num_fusion_nodes > 0 else None

    adj = torch.zeros((num_fusion_nodes, num_fusion_nodes), device=device)
    src, dst = fusion_edge_index
    valid_mask = (src < num_fusion_nodes) & (dst < num_fusion_nodes)
    src, dst = src[valid_mask], dst[valid_mask]
    adj[src, dst] = 1
    deg = torch.diag(torch.sum(adj, dim=1))
    laplacian = deg - adj
    return laplacian


def compute_spectral_loss(L_het, L_fus, k=10, device='cpu'):
    if L_het is None or L_fus is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if L_het.shape != L_fus.shape:
        logger.warning(f"Laplacian shapes mismatch: Het {L_het.shape}, Fus {L_fus.shape}. Skipping spectral loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)
    if L_het.size(0) < 2 or L_fus.size(0) < 2:  # Need at least 2 nodes for eigenvalues
        return torch.tensor(0.0, device=device, requires_grad=True)

    try:
        eigvals_het = torch.linalg.eigvalsh(L_het.float())
        stabilization_term = 1e-6
        L_fus_stabilized = L_fus + stabilization_term * torch.eye(L_fus.size(0), device=L_fus.device, dtype=L_fus.dtype)
        eigvals_fus = torch.linalg.eigvalsh(L_fus_stabilized.float())

        eigvals_het_sorted, _ = torch.sort(eigvals_het)
        eigvals_fus_sorted, _ = torch.sort(eigvals_fus)

        k_eff = min(k, len(eigvals_het_sorted), len(eigvals_fus_sorted))
        if k_eff == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if k_eff <= 1:
            logger.warning(f"k_eff ({k_eff}) too small to skip the first eigenvalue. Returning 0 loss.")
            return torch.tensor(0.0, device=device, requires_grad=True)
    
        vec_het = eigvals_het_sorted[1:k_eff]
        vec_fus = eigvals_fus_sorted[1:k_eff]
        loss = 1.0 - torch.nn.functional.cosine_similarity(vec_het, vec_fus, dim=0, eps=1e-8)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf detected in spectral loss. Returning 0.")
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss
    except torch.linalg.LinAlgError as e:
        logger.warning(f"Eigenvalue computation failed: {e}. Skipping spectral loss.")
        return torch.tensor(0.0, device=device, requires_grad=True)
    except Exception as e:
        logger.error(f"Unexpected error during spectral loss calculation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)


def collate_fn(batch):
    graph_sequences, targets, metadata, temporal_info = zip(*batch)
    batch_size = len(graph_sequences)

    if batch_size > 0:
        seq_length = len(graph_sequences[0]) if graph_sequences[0] else 0
        if seq_length == 0:
            return [], [], metadata, ([], [])

        batched_graphs = []
        for t in range(seq_length):
            time_t_graphs = [seq[t] for seq in graph_sequences if len(seq) > t]  # Check sequence length
            if time_t_graphs:
                batched_graphs.append(time_t_graphs)

        batched_targets = []
        if targets and targets[0]:
            target_len = len(targets[0])
            for t in range(target_len):
                time_t_targets = [seq[t] for seq in targets if len(seq) > t]  # Check target sequence length
                if time_t_targets:
                    batched_targets.append(time_t_targets)

        input_temporals = []
        target_temporals = []
        if temporal_info and all(temp is not None for temp in temporal_info):
            input_temporals = [temp[0] for temp in temporal_info if temp and len(temp) > 0]
            target_temporals = [temp[1] for temp in temporal_info if temp and len(temp) > 1]

        batched_temporal_info = (input_temporals, target_temporals)

        return batched_graphs, batched_targets, metadata, batched_temporal_info

    else:
        return [], [], metadata, ([], [])


def train(model, train_loader, optimizer, criterion, device, dataset, writer=None, epoch=None, fold=None,
          spectral_gamma=0.3, spectral_k=10):
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_spec_loss = 0
    num_batches = 0

    all_preds = []
    all_targets = []

    fusion_builder = model.fusion_builder if isinstance(model, FusionGNN) else None
    for batch_idx, (batched_graphs, batched_targets, _, batched_temporal_info) in enumerate(
            tqdm(train_loader, desc="Training")):

        if not batched_graphs or not batched_graphs[0]:
            continue
        batch_size = len(batched_graphs[0])
        if batch_size == 0:
            continue

        optimizer.zero_grad()

        device_graphs = []
        for time_graphs in batched_graphs:
            device_graphs.append([g.to(device) for g in time_graphs if g is not None])

        device_targets = []
        for time_targets in batched_targets:
            device_targets.append([t.to(device) for t in time_targets if t is not None])

        input_temporals, target_temporals = batched_temporal_info
        device_input_temporals = []
        device_output_temporals = []
        if input_temporals:
            for seq_temporal in input_temporals:
                if seq_temporal:
                    device_input_temporals.append(
                        [(y.to(device), w.to(device)) for y, w, _, _, _ in seq_temporal if y is not None and w is not None]
                    )

        if target_temporals:
            for seq_temporal in target_temporals:
                if seq_temporal:
                    device_output_temporals.append(
                        [(y.to(device), w.to(device)) for y, w, _, _, _ in seq_temporal if y is not None and w is not None]
                    )

        batch_outputs = model(device_graphs, device_input_temporals, device_output_temporals)

        if batch_outputs is None or not isinstance(batch_outputs, list) or not batch_outputs or any(
                output is None or not output.numel() for output in batch_outputs):
            logger.warning(f"Skipping batch {batch_idx} due to empty or invalid predictions")
            continue

        batch_loss = 0
        batch_pred_loss = 0
        batch_spec_loss = 0
        valid_samples_in_batch = 0

        for i in range(batch_size):
            outputs = batch_outputs[i]
            if not device_targets or len(device_targets) == 0:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to empty targets.")
                continue

            if i == 0:
                num_time_steps = len(device_targets)
                batch_size_actual = len(device_targets[0]) if num_time_steps > 0 else 0
                
                reorganized_targets = []
                
                for b in range(batch_size_actual):
                    reorganized_targets.append([])

                for t in range(num_time_steps):
                    for b in range(batch_size_actual):
                        if len(device_targets[t]) > b:
                            reorganized_targets[b].append(device_targets[t][b])
                
                device_targets = reorganized_targets

            sample_targets = device_targets[i] if i < len(device_targets) else []
            if not sample_targets:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to missing targets.")
                continue

            sample_pred_loss = 0
            valid_timepoints = 0

            for t, target_t in enumerate(sample_targets):
                if target_t is None or target_t.numel() == 0:
                    continue
                
                if t >= outputs.size(0):
                    logger.warning(f"Sample {i}, exceeds prediction horizon. Skipping time step {t}.")
                    continue
                
                preds_t = outputs[t]
                num_counties_pred = preds_t.size(0)
                num_counties_target = target_t.size(0)
                
                if num_counties_pred != num_counties_target:
                    logger.warning(f"Sample {i}, Time {t}: Mismatch counties pred={num_counties_pred}, target={num_counties_target}. Skipping time step.")
                    continue
                
                time_loss = criterion(preds_t, target_t)
                sample_pred_loss += time_loss
                valid_timepoints += 1

            if valid_timepoints == 0:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to no valid timepoints for prediction loss.")
                continue
            
            sample_pred_loss = sample_pred_loss / valid_timepoints

            sample_spec_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if spectral_gamma > 0 and fusion_builder is not None:
                if device_graphs and len(device_graphs[-1]) > i:
                    last_input_graph = device_graphs[-1][i]
                    if last_input_graph and hasattr(last_input_graph, 'x_dict') and hasattr(last_input_graph,
                                                                                            'edge_index_dict'):
                        L_het = compute_county_spatial_laplacian(
                            last_input_graph.x_dict, last_input_graph.edge_index_dict, device)

                        try:
                            encoded_x_dict = model.encode_heterograph(
                                last_input_graph.x_dict, last_input_graph.edge_index_dict)
                            corrected_x_dict = model.mrf_correction(
                                encoded_x_dict, last_input_graph.edge_index_dict)

                            fusion_features_i, fusion_edge_index_i, _ = fusion_builder(
                                corrected_x_dict, last_input_graph.edge_index_dict)

                            L_fus = compute_fusion_laplacian_learned(
                                fusion_features_i, fusion_edge_index_i, device)

                            if L_het is not None and L_fus is not None:
                                sample_spec_loss = compute_spectral_loss(L_het, L_fus, k=spectral_k, device=device)
                            else:
                                logger.warning(f"Could not compute Laplacians for sample {i}, batch {batch_idx}")

                        except Exception as e:
                            logger.error(
                                f"Error during per-sample fusion/Laplacian computation for sample {i}, batch {batch_idx}: {e}")
                            sample_spec_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    else:
                        logger.warning(
                            f"Missing graph data for spectral loss calculation for sample {i}, batch {batch_idx}.")
                else:
                    logger.warning(f"Missing last input graph for spectral loss for sample {i}, batch {batch_idx}.")

            sample_total_loss = sample_pred_loss + spectral_gamma * sample_spec_loss
            batch_loss += sample_total_loss
            batch_pred_loss += sample_pred_loss.item()
            batch_spec_loss += sample_spec_loss.item()
            valid_samples_in_batch += 1

        if valid_samples_in_batch > 0:
            avg_batch_loss = batch_loss / valid_samples_in_batch
            avg_batch_loss.backward()
            optimizer.step()
            total_loss += avg_batch_loss.item()
            total_pred_loss += (batch_pred_loss / valid_samples_in_batch)
            total_spec_loss += (batch_spec_loss / valid_samples_in_batch)
            num_batches += 1
        else:
            logger.warning(f"Batch {batch_idx} had no valid samples.")

    avg_epoch_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_pred_loss = total_pred_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_spec_loss = total_spec_loss / num_batches if num_batches > 0 else float('inf')

    if writer is not None and epoch is not None:
        tb_prefix = f"fold_{fold}/" if fold is not None else ""
        writer.add_scalar(f"{tb_prefix}train/total_loss", avg_epoch_loss, epoch)
        writer.add_scalar(f"{tb_prefix}train/pred_loss", avg_epoch_pred_loss, epoch)
        writer.add_scalar(f"{tb_prefix}train/spectral_loss", avg_epoch_spec_loss, epoch)

    logger.info(
        f"Epoch Avg Losses - Total: {avg_epoch_loss:.4f}, Prediction: {avg_epoch_pred_loss:.4f}, Spectral: {avg_epoch_spec_loss:.4f} (gamma={spectral_gamma})")

    return avg_epoch_pred_loss, (all_preds, all_targets)


def evaluate(model, val_loader, criterion, device, dataset, writer=None, epoch=None, fold=None, mode="val"):
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (batched_graphs, batched_targets, _, batched_temporal_info) in enumerate(
                tqdm(val_loader, desc="Evaluating")):

            if not batched_graphs or not batched_graphs[0]: continue
            batch_size = len(batched_graphs[0])
            if batch_size == 0: continue

            device_graphs = []
            for time_graphs in batched_graphs:
                device_graphs.append([g.to(device) for g in time_graphs if g is not None])

            device_targets = []
            for time_targets in batched_targets:
                device_targets.append([t.to(device) for t in time_targets if t is not None])

            input_temporals, _ = batched_temporal_info
            device_input_temporals = []
            if input_temporals:
                for seq_temporal in input_temporals:
                    if seq_temporal:
                        device_input_temporals.append(
                            [(y.to(device), w.to(device)) for y, w, _, _, _ in seq_temporal if y is not None and w is not None]
                        )

            batch_outputs = model(device_graphs, device_input_temporals)

            if batch_outputs is None or not isinstance(batch_outputs, list) or not batch_outputs or any(
                    output is None or not output.numel() for output in batch_outputs):
                logger.warning(f"Skipping batch {batch_idx} during eval due to empty or invalid predictions")
                continue

            num_time_steps = len(device_targets)
            batch_size_actual = len(device_targets[0]) if num_time_steps > 0 else 0
            
            reorganized_targets = []
            
            for b in range(batch_size_actual):
                reorganized_targets.append([])
            
            for t in range(num_time_steps):
                for b in range(batch_size_actual):
                    if len(device_targets[t]) > b:
                        reorganized_targets[b].append(device_targets[t][b])
            device_targets = reorganized_targets
            batch_loss = 0
            valid_samples_in_batch = 0

            for i in range(batch_size):
                outputs = batch_outputs[i]
                sample_targets = device_targets[i] if i < len(device_targets) else []
                if not sample_targets:
                    logger.warning(f"Skipping sample {i} in batch {batch_idx} during eval due to missing targets.")
                    continue

                sample_loss = 0
                valid_timepoints = 0

                for t, target_t in enumerate(sample_targets):
                    if target_t is None or target_t.numel() == 0: continue
                    
                    if t >= outputs.size(0):
                        continue
                        
                    preds_t = outputs[t]
                    num_counties_pred = preds_t.size(0)
                    num_counties_target = target_t.size(0)

                    if num_counties_pred != num_counties_target:
                        logger.warning(
                            f"Eval Sample {i}, Time {t}: Mismatch counties pred={num_counties_pred}, target={num_counties_target}. Skipping time step.")
                        continue

                    time_loss = criterion(preds_t, target_t)
                    sample_loss += time_loss
                    valid_timepoints += 1
                    
                    all_preds.append(preds_t.cpu())
                    all_targets.append(target_t.cpu())

                if valid_timepoints > 0:
                    sample_loss = sample_loss / valid_timepoints
                    batch_loss += sample_loss
                    valid_samples_in_batch += 1

            if valid_samples_in_batch > 0:
                batch_loss = batch_loss / valid_samples_in_batch
                total_loss += batch_loss.item()
                num_batches += 1
            else:
                logger.warning(f"Eval Batch {batch_idx} had no valid samples.")

    eval_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    if writer is not None and epoch is not None:
        tb_prefix = f"fold_{fold}/" if fold is not None else ""
        writer.add_scalar(f"{tb_prefix}{mode}/loss", eval_loss, epoch)
    return eval_loss, (all_preds, all_targets)


def main():
    parser = argparse.ArgumentParser(description='Train a temporal GNN model for infection prediction')
    parser.add_argument('--dataset', type=str, default='avian',
                        help='Dataset to use, japan or avian')
    parser.add_argument('--data_dir', type=str, default='/scratch/hn98/jd2651/processed_graphs',
                        help='Directory containing graph pickle files, for japan, it is /scratch/hn98/jd2651/processed_graphs, for avian, it is /scratch/hn98/jd2651/30_processed_graphs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=8, help='Hidden dimension size')
    parser.add_argument('--num_mrf', type=int, default=1, help='Iteration of MRF correction')
    parser.add_argument('--window_size', type=int, default=4, help='Input window size (weeks)')
    parser.add_argument('--pred_horizon', type=int, default=4, help='Prediction horizon (weeks)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--spectral_gamma', type=float, default=0.2,
                        help='Weight for the spectral regularization loss (default: 0.0, disabled)')
    parser.add_argument('--spectral_k', type=int, default=10,
                        help='Number of eigenvalues (k) to compare for spectral loss')
    parser.add_argument('--model_dir', type=str, default='./saved_results', help='Directory to save models')
    parser.add_argument('--model_type', type=str, default='FusionGNN',
                        choices=['FullHeteroGNN', 'FusionGNN'], help='Type of model to use')
    parser.add_argument('--link_threshold', type=float, default=0.5,
                        help='Threshold for learnable linking in FusionGNN')
    parser.add_argument('--use_top_k', type=bool, default=True,
                        help='Use top-k linking instead of threshold in FusionGNN')
    parser.add_argument('--top_k', type=int, default=100,
                        help='Value of K for top-k linking in FusionGNN')
    parser.add_argument('--use_temporal', type=bool, default=True,
                        help='Use temporal information (year/week) for prediction')
    parser.add_argument('--use_kfold', type=bool, default=True,
                        help='Use k-fold cross validation')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (used only if use_kfold=False)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation (used only if use_kfold=False)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing (used only if use_kfold=False)')
    parser.add_argument('--use_cuda', type=str, default='cuda',
                        help='Use CUDA for training')
    parser.add_argument('--device', type=int, default=1, 
                        help='Device to use for training')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                    help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--norm_mode', type=str, default='z_score',
                        choices=['minmax', 'z_score', 'log_minmax', 'log_plus_one'],
                        help='Normalization mode for dataset')
    parser.add_argument('--previous_weight', type=float, default=0.1,
                        help='Weight for the previous step in FusionGNN')
    parser.add_argument('--initial_weight', type=float, default=0.3,
                        help='Weight for the initial hidden state in FusionGNN')
    
    # 损失函数相关参数
    parser.add_argument('--loss_type', type=str, default='infection_weighted',
                        choices=['mse', 'focal', 'infection_weighted', 'hierarchical', 'huber', 'combined'],
                        help='Type of loss function to use')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--infection_zero_weight', type=float, default=1.0,
                        help='Weight for zero infection cases')
    parser.add_argument('--infection_low_weight', type=float, default=5.0,
                        help='Weight for low infection cases')
    parser.add_argument('--infection_med_weight', type=float, default=10.0,
                        help='Weight for medium infection cases')
    parser.add_argument('--infection_high_weight', type=float, default=20.0,
                        help='Weight for high infection cases')
    parser.add_argument('--infection_low_threshold', type=float, default=0.5,
                        help='Threshold for low infection cases')
    parser.add_argument('--infection_med_threshold', type=float, default=5.0,
                        help='Threshold for medium infection cases')
    parser.add_argument('--infection_high_threshold', type=float, default=20.0,
                        help='Threshold for high infection cases')
    parser.add_argument('--hierarchical_detection_weight', type=float, default=1.0,
                        help='Weight for detection task in hierarchical loss')
    parser.add_argument('--hierarchical_regression_weight', type=float, default=1.0,
                        help='Weight for regression task in hierarchical loss')
    parser.add_argument('--hierarchical_threshold', type=float, default=0.5,
                        help='Threshold for hierarchical loss detection task')
    
    # 后处理相关参数
    parser.add_argument('--use_post_processing', type=bool, default=True,
                        help='Whether to use post-processing for predictions')
    parser.add_argument('--detection_threshold', type=float, default=0.3,
                        help='Threshold for infection detection in post-processing')
    parser.add_argument('--min_prediction', type=float, default=0.0,
                        help='Minimum allowed prediction value')
    
    args = parser.parse_args()

    args.model_dir = args.model_dir + '_' + args.loss_type + '_' + str(args.spectral_gamma)
    os.makedirs(args.model_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device(args.use_cuda if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    logger.info("Creating datasets...")

    full_dataset = SimpleGraphDataset(
        graphs_dir=args.data_dir,
        window_size=args.window_size,
        prediction_horizon=args.pred_horizon,
        dataset=args.dataset,
        norm_mode=args.norm_mode
    )
    logger.info(f"Full dataset size: {len(full_dataset)}")
    if len(full_dataset) == 0:
        logger.error("Dataset is empty. Exiting.")
        return

    if args.use_kfold:
        logger.info(f"Using {args.num_folds}-fold cross validation")
        if args.num_folds < 2:
            logger.error("Number of folds must be at least 2 for K-fold cross validation.")
            return
        kf = KFold(n_splits=args.num_folds, shuffle=False)
        dataset_indices = list(range(len(full_dataset)))
        all_fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(dataset_indices)):
            logger.info(f"Starting fold {fold + 1}/{args.num_folds}")
            if len(train_idx) < 2:
                logger.warning(f"Fold {fold + 1} has insufficient data for train/val split. Skipping fold.")
                continue

            logger.info(f"Fold {fold + 1} split: Train={len(train_idx)}, Test={len(test_idx)}")

            train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )

            test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

            logger.info(f"Creating model for fold {fold + 1}...")

            sample_batch = next(iter(train_loader))
            if sample_batch and len(sample_batch) > 0 and sample_batch[0]:
                sample_graphs = sample_batch[0]
                if sample_graphs and len(sample_graphs) > 0:
                    sample_graph = sample_graphs[0][0]
                    county_input_dim = sample_graph['county'].x.size(-1)
                    logger.info(f"Detected county input dimension: {county_input_dim}")
                else:
                    county_input_dim = 1
                    logger.warning("Could not detect county input dimension, using default: 1")
            else:
                county_input_dim = 1
                logger.warning("Could not detect county input dimension, using default: 1")

            if args.model_type == 'FusionGNN':
                channel = 1
                model = FusionGNN(
                    channel=channel,
                    hidden_dim=args.hidden_dim,
                    num_layers=1,
                    dropout=args.dropout,
                    pred_horizon=args.pred_horizon,
                    link_threshold=args.link_threshold,
                    use_top_k=args.use_top_k,
                    top_k=args.top_k,
                    num_mrf=args.num_mrf,
                    device=args.device,
                    county_input_dim=county_input_dim  # Pass the detected dimension
                ).to(args.device)
            else:  # Default to FullHeteroGNN
                model = FullHeteroGNN(
                    hidden_dim=args.hidden_dim,
                    num_layers=1,
                    dropout=args.dropout,
                    pred_horizon=args.pred_horizon,
                    num_mrf=args.num_mrf,
                    device=args.device
                ).to(args.device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # 根据参数选择损失函数
            if args.loss_type == 'mse':
                criterion = nn.MSELoss()
                logger.info("Using MSE Loss")
            elif args.loss_type == 'focal':
                criterion = ImprovedFocalLoss(gamma=args.focal_gamma)
                logger.info(f"Using Improved Focal Loss (gamma={args.focal_gamma})")
            elif args.loss_type == 'infection_weighted':
                criterion = InfectionWeightedMSELoss(
                    zero_weight=args.infection_zero_weight,
                    low_weight=args.infection_low_weight,
                    med_weight=args.infection_med_weight,
                    high_weight=args.infection_high_weight,
                    low_threshold=args.infection_low_threshold,
                    med_threshold=args.infection_med_threshold,
                    high_threshold=args.infection_high_threshold
                )
                logger.info(f"Using Infection Weighted MSE Loss (weights: {args.infection_zero_weight}, {args.infection_low_weight}, {args.infection_med_weight}, {args.infection_high_weight})")
            elif args.loss_type == 'hierarchical':
                criterion = HierarchicalLoss(
                    detection_weight=args.hierarchical_detection_weight,
                    regression_weight=args.hierarchical_regression_weight,
                    threshold=args.hierarchical_threshold
                )
                logger.info(f"Using Hierarchical Loss (detection_weight={args.hierarchical_detection_weight}, regression_weight={args.hierarchical_regression_weight})")
            elif args.loss_type == 'huber':
                criterion = AdaptiveHuberLoss()
                logger.info("Using Adaptive Huber Loss")
            elif args.loss_type == 'combined':
                criterion = CombinedLoss(
                    focal_gamma=args.focal_gamma,
                    infection_weights=[
                        args.infection_zero_weight,
                        args.infection_low_weight,
                        args.infection_med_weight,
                        args.infection_high_weight
                    ]
                )
                logger.info("Using Combined Loss (MSE + Focal + Weighted)")
            else:
                criterion = nn.MSELoss()
                logger.warning(f"Unknown loss type '{args.loss_type}', using MSE Loss as default")

            logger.info(f"Starting training for fold {fold + 1}...")
            best_mse = float('inf')
            best_f1 = 0.0
            best_mae = float('inf')
            best_model_state = None
            
            # Early stopping parameters
            patience = 10
            counter = 0
            
            for epoch in range(args.epochs):
                train_pred_loss, _ = train(
                    model, train_loader, optimizer, criterion, args.device, full_dataset,
                    writer=None, epoch=epoch, fold=fold + 1,
                    spectral_gamma=args.spectral_gamma, spectral_k=args.spectral_k)
                multistep_train_metrics, avg_train_metrics = calculate_metrics(args,
                                                  model, train_loader, args.device, full_dataset,
                                                  pred_horizon=args.pred_horizon, mode=args.dataset, set='train',
                                                  save_predictions=False, save_dir=None)

                logger.info(
                    f"Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - Train Pred Loss: {train_pred_loss:.4f}")
                if avg_train_metrics:
                    logger.info(
                        f"Avg Train Metrics - MSE: {avg_train_metrics['mse']:.4f}, MAE: {avg_train_metrics['mae']:.4f}, F1: {avg_train_metrics['f1']:.4f}")
                    
                    # Display enhanced metrics if available
                    if 'detection_precision' in avg_train_metrics:
                        logger.info(
                            f"Enhanced Train Metrics - Detection P/R/F1: {avg_train_metrics['detection_precision']:.3f}/{avg_train_metrics['detection_recall']:.3f}/{avg_train_metrics['detection_f1']:.3f}, "
                            f"Regression MSE/MAE: {avg_train_metrics.get('regression_mse', 0):.4f}/{avg_train_metrics.get('regression_mae', 0):.4f}, "
                            f"Infection Rate: {avg_train_metrics.get('infection_rate', 0):.2%}")

                    if avg_train_metrics['regression_mse'] < best_mse:
                        print('save best model on validation MAE for this fold')
                        best_mse = avg_train_metrics['regression_mse']
                        best_f1 = avg_train_metrics['detection_f1']
                        best_mae = avg_train_metrics['regression_mae']
                        best_model_state = model.state_dict().copy()
                        
                        # Reset early stopping counter since we improved
                        counter = 0

                        save_path = os.path.join(args.model_dir, f'best_model_fold{fold + 1}.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': best_model_state,
                            'train_metrics': avg_train_metrics,
                            'best_mse': best_mse,
                            'best_f1': best_f1,
                            'best_mae': best_mae,
                            'args': vars(args)
                        }, save_path)

                        logger.info(f"Saved new best model for fold {fold + 1} to {save_path} with MAE: {best_mae:.4f}")
                    else:
                        print('not save best model on validation MAE for this fold')
                        # Increment early stopping counter
                        counter += 1
                        logger.info(f"Early stopping counter: {counter}/{patience}")
                        
                        # Check if we should stop training
                        if counter >= patience:
                            logger.info(f"Early stopping triggered after {epoch + 1} epochs for fold {fold + 1}")
                            break

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            else:
                logger.warning(f"No best model saved for fold {fold + 1}. Testing with final model state.")

            # Define save directory for this fold's predictions
            prediction_save_dir = os.path.join(args.model_dir, f'fold_{fold+1}_predictions')

            multistep_test_metrics, avg_test_metrics = calculate_metrics(args,
                                             model, test_loader, args.device, full_dataset,
                                             pred_horizon=args.pred_horizon, mode=args.dataset, set='test',
                                             save_predictions=True, save_dir=prediction_save_dir)

            if avg_test_metrics:
                logger.info(f"Fold {fold + 1} Test Metrics:")
                logger.info(f"MSE: {avg_test_metrics['mse']:.4f}, RMSE: {avg_test_metrics['rmse']:.4f}, "
                            f"MAE: {avg_test_metrics['mae']:.4f}, F1: {avg_test_metrics['f1']:.4f}")
                
                # Display enhanced metrics if available
                if 'detection_precision' in avg_test_metrics:
                    logger.info(f"Enhanced Test Metrics:")
                    logger.info(f"  Detection - Precision: {avg_test_metrics['detection_precision']:.3f}, "
                                f"Recall: {avg_test_metrics['detection_recall']:.3f}, "
                                f"F1: {avg_test_metrics['detection_f1']:.3f}, "
                                f"Accuracy: {avg_test_metrics['detection_accuracy']:.3f}")
                    logger.info(f"  Regression (infection samples) - MSE: {avg_test_metrics.get('regression_mse', 0):.4f}, "
                                f"MAE: {avg_test_metrics.get('regression_mae', 0):.4f}")
                    logger.info(f"  Data Distribution - Infection Rate: {avg_test_metrics.get('infection_rate', 0):.2%}")
                
                all_fold_metrics.append({
                    'fold': fold + 1,
                    'test_mse': avg_test_metrics['mse'],
                    'test_rmse': avg_test_metrics['rmse'],
                    'test_mae': avg_test_metrics['mae'],
                    'test_f1': avg_test_metrics['f1'],
                    'test_pearson': avg_test_metrics.get('pearson', 0.0),
                    'test_spearman': avg_test_metrics.get('spearman', 0.0),
                    # Add enhanced metrics if available
                    'test_detection_precision': avg_test_metrics.get('detection_precision', 0.0),
                    'test_detection_recall': avg_test_metrics.get('detection_recall', 0.0),
                    'test_detection_f1': avg_test_metrics.get('detection_f1', 0.0),
                    'test_detection_accuracy': avg_test_metrics.get('detection_accuracy', 0.0),
                    'test_regression_mse': avg_test_metrics.get('regression_mse', 0.0),
                    'test_regression_mae': avg_test_metrics.get('regression_mae', 0.0),
                    'test_infection_rate': avg_test_metrics.get('infection_rate', 0.0),
                })
            else:
                logger.warning(f"Could not calculate test metrics for fold {fold + 1}")

        if not all_fold_metrics:
            logger.error("No folds completed successfully. Cannot compute cross-validation summary.")
            return

        test_mse_values = [fold_data['test_mse'] for fold_data in all_fold_metrics]
        test_rmse_values = [fold_data['test_rmse'] for fold_data in all_fold_metrics]
        test_mae_values = [fold_data['test_mae'] for fold_data in all_fold_metrics]
        test_f1_values = [fold_data['test_f1'] for fold_data in all_fold_metrics]
        test_pearson_values = [fold_data['test_pearson'] for fold_data in all_fold_metrics]
        test_spearman_values = [fold_data['test_spearman'] for fold_data in all_fold_metrics]
        test_detection_precision_values = [fold_data['test_detection_precision'] for fold_data in all_fold_metrics]
        test_detection_recall_values = [fold_data['test_detection_recall'] for fold_data in all_fold_metrics]
        test_detection_f1_values = [fold_data['test_detection_f1'] for fold_data in all_fold_metrics]
        test_detection_accuracy_values = [fold_data['test_detection_accuracy'] for fold_data in all_fold_metrics]
        test_regression_mse_values = [fold_data['test_regression_mse'] for fold_data in all_fold_metrics]
        test_regression_mae_values = [fold_data['test_regression_mae'] for fold_data in all_fold_metrics]
        test_infection_rate_values = [fold_data['test_infection_rate'] for fold_data in all_fold_metrics]

        avg_test_mse = np.mean(test_mse_values)
        avg_test_rmse = np.mean(test_rmse_values)
        avg_test_mae = np.mean(test_mae_values)
        avg_test_f1 = np.mean(test_f1_values)
        avg_test_pearson = np.mean(test_pearson_values)
        avg_test_spearman = np.mean(test_spearman_values)
        avg_test_detection_precision = np.mean(test_detection_precision_values)
        avg_test_detection_recall = np.mean(test_detection_recall_values)
        avg_test_detection_f1 = np.mean(test_detection_f1_values)
        avg_test_detection_accuracy = np.mean(test_detection_accuracy_values)
        avg_test_regression_mse = np.mean(test_regression_mse_values)
        avg_test_regression_mae = np.mean(test_regression_mae_values)
        avg_test_infection_rate = np.mean(test_infection_rate_values)

        std_test_mse = np.std(test_mse_values)
        std_test_rmse = np.std(test_rmse_values)
        std_test_mae = np.std(test_mae_values)
        std_test_f1 = np.std(test_f1_values)
        std_test_pearson = np.std(test_pearson_values)
        std_test_spearman = np.std(test_spearman_values)
        std_test_detection_precision = np.std(test_detection_precision_values)
        std_test_detection_recall = np.std(test_detection_recall_values)
        std_test_detection_f1 = np.std(test_detection_f1_values)
        std_test_detection_accuracy = np.std(test_detection_accuracy_values)
        std_test_regression_mse = np.std(test_regression_mse_values)
        std_test_regression_mae = np.std(test_regression_mae_values)
        std_test_infection_rate = np.std(test_infection_rate_values)

        logger.info("===== Cross-Validation Results =====")
        logger.info(f"Average Test MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}")
        logger.info(f"Average Test RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}")
        logger.info(f"Average Test MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}")
        logger.info(f"Average Test F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}")
        logger.info(f"Average Test Pearson: {avg_test_pearson:.4f} ± {std_test_pearson:.4f}")
        logger.info(f"Average Test Spearman: {avg_test_spearman:.4f} ± {std_test_spearman:.4f}")
        logger.info(f"Average Test Detection Precision: {avg_test_detection_precision:.4f} ± {std_test_detection_precision:.4f}")
        logger.info(f"Average Test Detection Recall: {avg_test_detection_recall:.4f} ± {std_test_detection_recall:.4f}")
        logger.info(f"Average Test Detection F1: {avg_test_detection_f1:.4f} ± {std_test_detection_f1:.4f}")
        logger.info(f"Average Test Detection Accuracy: {avg_test_detection_accuracy:.4f} ± {std_test_detection_accuracy:.4f}")
        logger.info(f"Average Test Regression MSE: {avg_test_regression_mse:.4f} ± {std_test_regression_mse:.4f}")
        logger.info(f"Average Test Regression MAE: {avg_test_regression_mae:.4f} ± {std_test_regression_mae:.4f}")
        logger.info(f"Average Test Infection Rate: {avg_test_infection_rate:.4f} ± {std_test_infection_rate:.4f}")

        results_df = pd.DataFrame(all_fold_metrics)
        result_name = 'cross_validation_results_window' + str(args.window_size) + '_horizon' + str(
            args.pred_horizon) + '.csv'
        results_df.to_csv(os.path.join(args.model_dir, result_name), index=False)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = os.path.join(args.model_dir,
                                        f'{args.model_type}_w{args.window_size}_h{args.pred_horizon}_gamma{args.spectral_gamma}_k{args.spectral_k}_{timestamp}.log')

        with open(summary_filename, 'w') as log_file:
            log_file.write(f"=== {args.num_folds}-Fold Cross Validation Results for {args.model_type} ===\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Args: {vars(args)}\n\n")

            log_file.write("Results for each fold:\n")
            for fold_data in all_fold_metrics:
                fold = fold_data['fold']
                log_file.write(f"Fold {fold} - MSE: {fold_data['test_mse']:.4f}, RMSE: {fold_data['test_rmse']:.4f}, "
                               f"MAE: {fold_data['test_mae']:.4f}, F1: {fold_data['test_f1']:.4f}\n")
                log_file.write(f"Detection Precision: {fold_data['test_detection_precision']:.4f} ± {fold_data['test_detection_precision']:.4f}\n")
                log_file.write(f"Detection Recall: {fold_data['test_detection_recall']:.4f} ± {fold_data['test_detection_recall']:.4f}\n")
                log_file.write(f"Detection F1: {fold_data['test_detection_f1']:.4f} ± {fold_data['test_detection_f1']:.4f}\n")
                log_file.write(f"Detection Accuracy: {fold_data['test_detection_accuracy']:.4f} ± {fold_data['test_detection_accuracy']:.4f}\n")
                log_file.write(f"Regression MSE: {fold_data['test_regression_mse']:.4f} ± {fold_data['test_regression_mse']:.4f}\n")
                log_file.write(f"Regression MAE: {fold_data['test_regression_mae']:.4f} ± {fold_data['test_regression_mae']:.4f}\n")
                log_file.write(f"Infection Rate: {fold_data['test_infection_rate']:.4f} ± {fold_data['test_infection_rate']:.4f}\n")

            log_file.write("\nAverage Results across all folds:\n")
            log_file.write(f"MSE: {avg_test_mse:.4f} ± {std_test_mse:.4f}\n")
            log_file.write(f"RMSE: {avg_test_rmse:.4f} ± {std_test_rmse:.4f}\n")
            log_file.write(f"MAE: {avg_test_mae:.4f} ± {std_test_mae:.4f}\n")
            log_file.write(f"F1: {avg_test_f1:.4f} ± {std_test_f1:.4f}\n")
            log_file.write(f"Pearson Correlation: {avg_test_pearson:.4f} ± {std_test_pearson:.4f}\n")  # 打印皮尔逊相关系数
            log_file.write(f"Spearman Correlation: {avg_test_spearman:.4f} ± {std_test_spearman:.4f}\n")  # 打印斯皮尔曼相关系数
            log_file.write(f"Detection Precision: {avg_test_detection_precision:.4f} ± {std_test_detection_precision:.4f}\n")
            log_file.write(f"Detection Recall: {avg_test_detection_recall:.4f} ± {std_test_detection_recall:.4f}\n")
            log_file.write(f"Detection F1: {avg_test_detection_f1:.4f} ± {std_test_detection_f1:.4f}\n")
            log_file.write(f"Detection Accuracy: {avg_test_detection_accuracy:.4f} ± {std_test_detection_accuracy:.4f}\n")
            log_file.write(f"Regression MSE: {avg_test_regression_mse:.4f} ± {std_test_regression_mse:.4f}\n")
            log_file.write(f"Regression MAE: {avg_test_regression_mae:.4f} ± {std_test_regression_mae:.4f}\n")
            log_file.write(f"Infection Rate: {avg_test_infection_rate:.4f} ± {std_test_infection_rate:.4f}\n")

        logger.info(f"Cross-validation summary saved to {summary_filename}")

if __name__ == "__main__":
    main()