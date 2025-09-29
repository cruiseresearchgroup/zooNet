import torch
import numpy as np
import pandas as pd
import os
import time
import re
from scipy.stats import spearmanr
from Model_metrics import ModelEvaluator


def extract_time_from_filename(filename):
    """
    Extract year and week number from filename in format: hetero_graph_{region}_{year}_week{num}.pt
    
    Args:
        filename: string in format hetero_graph_{region}_{year}_week{num}.pt
        
    Returns:
        tuple: (year, week_num) or (None, None) if parsing fails
    """
    pattern = r'hetero_graph_.*_(\d{4})_week(\d+)\.pt'
    match = re.search(pattern, filename)
    if match:
        year = int(match.group(1))
        week_num = int(match.group(2))
        return year, week_num
    return None, None


def extract_simple_time_info(temporal_info, batch_idx, time_step, window_size=4):
    """
    Simplified time extraction with fallback to basic indexing.
    
    Args:
        temporal_info: temporal information for the batch
        batch_idx: current batch index
        time_step: current time step (prediction step: 0, 1, 2, 3...)
        window_size: size of input window (default 4)
        
    Returns:
        string: formatted time information representing the prediction target time
    """
    if temporal_info is None or batch_idx >= len(temporal_info):
        return f"batch_{batch_idx}_step_{time_step}"
    
    try:
        batch_temporal = temporal_info[batch_idx]
        
        # Case 1: 5-tuple format (most common new format)
        if isinstance(batch_temporal, tuple) and len(batch_temporal) >= 5:
            year, week = batch_temporal[2], batch_temporal[3]
            return format_real_time(year, week, time_step, window_size)
        
        # Case 2: Filename string
        elif isinstance(batch_temporal, str):
            year, week_num = extract_time_from_filename(batch_temporal)
            return format_real_time(year, week_num, time_step, window_size)
        
        # Case 3: Object with filename attribute
        elif hasattr(batch_temporal, 'filename'):
            year, week_num = extract_time_from_filename(batch_temporal.filename)
            return format_real_time(year, week_num, time_step, window_size)
        
        # Case 4: Any other format - use simple indexing
        else:
            return f"batch_{batch_idx}_step_{time_step}_format_unknown"
            
    except Exception:
        # Fallback for any parsing errors
        return f"batch_{batch_idx}_step_{time_step}_error"


def format_real_time(year, week_num, time_step_offset=0, window_size=4):
    """
    Format real time as year-week string for prediction targets.
    
    Args:
        year: year (int)
        week_num: week number of the first input week (int)
        time_step_offset: prediction step offset (0, 1, 2, 3...)
        window_size: size of input window (default 4)
        
    Returns:
        string: formatted as "YYYY-WXX" (e.g., "2023-W01")
    """
    if year is None or week_num is None:
        return "unknown"
    
    # Calculate actual prediction week: first_input_week + window_size + prediction_step
    actual_week = week_num + window_size + time_step_offset
    
    # Handle year overflow (assuming 52 weeks per year)
    if actual_week > 52:
        year += (actual_week - 1) // 52
        actual_week = ((actual_week - 1) % 52) + 1
    
    return f"{year}-W{actual_week:02d}"


def save_predictions_to_csv(forecast_denorm, groundtruth_denorm, save_dir, dataset_name, mode='test', temporal_info=None, window_size=4):
    """
    Save denormalized predictions and ground truth to CSV file.
    
    Args:
        forecast_denorm: numpy array of shape (batch_size, time_steps, num_counties, num_channels)
        groundtruth_denorm: numpy array of shape (batch_size, time_steps, num_counties, num_channels)
        save_dir: directory to save the CSV file
        dataset_name: name of the dataset ('avian' or 'japan')
        mode: mode name ('test', 'train', 'val')
        temporal_info: list of temporal information for each batch, containing real time data
        window_size: size of input window (default 4)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size, time_steps, num_counties, num_channels = forecast_denorm.shape
    
    data_rows = []
    
    for batch_idx in range(batch_size):
        for time_step in range(time_steps):
            for county_idx in range(num_counties):
                for channel_idx in range(num_channels):
                    # Extract values
                    pred_value = forecast_denorm[batch_idx, time_step, county_idx, channel_idx]
                    true_value = groundtruth_denorm[batch_idx, time_step, county_idx, channel_idx]
                    
                    # Simplified time processing
                    real_time = extract_simple_time_info(temporal_info, batch_idx, time_step, window_size)
                    
                    # Create data row
                    data_row = {
                        'sample_id': batch_idx,
                        'time_step': time_step,
                        'real_time': real_time,
                        'county_id': county_idx,
                        'channel_id': channel_idx,
                        'predicted_value': float(pred_value),
                        'true_value': float(true_value),
                        'absolute_error': float(abs(pred_value - true_value)),
                        'squared_error': float((pred_value - true_value) ** 2)
                    }
                    data_rows.append(data_row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Generate filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_{mode}_predictions_{timestamp}.csv"
    filepath = os.path.join(save_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Predictions saved to: {filepath}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Also save a summary CSV with aggregated statistics
    summary_data = []
    for batch_idx in range(batch_size):
        for time_step in range(time_steps):
            # Get data for this sample and time step (all counties, first channel)
            pred_slice = forecast_denorm[batch_idx, time_step, :, 0]  # Focus on first channel
            true_slice = groundtruth_denorm[batch_idx, time_step, :, 0]
            
            # Extract real time information for summary (simplified)
            real_time = extract_simple_time_info(temporal_info, batch_idx, time_step, window_size)
            
            # Calculate statistics
            mse = np.mean((pred_slice - true_slice) ** 2)
            mae = np.mean(np.abs(pred_slice - true_slice))
            
            summary_row = {
                'sample_id': batch_idx,
                'time_step': time_step,
                'real_time': real_time,
                'mse': float(mse),
                'mae': float(mae),
                'mean_predicted': float(np.mean(pred_slice)),
                'mean_true': float(np.mean(true_slice)),
                'std_predicted': float(np.std(pred_slice)),
                'std_true': float(np.std(true_slice))
            }
            summary_data.append(summary_row)
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{dataset_name}_{mode}_predictions_summary_{timestamp}.csv"
    summary_filepath = os.path.join(save_dir, summary_filename)
    summary_df.to_csv(summary_filepath, index=False)
    print(f"Prediction summary saved to: {summary_filepath}")
    
    return filepath, summary_filepath


def save_metrics_to_csv(multistep_metrics, avg_metrics, args, mode='test'):
    """
    Save multistep and average metrics to CSV file following the same pattern as Model_metrics.py
    
    Args:
        multistep_metrics: list of dictionaries containing metrics for each step
        avg_metrics: dictionary containing average metrics
        args: argument namespace containing model_dir, dataset, model_type etc.
        mode: mode name ('test', 'train', 'val')
    """
    # Create file path following the same pattern as Model_metrics.py
    file_path = args.model_dir + f'/{args.dataset}-{getattr(args, "model_type", "FusionGNN")}-{mode}-metrics.csv'
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a') as cf:
        print(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
        cf.write(' '.join(['*' * 10, f'Evaluation on {mode} set started at', time.ctime(), '*' * 10]))
        cf.write(f'*****, Evaluation starts, {mode}, {time.ctime()}, ***** \n')
        
        # Write parameters
        for param in vars(args).keys():
            cf.write(f'{param}: {getattr(args, param)},')
        cf.write('\n')
        
        # Write metrics in the same format as Model_metrics.py
        if multistep_metrics and len(multistep_metrics) > 0:
            col_names = [' '] + list(multistep_metrics[0].keys())
            cf.write(','.join(col_names) + '\n')
            
            # Write metrics for each step
            for step in range(len(multistep_metrics)):
                row_items = [f'Step {step}'] + [str(value) for value in multistep_metrics[step].values()]
                cf.write(','.join(row_items) + '\n')
            
            # Write horizon average (avg_metrics)
            if avg_metrics:
                # Calculate horizon averages for each metric
                horizon_avg = []
                for measure in multistep_metrics[0].keys():
                    if measure in avg_metrics:
                        horizon_avg.append(str(avg_metrics[measure]))
                    else:
                        # Calculate average if not provided
                        step_measures = [multistep_metrics[step][measure] for step in range(len(multistep_metrics))]
                        horizon_avg.append(str(np.mean(step_measures)))
                
                row_items = [f'Horizon avg.'] + horizon_avg
                cf.write(','.join(row_items) + '\n')
        
        cf.write(f'*****, Evaluation ends, {mode}, {time.ctime()}, ***** \n \n')
        print(' '.join(['*' * 10, f'Evaluation on {mode} set ended at', time.ctime(), '*' * 10]))
    
    print(f"Metrics saved to: {file_path}")
    return file_path


def calculate_metrics(args, model, data_loader, device, dataset, pred_horizon=4, mode='avian', set='test', 
                     save_predictions=True, save_dir=None):
    """
    Calculates various performance metrics for the model's predictions.
    
    Args:
        args: argument namespace
        model: trained model
        data_loader: data loader
        device: computation device
        dataset: dataset object
        pred_horizon: prediction horizon
        mode: dataset mode ('avian' or 'japan')
        set: dataset split ('train', 'test', 'val')
        save_predictions: whether to save predictions to CSV
        save_dir: directory to save predictions (if None, uses args.model_dir)
    """
    model.eval()
    
    with torch.no_grad():
        model_evaluator = ModelEvaluator(args)
        forecast, groundtruth, temporal_data = [], [], []
        for batch in data_loader:
            batched_graphs, batched_targets, _, batched_temporal_info = batch
            
            device_graphs = []
            for time_graphs in batched_graphs:
                device_graphs.append([g.to(device) for g in time_graphs])
                
            device_targets = []
            for time_targets in batched_targets:
                device_targets.append([t.to(device) for t in time_targets])
                
            input_temporals, _ = batched_temporal_info
            device_input_temporals = []
            for seq_temporal in input_temporals:
                device_input_temporals.append(
                    [(y.to(device), w.to(device)) for y, w, _, _, _ in seq_temporal]
                )
            
            # Collect temporal information for each sample in the batch
            # We'll extract the base temporal info from the first input time step
            batch_temporal_info = []
            for seq_temporal in input_temporals:
                if len(seq_temporal) > 0:
                    # Get the first time step's temporal info as the base
                    first_temporal = seq_temporal[0]
                    batch_temporal_info.append(first_temporal)
                else:
                    batch_temporal_info.append(None)
            temporal_data.append(batch_temporal_info)
            
            batch_outputs = model(device_graphs, device_input_temporals)
            y_preds = torch.stack(batch_outputs, dim=0)
            
            reshaped_targets = []
            batch_size = len(device_targets[0])
            num_time_steps = len(device_targets)
            for batch_idx in range(batch_size):
                batch_targets_over_time = []
                for time_idx in range(num_time_steps):
                    batch_targets_over_time.append(device_targets[time_idx][batch_idx])
                
                batch_tensor = torch.stack(batch_targets_over_time, dim=0) 
                reshaped_targets.append(batch_tensor)
            
            targets = torch.stack(reshaped_targets, dim=0) 

            # Add a feature dimension if it's missing (shape: [B, T, N] -> [B, T, N, 1])
            if y_preds.dim() == 3:
                y_preds = y_preds.unsqueeze(-1)
            if targets.dim() == 3:
                targets = targets.unsqueeze(-1)

            forecast_denorm = dataset.channel_wise_denormalize(y_preds.cpu().detach().numpy(), size=0)
            groundtruth_denorm = dataset.channel_wise_denormalize(targets.cpu().detach().numpy(), size=0)

            # The denormalized tensors are torch tensors, convert them to numpy
            if isinstance(forecast_denorm, torch.Tensor):
                forecast_denorm = forecast_denorm.cpu().detach().numpy()
            if isinstance(groundtruth_denorm, torch.Tensor):
                groundtruth_denorm = groundtruth_denorm.cpu().detach().numpy()

            forecast.append(forecast_denorm)
            groundtruth.append(groundtruth_denorm)
                        
        forecast = np.concatenate(forecast, axis=0)
        groundtruth = np.concatenate(groundtruth, axis=0)
        
        # Flatten temporal data to match the forecast/groundtruth structure
        flattened_temporal_data = []
        for batch_temporal in temporal_data:
            flattened_temporal_data.extend(batch_temporal)
        
        if save_predictions:
            if save_dir is None:
                save_dir = os.path.join(args.model_dir, 'predictions')
            
            # Get window_size from args or dataset, default to 4
            window_size = getattr(args, 'window_size', getattr(dataset, 'window_size', 4))
            
            prediction_file, summary_file = save_predictions_to_csv(
                forecast, groundtruth, save_dir, mode, set, temporal_info=flattened_temporal_data, window_size=window_size
            )
            print(f"Successfully saved predictions for {set} set")
        
        # Use enhanced evaluation if post-processing is enabled
        if hasattr(args, 'use_post_processing') and args.use_post_processing:
            detection_threshold = getattr(args, 'detection_threshold', 0.3)
            
            # Calculate step-wise enhanced metrics (like standard evaluation)
            multistep_metrics = []
            step_enhanced_metrics = []
            
            for step in range(args.pred_horizon):
                print(f'Enhanced evaluation step {step}:')
                
                # Get data for this specific step
                forecast_step = forecast[:, step, ...]  # [batch_size, num_counties, num_channels]
                groundtruth_step = groundtruth[:, step, ...]
                
                # Flatten arrays for this step only
                forecast_step_flat = forecast_step.reshape(-1)
                groundtruth_step_flat = groundtruth_step.reshape(-1)
                
                # Apply threshold strategy (simple post-processing)
                processed_predictions_step = forecast_step_flat.copy()
                processed_predictions_step[processed_predictions_step < detection_threshold] = 0.0
                processed_predictions_step = np.maximum(processed_predictions_step, 0.0)
                
                # Calculate detection metrics for this step
                pred_binary = (forecast_step_flat > detection_threshold).astype(int)
                true_binary = (groundtruth_step_flat > detection_threshold).astype(int)
                
                tp = np.sum((pred_binary == 1) & (true_binary == 1))
                tn = np.sum((pred_binary == 0) & (true_binary == 0))
                fp = np.sum((pred_binary == 1) & (true_binary == 0))
                fn = np.sum((pred_binary == 0) & (true_binary == 1))
                
                step_detection_precision = tp / (tp + fp + 1e-10)
                step_detection_recall = tp / (tp + fn + 1e-10)
                step_detection_f1 = 2 * step_detection_precision * step_detection_recall / (step_detection_precision + step_detection_recall + 1e-10)
                step_detection_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
                
                # Calculate regression metrics for infection samples in this step
                infection_mask_step = groundtruth_step_flat > detection_threshold
                if infection_mask_step.sum() > 0:
                    step_regression_mse = np.mean((processed_predictions_step[infection_mask_step] - groundtruth_step_flat[infection_mask_step]) ** 2)
                    step_regression_mae = np.mean(np.abs(processed_predictions_step[infection_mask_step] - groundtruth_step_flat[infection_mask_step]))
                else:
                    step_regression_mse = 0.0
                    step_regression_mae = 0.0
                
                # Calculate overall metrics for this step
                step_overall_mse = np.mean((processed_predictions_step - groundtruth_step_flat) ** 2)
                step_overall_mae = np.mean(np.abs(processed_predictions_step - groundtruth_step_flat))
                step_overall_rmse = np.sqrt(step_overall_mse)
                
                # Pearson correlation for this step
                step_correlation = np.corrcoef(processed_predictions_step, groundtruth_step_flat)[0, 1]
                if np.isnan(step_correlation):
                    step_correlation = 0.0
                
                # Spearman correlation for this step
                step_spearman, _ = spearmanr(processed_predictions_step, groundtruth_step_flat)
                if np.isnan(step_spearman):
                    step_spearman = 0.0
                
                # Store step metrics compatible with existing format
                step_metrics = {
                    'mse': step_overall_mse,
                    'rmse': step_overall_rmse,
                    'mae': step_overall_mae,
                    'mape': 0.0,  # Keep as 0 for consistency
                    'f1': step_detection_f1,
                    'precision': step_detection_precision,
                    'recall': step_detection_recall,
                    'pearson': step_correlation,
                    'spearman': step_spearman
                }
                multistep_metrics.append(step_metrics)
                
                # Store enhanced metrics for this step
                enhanced_step_metrics = {
                    'detection_precision': step_detection_precision,
                    'detection_recall': step_detection_recall,
                    'detection_f1': step_detection_f1,
                    'detection_accuracy': step_detection_accuracy,
                    'regression_mse': step_regression_mse,
                    'regression_mae': step_regression_mae,
                    'infection_rate': infection_mask_step.mean()
                }
                step_enhanced_metrics.append(enhanced_step_metrics)
                
                print(f'Step {step} - Detection F1: {step_detection_f1:.3f}, Precision: {step_detection_precision:.3f}, Recall: {step_detection_recall:.3f}, MSE: {step_overall_mse:.3f}, Pearson: {step_correlation:.3f}, Spearman: {step_spearman:.3f}')
            
            # Calculate average metrics across all steps
            avg_metrics = {}
            for measure in multistep_metrics[0].keys():
                step_measures = [multistep_metrics[step][measure] for step in range(len(multistep_metrics))]
                avg_metrics[measure] = np.mean(step_measures)
            
            # Add enhanced average metrics
            for measure in step_enhanced_metrics[0].keys():
                step_measures = [step_enhanced_metrics[step][measure] for step in range(len(step_enhanced_metrics))]
                avg_metrics[measure] = np.mean(step_measures)
            
            print(f"Enhanced evaluation for {mode}_{set} - Avg Detection F1: {avg_metrics['detection_f1']:.3f}, Avg Precision: {avg_metrics['precision']:.3f}, Avg Recall: {avg_metrics['recall']:.3f}, Avg MSE: {avg_metrics['mse']:.3f}, Avg Spearman: {avg_metrics['spearman']:.3f}")
            
            # Save metrics to CSV file following Model_metrics.py pattern
            metrics_file_path = save_metrics_to_csv(multistep_metrics, avg_metrics, args, set)
            print(f"Successfully saved enhanced step-wise metrics for {set} set to: {metrics_file_path}")
            
            return multistep_metrics, avg_metrics
        
        multistep_metrics, avg_metrics = model_evaluator.evaluate_numeric(
            y_pred=forecast, 
            y_true=groundtruth, 
            mode=set
        )
        
        # try:
        #     metrics_file_path = save_metrics_to_csv(multistep_metrics, avg_metrics, args, set)
        #     print(f"Successfully saved standard metrics for {set} set to: {metrics_file_path}")
        # except Exception as e:
        #     print(f"Warning: Failed to save standard metrics to CSV: {e}")
        
        return multistep_metrics, avg_metrics