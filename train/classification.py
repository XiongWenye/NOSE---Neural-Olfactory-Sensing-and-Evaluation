import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from utils.util_alignment import set_seeds, grand_average, average_over_subject, post_process_results_df
from utils.prepare_datasets import prepare_dataset, select_features
from constants import *
from utils.visualization_helper import post_process_dataframe

warnings.filterwarnings('ignore')

parent_dir = "/public/home/CS182/xiongwy2023-cs182/MoLFormer_N2024"
sys.path.append(parent_dir)


# base_path = '/local_storage/datasets/farzaneh/alignment_olfaction_datasets/'
base_path = 'datasets/'

seed = 2024
set_seeds(seed)
times = 30
n_components = 20

def pipeline_classification(X_train, y_train, X_test, seed, n_components=None):
    """Pipeline for classification with dimensionality reduction."""
    steps = []
    
    if n_components is not None:
        steps.append(('pca', PCA(n_components=n_components, random_state=seed)))
    
    steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', LogisticRegression(random_state=seed, max_iter=1000)))
    
    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    
    var = None
    if n_components is not None:
        var = pipe.named_steps['pca'].explained_variance_ratio_.sum()
    
    return pipe, X_test, var

def metrics_per_descriptor(X_test, y_test, model):
    """Calculate classification metrics for each descriptor."""
    predictions = model.predict(X_test)
    f1_scores = []
    accuracies = []
    precisions = []
    recalls = []
    
    for i in range(y_test.shape[1]):
        f1 = f1_score(y_test[:, i], predictions[:, i], average='weighted')
        acc = accuracy_score(y_test[:, i], predictions[:, i])
        prec = precision_score(y_test[:, i], predictions[:, i], average='weighted', zero_division=0)
        rec = recall_score(y_test[:, i], predictions[:, i], average='weighted', zero_division=0)
        
        f1_scores.append(f1)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
    
    return predictions, f1_scores, accuracies, precisions, recalls

def train_and_eval_classification(data_groupbyCID, times, n_components=None):
    f1_scores_cv = []
    accuracy_cv = []
    precision_cv = []
    recall_cv = []
    predicteds = []
    y_tests = []
    runs = []
    CIDs = []
    
    X = np.asarray(data_groupbyCID.embeddings.values.tolist())
    y = np.asarray(data_groupbyCID.y.values.tolist())
    
    for i in range(times):
        X_train, X_test, y_train, y_test, CID_train, CID_test = train_test_split(
            X, y, data_groupbyCID.CID, test_size=0.2, random_state=seed+i
        ) 
        
        model, X_test, var = pipeline_classification(X_train, y_train, X_test, seed, n_components=n_components)
        
        predicted, f1s, accs, precs, recs = metrics_per_descriptor(X_test, y_test, model)
        
        f1_scores_cv.append(f1s)
        accuracy_cv.append(accs)
        precision_cv.append(precs)
        recall_cv.append(recs)
        
        predicteds.extend(predicted)
        y_tests.extend(y_test)
        runs.extend([i] * len(y_test))
        CIDs.extend(CID_test)
        
    return CIDs, predicteds, y_tests, runs, f1_scores_cv, accuracy_cv, precision_cv, recall_cv

def post_process_classification_results(metrics_list):
    """Process classification metrics from cross-validation."""
    metrics_array = np.array(metrics_list)
    metrics_mean = np.mean(metrics_array, axis=0)
    metrics_std = np.std(metrics_array, axis=0)
    
    return metrics_mean, metrics_std

def pipeline_classification_model(model_name, input_file, times=30, n_components=None, ds="keller"):
    """Run classification pipeline for a specific model."""
    df = pd.read_csv(input_file)
    df = prepare_dataset(df, 'embeddings', 'y')
    df_groupbyCID = grand_average(df, ds)
    
    CIDs, predicteds, y_tests, runs, f1_scores_cv, accuracy_cv, precision_cv, recall_cv = train_and_eval_classification(
        df_groupbyCID, times=times, n_components=n_components
    )
    
    f1_mean, f1_std = post_process_classification_results(f1_scores_cv)
    acc_mean, acc_std = post_process_classification_results(accuracy_cv)
    prec_mean, prec_std = post_process_classification_results(precision_cv)
    rec_mean, rec_std = post_process_classification_results(recall_cv)
    
    # Create dataframes for metrics
    tasks = keller_tasks if ds.startswith('keller') else sagar_tasks
    
    df_f1 = pd.DataFrame(f1_mean.reshape(1, -1), columns=tasks)
    df_f1['model'] = model_name
    
    df_acc = pd.DataFrame(acc_mean.reshape(1, -1), columns=tasks)
    df_acc['model'] = model_name
    
    df_prec = pd.DataFrame(prec_mean.reshape(1, -1), columns=tasks)
    df_prec['model'] = model_name
    
    df_rec = pd.DataFrame(rec_mean.reshape(1, -1), columns=tasks)
    df_rec['model'] = model_name
    
    # Predictions dataframe
    tasks_length = len(tasks)
    df_predictions = pd.DataFrame(np.concatenate([
        np.asarray(CIDs).reshape(-1, 1),
        np.asarray(predicteds),
        np.asarray(y_tests),
        np.asarray(runs).reshape(-1, 1)
    ], axis=1))
    
    df_predictions['model'] = model_name
    df_predictions.columns = ['CID'] + [f'{i}_predicted' for i in range(tasks_length)] + \
                             [f'{i}_true' for i in range(tasks_length)] + ['run'] + ['model']
    
    return df_predictions, df_f1, df_acc, df_prec, df_rec

def compare_models_classification(times, n_components, input_file_molformer, input_file_pom, input_file_molformerfinetuned, ds="keller"):
    """Compare F1 scores of OpenPom, MolFormer, and Fine-tuned MolFormer."""
    print("Processing OpenPom model...")
    _, df_pom_f1, df_pom_acc, df_pom_prec, df_pom_rec = pipeline_classification_model(
        'openpom', input_file_pom, times=times, n_components=None, ds=ds
    )
    
    # Lists to store results for different MolFormer layers
    molformer_f1_list = []
    molformer_acc_list = []
    molformer_prec_list = []
    molformer_rec_list = []
    
    finetuned_f1_list = []
    finetuned_acc_list = []
    finetuned_prec_list = []
    finetuned_rec_list = []
    
    # Process MolFormer for different layers
    for layer in range(13):  # 0-12 layers
        print(f"Processing MolFormer layer {layer}...")
        layer_file = f"{input_file_molformer}{layer}.csv"
        
        _, layer_f1, layer_acc, layer_prec, layer_rec = pipeline_classification_model(
            f'molformer_layer{layer}', layer_file, times=times, n_components=n_components, ds=ds
        )
        
        layer_f1['layer'] = layer
        layer_acc['layer'] = layer
        layer_prec['layer'] = layer
        layer_rec['layer'] = layer
        
        molformer_f1_list.append(layer_f1)
        molformer_acc_list.append(layer_acc)
        molformer_prec_list.append(layer_prec)
        molformer_rec_list.append(layer_rec)
    
    # Process Fine-tuned MolFormer for different layers
    for layer in range(13):  # 0-12 layers
        print(f"Processing Fine-tuned MolFormer layer {layer}...")
        layer_file = f"{input_file_molformerfinetuned}{layer}.csv"
        
        _, layer_f1, layer_acc, layer_prec, layer_rec = pipeline_classification_model(
            f'molformer_finetuned_layer{layer}', layer_file, times=times, n_components=n_components, ds=ds
        )
        
        layer_f1['layer'] = layer
        layer_acc['layer'] = layer
        layer_prec['layer'] = layer
        layer_rec['layer'] = layer
        
        finetuned_f1_list.append(layer_f1)
        finetuned_acc_list.append(layer_acc)
        finetuned_prec_list.append(layer_prec)
        finetuned_rec_list.append(layer_rec)
    
    return (molformer_f1_list, molformer_acc_list, molformer_prec_list, molformer_rec_list,
            df_pom_f1, df_pom_acc, df_pom_prec, df_pom_rec,
            finetuned_f1_list, finetuned_acc_list, finetuned_prec_list, finetuned_rec_list)

def post_process_to_csv(metrics_list, tasks, title):
    """Process metrics from a list of dataframes into a single dataframe."""
    metrics_list[0]["layer"] = 0
    all_metrics = metrics_list[0]
    
    for i in range(1, 13):
        metrics_list[i]["layer"] = i
        all_metrics = pd.concat([all_metrics, metrics_list[i]])
    
    del all_metrics['model']
    all_metrics.columns = tasks + ["layer"]    
    all_metrics['model'] = title
    
    return all_metrics

def save_classification_results(ds, molformer_f1, molformer_acc, molformer_prec, molformer_rec,
                              pom_f1, pom_acc, pom_prec, pom_rec,
                              finetuned_f1, finetuned_acc, finetuned_prec, finetuned_rec):
    """Save classification results to CSV files."""
    if ds == "keller":
        tasks = keller_tasks
    else:
        tasks = sagar_tasks
    
    # Save OpenPom results
    pom_f1.columns = tasks + ["model"]
    pom_f1.to_csv(f'df_{ds}_f1_pom.csv', index=False)
    
    pom_acc.columns = tasks + ["model"]
    pom_acc.to_csv(f'df_{ds}_acc_pom.csv', index=False)
    
    pom_prec.columns = tasks + ["model"]
    pom_prec.to_csv(f'df_{ds}_prec_pom.csv', index=False)
    
    pom_rec.columns = tasks + ["model"]
    pom_rec.to_csv(f'df_{ds}_rec_pom.csv', index=False)
    
    # Process and save MolFormer results
    molformer_f1_df = post_process_to_csv(molformer_f1, tasks, "molformer")
    molformer_f1_df.to_csv(f'df_{ds}_f1_molformer.csv', index=False)
    
    molformer_acc_df = post_process_to_csv(molformer_acc, tasks, "molformer")
    molformer_acc_df.to_csv(f'df_{ds}_acc_molformer.csv', index=False)
    
    molformer_prec_df = post_process_to_csv(molformer_prec, tasks, "molformer")
    molformer_prec_df.to_csv(f'df_{ds}_prec_molformer.csv', index=False)
    
    molformer_rec_df = post_process_to_csv(molformer_rec, tasks, "molformer")
    molformer_rec_df.to_csv(f'df_{ds}_rec_molformer.csv', index=False)
    
    # Process and save Fine-tuned MolFormer results
    finetuned_f1_df = post_process_to_csv(finetuned_f1, tasks, "molformerfinetuned")
    finetuned_f1_df.to_csv(f'df_{ds}_f1_molformerfinetuned.csv', index=False)
    
    finetuned_acc_df = post_process_to_csv(finetuned_acc, tasks, "molformerfinetuned")
    finetuned_acc_df.to_csv(f'df_{ds}_acc_molformerfinetuned.csv', index=False)
    
    finetuned_prec_df = post_process_to_csv(finetuned_prec, tasks, "molformerfinetuned")
    finetuned_prec_df.to_csv(f'df_{ds}_prec_molformerfinetuned.csv', index=False)
    
    finetuned_rec_df = post_process_to_csv(finetuned_rec, tasks, "molformerfinetuned")
    finetuned_rec_df.to_csv(f'df_{ds}_rec_molformerfinetuned.csv', index=False)

def visualize_classification_results(ds, metric='f1'):
    """Visualize classification results."""
    import matplotlib.pyplot as plt
    
    results_path = 'dfs_result/classification/'
    
    # Load results based on metric
    if metric == 'f1':
        molformer_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_f1_molformer.csv")
        finetuned_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_f1_molformerfinetuned.csv")
        pom_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_f1_pom.csv")
        title = f"{ds.capitalize()} F1 Scores Comparison"
    elif metric == 'acc':
        molformer_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_acc_molformer.csv")
        finetuned_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_acc_molformerfinetuned.csv")
        pom_metrics = pd.read_csv(f"{base_path}{results_path}df_{ds}_acc_pom.csv")
        title = f"{ds.capitalize()} Accuracy Comparison"
    
    tasks = keller_tasks if ds.startswith('keller') else sagar_tasks
    
    # Placeholder for visualization (similar to regression visualization)
    trend_data = post_process_dataframe(
        molformer_metrics, None, 
        finetuned_metrics, None,
        pom_metrics, None, 
        None, None,
        tasks, f"figs/{ds}_classification_{metric}", 
        width=None, linewidth=0
    )

# Main execution
if __name__ == "__main__":
    # Define input files
    input_file_keller_pom = base_path + 'embeddings/pom/keller_pom_embeddings_.csv'
    input_file_keller_molformer = base_path + 'embeddings/molformer/keller_molformer_embeddings_'
    input_file_keller_molformerfinetuned = base_path + 'embeddings/molformerfinetuned/keller_molformerfinetuned_embeddings_'
    
    # Compare models
    molformer_f1, molformer_acc, molformer_prec, molformer_rec, \
    pom_f1, pom_acc, pom_prec, pom_rec, \
    finetuned_f1, finetuned_acc, finetuned_prec, finetuned_rec = compare_models_classification(
        times, n_components, input_file_keller_molformer, input_file_keller_pom, 
        input_file_keller_molformerfinetuned, ds="keller"
    )
    
    # Save results
    save_classification_results(
        "keller", 
        molformer_f1, molformer_acc, molformer_prec, molformer_rec,
        pom_f1, pom_acc, pom_prec, pom_rec,
        finetuned_f1, finetuned_acc, finetuned_prec, finetuned_rec
    )
    
    # Visualize results
    visualize_classification_results("keller", metric="f1")
    visualize_classification_results("keller", metric="acc")