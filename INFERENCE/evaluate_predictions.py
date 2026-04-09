import pandas as pd
import numpy as np
import os

def evaluate():
    # Load data
    results_file = 'INFERENCE/INFERENCE RESULTS/FUNDUS INFERENCE RESULTS V2/inference_results.csv'
    split_file = 'DATASET/filtered_data_split.csv'
    
    print("Loading csv files...")
    results_df = pd.read_csv(results_file)
    split_df = pd.read_csv(split_file)

    # keep only test split
    test_df = split_df[split_df['split'] == 'test'].copy()

    # extract basenames for joining
    results_df['basename'] = results_df['image'].apply(lambda x: os.path.basename(x))
    test_df['basename'] = test_df['image'].apply(lambda x: os.path.basename(x))

    # join the two dfs
    merged_df = pd.merge(test_df, results_df, on='basename', how='inner')

    # columns of interest for prediction. Let's find all _prob columns
    prob_cols = [c for c in results_df.columns if c.endswith('_prob')]
    classes = [c.replace('_prob', '') for c in prob_cols]

    total_evaluated = len(merged_df)
    
    exact_match_count = 0
    any_correct_count = 0
    argmax_correct_count = 0
    predictions_over_60 = 0

    for index, row in merged_df.iterrows():
        # Ground truth classes for this image (where column value is 1)
        # Note: some classes might not exist in split_df columns, we check if they are in columns
        gt_classes = set(c for c in classes if c in merged_df.columns and row[c] == 1)
        
        # Probabilities
        probs = {c: row[c + '_prob'] for c in classes}
        
        # Predicted classes with probability > 0.60
        pred_classes_60 = set(c for c, p in probs.items() if p > 0.60)
        
        if len(pred_classes_60) > 0:
            predictions_over_60 += 1
            
        # Top prediction
        best_class = max(probs, key=probs.get)
        best_prob = probs[best_class]
        
        # 1. Exact match: Predicted classes (>0.60) exactly match ground truth
        if pred_classes_60 == gt_classes and len(gt_classes) > 0:
            exact_match_count += 1
            
        # 2. Any correct: At least one predicted class (>0.60) is correct
        if len(pred_classes_60.intersection(gt_classes)) > 0:
            any_correct_count += 1
            
        # 3. Argmax correct: Highest probability class is > 0.60 and is correct
        if best_prob > 0.60 and best_class in gt_classes:
            argmax_correct_count += 1

    if total_evaluated > 0:
        print(f"Total test images from filtered_data_split.csv found in inference_results.csv: {total_evaluated}")
        print(f"Images with at least one prediction > 60% confidence: {predictions_over_60} ({(predictions_over_60/total_evaluated)*100:.2f}%)")
        print(f"Correct predictions (Argmax > 60% matches ground truth): {argmax_correct_count} ({(argmax_correct_count/total_evaluated)*100:.2f}%)")
        print(f"Correct predictions (At least one class > 60% matches ground truth): {any_correct_count} ({(any_correct_count/total_evaluated)*100:.2f}%)")
        print(f"Correct predictions (Classes > 60% EXACTLY match ground truth): {exact_match_count} ({(exact_match_count/total_evaluated)*100:.2f}%)")
    else:
        print("No test images found to evaluate.")

if __name__ == '__main__':
    evaluate()
