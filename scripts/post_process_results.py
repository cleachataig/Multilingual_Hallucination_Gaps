'''Script to post-process factscores results, both for the validation experiment and final experiments.'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import json
import glob
import numpy as np
import pandas as pd
import config

def merge_validation(results_dir, models=['ChatGPT', 'InstructGPT', 'PerplexityAI']):
    """
    Merges validation scores computed with different LLM evaluators with the human annotated data provided in the FactScore repository.

    Args:
        results_dir (str): The directory where the validation score files are located.
        models (list): The list of model names used for text generation.
    """
    labels={}
    for m in models:
        # Load the base validation file for the model
        with open(results_dir+f"scores_validation_{m}.json", 'r') as f:
            all_lab=json.load(f)
        
        # Iterate over additional validation files (each validation file contains scores computed with a different LM evaluator)
        for i in range(2, 7):
            with open(results_dir+f"scores_validation{i}_{m}.json", 'r') as f:
                lab=json.load(f)
                for person, labels_dict in lab.items():
                    if i==2:
                        # Retrieve human annotations only for one file
                        all_lab[person]["human"]=all_lab[person].pop("human_labels") 
                        all_lab[person]['human']=['NS' if l=='IR' else l for l in all_lab[person]["human"]]
                        all_lab[person]["AlwaysNS"]=['NS']*len(labels_dict['human_labels'])
                    
                    # Identify the key corresponding to the LLM evaluator labels in the current file
                    keys = set(labels_dict.keys())
                    keys.remove('human_labels')
                    try:
                        # Extract the LLM evaluator key and create a new key
                        model_key=keys.pop()
                        new_model_key=model_key.split('_')[0]
                        if i==3:
                            new_model_key="Llama3+NP"

                        # Assign the labels to the corresponding new model key
                        all_lab[person][new_model_key]=labels_dict[model_key]
                    except:
                        continue

        # Store the merged labels for the current model
        labels[m]=all_lab
    
    # Save merged results to output file
    with open(results_dir+f"scores_validation.json", 'w') as f:
        json.dump(labels, f)

def merge_json_files(file_pattern, output_file, delete=False, move_dir=None):
    """
    Merges JSON files of different batches into a single output file. 

    Args:
        file_pattern (str): The pattern to match JSON files.
        output_file (str): The path where the merged JSON data will be saved.
        delete (bool, optional): If True, delete the original files after merging. 
        move_dir (str, optional): If provided, move the original files to this directory after merging.
    """
    # Find all JSON files that match the file pattern:
    json_files = glob.glob(file_pattern+'*.json')

    # Merge content of each JSON file found
    merged_results = {}
    for file in json_files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
                merged_results.update(data)
            except:
                continue 
        if move_dir:
            # Move the file to the specified directory
            os.rename(file, os.path.join(move_dir, os.path.basename(file)))
        elif delete:
            # Remove the file after processing
            os.remove(file)

    # Save merged results to output file
    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=4)

    print(f'Results {len(merged_results)}')
    print(f'Merged results have been saved to {output_file}')

def process_scores(results_dir, experiment, exp_setting, model):
    """
    Processes scores for a given experiment and model. If the result file doesn't exist,
    it merges JSON batches files to create a merged result file.

    Args:
        results_dir (str): The directory where the results are located.
        experiment (str): The name of the experiment (multi, trans or enprompt).
        setting (tuple): A tuple with experiment settings (e.g., use of translation and prompt language).
        model (str): The name of the subject model.

    Returns:
        list: A list of dictionaries, each representing a row of processed data.
    """
    # Load or create the merged result file
    result_file=results_dir+f'{experiment}_nliscores_{model}.json'
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data=json.load(f)
    else:
        file_pattern=res_dir+ f'{exp_setting[0]}nliscores_{model}_{exp_setting[1]}_'
        merge_json_files(file_pattern, result_file)
        with open(result_file, 'r') as f:
            data=json.load(f)

    # Iterate over each entity and language for this experiment and model
    rows=[]
    for link, lang_dict in data.items():
        for lang, res_dict in lang_dict.items():
            if res_dict:
                # Extract relevant scores and statistics from the result dictionary
                factscores=res_dict['scores']
                nb_facts=res_dict["num_facts_per_response"]
                new_row={'link':link, 'experiment': experiment, 'lang': lang, 'model': model, 
                        'fs_mean': np.mean(factscores), 'ent_std': np.std(ent_scores),
                        'nb_facts_mean': np.mean(nb_facts), 'nb_facts_std': np.std(nb_facts),
                        "init_score": res_dict['init_score'], 'respond_ratio':  res_dict["respond_ratio"]}
                rows.append(new_row)
    return rows

def compile_result_df(results_dir, experiments, models):
    """
    Compiles all processed scores into a DataFrame and saves it as a CSV file.

    Args:
        results_dir (str): The directory where the results are located.
        experiments (dict): A dictionary where keys are experiment names and values are experimental settings.
        models (list): A list of the model names used to generate data.
    """
    all_rows=[]
    for exp, exp_setting in experiments.items():
        for model in models:
            # Process the scores for each model and experiment
            all_rows.extend(process_scores(results_dir, exp, exp_setting, model))

    # Create a DataFrame from the processed rows and export to CSV
    result_df=pd.DataFrame(all_rows, columns=['link', 'experiment', 'lang', 'model', 'ent_mean', 'ent_std', 'con_mean', 'con_std', 'diff_mean', 'diff_std', ])#'nb_facts_mean', 'nb_facts_std', 'init_score', 'respond_ratio'])
    result_df.to_csv(results_dir+"all_nliscores.csv", index=False)
    print(f'Results DataFrame has been saved to {results_dir+"all_nliscores.csv"}')

if __name__ == "__main__":
    res_dir='./result/scores/'
    experiments=config.experiments
    models=list(config.checkpoints.keys())
    compile_result_df(res_dir, experiments, models)