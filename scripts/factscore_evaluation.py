'''Script to perform the LLM evaluator experiment as well as
the FactScore experiments once the biographies have been generated.'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import os
import argparse
import pickle
import urllib
import json
from tqdm import tqdm
import numpy as np
from custom_factscore.factscorer import FactScorer
#from factscore.factscorer import FactScorer
from collections import defaultdict
import config

class CustomJSONizer(json.JSONEncoder):
    """ A custom JSON encoder for handling specific data types during JSON serialization."""
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)

def validation(people, atomic_facts, results_dict, model_dir, model_name, results_file, model_labels):
    """
    Validation experiment to evaluate the LLM evaluator.
    
    Args:
        people (list of str): List of names of individuals to compute FactScores about.
        atomic_facts (list of dict): List of atomic facts, each associated with the corresponding person.
        results_dict (dict): Dictionary to store results of the validation.
        model_dir (str): Directory where the model is stored.
        model_name (str): Name of the method to use for scoring.
        results_file (str): Path to the file where results are stored.
        model_labels (dict): Dictionary mapping model names to their respective labels.
    """
    fs = FactScorer(model_dir=model_dir,
                    model_name=model_name
                    ) # Initialize the fact scorer
    try:
        label = model_labels[model_name]
    except KeyError:
        raise ValueError("Model is incorrectly entered. Check the model_name in model_labels.")
    
    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            old_results=json.load(f)
    else:
        old_results=results_dict

    for i, name in tqdm(enumerate(people), total=len(people)):
        if label in old_results[name]:
            # If results for the person and label exist, reuse them
            results_dict[name][label]=old_results[name][label]
        else:
            # Compute factscores with atomic facts if results are not present
            out = fs.get_score([name], [""], atomic_facts=[atomic_facts[i]])
            results_dict[name][label]=["S" if fact['is_supported'] else "NS" for fact in out["decisions"][0]]

            # Update the results file
            with open(results_file, 'w') as f:
                json.dump(results, f)

def evaluation(term_dict, generations_dict, results_file, knowledge_dir, model_dir, english_wiki=False, 
               model_name="retrieval+mistral", knowledge_source="wiki_corpus_multi"):
    """
    Evaluates generated biographies against a knowledge source and computes fact scores in a results file.
    
    Args:
        term_dict (dict): Dictionary mapping links to entities names in all languages.
        generations_dict (dict): Dictionary mapping links to generated texts for different languages.
        results_file (str): Path to the file where fact scores are stored.
        knowledge_dir (str): Directory containing knowledge data.
        model_dir (str): Directory containing the model.
        english_wiki (bool): Whether to use English Wikipedia as the knowledge source.
        model_name (str): Name of the model to use for scoring.
        knowledge_source (str): Name of the knowledge source to use.
    """
    if english_wiki:
        print("Using Wikipedia in English")

    fs = FactScorer(data_dir=knowledge_dir,
                model_dir=model_dir,
                model_name=model_name
                )
    fs.register_knowledge_source(knowledge_source)

    # Load existing fact scores if the results file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            fact_scores=json.load(f)
    else:
        fact_scores={}
    
    # Iterate over each link in the generations dictionary
    for link in tqdm(list(generations_dict.keys())):
        lang_gen=generations_dict[link]
        if link not in fact_scores:
            fact_scores[link]={}
        
        if english_wiki:
            # Compute topic name in english
            term = urllib.parse.unquote(term_dict[link]['en'])
            topic=f"{term} en"

        for lang, generation in lang_gen.items():
            if len(generation)>0:
                if lang not in fact_scores[link]:
                    if not english_wiki:
                        # Compute topic name in target language
                        term = urllib.parse.unquote(term_dict[link][lang])
                        topic=f"{term} {lang}"
                    try:
                        #Compute FactScore
                        out = fs.get_score([topic]*len(generation), generation, knowledge_source=knowledge_source)
                    except IndexError:
                        print(f"IndexError for {link} in {lang}")
                        out={}
                    except AttributeError:
                        print(f"AttributeError for {link} in {lang}")
                        out={}
                    fact_scores[link][lang]=out
                    
                #Save results
                with open(results_file, 'w') as f:
                    json.dump(fact_scores, f, cls=CustomJSONizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #Subject LLM:
    parser.add_argument('--model',
                        type=str,
                        default="llama_8",
                        choices=['llama_8', 'llama_70', 'qwen_7','qwen_72','aya_8', 'aya_35', "InstructGPT", "ChatGPT", "PerplexityAI"])
    
    #Language of prompt (either english or original language):
    parser.add_argument('--prompt',
                        type=str,
                        default="en",
                        choices=['en', 'lang'])
    
    #To use the translated generations or not (and the Wikipedia in English):
    parser.add_argument('--translated',
                        type=str,
                        default="",
                        choices=['', 'trans_'])
    
    #Batch number (we divided the set of people into 6 batches)
    parser.add_argument('--batch_number',
                        type=int,
                        default=0)
    
    #To perform the validation experiment or the other experiments:
    parser.add_argument('--validation',
                        type=str,
                        default=None,
                        choices=['True', None])
    
    #Method to compute FactScore:
    parser.add_argument('--evaluator',
                        type=str,
                        default=None,
                        choices=[None, "retrieval+mistral", "retrieval+mistral+npm", "npm"])
    
    args = parser.parse_args()

    # Directories path
    data_dir='./data/'
    results_dir='./result/'

    if args.validation: # Running the validation experiment 
        # Mapping evaluator to labels names
        model_labels = {
            "retrieval+mistral": "Mistral_Labels",
            "retrieval+mistral+npm": "Mistral+NP_Labels",
            "npm": "NP_Labels"
        }

        # Data importation: we load human annotations and atomic facts
        topics=[]
        facts=[]
        results=defaultdict(dict)
    
        with open(f"./.cache/factscore/labeled/{args.model}.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                generation=json.loads(line)
                topic=generation['topic']
                atomic_facts=[]
                human_labels=[]
                if generation['annotations']:
                    for sentence in generation['annotations']:
                        if sentence['human-atomic-facts']:
                            for fact in sentence['human-atomic-facts']:
                                atomic_facts.append(fact['text'])
                                human_labels.append(fact['label'])
                    if len(atomic_facts)>0:
                        topics.append(topic)
                        facts.append(atomic_facts)
                        results[topic]['human_labels']=human_labels
        
        #Computing factscores for these atomic facts with the specified evaluator
        results=validation(topics, facts, results, 
                        model_dir="~/scratch/huggingface_models/hub/", 
                        model_name=args.evaluator, 
                        results_file=results_dir+f"scores/scores_validation6_{args.model}.json")
        
    else: # Running the evaluation experiments (multi, trans or enprompt)
        print(f"Evaluating the model {args.model} with prompt {args.prompt} for batch {args.batch_number}")
        
        # Configuration and file names
        batches_file=data_dir+'batches.pkl'
        results_file=results_dir+f'scores/{args.translated}scores_{args.model}_{args.prompt}_{args.batch_number}.json'
        languages=config.languages
        
        # Data importation
        with open(data_dir+'human_terms.json', 'r') as f:
            people=json.load(f)
        
        with open(batches_file, 'rb') as f:
            batches = pickle.load(f)

        with open(results_dir+f"generation/{args.translated}biographies_{args.model}_{args.prompt}.json", 'r') as f:
            generations=json.load(f)
        
        generations={k: generations[k] for k in generations.keys() if k in batches[args.batch_number]}

        # Evaluating FactScores
        evaluation(people, generations, 
                            results_file,
                            knowledge_dir=data_dir, 
                            model_dir="mistralai/Mistral-7B-Instruct-v0.3",
                            english_wiki=(args.translated=='trans_')
                            )