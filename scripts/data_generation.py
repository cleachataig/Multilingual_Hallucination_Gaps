'''Script to generate biographies with a specified model, 
for the list of people selected with the people_selection.py script.'''

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
import json
import argparse
import urllib
from tqdm import tqdm
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from accelerate.utils import gather_object

class Generator(object):
    """
    A class to handle text generation using a pre-trained model.

    Attributes:
        checkpoint (str): The path to the model checkpoint.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        model (AutoModelForCausalLM): The pre-trained model for causal language modeling.
    """
    def __init__(self, checkpoint, device=None):
        self.checkpoint = checkpoint
        self.load_model(device)

    def load_model(self, device):
        """Loads the model and tokenizer from the checkpoint."""
        print(f'Loading model from {self.checkpoint}')
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if device:
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype="auto")
            self.model.to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, torch_dtype="auto", device_map='auto')
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompts, device):
        """
        Generates responses for the given prompts using the pre-trained model.

        Args:
            prompts (list): A list of prompt strings to generate responses for.
            device (torch.device): The device to perform generation on.

        Returns:
            list: A list of generated text responses corresponding to the input prompts.
        """
        texts=[self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False) for prompt in prompts]
        inputs=self.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=500)
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

def fill_prompts(term_dict, languages, templates, english_prompting):
    """
    Fill prompt templates with entity name and language correctly 
    whether the prompts are in English or in the target language.
    """
    if not english_prompting:
        return [template.format(urllib.parse.unquote(term_dict[lang_code]).replace('_', ' '))
                for lang_code in languages.keys()
                for template in templates[lang_code]]
    else:
        term = urllib.parse.unquote(term_dict['en']).replace('_', ' ')
        return [template.format(term, language)
                for language in languages.values()
                for template in templates]

def process_results(generated_texts, language_codes):
    """
    Reorganize generated texts into a dictionnary.
    """
    reorganized_generations = [generated_texts[i:i+3] for i in range(0, len(generated_texts), 3)]
    ordered_gen = {language_codes[i]: reorganized_generations[i] for i in range(len(language_codes))}
    return ordered_gen

def bio_generation(templates, people, languages, model, results_file, english_prompting=False, distributed=False):
    """
    Generates biographies based on the input prompt templates and list of people, using the specified model, and saves the results to a file.
    
    Args:
    - templates (list of str or dict): List of templates for prompt generation or Dictionary of templates for each language code.
    - people (dict): Dictionary containing people data.
    - languages (dict): Dictionary mapping language codes to language names.
    - model: The model used for prompt generation.
    - results_file (str): Path to the results file.
    - english_prompting (bool): Whether the prompt is in English or in the target language `lang`.
    - distributed (bool): Whether to use distributed processing.
    """
    language_codes = list(languages.keys())
    if distributed:
        print("using distributed processes")
        distributed_state = PartialState()
        generator = Generator(model, distributed_state.device)
        # Split the keys of the people dictionary across different processes
        with distributed_state.split_between_processes(list(people.keys())) as people_keys:
            results = defaultdict(dict)
            for term_link in tqdm(people_keys):
                # Prepare prompts
                term_dict = people[term_link]
                prompts = fill_prompts(term_dict, languages, templates, english_prompting)

                #Generate text
                generated_texts = generator.generate(prompts, distributed_state.device)
                results[term_link] = process_results(generated_texts, language_codes)

                #Gather and save results
                distributed_state.wait_for_everyone()
                result_gathered = gather_object([results])
                if distributed_state.is_main_process:
                    merged_dict = defaultdict(dict)
                    for d in result_gathered:
                        for key, value in d.items():
                            merged_dict[key].update(value)
                    with open(results_file, 'w') as f:
                        json.dump(merged_dict, f)
    else:
        if os.path.exists(results_file):
            # Load existing results if file exists
            with open(results_file, 'r') as f:
                results=json.load(f)
        else:
            results = defaultdict(dict)
        generator = Generator(model)
        for term_link, term_dict in tqdm(people.items()):
            if term_link not in results.keys():
                # Prepare prompts
                prompts = fill_prompts(term_dict, languages, templates, english_prompting)
                #Generate text
                generated_texts = generator.generate(prompts, "cuda")
                results[term_link] = process_results(generated_texts, language_codes)
                #Save results
                with open(results_file, 'w') as f:
                    json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #Subject LLM:
    parser.add_argument('--model',
                        type=str,
                        default="llama_8",
                        choices=['llama_8', 'llama_70', 'qwen_7','qwen_72','aya_8', 'aya_35'])
    
    #Language of prompt (either english or original language):
    parser.add_argument('--prompt',
                        type=str,
                        default="english",
                        choices=['english', 'lang'])
    
    #To use process distribution or not:
    parser.add_argument('--distributed',
                        type=bool,
                        default=False)
    args = parser.parse_args()

    # Load configuration
    checkpoints_dict=config.checkpoints
    checkpoint=checkpoints_dict[args.model]
    languages=config.lang_names
    data_dir='./data/'
    results_dir='./result/generation/'

    # Data importation
    with open(data_dir+'human_terms.json', 'r') as f:
        people=json.load(f)
    
    if args.prompt=='english':
        results_file=results_dir+f'biographies_{args.model}_en.json'
        templates=config.en_templates

    elif args.prompt=='lang':
        results_file=results_dir+f'biographies_{args.model}_lang.json'
        with open(data_dir+'prompts_llama.json', 'r') as f:
            templates=json.load(f)
    
    # Biographies generation    
    bio_generation(templates, people, languages, checkpoint, results_file, args.prompt=='english', args.distributed)