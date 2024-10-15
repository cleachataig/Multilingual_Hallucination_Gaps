'''Script to translate prompts and generations with GPT 4.0.'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from openai import OpenAI
import config
import json
import tqdm
import os
import pickle
import argparse
from collections import defaultdict

def translate_template(template, language):
    '''Translates prompt template in target language.'''
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Translate the following text to {language}."},
            {"role": "user", "content": template}
        ]
    )
    translation = response.choices[0].message.content
    return translation

def create_batch_file(json_file, generations, lang_names, result_file=None, model="gpt-4o-mini"):
    '''Creates batch file with the generations.'''
    system_prompt = """You are given a text in {} and your job is to translate it into English.
    Do not make up any information, change anything, only translate the text. Do not add anything else. Only output the translation."""
    count=0
    if result_file:
        with open(result_file, "r") as f:
            results=json.load(f)
    with open(json_file, 'w') as f:
        for term_link, lang_generations in generations.items():
            entity_code=term_link.split('/')[-1]
            if term_link not in results.keys():
                if count<150:
                    for lang in lang_generations.keys():
                        if lang !='en':
                            language=lang_names[lang]
                            generations=lang_generations[lang]
                            for i, gen in enumerate(generations):
                                request={"custom_id": entity_code+'_'+lang+'_'+str(i), 
                                        "method": "POST", 
                                        "url": "/v1/chat/completions", 
                                        "body": {"model": model,
                                                "seed":42,
                                                "messages": [{"role": "system", "content": system_prompt.format(language)},
                                                            {"role": "user", "content": f"{gen}"}]}}
                                f.write(json.dumps(request) + "\n")
                    count+=1
                
def send_batch_gpt(batch_file, batch_info_file):
    '''Sends batch to GPT.'''
    client = OpenAI()
    batch_input_file = client.files.create(
    file=open(batch_file, "rb"),
    purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    info=client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "nightly eval job"
        }
    )
    print(info)
    info_dic={'batch_id':info.id}

    #Saving batch id to retrieve results and information
    with open(batch_info_file, 'w') as f:
        json.dump(info_dic, f)
    return info

def batch_info(batch_id):
    '''Prints batch information.'''
    client = OpenAI()
    #client.batches.cancel(batch_id)
    info=client.batches.retrieve(batch_id)
    print(info)
    return info

def get_batch_results(file_id, results_file, new_results_file, generations):
    '''Retrieves translations and post-processes it.'''
    if os.path.exists(results_file):
        os.remove(results_file)
    client = OpenAI()
    content = client.files.content(file_id)
    content_bytes = content.read()
    with open(results_file, "ab") as f:
        f.write(content_bytes)

    results=defaultdict(dict)
    with open(results_file) as f:
        for line in f:
            request=json.loads(line)
            request_id=request["custom_id"]
            entity_id, lang, i=request_id.split('_')
            link='http://www.wikidata.org/entity/'+entity_id
            if 'en' not in results[link]:
                results[link]['en']=generations[link]['en']
            if lang not in results[link]:
                results[link][lang]=[]
            results[link][lang].append(request['response']['body']['choices'][0]['message']['content'])
            
    with open(new_results_file, "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default="llama_8",
                        choices=['llama_8', 'llama_70', 'qwen_7','qwen_72','aya_8', 'aya_35'])
    parser.add_argument('--prompt',
                        type=str,
                        default="en",
                        choices=['en', 'lang'])
    parser.add_argument('--batch_number',
                        type=int,
                        default=0)
    parser.add_argument('--information',
                        type=str,
                        default="True",
                        choices=['True', 'False'])
    
    args = parser.parse_args()
    
    #Configuration
    data_dir='./data/'
    results_dir='./result/generation/'
    languages=config.languages
    lang_names=config.lang_names
    templates=config.templates
    batches_file=data_dir+'batches.pkl'

    """
    ## Prompts translation
    translations = {}
    for lang_code, language in lang_names.items():
        if lang_code=='en':
            translations[lang_code]=[template+ f" in {language}." for template in templates]
        else:
            translations[lang_code] = [translate_template(template + f" in {language}.", language) for template in templates]

    with open(data_dir+'prompts.json', 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=4)"""
    
    ## Biographies translation

    #Files names
    json_file=results_dir+f"translation/trans_batch_{args.model}_{args.prompt}_{args.batch_number}.jsonl"
    batch_file=results_dir+f"translation/info_trans_batch_{args.model}_{args.prompt}_{args.batch_number}.json"
    result_file=results_dir+f"translation/trans_biographies_{args.model}_{args.prompt}_{args.batch_number}.jsonl"
    result_file2=results_dir+f"translation/trans_biographies_{args.model}_{args.prompt}_{args.batch_number}.json"

    #Data importation
    with open(batches_file, 'rb') as f:
        batches = pickle.load(f)

    with open(results_dir+f"clean_biographies_{args.model}_{args.prompt}.json", 'r') as f:
        generations=json.load(f)
    
    generations={k: generations[k] for k in generations.keys() if k in batches[args.batch_number]}

    #To reduce cost we use batches with OpenAI API
    if args.information=='True':
        print("Giving batch information")
        if os.path.exists(batch_file):
            with open(batch_file, 'r') as f:
                batch_id=json.load(f)
            batch_id=batch_id['batch_id']
            info=batch_info(batch_id)
            if info.status=='completed':
                get_batch_results(info.output_file_id, result_file, result_file2, generations)
            elif info.status=='failed':
                print("Creating batch")
                create_batch_file(json_file, generations, lang_names)
                info=send_batch_gpt(json_file, batch_file)
        else:
            raise FileNotFoundError("The batch has not been created!")
    elif args.information=="False":
        print("Creating batch")
        create_batch_file(json_file, generations, lang_names)
        info=send_batch_gpt(json_file, batch_file)