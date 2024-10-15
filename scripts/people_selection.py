'''Script to select people that we will ask biographies for.'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import pickle
from tqdm import tqdm
import pandas as pd
from wikidata.client import Client
import requests
import config

def entities_per_lang(wikidata_url, headers, languages, lang_lim):
    """
    Retrieves and processes Wikidata human entities based on specified languages and their respective query limits.

    Parameters:
    wikidata_url (str): Wikidata API endpoint URL
    headers (dict): Headers for the HTTP request
    languages (list): List of language codes for which the human entities will be fetched
    lang_lim (dict): Dictionary mapping each lang code to the max number of entities to fetch for that language

    Returns:
    entity_langs (dict): Dictionary mapping each Wikidata entity URL to the list of available languages for this entity
    all_page_titles (dict): Dictionary mapping each Wikidata entity URL to the dictionary of Wikipedia titles in all languages
    """
    all_page_titles = {}
    entity_langs = {}
    for lang in tqdm(languages):
        query = """
            SELECT ?entity """ + f"?sitelink_{lang} ?label_{lang}" + """ WHERE {
            ?entity wdt:P31 wd:Q5 . 
            """ + f"""
                OPTIONAL {{
                    ?sitelink_{lang} schema:about ?entity ;
                                    schema:isPartOf <https://{lang}.wikipedia.org/> .
                    OPTIONAL {{ ?entity rdfs:label ?label_{lang} . FILTER(LANG(?label_{lang}) = '{lang}') }}
                }}""" + """
                FILTER(""" + f"(BOUND(?label_{lang}))" + """)
            }
            LIMIT""" +f"{lang_lim[lang]}" #Q5 is wikidata category for human

        # Make the request to the Wikidata API
        response = requests.get(wikidata_url, params={'query': query}, headers=headers)
        response.raise_for_status()
        
        # Parse the response JSON
        data = response.json()

        # Update the count of each entity
        for result in data['results']['bindings']:
            entity = result['entity']['value']
            entity_langs[entity] = entity_langs.get(entity, []) + [lang]
            entity_sitelinks = {
                lang: result.get(f"sitelink_{lang}", {}).get('value')
            }
            page_titles = {
                key: entity_sitelinks[key].split('/')[-1]
                    for key in entity_sitelinks}
            if entity in all_page_titles:
                all_page_titles[entity].update(page_titles)
            else:
                all_page_titles[entity] = page_titles

    return entity_langs, all_page_titles

def filter_entities(languages, entity_langs, all_page_titles):
    '''Filter entities that are available in all languages'''
    final_entities = [entity_link for entity_link, langs in entity_langs.items() if len(langs) == len(languages)]
    all_page_titles = {entity: all_page_titles[entity] for entity in final_entities}
    return all_page_titles

def build_lang_df(entity_langs, lang_csv_file):
    """Builds a Pandas DataFrame indicating the presence of entities in specified languages and saves it to a CSV file.
    """
    lang_present = pd.DataFrame([[link] + [l in langs for l in languages] for link, langs in entity_langs.items()],
        columns=['name'] + languages)
    lang_present.to_csv(lang_csv_file, index=False)

def divide_batches(all_page_titles, pickle_file):
    """ Divides entities into 6 batches."""
    entities=list(all_page_titles.keys())
    batch_size = len(entities) // 6
    batches = [entities[i*batch_size:(i+1)*batch_size] for i in range(6)]
    with open(pickle_file, 'wb') as f:
        pickle.dump(batches, f)

def compute_views(pageview_url, all_page_titles):
    """Computes the count of page views for each entity and each language."""
    page_views_count = {lang: {} for lang in languages}

    # Retrieve page views for each entity
    for entity in tqdm(all_page_titles):
        for lang in languages:
            page_title = all_page_titles[entity][lang]
            pv_response = requests.get(
                pageview_url.format(project=f"{lang}.wikipedia", title=page_title, start="20230601", end="20240601"),
                headers=headers
            )
            pv_data = pv_response.json()
            if 'items' in pv_data:
                for item in pv_data['items']:
                    page_views_count[lang].setdefault(entity, 0)
                    page_views_count[lang][entity] += item['views']
    return page_views_count

def build_pageview_df(page_views_count, views_csv_file):
    '''Builds a Pandas DataFrame counting page views for each entity in every language.'''
    page_views_df=pd.DataFrame(page_views_count)
    page_views_df=page_views_df.reset_index().rename(columns={'index':'link_wiki'})
    page_views_df.insert(0, 'name', page_views_df.link_wiki.apply(lambda x : all_page_titles[x]['en']))
    page_views_df.fillna(0, inplace=True)
    page_views_df.to_csv(views_csv_file, index=False)

def retrieve_pty(client, entity, pty_label):
    '''Retrieves the field associatied with the pty label for the specified entity.'''
    try:
        prop = client.get(pty_label)
        if pty_label=='P569' or pty_label=='P570':
            return entity[prop]
        elif pty_label=='P1412':
            return str([str(lang.label) for lang in entity.getlist(prop)])
        else:
            return str(entity[prop].label)
    except:
        return None

def retrieve_people_pties(all_page_titles, analysis_csv_file):
    '''Retrieves citizenship and spoken language of the personalities from Wikipedia.'''
    pers_df=pd.DataFrame(columns=['code', 'name', 'instance', 'citizenship', 'language', 'birth_place', 'birth_date', 'death_date', 'gender', 'occupation'])
    for link in tqdm(all_page_titles.keys()):
        wiki_link=link.split('/')[-1]
        client = Client()
        entity = client.get(wiki_link, load=True)
        instance=retrieve_pty(client, entity, 'P31')
        citizenship=retrieve_pty(client, entity, 'P27')
        lang=retrieve_pty(client, entity, 'P1412')
        birth_place=retrieve_pty(client, entity,'P19')
        birth_date=retrieve_pty(client, entity, 'P569')
        death_date=retrieve_pty(client, entity, 'P570')
        gender=retrieve_pty(client, entity, 'P21')
        occupation=retrieve_pty(client, entity, 'P106')
        name=all_page_titles[link]['en']
        row=[wiki_link, name, instance, citizenship, lang, birth_place, birth_date, death_date, gender, occupation]
        pers_df.loc[len(pers_df)]=row
    pers_df.to_csv(analysis_csv_file, index=False)

if __name__ == '__main__':
    data_dir='./data/'
    languages=config.languages
    headers=config.headers
    wikidata_url=config.wikidata_url
    lang_lim=config.lang_lim
    pageview_url=config.pageview_url

    batches_file=data_dir+'batches.pkl'
    lang_csv_file=data_dir+'presence_lang.csv'
    views_csv_file=data_dir+'pageviews.csv'
    analysis_csv_file=data_dir+'raw_people_analysis.csv'

    #COMPUTING THE LIST OF ENTITIES
    entities_langs, all_page_titles=entities_per_lang(wikidata_url, headers, languages, lang_lim)
    all_page_titles=filter_entities(languages, entities_langs, all_page_titles)
    build_lang_df(languages, entities_langs, lang_csv_file)

    with open(data_dir+f'human_terms.json', 'w', encoding='utf-8') as f:
        json.dump(all_page_titles, f, ensure_ascii=False, indent=4)
    
    #DIVIDING IN BATCHES
    divide_batches(all_page_titles, batches_file)

    #RETRIEVING VIEW COUNTS
    page_views_count=compute_views(pageview_url, all_page_titles)
    build_pageview_df(page_views_count, views_csv_file)
    
    with open(data_dir+'page_views.json', 'w', encoding='utf-8') as f:
        json.dump(page_views_count, f, ensure_ascii=False, indent=4)
    
    #RETRIEVING PROPERTIES ABOUT OUR ENTITIES
    retrieve_people_pties(all_page_titles, analysis_csv_file)
    
