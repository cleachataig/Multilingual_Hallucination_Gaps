'''Script to build our knowledge source from Wikipedia summaries'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import sqlite3
from tqdm import tqdm
import pandas as pd
import wikipediaapi
import urllib
import time
import re
import config

def pages_retrieval(summary_file, term_dict, languages, headers):
    '''Retrieving Wikipedia summaries for all the entities in `term_dict` in every language of the list `languages`.'''
    max_retries = 5
    with open(summary_file, "w", buffering=1) as fo:
        with tqdm(total=len(languages), desc="Languages") as lang_bar:
            for lang in languages:
                wiki_wiki=wikipediaapi.Wikipedia(
                            user_agent=headers['User-Agent'],
                            language=lang)
                for link, titles in term_dict.items():
                    success=False
                    for _ in range(max_retries):
                        try:
                            term=titles[lang]
                            term = urllib.parse.unquote(term)
                            page = wiki_wiki.page(term)
                            summary=page.summary.replace("\n", " ").replace('\t', ' ')
                            print(f"{link}\t{lang}\t{term}\t{summary}\t{page.length}", file=fo)
                            success = True
                            break 
                        except:
                            print("Error caught, retrying")
                            time.sleep(1)
                            wiki_wiki=wikipediaapi.Wikipedia(
                                user_agent=headers['User-Agent'],
                                language=lang)
                    if not success:
                        raise TimeoutError
                lang_bar.update(1)

def build_knowledge_source(term_dict, summary_file, knowledge_csv_file, database_file, views_csv_file=None):
    '''Building our DB file that will be used as knowledge source.'''
    wiki_df=pd.read_csv(summary_file, sep='\t', header=None)
    wiki_df.columns=['link_wiki', 'lang', 'name', 'text', 'length']
    wiki_df.text=wiki_df.text.astype(str)
    wiki_df['sent_nb']=wiki_df['text'].apply(lambda x: len(re.split(r'[.!?。｡।]', x)))
    wiki_df['length_sum']=wiki_df['text'].apply(lambda x: len(x))
    if views_csv_file:
        page_views_df=pd.read_csv(views_csv_file)
        page_views_melt=pd.melt(page_views_df, id_vars=['link_wiki'], value_vars=languages, var_name='lang', value_name='page_views')
        wiki_df=wiki_df.merge(page_views_melt, on=['link_wiki', 'lang']).sort_values('lang')
    
    #Cleaning entities with NaN pages
    for link in list(wiki_df.loc[wiki_df.text.isna(), 'link_wiki']):
        term_dict.pop(link)
    wiki_df=wiki_df.dropna(subset='text')

    #Exporting to CSV and DB
    wiki_df.to_csv(knowledge_csv_file, index=False)
    wiki_df['title'] = (wiki_df['name'] +' '+wiki_df['lang'])
    connection = sqlite3.connect(database_file)
    wiki_df[['title', 'text']].to_sql('documents', connection, if_exists='replace')

    #Return cleaned term_dict
    return term_dict

if __name__ == '__main__':
    data_dir='./data/'
    languages=config.languages
    headers=config.headers
    summary_file=data_dir+"wiki_summaries.txt"
    knowledge_csv_file=data_dir+"knowledge_source.csv"
    database_file=data_dir+'wiki_corpus_multi.db'

    with open(data_dir+'human_terms.json', 'r', encoding='utf-8') as f:
        all_page_titles=json.load(f)
    
    #pages_retrieval(summary_file, all_page_titles, languages, headers)
    all_page_titles=build_knowledge_source(all_page_titles, summary_file, knowledge_csv_file, database_file)

    with open(data_dir+'human_terms.json', 'w', encoding='utf-8') as f:
        json.dump(all_page_titles, f)