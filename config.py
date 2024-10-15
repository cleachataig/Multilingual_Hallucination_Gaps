# Languages list
languages = ['en', 'ja', 'zh', 'es', 'fr', 'pl', 'vi', 'tr', 'fa', 'ko', 'ar', 'hu', 'th', 'hi', 'bn', 'ms', 'ta', 'sw', 'jv']
nli_languages=['en', 'zh', 'es', 'fr', 'vi', 'ar', 'th', 'hi', 'sw']

lang_names = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "pl": "Polish",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "fa": "Persian",
    "ko": "Korean",
    "ar":"Arabic", 
    "hu": "Hungarian",
    "th": "Thai",
    "hi": "Hindi",
    "bn": "Bengali",
    "ms": "Malay",
    "ta": "Tamil",
    "sw": "Swahili",
    "jv": "Javanese",
}

lang_cat = {
    "en": "Very-High",
    "ja": "High",
    "zh": "High",
    "es": "High",
    "fr": "High",
    "pl": "High",
    "vi": "Medium",
    "tr": "Medium",
    "fa": "Medium",
    "ko": "Medium",
    "ar":"Medium", 
    "hu": "Medium",
    "th": "Medium",
    "hi": "Medium",
    "bn": "Low",
    "ms": "Low",
    "ta": "Low",
    "sw": "Low",
    "jv": "Low",
}


# Languages limits
lang_lim={'en':30000, 'fr':30000, 
    'pl':20000, 'ja':20000, 'zh':20000, 'es':20000, 'vi':20000, 'fa':20000, 'ar':20000,  
    'tr':10000, 'ko':10000, 'hu': 10000, 
    'ms':6000,
    'th':5000, 'hi':5000, 'bn':5000, 'ta':5000, 
    'sw':4000, 'jv':4000}

# Request headers
headers = {
    'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)',
    "Accept": "application/json"
}

# Wikidata API endpoint
wikidata_url = 'https://query.wikidata.org/sparql'

# Pageview API endpoint
pageview_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{title}/monthly/{start}/{end}'

#Checkpoints
checkpoints={
    'llama_8':"/network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct",
    'llama_70': "/network/weights/llama.var/llama_3/Meta-Llama-3-70B-Instruct",
    'qwen_7':"Qwen/Qwen2-7B-Instruct",
    'qwen_72': "Qwen/Qwen2-72B-Instruct",
    "aya_8": "CohereForAI/aya-23-8B",
    "aya_35": "CohereForAI/aya-23-35B"
}

#Prompt templates
templates=["Tell me a biography of {}", 
           "Give me a biography of {}",
           "Please give me a biography of {}"]

en_templates=["Tell me a biography of {} in {}. Do not translate.", 
             "Give me a biography of {} in {}. Do not translate.", 
             "Please give me a biography of {} in {}. Do not translate."]

#Experiments
experiments={
    'multi': ('', 'lang'),
    'trans': ('trans_', 'lang'),
    'enprompt': ('trans_', 'en'),
    'enpromptwiki': ('', 'en')
}