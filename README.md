# Multilingual Hallucination Gaps in Large Language Models

This code is related with the paper *Multilingual Hallucination Gaps in Large Language Models*. 

## Description

Large language models (LLMs) are increasingly used as alternatives to traditional search engines given their capacity to generate text that resembles human language. However, this shift is concerning, as LLMs often generate hallucinations—misleading or false information that appears highly credible.
In this study, we explore the phenomenon of hallucinations across multiple languages in free-form text generation, focusing on what we call \textit{multilingual hallucination gaps}. These gaps reflect differences in the frequency of hallucinated answers depending on the prompt and language used. To quantify such hallucinations, we used the \fs metric and extended its framework to a multilingual setting. We conducted experiments using LLMs from the LLaMA, Qwen, and Aya families, generating biographies in 19 languages and comparing the results to Wikipedia pages. Our results reveal variations in hallucination rates, especially between high- and low-resource languages, raising important questions about LLM multilingual performance and the challenges in evaluating hallucinations in multilingual free-form text generation.

## Environment Setup

The project was built using Python 3.10. A list of required packages can be found in the file `requirements.txt`. To translate content, we use the OpenAI API which requires an API key. A HuggingFace token might also be needed to access some of the LLM repositories. Note that Git LFS is needed to download the result ZIP file in the repository.

## Directory Structure 
    .
    ├── custom_factscore        # FActScore, adapted to Mistral model
    ├── data                    # Knowledge source data and list of selected personalities
    ├── notebooks               # Notebooks to analyze results
    ├── result                  # Generation and score results files
    ├── scripts                 # Scripts to reproduce experiments
    └── README.md

## Data generation

To generate biographies, run the following command:

```python scripts/data_generation.py --model "llama_8" --prompt 'lang' ```

or for distributed processing:

```accelerate launch scripts/data_generation.py --model "llama_8" --prompt 'lang' --distributed True```

Generations can be analyzed in the notebook `data_generation.ipynb`, which also includes sanity checks.

## FactScore evaluation

After generating the biographies, compute the FactScore with the following command:

```python scripts/factscore_evaluation.py --model "llama_8" --prompt 'lang' --translated '' --batch_number 0```

Process the results with
```python scripts/post_process_results.py```

This will compile the results into a DataFrame, which is necessary for further analysis in the notebook `factscore_analysis.ipynb`.

## Additional content

Scripts for selecting people, retrieving Wikipedia pages, and translating generations are available in `people_selection.py`, `knowledge_source.py` and `translation.py` respectively.Data analysis of the selected people can be reproduced with the notebook `people_data_analysis.ipynb`.
