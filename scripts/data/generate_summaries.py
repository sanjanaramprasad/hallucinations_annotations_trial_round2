import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from run_model import *
import math 
from tqdm import tqdm




def run_model(df,
              prompt_template,
             instruction,
             model_name):
    model_class = ModelPrompt(model_name = 'llama3_8b')
    model_class.init_prompt_template(prompt_template)

    
    df_docs = df.drop_duplicates(subset=['source'], keep='last')
    model_name = 'llama3_8b'
    added_summaries = []
    seen_docids = []
    
    for idx, row in tqdm(df_docs.iterrows(), total=df_docs.shape[0]):
        docid = row['id']
        init_dict = {k :v for k , v in row.items()}
        if docid not in seen_docids:
            seen_docids.append(docid)
            prompt_values = {
                'instruction': instruction,
                'source': row['source']
                }
            summary = model_class.generate(prompt_values)
            summary = summary.strip()
            init_dict['summary'] = summary
            init_dict['model'] = model_name
            init_dict['error_spans'] = float('nan')
            init_dict['error_category'] = float('nan')
            init_dict['faithful'] = float('nan')
            init_dict['corrected_summary'] = float('nan')
            init_dict['id'] = str(uuid.uuid4())
            added_summaries.append(init_dict)
    return pd.DataFrame(added_summaries)

if __name__ == '__main__:
    