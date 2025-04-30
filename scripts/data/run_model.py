from transformers import AutoTokenizer, AutoModelForCausalLM
import re

model_map = {
 'llama7b': 'llama_2_7b_chat_hf',
 'llama3_8b': 'llama3-8B-Instruct-HF'
} 


class ModelPrompt:

    def __init__(self,
                model_name,
                model_dir = '/work/frink/models'):
        
        self.model = AutoModelForCausalLM.from_pretrained(f"{model_dir}/{model_map[model_name]}")
        self.model.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{model_map[model_name]}")
                                                       
        return 

    def init_prompt_template(self,
                             prompt_template):
        self.prompt_template_keys = re.findall(r'{(.*?)}', prompt_template)
        self.prompt_template = prompt_template
        print('PROMPT', self.prompt_template)
        return


    def init_instruction(self,
                         instruction):
        self.instruction = instruction
        return


    def generate(self,
                prompt_fillers):
        prompt = self.prompt_template.format(**prompt_fillers)
        # print(prompt)
        # print('**'* 13)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # self.
        gen_len = inputs['input_ids'].shape[-1] + 500
        generate_ids = self.model.generate(inputs.input_ids.to('cuda'), max_length=gen_len)
        generate_ids = generate_ids[0][inputs['input_ids'].shape[-1]:]
        # print(summary_ids)
        generated_text = self.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(generated_text)
        # print('***'*13)
        return generated_text