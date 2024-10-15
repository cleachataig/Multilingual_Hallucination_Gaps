# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from custom_factscore.lm import LM
from custom_factscore.few_shot_examples import q1, r1, q2, r2, q3, r3, q4, r4, q5, r5

class CLM(LM):
    def __init__(self, model_dir, cache_dir, device, cache_file=None):
        if not cache_file:
            cache_file=os.path.join(self.cache_dir, "Model.pkl")
        super().__init__(cache_file)
        self.model_dir = model_dir
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir=cache_dir
        self.save_interval = 100
        self.load_model()
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("True")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("False")

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        #self.tokenizer.pad_token = self.tokenizer.eos_token

    # for generating atomic facts
    def generate(self, q, max_new_tokens=256,
                  do_sample=True, temperature=0.7, top_p=0.95):
        
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        self.add_n += 1
            
        messages_to_pass = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": r1},
        {"role": "user", "content": q2},
        {"role": "assistant", "content": r2},
        {"role": "user", "content": q3},
        {"role": "assistant", "content": r3},
        {"role": "user", "content": q4},
        {"role": "assistant", "content": r4},
        {"role": "user", "content": q5},
        {"role": "assistant", "content": r5},
        {"role": "user", "content": q}
        ]
        input_ids = self.tokenizer.apply_chat_template(messages_to_pass, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id,
                                             max_new_tokens=max_new_tokens, do_sample=do_sample, 
                                             temperature=temperature, top_p=top_p)
        decoded = self.tokenizer.batch_decode(generated_ids)
        final_ans = decoded[0]
        final_ans = re.split(r"\[/INST\]", final_ans)[-1]
        final_ans = final_ans.replace("</s>", "")
        return final_ans
    
    # for fact checking generate only one token and its logit
    def _generate(self, prompt, max_new_tokens=1):
        messages = [{"role": "user", "content": prompt}]
        tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt",
                                                    add_generation_prompt=True)
        tokens = tokens.to(self.device)
        out = self.model.generate(tokens, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens, 
                     output_scores=True, return_dict_in_generate=True)
        #gen_tokens = out["sequences"]
        #return self.tokenizer.decode(gen_tokens[0])

        return out["scores"][0].detach().cpu().numpy()