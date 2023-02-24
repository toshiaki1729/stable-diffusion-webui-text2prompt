from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import scripts.t2p.settings as settings

if settings.DEVELOP:
    import scripts.t2p.prompt_generator as pgen
    import scripts.t2p.prompt_generator.database_loader as dloader
else:
    from scripts.t2p.dynamic_import import dynamic_import
    pgen = dynamic_import('scripts/t2p/prompt_generator/__init__.py')
    dloader = dynamic_import('scripts/t2p/prompt_generator/database_loader.py')

NUM_CHOICE = 10
K_VALUE = 50

# brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class WDLike(pgen.PromptGenerator):
    def __init__(self):
        self.clear()
        self.database_loader = dloader.DatabaseLoader(settings.DATABASE_PATH_DANBOORU, settings.RE_TOKENFILE_DANBOORU)

    def get_model_names(self):
        return sorted(self.database_loader.datas.keys())

    def clear(self):
        self.database_loader = None
        self.database = None
        self.tags = None
        self.tokens = None
        self.tokenizer = None
        self.model = None
        self.loaded_model_name = None
    
    def load_data(self, database_name: str):
        print(f'[text2prompt] Loading database with name "{database_name}"...')
        self.database = self.database_loader.load(database_name)
        print('[text2prompt] Database loaded')
    
    def load_model(self):
        if self.database is None:
            print('[text2prompt] Cannot load model; Database is not loaded.')
            return
        from modules.devices import device
        # brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
        # Load model from HuggingFace Hub
        if self.loaded_model_name and self.loaded_model_name == self.database.model_name:
            return
        else:
            print(f'[text2prompt] Loading model with name "{self.database.model_name}"...')
            self.tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER_MODELS[self.database.model_name])
            self.model = AutoModel.from_pretrained(settings.TOKENIZER_MODELS[self.database.model_name]).to(device)
            self.loaded_model_name = self.database.model_name
            print('[text2prompt] Model loaded')

    def unload_model(self):
        if self.tokenizer is not None:
            del self.tokenizer
        if self.model is not None:
            del self.model

    def ready(self) -> bool:
        return self.database is not None \
            and self.database.loaded() \
            and self.model is not None \
            and self.tokenizer is not None \
            and self.loaded_model_name and self.loaded_model_name == self.database.model_name

    def __call__(self, text: str, text_neg: str, neg_weight: float, opts: pgen.GenerationSettings) -> List[str]:
        if not self.ready(): return ''

        i = max(0, min(opts.tag_range, len(self.database.tag_idx) - 1))
        r = self.database.tag_idx[i][1]
        self.tokens = self.database.data[:r, :]
        self.tags = self.database.tags[:r]
        
        from modules.devices import device
        
        # --------------------------------------------------------------------------------------------------------------------------------
        # brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
        # Tokenize sentences
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        if text_neg:
            encoded_input_neg = self.tokenizer(text_neg, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            if text_neg:
                model_output_neg = self.model(**encoded_input_neg)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        if text_neg:
            sentence_embeddings -= neg_weight*mean_pooling(model_output_neg, encoded_input_neg['attention_mask'])

        # --------------------------------------------------------------------------------------------------------------------------------

        # Get cosine similarity between given text and tag descriptions
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        tag_tokens_dev = torch.from_numpy(self.tokens).type(torch.float32).to(device)
        similarity: torch.Tensor = cos(sentence_embeddings[0], tag_tokens_dev)

        # Convert similarity into probablity
        if opts.conversion is pgen.ProbabilityConversion.CUTOFF_AND_POWER:
            probs_cpu = torch.clamp(similarity.detach().cpu(), 0, 1) ** opts.prob_power
        elif opts.conversion is pgen.ProbabilityConversion.SOFTMAX:
            probs_cpu = torch.softmax(similarity.detach().cpu(), dim=0)

        probs_cpu = probs_cpu / probs_cpu.sum(dim=0)
        probs_cpu = probs_cpu.nan_to_num()

        results = None

        if opts.sampling is pgen.SamplingMethod.NONE:
            tags_np = np.array(self.tags)
            opts.n = min(tags_np.shape[0], opts.n)
            if opts.n <= 0: return []
            if opts.weighted:
                probs_np = probs_cpu.detach().numpy()
                num_nonzero = np.count_nonzero(probs_np)
                if num_nonzero <= opts.n:
                    if num_nonzero > 0:
                        results=np.random.choice(tags_np, num_nonzero, replace=False, p=probs_np)
                    else:
                        results = np.random.choice(tags_np, opts.n, replace=False)
                else:
                    results = np.random.choice(a=tags_np, size=opts.n, replace=False, p=probs_np)
            else:
                # Just sample randomly
                results = np.random.choice(a=tags_np, size=opts.n, replace=False)

        elif opts.sampling is pgen.SamplingMethod.TOP_K:
            probs, indices = probs_cpu.topk(opts.k)
            indices = indices.detach().numpy().tolist()
            if len(indices) <= 0: return []

            tags_np = np.array([self.tags[i] for i in indices])
            opts.n = min(tags_np.shape[0], opts.n)
            if opts.weighted:
                probs_np = probs.detach().numpy()
                probs_np /= np.sum(probs_np)
                probs_np = np.nan_to_num(probs_np)
                num_nonzero = np.count_nonzero(probs_np)
                if num_nonzero <= opts.n:
                    if num_nonzero > 0:
                        results=np.random.choice(tags_np, num_nonzero, replace=False, p=probs_np)
                    else:
                        results = np.random.choice(tags_np, opts.n, replace=False)
                else:
                    results = np.random.choice(tags_np, opts.n, replace=False, p=probs_np)
            else:
                results = np.random.choice(tags_np, opts.n, replace=False)
        
        # brought from https://nn.labml.ai/sampling/nucleus.html
        elif opts.sampling is pgen.SamplingMethod.TOP_P:
            sorted_probs, sorted_indices = probs_cpu.sort(descending=True)
            cs_probs = torch.cumsum(sorted_probs, dim=0)
            nucleus = cs_probs < opts.p
            nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]])
            sorted_indices[~nucleus] = -1

            indices_np = sorted_indices.detach().numpy()
            indices = [i for i in indices_np if i >= 0]
            if len(indices) <= 0: return []

            tags_np = np.array([self.tags[i] for i in indices])
            opts.n = min(tags_np.shape[0], opts.n)
            if opts.weighted:
                probs_np = np.array([sorted_probs[i] for i in indices])
                probs_np /= np.sum(probs_np)
                probs_np = np.nan_to_num(probs_np)
                num_nonzero = np.count_nonzero(probs_np)
                if num_nonzero <= opts.n:
                    if num_nonzero > 0:
                        results=np.random.choice(tags_np, num_nonzero, replace=False, p=probs_np)
                    else:
                        results = np.random.choice(tags_np, opts.n, replace=False)
                else:
                    results = np.random.choice(tags_np, opts.n, replace=False, p=probs_np)
            else:
                results = np.random.choice(tags_np, opts.n, replace=False)

        return [] if results is None else results.tolist()