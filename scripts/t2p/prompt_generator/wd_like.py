from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from . import PromptGenerator, GenerationSettings, SamplingMethod, ProbabilityConversion
from .. import settings


NUM_CHOICE = 10
K_VALUE = 50

# brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class WDLike(PromptGenerator):
    def __init__(self):
        self.tags = []
        self.tokens = None
        self.tokenizer = None
        self.model = None
        self.load_data()
    
    def load_data(self):
        with open(settings.WDLIKE_TAG_PATH, mode='r', encoding='utf8', newline='\n') as f:
            self.tags = [l.strip() for l in f.readlines()]
        with open(settings.WDLIKE_TOKEN_PATH, mode='rb') as f:
            self.tokens = np.load(f)
    
    def load_model(self):
        from modules.devices import device
        # brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(settings.WDLIKE_MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.WDLIKE_MODEL_NAME).to(device)

    def __call__(self, text: str, opts: GenerationSettings) -> List[str]:
        if not self.model or not self.tokenizer:
            return ''
        
        from modules.devices import device
        
        # --------------------------------------------------------------------------------------------------------------------------------
        # brought from https://huggingface.co/sentence-transformers/all-mpnet-base-v2#usage-huggingface-transformers
        # Tokenize sentences
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # --------------------------------------------------------------------------------------------------------------------------------

        # Get cosine similarity between given text and tag descriptions
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        tag_tokens_dev = torch.from_numpy(self.tokens).to(device)
        similarity: torch.Tensor = cos(sentence_embeddings[0], tag_tokens_dev)

        # Convert similarity into probablity
        if opts.conversion == ProbabilityConversion.CUTOFF_AND_POWER:
            probs_cpu = torch.clamp(similarity.detach().cpu(), 0, 1) ** opts.prob_power
        elif opts.conversion == ProbabilityConversion.SOFTMAX:
            probs_cpu = torch.softmax(similarity.detach().cpu(), dim=0)

        probs_cpu = probs_cpu / probs_cpu.sum(dim=0)

        results = None

        if opts.sampling == SamplingMethod.NONE:
            tags_np = np.array(self.tags)
            opts.n = min(tags_np.shape[0], opts.n)
            if opts.n <= 0: return []
            if opts.weighted:
                probs_np = probs_cpu.detach().numpy()
                probs_np /= np.sum(probs_np)

                if np.count_nonzero(probs_np) <= opts.n:
                    results = tags_np
                else:
                    results = np.random.choice(a=tags_np, size=opts.n, replace=False, p=probs_np)
            else:
                # Just sample randomly
                results = np.random.choice(a=tags_np, size=opts.n, replace=False)

        elif opts.sampling == SamplingMethod.TOP_K:
            probs, indices = probs_cpu.topk(opts.k)
            indices = indices.detach().numpy().tolist()
            if len(indices) <= 0: return []

            tags_np = np.array([self.tags[i] for i in indices])
            opts.n = min(tags_np.shape[0], opts.n)
            if opts.weighted:
                probs_np = probs.detach().numpy()
                probs_np /= np.sum(probs_np)

                if np.count_nonzero(probs_np) <= opts.n:
                    results = tags_np
                else:
                    results = np.random.choice(tags_np, opts.n, replace=False, p=probs_np)
            else:
                results = np.random.choice(tags_np, opts.n, replace=False)
        
        # brought from https://nn.labml.ai/sampling/nucleus.html
        elif opts.sampling == SamplingMethod.TOP_P:
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
                
                if np.count_nonzero(probs_np) <= opts.n:
                    results = tags_np
                else:
                    results = np.random.choice(tags_np, opts.n, replace=False, p=probs_np)
            else:
                results = np.random.choice(tags_np, opts.n, replace=False)

        return [] if results is None else results.tolist()