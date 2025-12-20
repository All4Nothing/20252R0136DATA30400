import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ClassModel(nn.Module):
    def __init__(self, encoder_name, enc_dim, class_embeddings, temperature=0.1):
        super(ClassModel, self).__init__()
        self.doc_encoder = AutoModel.from_pretrained(encoder_name)
        self.doc_dim = enc_dim

        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.num_classes, self.label_dim = class_embeddings.size()
        self.label_embedding_weights = nn.Parameter(class_embeddings.clone(), requires_grad=True)

        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids, attention_mask=attention_mask)
        doc_vector = self.mean_pooling(outputs, attention_mask)
        doc_norm = F.normalize(doc_vector, p=2, dim=1)
        label_norm = F.normalize(self.label_embedding_weights, p=2, dim=1)
        scores = torch.matmul(doc_norm, label_norm.T)
        logit_scale = self.logit_scale.exp().clamp(max=100) 
        scores = torch.matmul(doc_norm, label_norm.T) * logit_scale
        return scores

def multilabel_bce_loss_w(output, target, weight=None):
    if weight is None:
        weight = torch.ones_like(output)
    loss = F.binary_cross_entropy_with_logits(output, target, weight, reduction="sum")
    return loss / output.size(0)


class DataLoaderSimple:
    def __init__(self, data_dir="Amazon_products"):
        self.data_dir = data_dir
        self.test_corpus = []
        self.test_ids = []
        self.hierarchy_graph = nx.DiGraph()
        self.idx_to_class = {}
        self.all_classes = []

    def load(self):
        logger.info("Loading test data and metadata...")
        with open(os.path.join(self.data_dir, "test", "test_corpus.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2: 
                    self.test_ids.append(parts[0])
                    self.test_corpus.append(parts[1])
        with open(os.path.join(self.data_dir, "classes.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                cid, cname = line.strip().split('\t')
                self.all_classes.append(cname)
                self.idx_to_class[int(cid)] = cname
        with open(os.path.join(self.data_dir, "class_hierarchy.txt"), 'r', encoding='utf-8') as f:
            for line in f:
                p, c = line.strip().split('\t')
                self.hierarchy_graph.add_edge(int(p), int(c))
        logger.info(f"Loaded {len(self.test_corpus)} test documents and {len(self.all_classes)} classes.")

class HierarchyExpander:
    def __init__(self, graph):
        self.graph = graph
        self.ancestors_cache = {}
        
    def get_ancestors(self, node):
        if node in self.ancestors_cache: return self.ancestors_cache[node]
        try:
            ancestors = nx.ancestors(self.graph, node)
        except:
            ancestors = set()
        self.ancestors_cache[node] = ancestors
        return ancestors

    def expand(self, labels_list):
        expanded = []
        for labels in tqdm(labels_list, desc="Expanding Hierarchy"):
            label_set = set(labels)
            for l in labels:
                label_set.update(self.get_ancestors(l))
            expanded.append(sorted(list(label_set)))
        return expanded

class WeightedMultiLabelDataset(Dataset):
    def __init__(self, texts, tokenizer, hierarchy_graph, num_classes, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        self.max_length = max_length
        self.descendants_cache = {}
        for node in range(num_classes):
            if hierarchy_graph.has_node(node):
                self.descendants_cache[node] = nx.descendants(hierarchy_graph, node)
            else:
                self.descendants_cache[node] = set()

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def select_top_down_beam(probs, graph, beam_width=5, min_labels=2, max_labels=3, alpha=3):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    beam = []
    for r in roots:
        if r < len(probs):
            score = probs[r]
            beam.append((score, [r]))
    beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]
    completed_paths = []
    for _ in range(max_labels - 1):
        candidates = []
        for score, path in beam:
            curr_node = path[-1]
            if min_labels <= len(path) <= max_labels:
                completed_paths.append((score, path))
            children = list(graph.successors(curr_node))
            if not children:
                continue
            for child in children:
                if child >= len(probs): continue
                new_score = score * probs[child]
                new_path = path + [child]
                candidates.append((new_score, new_path))
        if not candidates:
            break
        beam = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
    completed_paths.extend(beam)
    valid_paths = [
        (s, p) for s, p in completed_paths 
        if min_labels <= len(p) <= max_labels
    ]
    if not valid_paths:
        print(f"!!!! WARNING: no valid paths")
        best_idx = np.argmax(probs)
        return [int(best_idx)]
    best_path = sorted(valid_paths, key=lambda x: math.pow(x[0], 1.0 / (len(x[1]) ** alpha)), reverse=True)[0][1]
    return sorted(best_path)


class InferenceEngine:
    def __init__(self, model_dir, hierarchy_graph, num_classes, device_id=0):
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        logger.info(f"Loading model architecture and weights from {model_dir}...")
        dummy_embeddings = torch.zeros(num_classes, 768)
        target_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = ClassModel(target_model_name, 768, dummy_embeddings, temperature=0.07)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except:
            logger.warning("Tokenizer not found in model_dir, loading from huggingface hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)

    def predict(self, texts, batch_size=64, method="top_down_beam", alpha=3):
        dataset = WeightedMultiLabelDataset(
            texts, self.tokenizer, self.hierarchy_graph, self.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_preds = []
        logger.info(f"Starting Inference with Method {method}...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                for doc_probs in probs:
                    path = select_top_down_beam(
                        doc_probs, 
                        self.hierarchy_graph, 
                        beam_width=10,
                        min_labels=2, 
                        max_labels=3,
                        alpha=alpha
                    )

                    all_preds.append(path)
        return all_preds

    def save_submission(self, predictions, ids, output_path):
        logger.info(f"Saving submission to {output_path}")
        data = []
        for doc_id, label_indices in zip(ids, predictions):
            label_str = ",".join(map(str, label_indices))
            data.append({'id': doc_id, 'labels': label_str})
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        avg_len = df['labels'].apply(lambda x: len(x.split(','))).mean()
        logger.info(f"Submission Stats: Average Labels per Doc = {avg_len:.2f}")


if __name__ == "__main__":
    method = "top_down_beam"
    alpha=3
    DATA_DIR = "../Amazon_products"
    MODEL_DIR = "outputs/models/best_model"
    OUTPUT_FILE = f"outputs/submission.csv"
    
    if not os.path.exists(DATA_DIR):
        if os.path.exists(f"../{DATA_DIR}"): DATA_DIR = f"../{DATA_DIR}"
        else: raise FileNotFoundError("Data directory not found")
    loader = DataLoaderSimple(DATA_DIR)
    loader.load()
    engine = InferenceEngine(
        model_dir=MODEL_DIR,
        hierarchy_graph=loader.hierarchy_graph,
        num_classes=len(loader.all_classes)
    )
    final_preds = engine.predict(loader.test_corpus, batch_size=128, method=method, alpha=alpha)
    engine.save_submission(final_preds, loader.test_ids, OUTPUT_FILE)
    logger.info("Done!")