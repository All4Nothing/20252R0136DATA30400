import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    from dotenv import load_dotenv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(script_dir), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env file from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed. Install it with: pip install python-dotenv")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
import json
import logging
import openai
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    pos_weight = torch.ones_like(output) * 5.0
    loss = F.binary_cross_entropy_with_logits(
        output, 
        target, 
        weight=weight, 
        pos_weight=pos_weight,
        reduction="mean" 
    )
    return loss

def custom_focal_loss(inputs, targets, weight=None, alpha=1, gamma=2.0):
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1 - pt) ** gamma * bce_loss
    if weight is not None:
        f_loss = f_loss * weight
    return f_loss.mean()

class WeightedMultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, hierarchy_graph, num_classes, 
                 is_augmented_mask=None, augmentation_scaling=1.0, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hierarchy_graph = hierarchy_graph
        self.num_classes = num_classes
        self.max_length = max_length
        self.is_augmented_mask = is_augmented_mask
        self.augmentation_scaling = augmentation_scaling
        self.descendants_cache = {}
        for node in range(num_classes):
            self.descendants_cache[node] = nx.descendants(hierarchy_graph, node)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_indices = self.labels[idx]
        is_aug = self.is_augmented_mask[idx] if self.is_augmented_mask else False
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        target = torch.zeros(self.num_classes)
        target[label_indices] = 0.9
        mask = torch.ones(self.num_classes)
        if is_aug:
            scaling_value = self.augmentation_scaling[idx] if isinstance(self.augmentation_scaling, (list, tuple, np.ndarray)) else self.augmentation_scaling
            mask = mask * scaling_value
        else:
            mask[label_indices] *= 2.0
            for pos_cls in label_indices:
                descendants = self.descendants_cache.get(pos_cls, set())
                for desc in descendants:
                    if desc not in label_indices:
                        mask[desc] = 0.0
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': target,
            'sample_mask': mask
        }

def generate_bert_class_embeddings(model_name, class_list, class_descriptions, device):
    logger.info(f"Generating class embeddings using {model_name} with LLM Descriptions...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    texts_to_encode = []
    for cls_name in class_list:
        if class_descriptions and cls_name in class_descriptions:
            desc = class_descriptions[cls_name]
            texts_to_encode.append(desc) 
        else:
            texts_to_encode.append(f"The product category is {cls_name.replace('_', ' ')}")
    class_emb = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            masked_embeddings = outputs.last_hidden_state * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            class_emb.append(mean_embeddings.cpu())
    return torch.cat(class_emb, dim=0)

class TELEClassTrainer:
    def __init__(self, num_classes, model_name, hidden_dim, class_embeddings, temperature=0.07, device_ids=None):
        self.available_gpus = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.primary_device = f'cuda:{self.available_gpus[0]}' if self.available_gpus else 'cpu'
        self.hidden_dim = hidden_dim
        self.model = ClassModel(model_name, self.hidden_dim, class_embeddings, temperature=0.07)
        modules = [self.model.doc_encoder.embeddings, *self.model.doc_encoder.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Frozen MPNet layers 0-7. Training only top layers")
        self.model.to(self.primary_device)
        if len(self.available_gpus) > 1:
            self.model = DataParallel(self.model, device_ids=self.available_gpus)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_classes = num_classes

    def train(self, train_texts, train_labels, train_mask, train_scaling,
              val_texts, val_labels, val_mask, val_scaling,
              hierarchy_graph, epochs=5, lr=2e-5, batch_size=16, 
              output_dir="outputs", patience=5, save_interval=5):
        eff_batch_size = batch_size * max(1, len(self.available_gpus))
        train_dataset = WeightedMultiLabelDataset(
            train_texts, train_labels, self.tokenizer, hierarchy_graph, self.num_classes,
            is_augmented_mask=train_mask, augmentation_scaling=train_scaling
        )
        val_dataset = WeightedMultiLabelDataset(
            val_texts, val_labels, self.tokenizer, hierarchy_graph, self.num_classes,
            is_augmented_mask=val_mask, augmentation_scaling=val_scaling
        )
        train_loader = TorchDataLoader(train_dataset, batch_size=eff_batch_size, shuffle=True, num_workers=4)
        val_loader = TorchDataLoader(val_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=4)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        num_training_steps = len(train_loader) * epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        self.save_model(os.path.join(output_dir, f"checkpoint_epoch_0"))
        self.model.train()
        best_f1_score = -1.0 
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.primary_device)
                attention_mask = batch['attention_mask'].to(self.primary_device)
                labels = batch['labels'].to(self.primary_device)
                sample_mask = batch['sample_mask'].to(self.primary_device)
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = custom_focal_loss(outputs, labels, sample_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            avg_train_loss = total_loss / len(train_loader)
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    input_ids = batch['input_ids'].to(self.primary_device)
                    attention_mask = batch['attention_mask'].to(self.primary_device)
                    labels = batch['labels'].to(self.primary_device)
                    sample_mask = batch['sample_mask'].to(self.primary_device)
                    outputs = self.model(input_ids, attention_mask)
                    loss = custom_focal_loss(outputs, labels, weight=sample_mask, gamma=2.0)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).long().cpu().numpy()
                    targets = (labels > 0.5).long().cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(targets)
            avg_val_loss = val_loss / len(val_loader)
            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            val_samples_f1 = f1_score(all_targets, all_preds, average='samples')
            val_micro_f1 = f1_score(all_targets, all_preds, average='micro')
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            logger.info(f"          Samples F1 (Metric)={val_samples_f1:.4f}, Micro F1={val_micro_f1:.4f}")
            logger.info(f"Saving model to {output_dir}/checkpoint_epoch_{epoch+1}")
            self.save_model(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}"))
            if val_samples_f1 > best_f1_score:
                best_f1_score = val_samples_f1
                patience_counter = 0
                logger.info(f"New Best Samples F1! ({best_f1_score:.4f}) Saving model to {output_dir}/best_model")
                self.save_model(os.path.join(output_dir, "best_model"))
            else:
                patience_counter += 1
                logger.info(f"No F1 improvement. Patience {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at score {best_f1_score:.4f}")
                break
        logger.info(f"Training complete. Best Samples F1: {best_f1_score:.4f}")

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(path)
        model_to_save.doc_encoder.config.save_pretrained(path)

class CustomInference:
    def __init__(self, model_path, class_embeddings, hierarchy_graph, device_ids=None):
        self.device = f'cuda:{device_ids[0]}' if device_ids else 'cpu'
        self.hierarchy_graph = hierarchy_graph
        self.model = ClassModel("bert-base-uncased", 768, class_embeddings)
        self.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict(self, texts, batch_size=32, threshold=0.15):
        dataset = WeightedMultiLabelDataset(
            texts, [[]]*len(texts), self.tokenizer, self.hierarchy_graph, self.model.num_classes
        )
        dataloader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).cpu().numpy()
                for i, p in enumerate(preds):
                    indices = np.where(p)[0].tolist()
                    if not indices:
                        indices = [torch.argmax(probs[i]).item()]
                    all_preds.append(indices)
        return all_preds

    def generate_submission(self, predictions, idx_to_class, output_path="submission.csv"):
        logger.info(f"Generating submission file: {output_path}")
        submission_data = []
        for doc_id, label_indices in enumerate(predictions):
            label_str = ",".join(map(str, label_indices))
            submission_data.append({'id': doc_id, 'labels': label_str})

        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Submission saved: {output_path}")

def get_available_gpus():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

class DataLoader:
    def __init__(self, data_dir: str = "Amazon_products"):
        self.data_dir = data_dir
        self.train_corpus = []
        self.test_corpus = []
        self.all_corpus = []
        self.all_indices = []
        self.train_indices = []
        self.test_indices = []
        self.hierarchy_graph = nx.DiGraph()
        self.class_keywords = {}
        self.all_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_all(self):
        logger.info("Loading all data files...")
        self._load_corpora()
        self._load_taxonomy()
        self._load_keywords()
        self._load_classes()
        logger.info(f"Data loading complete:")
        logger.info(f"  - Train corpus: {len(self.train_corpus)} documents")
        logger.info(f"  - Test corpus: {len(self.test_corpus)} documents")
        logger.info(f"  - Combined corpus: {len(self.all_corpus)} documents")
        logger.info(f"  - Classes: {len(self.all_classes)}")
        logger.info(f"  - Hierarchy edges: {self.hierarchy_graph.number_of_edges()}")
        
    def _load_corpora(self):
        train_path = os.path.join(self.data_dir, "train", "train_corpus.txt")
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.train_corpus.append(text)        
        test_path = os.path.join(self.data_dir, "test", "test_corpus.txt")
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id, text = parts
                    self.test_corpus.append(text)
        self.all_corpus = self.train_corpus + self.test_corpus
        self.all_indices = list(range(len(self.all_corpus)))
        self.train_indices = list(range(len(self.train_corpus)))
        self.test_indices = list(range(len(self.train_corpus), len(self.all_corpus)))
        
    def _load_taxonomy(self):
        hierarchy_path = os.path.join(self.data_dir, "class_hierarchy.txt")
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parent_id, child_id = parts
                    self.hierarchy_graph.add_edge(int(parent_id), int(child_id))
                    
    def _load_keywords(self):
        keywords_path = os.path.join(self.data_dir, "class_related_keywords.txt")
        with open(keywords_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    class_name, keywords = parts
                    keyword_list = [kw.strip() for kw in keywords.split(',')]
                    self.class_keywords[class_name] = keyword_list
                    
    def _load_classes(self):
        classes_path = os.path.join(self.data_dir, "classes.txt")
        with open(classes_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id, class_name = parts
                    self.all_classes.append(class_name)
                    self.class_to_idx[class_name] = int(class_id)
                    self.idx_to_class[int(class_id)] = class_name

class MultiGPUClassRepresentation:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", 
                 device_ids: Optional[List[int]] = None, output_dir: str = "outputs"):
        self.available_gpus = get_available_gpus()
        self.output_dir = output_dir
        if not self.available_gpus:
            logger.warning("No GPUs available, using CPU")
            self.device_ids = []
            self.primary_device = 'cpu'
        else:
            self.device_ids = device_ids or self.available_gpus
            self.primary_device = f'cuda:{self.device_ids[0]}'
            logger.info(f"Using GPUs: {self.device_ids}")
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        if len(self.device_ids) > 1:
            logger.info(f"Enabling multi-GPU encoding across {len(self.device_ids)} GPUs")
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.device_ids))
        self.model.to(self.primary_device)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM features may not work.")
            self.openai_client = None
    
    def _call_llm_api(self, prompt: str) -> str:
        if self.openai_client is None:
            logger.error("OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
            return ""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return ""
        
    def create_enriched_class_descriptions(self, class_keywords, hierarchy_graph, idx_to_class):
        logger.info("Generating enriched class descriptions via LLM...")
        descriptions = {}
        cache_file = os.path.join(self.output_dir, "enriched_class_descriptions.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            logger.info(f"Loaded {len(descriptions)} cached descriptions")
            return descriptions
        class_to_idx = {v: k for k, v in idx_to_class.items()}
        cnt = 0
        save_interval = 50
        for class_name, keywords in tqdm(class_keywords.items(), desc="LLM Enrichment"):
            cid = class_to_idx.get(class_name)
            parent_name = "None"
            siblings = []
            
            if cid is not None and hierarchy_graph.has_node(cid):
                preds = list(hierarchy_graph.predecessors(cid))
                if preds:
                    parent_id = preds[0] 
                    parent_name = idx_to_class.get(parent_id, "Unknown")
                    siblings_ids = list(hierarchy_graph.successors(parent_id))
                    siblings = [idx_to_class[s] for s in siblings_ids if s != cid]
            
            sibling_str = ", ".join(siblings[:3]) if siblings else "None"
            keyword_str = ", ".join(keywords)
            
            prompt = f"""
            Task: Define the Amazon product category '{class_name}'.
            Context:
            - Parent Category: {parent_name}
            - Sibling Categories (Distinct from): {sibling_str}
            - Associated Keywords: {keyword_str}
            
            Write a sharp, 1-sentence definition focusing on visual/functional attributes 
            that distinguish '{class_name}' from its siblings. 
            Do NOT mention the class name directly in the beginning.
            """
            
            llm_desc = self._call_llm_api(prompt)

            if not llm_desc:
                logger.warning(f"Failed to generate description for {class_name}, using keywords only")

            with open("llm_desc.txt", "a", encoding="utf-8") as f:
                f.write(f"{llm_desc}")

            final_desc = f"Class Name: {class_name} ; Keywords: {keyword_str} ; Description: {llm_desc}"
            descriptions[class_name] = final_desc

            cnt += 1
            if cnt % save_interval == 0:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(descriptions, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(descriptions)} enriched descriptions to {cache_file}")
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(descriptions)} enriched descriptions to {cache_file}")
        return descriptions
    
    def encode_classes(self, class_descriptions: Dict[str, str], all_classes: List[str]) -> torch.Tensor:
        logger.info("Encoding class descriptions...")
        descriptions_list = [class_descriptions.get(cls, f"The product category is {cls}") 
                            for cls in all_classes]
        embeddings = self.model.encode(
            descriptions_list,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.primary_device,
            batch_size=64
        )
        return embeddings
    
    def encode_documents_parallel(self, documents: List[str], batch_size: int = 64) -> torch.Tensor:
        logger.info(f"Encoding {len(documents)} documents across {len(self.device_ids) or 1} GPU(s)...")
        
        if len(self.device_ids) <= 1:
            embeddings = self.model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
                device=self.primary_device
            )
        else:
            embeddings = self.model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size * len(self.device_ids),
                device=self.primary_device
            )
        
        return embeddings

class IterativePseudoLabeler:
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_similarity(self, doc_embeddings: torch.Tensor, class_embeddings: torch.Tensor) -> torch.Tensor:
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=1)
        similarity = torch.mm(doc_embeddings, class_embeddings.t())
        return similarity
    
    def refine_class_embeddings(
        self,
        doc_embeddings: torch.Tensor,
        class_embeddings: torch.Tensor,
        num_iterations: int = 3,
        top_n_reliable: int = 20,
        initial_top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.info(f"Starting iterative refinement for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            similarity = self.compute_similarity(doc_embeddings, class_embeddings)
            
            new_class_embeddings = []
            for class_idx in range(class_embeddings.shape[0]):
                class_scores = similarity[:, class_idx]
                top_n_indices = torch.topk(class_scores, min(top_n_reliable, len(class_scores))).indices
                reliable_embeddings = doc_embeddings[top_n_indices]
                new_class_embedding = reliable_embeddings.mean(dim=0)
                new_class_embeddings.append(new_class_embedding)
            
            class_embeddings = torch.stack(new_class_embeddings)
            logger.info(f"  Class embeddings updated based on reliable document centroids")
        
        final_similarity = self.compute_similarity(doc_embeddings, class_embeddings)
        return class_embeddings, final_similarity
    
    def assign_labels_with_gap(
        self,
        similarity: torch.Tensor,
        min_labels: int = 2,
        max_gap_search: int = 5
    ) -> Tuple[List[List[int]], List[List[float]]]:
        logger.info("Assigning pseudo-labels with gap-based cutoff...")
        
        pseudo_labels = []
        pseudo_scores = []
        
        for doc_idx in range(similarity.shape[0]):
            scores = similarity[doc_idx]
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            diffs = sorted_scores[:-1] - sorted_scores[1:]
            valid_range = diffs[:max_gap_search]
            best_gap_idx = torch.argmax(valid_range).item()
            num_labels = max(min_labels, best_gap_idx + 1)
            
            selected_indices = sorted_indices[:num_labels].cpu().tolist()
            selected_scores = sorted_scores[:num_labels].cpu().tolist()
            
            pseudo_labels.append(selected_indices)
            pseudo_scores.append(selected_scores)
        
        avg_labels = np.mean([len(labels) for labels in pseudo_labels])
        logger.info(f"  Average labels per document: {avg_labels:.2f}")
        
        return pseudo_labels, pseudo_scores

class AugmentationModule:
    def __init__(self, data_loader: DataLoader, augmentation_dir: str = "augmented_data"):
        self.data_loader = data_loader
        self.augmentation_dir = augmentation_dir
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found in environment variables. LLM features may not work.")
            self.openai_client = None
        os.makedirs(augmentation_dir, exist_ok=True)
        
    def identify_starved_classes(
        self,
        pseudo_labels: List[List[int]],
        train_indices: List[int],
        threshold: int = 10
    ) -> List[int]:
        class_counts = {}
        for idx in train_indices:
            for class_idx in pseudo_labels[idx]:
                class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
        
        starved_classes = [
            class_idx for class_idx in range(len(self.data_loader.all_classes))
            if class_counts.get(class_idx, 0) < threshold
        ]
        print(f"starved_classes: {starved_classes}")
        logger.info(f"Identified {len(starved_classes)} starved classes (< {threshold} documents)")
        return starved_classes
    
    def generate_synthetic_data(self, class_idx, class_name, count=5):
        if not self.openai_client:
            return []
            
        prompt = f"""
        Act as a real Amazon customer writing a review.
        Target Category: "{class_name}"
        
        Your goal is to generate {count} raw, authentic datasets that look exactly like the "Real Examples" below.
        
        --- REAL EXAMPLES (Mimic this style) ---
        1. bigelow tea ( 6 pack ) the flavor of bigelow 's earl gray tea is the best i have ever tasted . sophisticated , smooth , and mellow , with just the right blend of bergamot . i drink this decaffeinated version every evening . wonderful !
        2. vtech build discover workbench my son got this for christmas and he loves it ! there are lots of things to do , even for a 26 month old who ca n't quite follow the directions yet . he enjoys being able to turn the screws and bolts . my only complaint is that it has no volume control .
        3. moisturizing shampoo from kenra 10 . 1 oz good price for a great product . it 's less expensive than buying in a store or shop . i also use the kenra conditioner .
        ---------------------------------------

        CRITICAL WRITING RULES:
        1. **Variable Structure:** Do NOT use the same format for every line. Some product titles should be short, some long with specs (oz, pack, watts).
        2. **Messy Titles:** Put specs like "10.1 oz" or "( 6 pack )" sometimes at the end, sometimes in the middle of the title.
        3. **Diverse Content:** - Don't mention price ($) or shipping in every review. Focus on usage, smell, texture, family reaction, or specific pros/cons.
           - Vary length! Write some short reviews (1 sentence) and some long detailed stories (3-4 sentences).
        4. **Natural Tone:** Use lowercase. It's okay to be conversational. Use "i" instead of "I" sometimes.
        5. **Spacing:** Try to put spaces before punctuation like " .", " ,", " !" to match the examples.
        
        Generate {count} reviews now (one per line, no numbering):
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            text = response.choices[0].message.content.strip()
            with open("augmentation_data.txt", "a", encoding="utf-8") as f:
                f.write(f"{text}")
            return [line.strip() for line in text.split('\n') if line.strip()]
        except Exception as e:
            logger.error(f"Augmentation failed for {class_name}: {e}")
            return []

    def calibrate_text_style(self, text):
        text = text.lower()
        text = re.sub(r'([a-zA-Z0-9])([.,!?])', r'\1 \2', text)
        text = text.replace("n't", " n't")
        text = text.replace("'s", " 's")
        text = text.replace("'ve", " 've")
        text = text.replace("'re", " 're")
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_augmentation_data(
        self,
        starved_classes: List[int],
        num_samples_per_class: int = 15
    ) -> Tuple[List[str], List[List[int]]]:
        logger.info(f"Starting LLM Augmentation for {len(starved_classes)} classes...")
        aug_texts = []
        aug_labels = []
        for class_idx in tqdm(starved_classes, desc="Augmenting"):
            class_name = self.data_loader.all_classes[class_idx]
            new_reviews = self.generate_synthetic_data(class_idx, class_name, count=5)
            for review in new_reviews:
                aug_texts.append(self.calibrate_text_style(review))
                aug_labels.append([class_idx])
        logger.info(f"Generated {len(aug_texts)} synthetic samples.")
        return aug_texts, aug_labels

class HierarchyExpander:
    def __init__(self, hierarchy_graph: nx.DiGraph, class_to_idx: Dict[str, int]):
        self.hierarchy_graph = hierarchy_graph
        self.class_to_idx = class_to_idx
        
    def get_ancestors(self, class_idx: int) -> Set[int]:
        ancestors = set()
        queue = [class_idx]
        visited = {class_idx}
        while queue:
            current = queue.pop(0)
            for node in self.hierarchy_graph.nodes():
                if self.hierarchy_graph.has_edge(node, current) and node not in visited:
                    ancestors.add(node)
                    queue.append(node)
                    visited.add(node)
        return ancestors
    
    def expand_labels_with_hierarchy(
        self,
        pseudo_labels: List[List[int]]
    ) -> List[List[int]]:
        logger.info("Expanding labels with hierarchy...")
        expanded_labels = []
        for doc_labels in tqdm(pseudo_labels, desc="Expanding labels"):
            expanded = set(doc_labels)
            for class_idx in doc_labels:
                ancestors = self.get_ancestors(class_idx)
                expanded.update(ancestors)
            expanded_labels.append(sorted(list(expanded)))
        avg_original = np.mean([len(labels) for labels in pseudo_labels])
        avg_expanded = np.mean([len(labels) for labels in expanded_labels])
        logger.info(f"  Average labels: {avg_original:.2f} -> {avg_expanded:.2f}")
        return expanded_labels

class MultiGPUTELEClassPipeline:
    def __init__(self, 
                 data_dir: str = "Amazon_products", 
                 output_dir: str = "outputs", 
                 seed: int = 42,
                 device_ids: Optional[List[int]] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = seed
        
        available_gpus = get_available_gpus()
        self.device_ids = device_ids or available_gpus
        
        if self.device_ids:
            logger.info(f"Pipeline will use GPUs: {self.device_ids}")
        else:
            logger.info("No GPUs available, pipeline will use CPU")
        
        set_seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
        
    def run(self):
        logger.info("="*80)
        logger.info("MULTI-GPU TELECLASS PIPELINE")
        logger.info(f"Available GPUs: {len(self.device_ids)}")
        logger.info("="*80)
        
        logger.info("\n" + "="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        data_loader = DataLoader(self.data_dir)
        data_loader.load_all()
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: MULTI-GPU CLASS REPRESENTATION")
        logger.info("="*80)
        class_repr = MultiGPUClassRepresentation(device_ids=self.device_ids, output_dir=self.output_dir)
        
        class_descriptions = class_repr.create_enriched_class_descriptions(
            class_keywords=data_loader.class_keywords,
            hierarchy_graph=data_loader.hierarchy_graph,
            idx_to_class=data_loader.idx_to_class
        )

        class_embeddings = class_repr.encode_classes(class_descriptions, data_loader.all_classes)
        doc_embeddings = class_repr.encode_documents_parallel(data_loader.all_corpus, batch_size=64)
        
        logger.info(f"Class embeddings shape: {class_embeddings.shape}")
        logger.info(f"Document embeddings shape: {doc_embeddings.shape}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: ITERATIVE PSEUDO-LABELING")
        logger.info("="*80)
        
        primary_device = f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu'
        labeler = IterativePseudoLabeler(device=primary_device)
        refined_class_embeddings, final_similarity = labeler.refine_class_embeddings(
            doc_embeddings,
            class_embeddings,
            num_iterations=0,
            top_n_reliable=20
        )
        
        pseudo_labels, pseudo_scores = labeler.assign_labels_with_gap(
            final_similarity,
            min_labels=1,
            max_gap_search=3
        )
        
        torch.save({
            'pseudo_labels': pseudo_labels,
            'pseudo_scores': pseudo_scores,
            'class_embeddings': refined_class_embeddings,
            'doc_embeddings': doc_embeddings
        }, os.path.join(self.output_dir, "intermediate", "phase2_outputs.pt"))
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: DATA AUGMENTATION")
        logger.info("="*80)
        
        aug_module = AugmentationModule(data_loader)
        starved_classes = aug_module.identify_starved_classes(
            pseudo_labels,
            data_loader.all_indices,
            threshold=15
        )
        
        aug_save_path = os.path.join(self.output_dir, "augmented_data.txt")
        if os.path.exists(aug_save_path):
            logger.info(f"Loading existing augmented data from {aug_save_path}")
            augmented_texts = []
            augmented_labels = []
            with open(aug_save_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) == 2:
                        label_part = parts[0].replace("label_str: ", "")
                        text_part = parts[1].replace("text: ", "")
                        label = [int(x) for x in label_part.split(",") if x.strip()]
                        augmented_labels.append(label)
                        augmented_texts.append(text_part)
            logger.info(f"Loaded {len(augmented_texts)} augmented samples from existing file")
        else:
            augmented_texts, augmented_labels = aug_module.generate_augmentation_data(starved_classes)
            with open(aug_save_path, "w", encoding="utf-8") as f:
                for txt, label in zip(augmented_texts, augmented_labels):
                    label_str = ",".join(map(str, label))
                    f.write(f"label_str: {label_str}\ttext: {txt}\n")
            logger.info(f"Augmented data saved to {aug_save_path}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: HIERARCHY EXPANSION")
        logger.info("="*80)
        
        expander = HierarchyExpander(data_loader.hierarchy_graph, data_loader.class_to_idx)
        expanded_labels = expander.expand_labels_with_hierarchy(pseudo_labels)

        avg_labels = np.mean([len(l) for l in expanded_labels])
        print(f"Average Labels after Expansion: {avg_labels}")
        
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: CUSTOM MODEL TRAINING")
        logger.info("="*80)

        primary_device = f'cuda:{self.device_ids[0]}' if self.device_ids else 'cpu'
        
        final_train_texts = data_loader.all_corpus + augmented_texts
        final_train_labels = expanded_labels + augmented_labels
        is_augmented_mask = [False] * len(data_loader.all_corpus) + [True] * len(augmented_texts)
        num_real = len(data_loader.all_corpus)
        num_aug = len(augmented_texts)
        aug_scaling = 1.0
        logger.info(f"Augmentation Scaling Factor: {aug_scaling:.4f}")

        logger.info("\n" + "="*80)
        logger.info("PHASE 5: CUSTOM MODEL TRAINING (Using MPNet Backbone)")
        logger.info("="*80)

        target_model_name = "sentence-transformers/all-mpnet-base-v2"
        
        if 'class_repr' not in locals():
            class_repr = MultiGPUClassRepresentation(
                model_name=target_model_name, 
                device_ids=self.device_ids, 
                output_dir=self.output_dir
            )
            
        final_class_embeddings = class_repr.encode_classes(
            class_descriptions,
            data_loader.all_classes
        )
        
        real_indices = np.arange(len(data_loader.all_corpus))
        train_idx, val_idx = train_test_split(real_indices, test_size=0.1, random_state=42)
        real_train_texts = [data_loader.all_corpus[i] for i in train_idx]
        real_train_labels = [expanded_labels[i] for i in train_idx]
        val_texts = [data_loader.all_corpus[i] for i in val_idx]
        val_labels = [expanded_labels[i] for i in val_idx]
        val_mask = [False] * len(val_texts)
        val_scaling = [1.0] * len(val_texts) 
        final_train_texts = real_train_texts + augmented_texts
        final_train_labels = real_train_labels + augmented_labels
        train_mask = [False] * len(real_train_texts) + [True] * len(augmented_texts)
        aug_scaling_value = 1.0 
        train_scaling = [1.0] * len(real_train_texts) + [aug_scaling_value] * len(augmented_texts)
        logger.info(f"  - Final Train Size: {len(final_train_texts)} (Real: {len(real_train_texts)} + Aug: {len(augmented_texts)})")
        logger.info(f"  - Clean Val Size:   {len(val_texts)} (All Real)")
        config = AutoConfig.from_pretrained(target_model_name)
        trainer = TELEClassTrainer(
            num_classes=len(data_loader.all_classes),
            hidden_dim=config.hidden_size,
            model_name=target_model_name,
            class_embeddings=final_class_embeddings,
            device_ids=self.device_ids,
            temperature=0.1
        )
        trainer.train(
            train_texts=final_train_texts,
            train_labels=final_train_labels,
            train_mask=train_mask,
            train_scaling=train_scaling,
            val_texts=val_texts,
            val_labels=val_labels,
            val_mask=val_mask,
            val_scaling=val_scaling,
            hierarchy_graph=data_loader.hierarchy_graph,
            epochs=1,
            lr=2e-5,
            output_dir=os.path.join(self.output_dir, "models")
        )
        logger.info("\n" + "="*80)
        logger.info("MULTI-GPU PIPELINE COMPLETE!")
        logger.info("="*80)

if __name__ == "__main__":
    import sys
    if os.path.exists("Amazon_products"):
        data_dir = "Amazon_products"
    elif os.path.exists("../Amazon_products"):
        data_dir = "../Amazon_products"
    else:
        print("Error: Cannot find Amazon_products directory")
        print("Please run from the project root or taxoclass directory")
        sys.exit(1)
    device_ids = None
    pipeline = MultiGPUTELEClassPipeline(
        data_dir=data_dir,
        output_dir="outputs",
        seed=42,
        device_ids=device_ids
    )
    pipeline.run()
