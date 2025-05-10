from llmlingua import PromptCompressor
from typing import List, Union, Tuple, Dict
import math
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import re
from transformers import AutoTokenizer
class TokenClfDataset(Dataset):
    def __init__(
        self,
        texts,
        max_len=512,
        tokenizer=None,
        model_name="bert-base-multilingual-cased",
    ):
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        if "bert-base-multilingual-cased" in model_name:
            self.cls_token = "[CLS]"
            self.sep_token = "[SEP]"
            self.unk_token = "[UNK]"
            self.pad_token = "[PAD]"
            self.mask_token = "[MASK]"
        elif "xlm-roberta-large" in model_name:
            self.bos_token = "<s>"
            self.eos_token = "</s>"
            self.sep_token = "</s>"
            self.cls_token = "<s>"
            self.unk_token = "<unk>"
            self.pad_token = "<pad>"
            self.mask_token = "<mask>"
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer.tokenize(text)

        tokenized_text = (
            [self.cls_token] + tokenized_text + [self.sep_token]
        )  # add special tokens

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
        else:
            tokenized_text = tokenized_text + [
                self.pad_token for _ in range(self.max_len - len(tokenized_text))
            ]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len

class AnalyzerSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = TokenProbabilityAnalyzer(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
            )
        return cls._instance
class TokenProbabilityAnalyzer(PromptCompressor):
    def __init__(
        self,
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
        model_config: dict = {},
        open_api_config: dict = {},
        use_llmlingua2: bool = False,
        llmlingua2_config: dict = {},
        ):
        super().__init__(model_name,device_map,model_config,open_api_config,use_llmlingua2,llmlingua2_config)
        self.qwen_tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    def get_token_importance(
            self, 
            text: str, 
            token_to_word: str = "mean",
            force_tokens: List[str] = [],
            force_reserve_digit: bool = False
        ) -> Dict:
        """
        计算文本中每个token的重要性得分，使用与LLMLingua相同的算法。
        """
        # 清理系统 token，使用 Qwen2.5 的 qwen_tokenizer 过滤特殊 token
        token_ids = self.qwen_tokenizer.encode(text, add_special_tokens=False)
        filtered_ids = [
            tid for tid in token_ids 
            if self.qwen_tokenizer.convert_ids_to_tokens(tid) not in {"<|im_start|>", "<|im_end|>", "<|system|>", "<|endoftext|>"}
        ]
        clean_text = self.qwen_tokenizer.decode(filtered_ids, skip_special_tokens=True)

        # 将文本分成chunks - 正确访问父类的私有方法
        chunks = self._PromptCompressor__chunk_context(clean_text, chunk_end_tokens=set([".", "\n"]))

        # 构造 token_map
        token_map = {}
        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]

        # 将chunks转换为列表形式以匹配__get_context_prob的输入格式
        context_chunked = [chunks]

        # 计算 token 概率
        probs, _ = self.__get_context_prob(
            context_chunked,
            token_to_word=token_to_word,
            force_tokens=force_tokens,
            token_map=token_map,
            force_reserve_digit=force_reserve_digit,
        )

        return probs,len(token_ids)
    
    def __get_context_prob(
        self,
        context_list: list,
        token_to_word="mean",
        force_tokens: List[str] = [],
        token_map: dict = {},
        force_reserve_digit: bool = False,
    ):
        chunk_list = []
        for chunks in context_list:
            for c in chunks:
                chunk_list.append(c)

        dataset = TokenClfDataset(
            chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
        )
        dataloader = DataLoader(
            dataset, batch_size=self.max_batch_size, shuffle=False, drop_last=False
        )

        chunk_probs = []
        chunk_words = []
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long) == 1

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss, logits = outputs.loss, outputs.logits
                probs = F.softmax(logits, dim=-1)

                for j in range(ids.shape[0]):
                    _probs = probs[j, :, 1]
                    _ids = ids[j]
                    _mask = mask[j]

                    active_probs = torch.masked_select(_probs, _mask)
                    active_ids = torch.masked_select(_ids, _mask)

                    tokens = self.tokenizer.convert_ids_to_tokens(
                        active_ids.squeeze().tolist()
                    )
                    token_probs = [prob for prob in active_probs.cpu().numpy()]

                    (
                        words,
                        valid_token_probs,
                        valid_token_probs_no_force,
                    ) = self._PromptCompressor__merge_token_to_word(
                        tokens,
                        token_probs,
                        force_tokens=force_tokens,
                        token_map=token_map,
                        force_reserve_digit=force_reserve_digit,
                    )
                    word_probs_no_force = self._PromptCompressor__token_prob_to_word_prob(
                        valid_token_probs_no_force, convert_mode='mean'
                    )

                    if "xlm-roberta-large" in self.model_name:
                        for i in range(len(words)):
                            words[i] = words[i].lstrip("▁")
                    chunk_words.append(words)
                    chunk_probs.append(word_probs_no_force)

        prev_idx = 0
        context_probs = []
        context_words = []
        for chunk_list in context_list:
            n_chunk = len(chunk_list)
            context_probs.append([])
            context_words.append([])
            for i in range(n_chunk):
                context_probs[-1].extend(chunk_probs[prev_idx + i])
                context_words[-1].extend(chunk_words[prev_idx + i])
            prev_idx = prev_idx + n_chunk

        eps = 1e-8

        power_match = re.match(r"power(-?\d+)_mean", token_to_word)
        if power_match:
            p = int(power_match.group(1))
            if p == 0:
                # 几何平均
                context_probs = [
                    math.exp(sum(math.log(max(prob, eps)) for prob in probs) / len(probs))
                    for probs in context_probs
                ]
            else:
                context_probs = [
                    (sum(prob**p for prob in probs) / len(probs))**(1 / p)
                    for probs in context_probs
                ]
        else:
            context_probs = [sum(probs) / len(probs) for probs in context_probs]

        return context_probs, context_words