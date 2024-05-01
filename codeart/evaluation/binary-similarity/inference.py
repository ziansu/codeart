import sys
sys.path.append('../../code')

from transformers import (
    HfArgumentParser,
    set_seed,
)
from typing import Optional
import shlex
import os
import numpy as np
import json
from dataclasses import dataclass, field
import argparse
import torch
from utils import AverageMeter, BinaryRetriever, BinaryRetrieverForGCBLike
from eval import eval_from_dict, eval_from_file, TOP_K
from modeling_utils import MaskBuilder
from models import (
    RabertConfig,
    CodeArtConfig,
    RabertModel,
    CodeArtModel,
    RabertTokenizer,
    CodeArtTokenizer,
    CodeArtForBinSim,
    RabertForBinSim,
    GCBLikeTokenizer,
)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    # position arguments
    position_embedding_type: str = field(
        default='absolute',
        metadata={
            "help": (
                "Type of positional embedding to use, can be either 'absolute' or 'relative'"
            ),
            "choices": ['absolute', 'mixed'],
        }
    )
    max_relative_position_embeddings: Optional[int] = field(
        default=5,
        metadata={
            "help": (
                "Max relative postion distance to consider in relative embeddings"
            )
        }
    )

    # masking arguments
    masking_preset: Optional[str] = field(
        default=None,
        metadata={
            "help": ("Preset masking strategy"),
            "choices": [None, 'aggressive', 'conservative'],
        }
    )
    masking_enable_global_memory_patterns: bool = field(
        default=True,
        metadata={
            "help": ("enable global memory patterns")
        }
    )
    masking_enable_bridge_patterns: bool = field(
        default=True,
        metadata={
            "help": ("enable bridge patterns")
        }
    )
    masking_enable_graph_patterns: bool = field(
        default=True,
        metadata={
            "help": ("enable graph patterns")
        }
    )
    masking_enable_local_patterns: bool = field(
        default=True,
        metadata={
            "help": ("enable local patterns")
        }
    )

    with_transitive_closure: bool = field(
        default=True,
        metadata={
            "help": ("Whether to include transitive closure in the masking process")
        }
    )

    max_transitions: Optional[int] = field(
        default=None,
    )

    normalize_embed: bool = field(
        default=True
    )

    zero_shot: bool = field(
        default=False
    )

    gcb_like: bool = field(
        default=False
    )


@dataclass
class DataArguments:

    batch_size: int = field(
        default=16
    )
    source_file: str = field(
        default='coreutils.clang.O0.jsonl'
    )
    target_file: str = field(
        default='coreutils.gcc.O3.jsonl'
    )
    source_embed_save_file: str = field(
        default='src_embedding.npy'
    )
    target_embed_save_file: str = field(
        default='tgt_embedding.npy'
    )
    save_file: str = field(
        default='retrieval_results.json'
    )
    top_k: int = field(
        default=200
    )
    cpu: Optional[bool] = field(
        default=False
    )
    pool_size: int = field(
        default=100
    )

    def __post_init__(self):
        self.source_idx_file = self.source_file.replace(".jsonl", ".id")
        self.target_idx_file = self.target_file.replace(".jsonl", ".id")


def main():
    set_seed(42)
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    if not model_args.with_transitive_closure:
        if model_args.gcb_like:
            tokenizer = GCBLikeTokenizer.from_pretrained(
                model_args.model_name_or_path
            )

            binsim_model = RabertForBinSim.from_pretrained(
                model_args.model_name_or_path
            )
            model = binsim_model.rabert

        else:

            tokenizer = RabertTokenizer.from_pretrained(
                model_args.model_name_or_path
            )
            model = RabertModel.from_pretrained(
                model_args.model_name_or_path,
            )
    else:
        tokenizer = CodeArtTokenizer.from_pretrained(
            model_args.model_name_or_path
        )
        # model = CodeArtModel.from_pretrained(
        #     model_args.model_name_or_path,
        # )
        # loaded_weights = torch.load(
        #     os.path.join(model_args.model_name_or_path, 'pytorch_model.bin'),
        #     map_location='cpu'
        # )
        binsim_model = CodeArtForBinSim.from_pretrained(
            model_args.model_name_or_path
        )
        model = binsim_model.codeart
        # mask builder
    maskbuilder = MaskBuilder(
        preset=model_args.masking_preset,
        enable_global_memory_patterns=model_args.masking_enable_global_memory_patterns,
        enable_bridge_patterns=model_args.masking_enable_bridge_patterns,
        enable_graph_patterns=model_args.masking_enable_graph_patterns,
        device='cpu' if data_args.cpu else 'cuda'
    )

    tokenizer.add_tokens('<INST>')
    tokenizer.maskbuilder = maskbuilder

    pooler_method = 'cls'
    if not model_args.zero_shot:
        pooler_method = 'pooler'

    if not model_args.gcb_like:
        searcher = BinaryRetriever
        print("\033[93m searcher = BinaryRetriever \033[0m")
    else:
        searcher = BinaryRetrieverForGCBLike
        print("\033[93m searcher = BinaryRetrieverForGCBLike \033[0m")

    searcher = searcher(
        tokenizer=tokenizer,
        encoder=model,
        pooler=pooler_method
    )

    searcher.encode_file(
        data_args.source_file,
        data_args.source_embed_save_file,
        normalize_embed=model_args.normalize_embed,
        max_transitions=model_args.max_transitions
    )
    searcher.encode_file(
        data_args.target_file,
        data_args.target_embed_save_file,
        normalize_embed=model_args.normalize_embed,
        max_transitions=model_args.max_transitions
    )

    with open(data_args.source_idx_file, 'r') as f:
        source_id_map = {}
        for idx, line in enumerate(f):
            source_id_map[idx] = line.strip()

    with open(data_args.target_idx_file, 'r') as f:
        target_id_map = {}
        for idx, line in enumerate(f):
            target_id_map[idx] = line.strip()

    source_embed = np.load(data_args.source_embed_save_file)
    target_embed = np.load(data_args.target_embed_save_file)
    assert (len(source_id_map) == source_embed.shape[0])
    assert (len(target_id_map) == target_embed.shape[0])

    results = {
        'recall': {k: AverageMeter() for k in TOP_K},
        'mrr': AverageMeter()
    }
    total_pools = len(source_id_map) // data_args.pool_size
    for i in range(total_pools):
        pool_source_embed = source_embed[i *
                                         data_args.pool_size: (i + 1) * data_args.pool_size]
        pool_target_embed = target_embed[i *
                                         data_args.pool_size: (i + 1) * data_args.pool_size]
        pool_source_id_map, pool_target_id_map = {}, {}
        for j in range(data_args.pool_size):
            pool_source_id_map[j] = source_id_map[i * data_args.pool_size + j]
            pool_target_id_map[j] = target_id_map[i * data_args.pool_size + j]
        pool_results = searcher.retrieve(
            pool_source_embed,
            pool_target_embed,
            pool_source_id_map,
            pool_target_id_map,
            data_args.top_k
        )
        pool_results = eval_from_dict(pool_results)
        results['mrr'].update(pool_results['mrr'])
        for k in TOP_K:
            results['recall'][k].update(pool_results['recall'][k])

    print(results)


if __name__ == '__main__':
    main()
