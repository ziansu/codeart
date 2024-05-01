import time
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import (
    EvalPrediction,
    has_length,
    speed_metrics
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_pt_utils import (
    find_batch_size
)
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
import torch
from typing import Callable, Dict, List, Optional, Tuple
from torch.utils.data.dataset import Dataset
from transformers.utils import logging
import numpy as np
import torch.nn.functional as F
from models.modeling_codeart import CodeArtForBinSim

logger = logging.get_logger(__name__)


class BinSimTrainer(Trainer):
    def __init__(
        self,
        model: CodeArtForBinSim = None,
        args: TrainingArguments = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super(BinSimTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )        

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "valid",
    ) -> Dict[str, float]:
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # 
        # BEGIN MAGIC
        # 
        # # # # # # # # # # # # # # # # # # # # # # # # #
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        if not self.is_in_train:            
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running Validation *****")
        if has_length(eval_dataloader):
            logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()
        self.callback_handler.eval_dataloader = eval_dataloader
        # Do this before wrapping.
        eval_dataset = getattr(eval_dataloader, "dataset", None)

        if self.args.past_index >= 0:
            self._past = None

        # # # # # # # # # # # # # # # # # # # # # # # # #
        # 
        # END MAGIC
        # 
        # # # # # # # # # # # # # # # # # # # # # # # # #
        # torch.cuda.empty_cache()
        pr1_all=[]
        mrr_all=[]
        pr1_wo_pooler_all=[]
        mrr_wo_pooler_all=[]
        observed_batch_size = 0
        for step, inputs in enumerate(eval_dataloader):
            if observed_batch_size == 0:
              observed_batch_size = find_batch_size(inputs)
            pr1, mrr, pr1_wo_pooler, mrr_wo_pooler = self.validate(model, inputs)
            pr1_all.append(pr1)
            mrr_all.append(mrr)
            pr1_wo_pooler_all.append(pr1_wo_pooler)
            mrr_wo_pooler_all.append(mrr_wo_pooler)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        pr1_all = np.array(pr1_all)
        mrr_all = np.array(mrr_all)
        pr1_wo_pooler_all = np.array(pr1_wo_pooler_all)
        mrr_wo_pooler_all = np.array(mrr_wo_pooler_all)
        pr1 = np.mean(pr1_all)
        mrr = np.mean(mrr_all)
        pr1_wo_pooler = np.mean(pr1_wo_pooler_all)
        mrr_wo_pooler = np.mean(mrr_wo_pooler_all)

        ret_metrics = {"valid_pr1": pr1, "valid_mrr": mrr, 
                        "valid_pr1_wo_pooler": pr1_wo_pooler,
                        "valid_mrr_wo_pooler": mrr_wo_pooler,
                       'batch_size': observed_batch_size}
        ret_metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=self.num_examples(eval_dataloader),
                num_steps=step                
            )
        )
        self.log(ret_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, ret_metrics)
        self._memory_tracker.stop_and_update_metrics(ret_metrics)
        return ret_metrics

    def validate(self, model, valid_data):
        valid_data = self._prepare_inputs(valid_data)
        batch_size = valid_data["input_ids"].shape[0]
        view_size = valid_data["input_ids"].shape[1]

        with torch.no_grad():            
            outputs = model(**valid_data)
            embs = outputs.embeddings        
            embs_wo_pooler = outputs.embs_wo_pooler
        # reshape back to batch_size * view_size
        embs = embs.view(batch_size, view_size, -1)
        # take the first emb in each batch
        anchor_embs = embs[:, 0, :]
        # emb_pool
        emb_pool = embs[:, 1, :]
        # normalize both
        anchor_embs_normalized = F.normalize(anchor_embs, dim=-1)
        emb_pool_normalized = F.normalize(emb_pool, dim=-1)
        sim = torch.matmul(anchor_embs_normalized, emb_pool_normalized.transpose(0, 1))
        rank = []
        for i in range(batch_size):
            rank.append(torch.sum(sim[i, :] > sim[i, i]).item())
        rank_arr = np.array(rank)
        mrr = np.mean(1 / (rank_arr + 1))
        pr1 = np.mean(rank_arr == 0)

        embs_wo_pooler = embs_wo_pooler.view(batch_size, view_size, -1)
        anchor_embs_wo_pooler = embs_wo_pooler[:, 0, :]
        emb_pool_wo_pooler = embs_wo_pooler[:, 1, :]
        anchor_embs_wo_pooler_normalized = F.normalize(anchor_embs_wo_pooler, dim=-1)
        emb_pool_wo_pooler_normalized = F.normalize(emb_pool_wo_pooler, dim=-1)
        sim_wo_pooler = torch.matmul(anchor_embs_wo_pooler_normalized, emb_pool_wo_pooler_normalized.transpose(0, 1))
        rank_wo_pooler = []
        for i in range(batch_size):
            rank_wo_pooler.append(torch.sum(sim_wo_pooler[i, :] > sim_wo_pooler[i, i]).item())
        rank_wo_pooler_arr = np.array(rank_wo_pooler)
        mrr_wo_pooler = np.mean(1 / (rank_wo_pooler_arr + 1))
        pr1_wo_pooler = np.mean(rank_wo_pooler_arr == 0)        
        return pr1, mrr, pr1_wo_pooler, mrr_wo_pooler
        