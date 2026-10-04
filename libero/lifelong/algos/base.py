import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *

REGISTERED_ALGOS = {}

from torch.utils.tensorboard import SummaryWriter


def register_algo(policy_class):
    """Register a policy class with the registry."""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """Get the policy class from the registry."""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all algorithms that should not be registered here.
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class Sequential(nn.Module, metaclass=AlgoMeta):
    """
    The sequential finetuning BC baseline, also the superclass of all lifelong
    learning algorithms.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir
        self.algo = cfg.lifelong.algo

        self.policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        self.current_task = -1

    def end_task(self, dataset, task_id, benchmark, env=None):
        """
        What the algorithm does at the end of learning each lifelong task.
        """
        pass

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        self.optimizer = eval(self.cfg.train.optimizer.name)(
                self.policy.parameters(),
                **self.cfg.train.optimizer.kwargs,
            )

        self.scheduler = None
        
        self.scaler = amp.GradScaler()

        self.summary_writer = SummaryWriter(log_dir=self.experiment_dir+'/tblog/'+str(task))

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)  # detect anomaly       
        with amp.autocast('cuda', dtype=torch.float16):
            loss = self.policy.compute_loss(data)
        # with torch.autograd.detect_anomaly():
        self.scaler.scale(self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
            # print(f'[info] Gradient norm: {grad_norm}')
            # for name, tensor in self.policy.named_parameters():
            #     if tensor.grad is not None:
            #         tensor_grad_norm = torch.norm(tensor.grad)
            #         print(f'[info] {name} grad norm: {tensor_grad_norm}')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.float16):
                loss = self.policy.compute_loss(data)
        return loss.item()

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )
        multiprocessing.set_start_method("fork", force=True)
        if self.cfg.use_ddp:
            torch.distributed.barrier()
            train_dataloader = DataLoader(
                    dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    shuffle=False,
                    sampler=DistributedSampler(dataset),
                    persistent_workers=True,
            )
        else:
            train_dataloader = DataLoader(
                    dataset,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(dataset),
                    persistent_workers=True,
                )

        self.scheduler = None
        try:
            if self.cfg.train.scheduler is not None:
                self.cfg.train.scheduler.kwargs['epochs'] = self.cfg.train.n_epochs
                self.scheduler = eval(self.cfg.train.scheduler.name)(
                    self.optimizer,
                    steps_per_epoch = len(train_dataloader),
                    **self.cfg.train.scheduler.kwargs,
                )
        except:
            pass

        if not self.cfg.use_ddp or int(os.environ["RANK"]) == 0:
            successes = []
            losses = []
            epochs = []
            peak_memories = []
            
            # for evaluate how fast the agent learns on current task, this corresponds
            # to the area under success rate curve on the new task.
            cumulated_counter = 0.0
            idx_at_best_succ = 0

            prev_success_rate = -1.0
            # best_state_dict = self.policy.state_dict()  # currently save the best model

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if self.cfg.use_ddp:
                train_dataloader.sampler.set_epoch(epoch)

            if epoch > 0:  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
            t1 = time.time()

            if torch.cuda.is_available():
                MB = 1024.0 * 1024.0
                peak_memory = (torch.cuda.max_memory_allocated() / MB) / 1000 
            else:
                peak_memory = 0

            if self.cfg.use_ddp:
                training_loss = torch.as_tensor(training_loss, device=torch.device("cuda:"+os.environ["RANK"]))
                peak_memory = torch.as_tensor(peak_memory, device=torch.device("cuda:"+os.environ["RANK"]))
                dataloader_len = torch.as_tensor(len(train_dataloader), device=torch.device("cuda:"+os.environ["RANK"]))
                
                training_loss_gather_list = [torch.zeros_like(training_loss) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                peak_memory_gather_list = [torch.zeros_like(peak_memory) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None
                dataloader_len_gather_list = [torch.zeros_like(dataloader_len) for _ in range(dist.get_world_size())] if int(os.environ["RANK"]) == 0 else None

                dist.gather(training_loss, training_loss_gather_list, dst=0)
                dist.gather(peak_memory, peak_memory_gather_list, dst=0)
                dist.gather(dataloader_len, dataloader_len_gather_list, dst=0)
                
                if int(os.environ["RANK"]) == 0:
                    training_loss = sum(training_loss_gather_list).item() / sum(dataloader_len_gather_list).item()
                    peak_memory = sum(peak_memory_gather_list).item()
                    print(
                        f'[info] # Batch: {sum(dataloader_len_gather_list).item()} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                        f'Memory utilization: %.3f GB' % peak_memory
                    )
            else:
                training_loss /= len(train_dataloader)
                print(
                    f'[info] # Batch: {len(train_dataloader)} | Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f} | '
                    f'Memory utilization: %.3f GB' % peak_memory
                )
            
            if (not self.cfg.use_ddp) or (int(os.environ["RANK"]) == 0):
                self.summary_writer.add_scalar("train_loss", training_loss, epoch)
                # for name, param in self.policy.named_parameters():
                    # if not param.requires_grad:
                        # self.summary_writer.add_histogram(f'value/{name}', param, epoch)
                        # self.summary_writer.add_histogram(f'grad/{name}.grad', param.grad, epoch)
                        # self.summary_writer.add_histogram(f'grad/{name}.grad', param.grad, epoch)

            if epoch % self.cfg.eval.eval_every == 0 and (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):  # evaluate BC loss
                # every eval_every epoch, we evaluate the agent on the current task,
                # then we pick the best performant agent on the current task as
                # if it stops learning after that specific epoch. So the stopping
                # criterion for learning a new task is achieving the peak performance
                # on the new task. Future work can explore how to decide this stopping
                # epoch by also considering the agent's performance on old tasks.
                
                t0 = time.time()

                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str="",
                )
                
                epochs.append(epoch)
                peak_memories.append(peak_memory)
                losses.append(training_loss)
                successes.append(success_rate)
                
                self.summary_writer.add_scalar("success_rate", success_rate, epoch)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                
                t1 = time.time()
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )

            if self.cfg.use_ddp:
                torch.distributed.barrier()
        
        # load the best performance agent on the current task
        if (not self.cfg.debug_no_eval):
            msg = self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0], strict=False)
            # print(f'[info] {msg}')

        torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg, learnable_only=True)
        # end learning the current task, some algorithms need post-processing
        self.end_task(dataset, task_id, benchmark)
        
        if (not self.cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not self.cfg.debug_no_eval):
            # return the metrics regarding forward transfer
            losses = np.array(losses)
            successes = np.array(successes)
            epochs = np.array(epochs)
            peak_memories = np.array(peak_memories)
            auc_checkpoint_name = os.path.join(
                self.experiment_dir, f"task{task_id}_auc.log"
            )
            torch.save(
                {   "epochs": epochs,
                    "success": successes,
                    "loss": losses,
                    "peak_memories": peak_memories,
                },
                auc_checkpoint_name,
            )

            # pretend that the agent stops learning once it reaches the peak performance
            losses[idx_at_best_succ:] = losses[idx_at_best_succ]
            successes[idx_at_best_succ:] = successes[idx_at_best_succ]
            return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter
        else:
            return 0.0, 0.0

    def reset(self):
        self.policy.reset()
