import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['MUJOCO_EGL_DEVICE_ID'] = "0"
# os.environ["WANDB_MODE"] = "offline"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import sys
import json
import multiprocessing
import pprint
import time
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import copy

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import get_dataset, GroupedTaskDataset, SequenceVLDataset, SkillLearningDataset, MetaPolicyDataset, MetaPolicySequenceDataset
from libero.lifelong.metric import evaluate_loss, evaluate_success, confidence_interval, evaluate_multitask_training_success
from libero.lifelong.utils import (
    NpEncoder,
    count_policy_parameters,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)

import glob
import h5py
from libero.lifelong.algos.lotus_skill_learning.models.model_utils import safe_cuda
from libero.lifelong.algos.lotus_skill_learning.models.conf_utils import *
from libero.lifelong.algos.lotus_skill_learning.models.torch_utils import *
from libero.lifelong.algos.lotus import SubSkill, MetaController
from libero.lifelong.models.bc_hierarchical_policy.cvae_policy import MaskingLayer
from libero.lifelong.algos.lotus_skill_learning.utils import get_subtask_label, save_subgoal_embedding


class CustomDDP(DDP):
    """
    The default DistributedDataParallel enforces access to class the module attributes via self.module. 
    This is impractical for our use case, as we need to access certain module access throughout. 
    We override the __getattr__ method to allow access to the module attributes directly.
    
    For example: 
    ```
        # default behaviour
        model = OnlineDecisionTransformerModel()
        model = DistributedDataParallel(model)
        model.module.some_attribute
        
        # custom behaviour using this class
        model = OnlineDecisionTransformerModel()
        model = CustomDDP(model)
        model.some_attribute
        
    ```        
    Shoudl not cause any inconsistencies: 
    https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572
    
    """
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
                demos=range(cfg.data.n_demos_per_task),
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    gsz = cfg.data.task_group_size
    if gsz == 1:  # each manipulation task is its own lifelong learning task
        datasets = []
        for i in range(n_manip_tasks):
            datasets.append(SequenceVLDataset(manip_datasets[i], task_embs[i]))
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # group gsz manipulation tasks into a lifelong task, currently not used
        raise NotImplementedError
    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks

    # prepare experiment and update the config
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta
    
    if not cfg.use_ddp or int(os.environ["RANK"]) == 0:
        print("\n=================== Lifelong Benchmark Information  ===================")
        print(f" Name: {benchmark.name}")
        print(f" # Tasks: {n_manip_tasks // gsz}")
        for i in range(n_tasks):
            print(f"    - Task {i+1}:")
            for j in range(gsz):
                print(f"        {benchmark.get_task(i*gsz+j).language}")
        print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
        print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
        print("=======================================================================\n")
    
    if cfg.use_ddp:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))

    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
        "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
        "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    }

    if cfg.eval.save_sim_states:
        # for saving the evaluate simulation states, so we can replay them later
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # for testing task p when the agent learns to task k
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # for testing task k at the e-th epoch when the agent learns on task k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]
    
    # define lifelong algorithm
    if cfg.lifelong.algo != "LOTUS" and cfg.lifelong.algo != "DMPEL":
        algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
        if cfg.use_ddp:
            device = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(device)
            algo = CustomDDP(algo.to(device), device_ids=[device], output_device=device)
        else:
            algo = safe_device(algo, cfg.device)

        if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
            try:
                sd = torch_load_model(cfg.pretrain_model_path)[0]
                msg = algo.policy.load_state_dict(sd, strict=False)
                print(msg)
            except:
                print(
                    f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
                )
                sys.exit(0)

    elif cfg.lifelong.algo == "DMPEL":
        algo = get_algo_class(cfg.lifelong.algo)(n_tasks, cfg)
        if cfg.pretrain_model_path != "":
            try:
                sd = torch_load_model(cfg.pretrain_model_path)[0]
                msg = algo.policy.load_state_dict(sd, strict=False)
                print(msg)
            except:
                print(
                    f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
                )
                sys.exit(0)
            algo.policy.init_moe_policy()
        else:
            algo.policy.init_moe_policy()
        if cfg.use_ddp:
            device = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(device)
            algo = CustomDDP(algo.to(device), device_ids=[device], output_device=device)
        else:
            algo = safe_device(algo, cfg.device)
        
        print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    else:
        pass
        
    if not cfg.use_ddp or int(os.environ["RANK"]) == 0:
        _, _, _ = count_policy_parameters(algo, cfg)
        # save the experiment config file, so we can resume or replay later
        with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
            json.dump(cfg, f, cls=NpEncoder, indent=4)
    
    if cfg.lifelong.algo == "Multitask":
        algo.train()
        s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary)
        result_summary["L_fwd"][-1] = l_fwd
        result_summary["S_fwd"][-1] = s_fwd

        torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
    
    elif cfg.lifelong.algo == "LOTUS":
        if cfg.lifelong.phase == 'base':
            task_names = benchmark.get_task_names()
            skill_learning_cfg = cfg.lifelong.lotus
            skill_exp_name = skill_learning_cfg.exp_name
            exp_dir = f"path_to_dataset/lotus/{skill_exp_name}/skill_data" # TODO: replace path_to_dataset
            data_file_name_list = []
            subtasks_file_name_list = []
            used_data_file_name_list_skill = task_names #[]
            used_data_file_name_list_meta = task_names
            for dataset_category in os.listdir(exp_dir):
                dataset_category_path = os.path.join(exp_dir, dataset_category)
                if os.path.isdir(dataset_category_path) and dataset_category in ['libero_object','libero_spatial','libero_goal', "libero_10", "libero_90", "rw_all"]:
                    for dataset_name in os.listdir(dataset_category_path):
                        dataset_name = dataset_name.split("_demo_")[0] + '_demo'
                        data_file_name_list.append(f"path_to_dataset/{dataset_category}/{dataset_name}.hdf5")
                        file_pattern = f"path_to_dataset/lotus/{skill_exp_name}/skill_data/{dataset_category}/{dataset_name}*"
                        matching_files = glob.glob(file_pattern)
                        assert len(matching_files)==1
                        subtasks_file_name_list.append(matching_files[0])

            skill_dataset = SkillLearningDataset(data_file_name_list=data_file_name_list,
                                        subtasks_file_name_list=subtasks_file_name_list,
                                        subtask_id=[],
                                        data_modality=skill_learning_cfg.skill_training.data_modality,
                                        use_eye_in_hand=skill_learning_cfg.skill_training.use_eye_in_hand,
                                        subgoal_cfg=skill_learning_cfg.skill_subgoal_cfg,
                                        seq_len=cfg.data.seq_len,
                                        task_embs=task_embs,
                                        used_data_file_name_list=used_data_file_name_list_skill
                                        )
            # skill policy training
            skill_policies = {}
            for i in range(skill_dataset.num_subtasks):
                dataset = skill_dataset.get_dataset(idx=i)
                print(f"Subtask id: {i}")
                sub_skill_policy = safe_device(SubSkill(n_tasks, cfg), cfg.device)
                params = sum(p.numel() for p in sub_skill_policy.policy.parameters() if p.requires_grad)
                trainable_params = params / 1e6
                print(f"[info] policy has {trainable_params:.1f} M Trainable Params\n")
                if dataset is None:
                    print(f"No Data on Subtask {i}")
                else:
                    if i in skill_dataset.train_dataset_id:
                        sub_skill_policy.train()
                        loss = sub_skill_policy.learn_one_skill(dataset, benchmark, result_summary, i)
                skill_policies[i] = sub_skill_policy.policy
                del sub_skill_policy

            # set eval mode for all skill policies
            for skill_policy in skill_policies.values():
                skill_policy.eval()
            
            del skill_policy
            del skill_dataset
            del dataset

            # save the subgoal embedding
            save_subgoal_embedding(cfg, skill_policies, data_file_name_list, skill_learning_cfg, skill_exp_name)

            # meta policy training

            meta_dataset = MetaPolicySequenceDataset(data_file_name_list=data_file_name_list,
                                            embedding_file_name=os.path.join(cfg.experiment_dir, f"subgoal_embedding.hdf5"),
                                            subtasks_file_name_list=subtasks_file_name_list,
                                            use_eye_in_hand=skill_learning_cfg.meta.use_eye_in_hand,
                                            task_names = task_names, # include task order infos
                                            task_embs=task_embs,
                                            new_task_name="@default@",
                                            demo_range=range(0, 50),
                                            used_data_file_name_list=used_data_file_name_list_meta)

            cfg.lifelong.lotus.num_subtasks = meta_dataset.num_subtasks
            cfg.lifelong.lotus.subgoal_embedding_dim = meta_dataset.subgoal_embedding_dim
            # save the experiment config file, so we can resume or replay later
            with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
                json.dump(cfg, f, cls=NpEncoder, indent=4)
            meta_policy = safe_device(MetaController(n_tasks, cfg, skill_policies), cfg.device)

            # GFLOPs, MParams = compute_flops(meta_policy, meta_dataset, cfg)
            # print(f"[info] Subskill policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")
            params = sum(p.numel() for p in meta_policy.policy.parameters() if p.requires_grad)
            trainable_params = params / 1e6
            print(f"[info] policy has {trainable_params:.1f} M Trainable Params\n")

            meta_policy.train()
            if cfg.pretrain_model_path != "":
                meta_policy.load_meta_policy(experiment_dir=cfg.pretrain_model_path)

            s_fwd, l_fwd, kl_loss, ce_loss, embedding_loss = meta_policy.learn_multi_task(meta_dataset, benchmark, result_summary)
            result_summary["L_fwd"][-1] = l_fwd
            result_summary["S_fwd"][-1] = s_fwd

            # evalute on all seen tasks at the end if eval.eval is true
            if cfg.eval.eval:
                all_tasks = list(range(benchmark.n_tasks))
                success_rates = evaluate_multitask_training_success(
                            cfg, meta_policy, benchmark, all_tasks
                        )
                result_summary["S_conf_mat"][-1] = success_rates
                print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(success_rates))

                torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
        elif cfg.lifelong.phase == 'lifelong':

            for i in range(n_tasks):
                print(f"[info] start training on task {i}")
                t0 = time.time()
                benchmark_name = benchmark.name
                task_description = benchmark.get_task(i).language
                print(f"Task description: {benchmark_name} - {task_description}")

                task_names = benchmark.get_task_names()[:i+1]
                task_embs = benchmark.task_embs[:i+1]

                skill_learning_cfg = cfg.lifelong.lotus
                skill_exp_name = skill_learning_cfg.exp_name
                exp_dir = f"path_to_dataset/lotus/{skill_exp_name}/task_" + str(91+i) + "/skill_data"
                data_file_name_list = []
                subtasks_file_name_list = []
                used_data_file_name_list_skill = task_names #[]
                used_data_file_name_list_meta = task_names
                for dataset_category in os.listdir(exp_dir):
                    dataset_category_path = os.path.join(exp_dir, dataset_category)
                    if os.path.isdir(dataset_category_path) and dataset_category in ['libero_object','libero_spatial','libero_goal', "libero_10", "libero_90", "rw_all"]:
                        for dataset_name in os.listdir(dataset_category_path):
                            dataset_name = dataset_name.split("_demo_")[0] + '_demo'
                            data_file_name_list.append(f"path_to_dataset/{dataset_category}/{dataset_name}.hdf5")
                            file_pattern = f"path_to_dataset/lotus/{skill_exp_name}/task_" + str(91+i) + "/skill_data" + f"/{dataset_category}/{dataset_name}*"
                            matching_files = glob.glob(file_pattern)
                            assert len(matching_files)==1
                            subtasks_file_name_list.append(matching_files[0])

                if i > 0:
                    cfg = copy.deepcopy(cfg_for_skill)
                skill_dataset = SkillLearningDataset(data_file_name_list=data_file_name_list,
                                            subtasks_file_name_list=subtasks_file_name_list,
                                            subtask_id=[],
                                            data_modality=skill_learning_cfg.skill_training.data_modality,
                                            use_eye_in_hand=skill_learning_cfg.skill_training.use_eye_in_hand,
                                            subgoal_cfg=skill_learning_cfg.skill_subgoal_cfg,
                                            seq_len=cfg.data.seq_len,
                                            new_task_name=task_description.replace(" ", "_"),
                                            task_embs=task_embs,
                                            used_data_file_name_list=used_data_file_name_list_skill
                                            )
                
                # skill policy training
                skill_policies = {}
                for j in range(skill_dataset.num_subtasks):
                    dataset = skill_dataset.get_dataset(idx=j)
                    print(f"Subtask id: {j}")
                    sub_skill_policy = safe_device(SubSkill(n_tasks, cfg), cfg.device)
                    if i == 0:
                        sub_skill_policy.load_skill(skill_id=j, experiment_dir=cfg.pretrain_model_path)
                    else:
                        sub_skill_policy.load_skill(skill_id=j, experiment_dir=cfg.experiment_dir)
                    # sub_skill_policy.eval()
                    if dataset is None:
                        print(f"No Data on Subtask {j}")
                    else:
                        if j in skill_dataset.train_dataset_id:
                            sub_skill_policy.train()
                            loss = sub_skill_policy.learn_one_skill(dataset, benchmark, result_summary, j)
                    skill_policies[j] = sub_skill_policy.policy
                    del sub_skill_policy

                # set eval mode for all skill policies
                for skill_policy in skill_policies.values():
                    skill_policy.eval()
                
                del skill_policy
                del skill_dataset
                del dataset

                # save the subgoal embedding
                save_subgoal_embedding(cfg, skill_policies, data_file_name_list, skill_learning_cfg, skill_exp_name, i)
                cfg_for_skill = copy.deepcopy(cfg)
                # meta policy training
                meta_dataset = MetaPolicySequenceDataset(data_file_name_list=data_file_name_list,
                                                        embedding_file_name=os.path.join(cfg.experiment_dir, f"subgoal_embedding_{i}.hdf5"),
                                                        subtasks_file_name_list=subtasks_file_name_list,
                                                        use_eye_in_hand=skill_learning_cfg.meta.use_eye_in_hand,
                                                        task_names = task_names, # include task order infos
                                                        task_embs=task_embs,
                                                        new_task_name=task_description.replace(" ", "_"),
                                                        demo_range=range(0, 50),
                                                        used_data_file_name_list=used_data_file_name_list_meta)

                cfg.lifelong.lotus.num_subtasks = meta_dataset.num_subtasks
                cfg.lifelong.lotus.subgoal_embedding_dim = meta_dataset.subgoal_embedding_dim
                
                meta_policy = safe_device(MetaController(n_tasks, cfg, skill_policies), cfg.device)
                if i == 0:
                    meta_policy.load_meta_policy(cfg.pretrain_model_path + "/meta_controller_model_ep10.pth")
                    meta_policy.policy.meta_id_layer.max_subtasks_num = 30
                    old_linear_layer = meta_policy.policy.meta_id_layer._id_layers[-3]
                    old_linear_layer_weight = old_linear_layer.weight.data.clone()
                    old_linear_layer_bias = old_linear_layer.bias.data.clone()
                    new_linear_layer = torch.nn.Linear(old_linear_layer.in_features, 30)
                    new_linear_layer.weight.data[:20, :] = old_linear_layer_weight
                    new_linear_layer.bias.data[:20] = old_linear_layer_bias
                    meta_policy.policy.meta_id_layer._id_layers[-3] = new_linear_layer
                    num_subtasks = meta_policy.policy.meta_id_layer.num_subtasks
                    meta_policy.policy.meta_id_layer._id_layers[-2] = MaskingLayer(30, num_subtasks)
                    meta_policy.policy.meta_id_layer.id_layers = torch.nn.Sequential(*meta_policy.policy.meta_id_layer._id_layers)
                else:
                    meta_policy.load_meta_policy_with_id(experiment_dir=cfg.experiment_dir, task_id=i-1)
                
                params = sum(p.numel() for p in meta_policy.policy.parameters() if p.requires_grad)
                trainable_params = params / 1e6
                print(f"[info] policy has {trainable_params:.2f} M Trainable Params\n")
                meta_policy.train()
                s_fwd, l_fwd, kl_loss, ce_loss, embedding_loss = meta_policy.learn_one_task(meta_dataset, i, benchmark, result_summary)
                result_summary["S_fwd"][i] = s_fwd
                result_summary["L_fwd"][i] = l_fwd
                t1 = time.time()
                # evalute on all seen tasks at the end of learning each task
                if cfg.eval.eval:

                    S = evaluate_success(
                            cfg=cfg,
                            algo=meta_policy,
                            benchmark=benchmark,
                            task_ids=list(range((i + 1) * gsz)),
                            result_summary=result_summary if cfg.eval.save_sim_states else None,
                        )
                    t3 = time.time()
                    result_summary["S_conf_mat"][i][: i + 1] = S

                    print(
                        f"[info] train time (min) {(t1-t0)/60:.1f} "
                        + f"eval success time {(t3-t1)/60:.1f}"
                    )
                    print(("[Task %2d succ.] " + " %4.2f |" * (i + 1)) % (i, *S))
            torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
        else:
            raise NotImplementedError
    else:
        for i in range(n_tasks):
            if not cfg.use_ddp or int(os.environ["RANK"]) == 0:
                print(f"[info] start training on task {i}", flush=True)
            
            algo.train()
            t0 = time.time()
            
            s_fwd, l_fwd = algo.learn_one_task(
                datasets[i], i, benchmark, result_summary
            )

            if not cfg.use_ddp or int(os.environ["RANK"]) == 0:
                result_summary["S_fwd"][i] = s_fwd
                result_summary["L_fwd"][i] = l_fwd
                t1 = time.time()

            # evalute on all seen tasks at the end of learning each task
            if cfg.eval.eval and (not cfg.use_ddp or int(os.environ["RANK"]) == 0) and (not cfg.debug_no_eval):
                L = evaluate_loss(cfg, algo, benchmark, datasets[: i + 1])
                t2 = time.time()
                S = evaluate_success(
                    cfg=cfg,
                    algo=algo,
                    benchmark=benchmark,
                    task_ids=list(range((i + 1) * gsz)),
                    result_summary=result_summary if cfg.eval.save_sim_states else None,
                )
                t3 = time.time()
                result_summary["L_conf_mat"][i][: i + 1] = L
                result_summary["S_conf_mat"][i][: i + 1] = S

                print(
                    f"[info] train time (min) {(t1-t0)/60:.1f} "
                    + f"eval loss time {(t2-t1)/60:.1f} "
                    + f"eval success time {(t3-t2)/60:.1f}"
                )
                print(("[Task %2d loss ] " + " %4.2f |" * (i + 1)) % (i, *L))
                print(("[Task %2d succ.] " + " %4.2f |" * (i + 1)) % (i, *S))
                torch.save(
                    result_summary, os.path.join(cfg.experiment_dir, f"result.pt")
                )
    
    print("[info] finished learning\n")
    
    if cfg.use_ddp: 
        dist.destroy_process_group()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":  
    #     multiprocessing.set_start_method("spawn", force=True)
    main()
