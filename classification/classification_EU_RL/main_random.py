import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
import wandb

from weighted_methods.utils import extract_weight_method_parameters_from_args
from weighted_methods.weight_methods import WeightMethods

# import pruner
from trainer import validate


def canonicalize_method_id(method_id: str) -> str:
    """Canonicalize method names to their canonical forms for unified directory naming."""
    canonical_map = {
        "omd_tch": "omd_tch_eg",
        "afleg": "omd_tch_eg",
        "afl": "omd_tch_pgd",
        "ada_afleg": "ada_omd_tch_eg",
    }
    return canonical_map.get(method_id, method_id)


def evaluate_model_state(model, state_dict, unlearn_data_loaders, criterion, args, device):
    original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(state_dict, strict=False)

    accuracy = {}
    for name, loader in unlearn_data_loaders.items():
        utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args, name, device)
        if name == "forget":
            accuracy[name] = round(100 - val_acc, 2)
        else:
            accuracy[name] = round(val_acc, 2)

    model.load_state_dict(original_state, strict=False)
    return accuracy


def main(args):

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        _,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.to(device)



    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)

    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    
    forget_ratio = len(forget_dataset) / (len(retain_dataset) + len(forget_dataset))
    save_components = [
        args.save_dir,
        args.arch,
        args.dataset,
        "forget_" + str(round(forget_ratio * 100, 2)) + "%",
        args.unlearn,
    ]
    if args.mtl and args.mtl_method is not None:
        canonical_method = canonicalize_method_id(args.mtl_method)
        save_components.append(canonical_method)
    save_components.append(f"{args.wandb_entity}")
    args.save_dir = os.path.join(*save_components)

    if args.path!=None:
        map_ratio=os.path.basename(args.path)
        args.save_dir = os.path.join(args.save_dir,map_ratio.split(".pt")[0])
        args.path = os.path.join(os.path.dirname(args.path),map_ratio)

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, test=test_loader
    )

    # --- Gap 1: forget set composition (logged once at startup) ---
    try:
        _forget_targets = torch.tensor(forget_dataset.targets)
    except AttributeError:
        _forget_targets = torch.tensor(forget_dataset.labels)
    _forget_class_dist = torch.bincount(_forget_targets, minlength=getattr(args, "num_classes", 10)).tolist()
    forget_set_info = {
        "size": len(forget_dataset),
        "retain_size": len(retain_dataset),
        "class_distribution": _forget_class_dist,
    }
    print(f"[SEED {args.seed}] Forget set size: {forget_set_info['size']}, "
          f"class dist: {_forget_class_dist}")

    args.training_log = {
        "method": args.unlearn,
        "seed": args.seed,
        "train_seed": args.train_seed,
        "forget_set_info": forget_set_info,
        "epochs": [],
        "final_validation": {},
    }

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None


    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.path:
            mask = torch.load(args.path, map_location=device)
        else:
            mask = None

        if args.unlearn != "retrain":
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        if wandb.run is not None:
            wandb.watch(model, log="all")

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        if args.mtl:
            # weight method
            weight_methods_parameters = extract_weight_method_parameters_from_args(args)
            weight_method = WeightMethods(args.mtl_method, n_tasks=2, device=device, **weight_methods_parameters[args.mtl_method])
            unlearn_method(unlearn_data_loaders, model, criterion, args, mask, device, weight_method)
        else:
            unlearn_method(unlearn_data_loaders, model, criterion, args, mask, device)
            

        unlearn.save_unlearn_checkpoint(model, None, args)


    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        # --- Gap 3: verify forget set targets are original (non-negative) at eval time ---
        _eval_forget_loader = unlearn_data_loaders["forget"]
        try:
            _eval_forget_targets = np.array(_eval_forget_loader.dataset.targets)
        except AttributeError:
            _eval_forget_targets = np.array(_eval_forget_loader.dataset.labels)
        _negative_count = int((_eval_forget_targets < 0).sum())
        print(f"[SEED {args.seed}] Forget targets at eval: "
              f"{_negative_count} negative (should be 0), "
              f"min={_eval_forget_targets.min()}, max={_eval_forget_targets.max()}")
        if _negative_count > 0:
            print(f"  WARNING: forget dataset still contains {_negative_count} marked (negative) targets at eval time!")
        args.training_log["forget_target_integrity"] = {
            "negative_count_at_eval": _negative_count,
            "target_min": int(_eval_forget_targets.min()),
            "target_max": int(_eval_forget_targets.max()),
        }

        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_metrics = validate(
                loader, model, criterion, args, name, device, return_metrics=True
            )
            val_acc = float(val_metrics["accuracy"])
            val_loss = float(val_metrics["loss"])

            if name == "forget":
                accuracy[name] = round(100 - val_acc, 2)
            else:
                accuracy[name] = round(val_acc, 2)
            args.training_log["final_validation"][name] = {
                "accuracy": val_acc,
                "loss": val_loss,
                "reported_accuracy": accuracy[name],
            }
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy

        if args.mtl and getattr(args, "mtl_method", None) in {"omd_tch_eg", "omd_tch_pgd", "afleg", "afl"} and getattr(args, "omd_tch_avg_state", None) is not None:
            avg_accuracy = evaluate_model_state(
                model,
                args.omd_tch_avg_state,
                unlearn_data_loaders,
                criterion,
                args,
                device,
            )
            evaluation_result["avg_accuracy"] = avg_accuracy

        if args.mtl and getattr(args, "mtl_method", None) in {"ada_omd_tch_eg", "ada_afleg"} and getattr(args, "ada_omd_result_state", None) is not None:
            adaptive_accuracy = evaluate_model_state(
                model,
                args.ada_omd_result_state,
                unlearn_data_loaders,
                criterion,
                args,
                device,
            )
            evaluation_result["adaptive_accuracy"] = adaptive_accuracy

        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if not args.training_log["final_validation"]:
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_metrics = validate(
                loader, model, criterion, args, name, device, return_metrics=True
            )
            val_acc = float(val_metrics["accuracy"])
            val_loss = float(val_metrics["loss"])
            reported_accuracy = (
                round(100 - val_acc, 2) if name == "forget" else round(val_acc, 2)
            )
            args.training_log["final_validation"][name] = {
                "accuracy": val_acc,
                "loss": val_loss,
                "reported_accuracy": reported_accuracy,
            }

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
            device=device
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    # SVC_MIA on OMD-TCH averaged model
    for avg_key, state_attr, result_key in [
        ("omd_tch", "omd_tch_avg_state", "avg_SVC_MIA_forget_efficacy"),
        ("ada_omd_tch", "ada_omd_result_state", "adaptive_SVC_MIA_forget_efficacy"),
    ]:
        avg_state = getattr(args, state_attr, None) if getattr(args, "mtl", False) else None
        if avg_state is not None and result_key not in evaluation_result:
            original_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            model.load_state_dict(avg_state, strict=False)

            evaluation_result[result_key] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=forget_loader,
                model=model,
                device=device
            )

            model.load_state_dict(original_state, strict=False)
            unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    # if "SVC_MIA_training_privacy" not in evaluation_result:
    #     test_len = len(test_loader.dataset)
    #     retain_len = len(retain_dataset)
    #     num = test_len // 2
    
    #     utils.dataset_convert_to_test(retain_dataset, args)
    #     utils.dataset_convert_to_test(forget_loader, args)
    #     utils.dataset_convert_to_test(test_loader, args)
    
    #     shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
    #     target_train = torch.utils.data.Subset(retain_dataset, list(range(num, retain_len)))
    #     shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
    #     target_test = torch.utils.data.Subset(test_loader.dataset, list(range(num, test_len)))
    
    #     shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=args.batch_size, shuffle=False)
    #     shadow_test_loader = torch.utils.data.DataLoader(shadow_test, batch_size=args.batch_size, shuffle=False)
    #     target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=args.batch_size, shuffle=False)
    #     target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=args.batch_size, shuffle=False)
    
    #     evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader,
    #         shadow_test=shadow_test_loader,
    #         target_train=target_train_loader,
    #         target_test=target_test_loader,
    #         model=model,
    #         device=device
    #     )
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    unlearn.save_training_log(args.training_log, args)

    if wandb.run is not None:

        wandb.log({"Retain Acc (RA)": evaluation_result["accuracy"]["retain"]})
        wandb.log({"Unlearn Acc (UA)": evaluation_result["accuracy"]["forget"]})
        wandb.log({"Test Acc (TA)": evaluation_result["accuracy"]["test"]})

        wandb.log({"Forget Correctness": evaluation_result["SVC_MIA_forget_efficacy"]["correctness"]})
        wandb.log({"Forget Confidence (MIA)": evaluation_result["SVC_MIA_forget_efficacy"]["confidence"]})
        wandb.log({"Forget Entropy": evaluation_result["SVC_MIA_forget_efficacy"]["entropy"]})
        wandb.log({"Forget M_Entropy": evaluation_result["SVC_MIA_forget_efficacy"]["m_entropy"]})
        wandb.log({"Forget Prob": evaluation_result["SVC_MIA_forget_efficacy"]["prob"]})

        for result_key, prefix in [
            ("avg_SVC_MIA_forget_efficacy", "Avg"),
            ("adaptive_SVC_MIA_forget_efficacy", "Adaptive"),
        ]:
            if result_key in evaluation_result:
                wandb.log({f"{prefix} Forget Correctness": evaluation_result[result_key]["correctness"]})
                wandb.log({f"{prefix} Forget Confidence (MIA)": evaluation_result[result_key]["confidence"]})
                wandb.log({f"{prefix} Forget Entropy": evaluation_result[result_key]["entropy"]})
                wandb.log({f"{prefix} Forget M_Entropy": evaluation_result[result_key]["m_entropy"]})
                wandb.log({f"{prefix} Forget Prob": evaluation_result[result_key]["prob"]})

        print("Finish Wandb Login")

    print("############################Final Result################################")
    for name, result in evaluation_result.items():
        print(name, ":", result)
        print("----------------------------------------")
    print("########################################################################")



if __name__ == "__main__":
    args = arg_parser.parse_args()
    if args.wandb_project is not None:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_entity, config=args)

    main(args)

    if wandb.run is not None:
        wandb.finish()
