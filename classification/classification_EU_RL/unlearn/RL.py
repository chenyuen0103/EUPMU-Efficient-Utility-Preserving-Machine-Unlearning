import pdb
import os
import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .impl import iterative_unlearn, update_omd_tch_average, update_ada_omd_marked_state
import wandb
import random
from torch.utils.data import Dataset, Subset


def _resolve_targets_owner(dataset):
    if hasattr(dataset, "targets"):
        return dataset, "targets"
    if hasattr(dataset, "labels"):
        return dataset, "labels"
    if hasattr(dataset, "dataset"):
        nested_dataset = dataset.dataset
        if hasattr(nested_dataset, "targets"):
            return nested_dataset, "targets"
        if hasattr(nested_dataset, "labels"):
            return nested_dataset, "labels"
    raise AttributeError("Could not find targets/labels on forget dataset")


def _targets_to_numpy(targets):
    if torch.is_tensor(targets):
        return targets.detach().cpu().numpy().copy()
    return np.array(targets, copy=True)


def _log_rl_label_debug(args, epoch, true_targets, random_targets):
    prev_random_targets = getattr(args, "_prev_rl_random_targets", None)
    sample_count = min(getattr(args, "rl_label_debug_samples", 0), len(random_targets))

    if prev_random_targets is not None and len(prev_random_targets) == len(random_targets):
        changed_ratio = float(np.mean(random_targets != prev_random_targets))
        print(
            f"[SEED {args.seed} | Epoch {epoch}] RL labels changed vs prev epoch: "
            f"{changed_ratio:.4f}"
        )
    else:
        changed_ratio = float("nan")

    if sample_count > 0:
        sample_indices = np.linspace(
            0, len(random_targets) - 1, num=sample_count, dtype=int
        )
        preview = []
        for idx in sample_indices:
            item = f"{idx}: {int(true_targets[idx])}->{int(random_targets[idx])}"
            if prev_random_targets is not None and len(prev_random_targets) == len(random_targets):
                item += f" (prev {int(prev_random_targets[idx])})"
            preview.append(item)
        print(
            f"[SEED {args.seed} | Epoch {epoch}] RL label sample: "
            + ", ".join(preview)
        )

    if getattr(args, "rl_label_debug_dump", False):
        debug_dir = os.path.join(args.save_dir, "rl_label_debug")
        os.makedirs(debug_dir, exist_ok=True)
        dump_path = os.path.join(debug_dir, f"epoch_{epoch:03d}.npz")
        np.savez_compressed(
            dump_path,
            forget_index=np.arange(len(random_targets), dtype=np.int64),
            true_targets=true_targets,
            random_targets=random_targets,
            prev_random_targets=prev_random_targets
            if prev_random_targets is not None and len(prev_random_targets) == len(random_targets)
            else np.full_like(random_targets, -1),
            changed_from_prev=random_targets != prev_random_targets
            if prev_random_targets is not None and len(prev_random_targets) == len(random_targets)
            else np.zeros(len(random_targets), dtype=bool),
        )
        print(f"[SEED {args.seed} | Epoch {epoch}] Saved RL label dump to {dump_path}")

    setattr(args, "_prev_rl_random_targets", random_targets.copy())


@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, mask=None, device=None, weight_method=None):
    forget_loader = data_loaders["forget"]
    retain_loader = deepcopy(data_loaders["retain"])
    forget_dataset = deepcopy(forget_loader.dataset)

    # if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "TinyImagenet":
    if True:

        _forget_loss_type = getattr(args, "forget_loss_type", "rl")

        if _forget_loss_type == "rl":
            _targets_owner, _targets_attr = _resolve_targets_owner(forget_dataset)
            _true_targets = _targets_to_numpy(getattr(_targets_owner, _targets_attr))
            _new_random_targets = np.random.randint(
                0, args.num_classes, size=_true_targets.shape
            )
            setattr(_targets_owner, _targets_attr, _new_random_targets)
            _random_targets = _targets_to_numpy(getattr(_targets_owner, _targets_attr))

            # --- Gap 2: random-label / true-label overlap (key RL diagnostic) ---
            _overlap = float(np.mean(_random_targets == _true_targets))
            _expected_overlap = 1.0 / args.num_classes
            print(f"[SEED {args.seed} | Epoch {epoch}] RL label overlap: "
                  f"{_overlap:.4f} (expected ~{_expected_overlap:.4f}, "
                  f"ratio={_overlap/_expected_overlap:.2f}x)")
            _log_rl_label_debug(args, epoch, _true_targets, _random_targets)
        else:
            # GA mode: keep true labels; overlap metric is not meaningful
            _overlap = float("nan")
            _expected_overlap = 1.0 / args.num_classes
            print(f"[SEED {args.seed} | Epoch {epoch}] forget_loss_type=ga: using gradient ascent on true labels")

        # if  args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "TinyImagenet":
        #     try:
        #         forget_dataset.targets = (forget_dataset.targets + np.random.randint(1, args.num_classes, forget_dataset.targets.shape)) % args.num_classes
        #     except:
        #         print(forget_dataset.dataset.targets[:10])
        #         forget_dataset.dataset.targets = (forget_dataset.dataset.targets + np.random.randint(1, args.num_classes, forget_dataset.dataset.targets.shape)) % args.num_classes
        #         print(forget_dataset.dataset.targets[:10])
        # else:
        #     forget_dataset.labels = (forget_dataset.labels + np.random.randint(1, args.num_classes, forget_dataset.labels.shape)) % args.num_classes


        retain_dataset = retain_loader.dataset
        _needs_split_labels = args.retainwithAllParamUpdate or args.mtl or _forget_loss_type == "ga"

        if _needs_split_labels:
            try:
                forget_dataset.targets = list(zip(forget_dataset.targets, len(forget_dataset) * ["forget"]))
                retain_dataset.targets = list(zip(retain_dataset.targets, len(retain_dataset) * ["retain"]))
            except:
                forget_dataset.temp = len(forget_dataset) * ["forget"]
                retain_dataset.temp = len(retain_dataset) * ["retain"]

        def _scaled_subset_loss(logits, labels, indexes, batch_size):
            if len(indexes) == 0:
                return logits.new_zeros(())
            return criterion(logits[indexes], labels[indexes]) * (len(indexes) / batch_size)

        if args.only_trainForgetSet:
            train_dataset = forget_dataset
        elif args.only_trainForgetSet_and_samesizeOfretain:
            indices = random.sample(range(len(retain_dataset)), len(forget_dataset))
            subset_retain_dataset = Subset(retain_dataset, indices)
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, subset_retain_dataset])
        else:
            train_dataset = torch.utils.data.ConcatDataset([forget_dataset, retain_dataset])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        grad_norms = utils.AverageMeter()

        # switch to train mode
        model.train()

        start = time.time()
        if args.only_trainForgetSet:
            loader_len = len(forget_loader)
        elif args.only_trainForgetSet_and_samesizeOfretain:
            loader_len = len(forget_loader) * 2
        else:
            loader_len = len(forget_loader) + len(retain_loader)
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i + 1, optimizer,
                            one_epoch_step=loader_len, args=args)

        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.to(device)

            if _needs_split_labels:
                target_label=target[1]
                target=target[0]
            target = target.to(device)

            if args.arch == "clip":
                image = preprocess(image)
                # Calculate features
                image_features = model.encode_image(image)
                with torch.no_grad():
                    text_features = model.encode_text(text_inputs)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                output_clean = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            else:
                # compute output
                output_clean = model(image)

            if args.mtl:
                optimizer.zero_grad()
                retain_indexes = [index for index, value in enumerate(target_label) if value == "retain"]
                forget_indexes = [index for index, value in enumerate(target_label) if value == "forget"]
                loss_retain = _scaled_subset_loss(output_clean, target, retain_indexes, len(target_label))
                _raw_loss_forget = _scaled_subset_loss(output_clean, target, forget_indexes, len(target_label))
                loss_forget = -_raw_loss_forget if _forget_loss_type == "ga" else _raw_loss_forget

                shared_parameters = [param for param in model.parameters() if param.requires_grad]
                if hasattr(weight_method.method, "set_task_gradients"):
                    task_grads = []
                    for task_loss in (loss_retain, loss_forget):
                        grads = torch.autograd.grad(
                            task_loss,
                            shared_parameters,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        flat_grads = []
                        for param, grad in zip(shared_parameters, grads):
                            if grad is None:
                                flat_grads.append(torch.zeros_like(param).reshape(-1))
                            else:
                                flat_grads.append(grad.detach().reshape(-1))
                        task_grads.append(torch.cat(flat_grads))
                    weight_method.method.set_task_gradients(torch.stack(task_grads, dim=0))

                loss, extra_outputs = weight_method.backward(
                    losses=torch.stack([loss_retain, loss_forget]),
                    shared_parameters=list(model.parameters()),
                )

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                _gn = sum(
                    p.grad.detach().norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                grad_norms.update(_gn, 1)

                update_ada_omd_marked_state(args, model, [loss_retain.detach(), loss_forget.detach()])
                # The OMD-TCH paper averages optimization iterates over rounds; we therefore
                # snapshot the current iterate before each optimizer step.
                update_omd_tch_average(args, model)
                optimizer.step()

                if ("famo" in args.mtl_method):
                    with torch.no_grad():
                        output_clean_ = model(image)
                        loss_retain = criterion(output_clean_[retain_indexes], target[retain_indexes]) * (
                                    len(retain_indexes) / len(target_label))
                        _raw_lf = criterion(output_clean_[forget_indexes], target[forget_indexes]) * (
                                    len(forget_indexes) / len(target_label))
                        loss_forget = -_raw_lf if _forget_loss_type == "ga" else _raw_lf
                        new_losses = torch.stack(
                            (
                                loss_retain,
                                loss_forget,
                            )
                        )
                        weight_method.method.update(new_losses.detach())
                if ("eu" == args.mtl_method):
                    with torch.no_grad():
                        output_clean_ = model(image)
                        loss_retain2 = criterion(output_clean_[retain_indexes], target[retain_indexes]) * (
                                    len(retain_indexes) / len(target_label))
                        weight_method.method.update(loss_retain2.detach())
                        if wandb.run is not None:
                            wandb.log({"EU_weight": weight_method.method.w})
                            wandb.log({"retain_loss": loss_retain})
                            wandb.log({"forget_loss": loss_forget})

                # log OMD-TCH simplex weights
                if extra_outputs is not None and "updated_omd_weights" in extra_outputs:
                    if wandb.run is not None:
                        with torch.no_grad():
                            wandb.log({
                                "omd/weight_retain": extra_outputs["updated_omd_weights"][0].item(),
                                "omd/weight_forget": extra_outputs["updated_omd_weights"][1].item()
                            })
                            wandb.log({
                                "retain_loss": loss_retain.item(),
                                "forget_loss": loss_forget.item()
                            })



            elif args.retainwithAllParamUpdate:

                retain_indexes = [index for index, value in enumerate(target_label) if value == "retain"]
                forget_indexes = [index for index, value in enumerate(target_label) if value == "forget"]
                loss_retain = _scaled_subset_loss(output_clean, target, retain_indexes, len(target_label))
                _raw_loss_forget = _scaled_subset_loss(output_clean, target, forget_indexes, len(target_label))
                loss_forget = -_raw_loss_forget if _forget_loss_type == "ga" else _raw_loss_forget

                optimizer.zero_grad()
                loss_forget.backward(retain_graph=True)

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                loss_retain.backward()

                _gn = sum(
                    p.grad.detach().norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                grad_norms.update(_gn, 1)

                optimizer.step()
                loss =  loss_forget+loss_retain
            else:
                if _forget_loss_type == "ga":
                    retain_indexes = [index for index, value in enumerate(target_label) if value == "retain"]
                    forget_indexes = [index for index, value in enumerate(target_label) if value == "forget"]
                    loss_retain = _scaled_subset_loss(output_clean, target, retain_indexes, len(target_label))
                    loss_forget = -_scaled_subset_loss(output_clean, target, forget_indexes, len(target_label))
                    loss = loss_retain + loss_forget
                else:
                    loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]

                _gn = sum(
                    p.grad.detach().norm(2).item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
                grad_norms.update(_gn, 1)

                optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Time {3:.2f}'.format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1))
                start = time.time()



    if wandb.run is not None:
        lrl = [param_group['lr'] for param_group in optimizer.param_groups if param_group["params"] != []]
        lr = sum(lrl) / len(lrl)
        # Keep all W&B logs on a single auto-incrementing step stream.
        # Mixing auto-step logs with explicit step=epoch causes monotonic-step warnings.
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            "Train Top1 Acc": top1.avg,
            "Train Loss": losses.avg,
            "Grad Norm (train avg)": grad_norms.avg,
        })

    return {
        "train_acc": top1.avg,
        "train_loss": losses.avg,
        "grad_norm_avg": grad_norms.avg,
        "rl_label_overlap": _overlap,
        "rl_label_overlap_ratio": _overlap / (1.0 / args.num_classes),
    }
