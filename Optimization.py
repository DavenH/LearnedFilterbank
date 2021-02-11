import copy
import os
import sys
from random import random
from typing import List

import Genetics as genetics
import Utils as util
import math

from FilterbankModule import FilterbankModule


def exp_space(starting_const: float, index:int, grid_size:int):
    return math.exp(starting_const + 0.1 * (index - grid_size / 2))


def grid_search_hpo(**model_config):
    grid_len = 3
    train_tensor, eval_tensor, _ = util.load_datasets(model_config["device"],
                                                      max_train_samples=512 * 100,
                                                      max_train_files=10)
    step_levels = 3
    best_eval_loss = [1e10 for _ in range(step_levels)]
    best_large_penalty = [10000000 for _ in range(step_levels)]
    best_small_penalty = copy.deepcopy(best_large_penalty)
    best_sparse_reward = copy.deepcopy(best_large_penalty)

    # start with best-known constants
    const_lp = math.log(0.00000086)
    const_sp = math.log(0.00000142)
    const_spr = math.log(0.00000145)

    for large_pen in range(grid_len):
        for small_pen in range(grid_len):
            for sparse_r in range(grid_len):
                print (f"Trying config {large_pen} {small_pen} {sparse_r}")

                model = FilterbankModule(train_data=train_tensor, eval_data=eval_tensor, **model_config)
                model.config["arch"]["large_penalty"] = exp_space(const_lp, large_pen, grid_len)
                model.config["arch"]["small_penalty"] = exp_space(const_sp, small_pen, grid_len)
                model.config["arch"]["sparse_reward"] = exp_space(const_spr, sparse_r, grid_len)

                # exponentially increase the number of evaluation steps, allowing us
                # to bail in log(n) time if the configuration sucks
                for step in range(step_levels):
                    num_steps = 2 ** step

                    for i in range(num_steps):
                        model.step()

                    eval_loss = model.evaluate()

                    if eval_loss < best_eval_loss[step]:
                        best_eval_loss[step] = eval_loss
                        best_large_penalty[step] = large_pen
                        best_small_penalty[step] = small_pen
                        best_sparse_reward[step] = sparse_r

                    # short circuit if loss isn't good enough
                    elif eval_loss > best_eval_loss[step] * 1.1:
                        break

    print(
        f"Best loss at final step:\t{best_eval_loss[-1]:2.4f}\t\tlarge_pen:\t"
        f"{exp_space(const_lp, best_large_penalty[-1], grid_len):0.8f}\t\t"
        f"{exp_space(const_sp, best_small_penalty[-1], grid_len):0.8f}\t\t"
        f"{exp_space(const_spr, best_sparse_reward[-1], grid_len):0.8f}")


def upsert_sorted(queue: List, value: float, arg: dict, hsh: int, keep_n: int) -> List:
    for i, tup in enumerate(queue):
        if tup[2] == hsh and value < tup[0]:
            queue[i] = (value, copy.deepcopy(arg), hsh)
            return queue

    if not queue or queue[-1][0] > value or len(queue) < keep_n:
        queue.append((value, copy.deepcopy(arg), hsh))

    queue = sorted(queue, key=lambda pair: pair[0])
    queue = queue[0:keep_n]
    return queue


def perform_architecture_search(configuration: dict,
                                init_pop_size: int,
                                num_generations: int,
                                checkpoint_directory:str,
                                param_budget: int = 2000000,
                                trials_before_bailing: int = 100000,
                                culling_fraction=0.3,
                                arch_temp=1.,
                                hyperparam_temp=1.,
                                ):
    train_data, eval_data, _ = util.load_datasets(configuration["device"],
                                                  max_train_samples=512 * 100,
                                                  max_eval_samples=512 * 50)

    m = FilterbankModule(train_data, eval_data, **configuration)
    m.config["restore_from_config"] = True

    window_size = configuration["window_size"]
    config_population = genetics.make_random_config_population(population_size=init_pop_size,
                                                               window_size=window_size,
                                                               param_budget=param_budget,
                                                               trials_before_bailing=trials_before_bailing)

    # put some good configs in the population to act as a reference, and seed good 'genetics'

    config_population[0] = {"conv_config": [[6, 1, 4], [8, 5, 4], [8, 1, 1], [5, 8, 2], [7, 1, 1], [8, 1, 2]],
                            "resid_cxns": [(0, 6)]}
    config_population[1] = {"conv_config": [[5, 32, 1], [7, 32, 1], [5, 1, 10], [8, 1, 1]], "resid_cxns": [(0, 4)]}
    config_population[2] = {"conv_config": [[9, 38, 4], [5, 3, 2], [5, 2, 2], [10, 3, 2], [8, 1, 1]],
                            "resid_cxns": [(2, 4), (0, 5)]}
    # config_pop[3]= { "conv_config": [[8, 32, 4], [4, 6, 1], [4, 2, 1], [8, 1, 7]], "resid_cxns": [(0, 4)]}

    # these constants were found from grid search
    for config in config_population:
        config["lr"] = 0.00222
        config["large_penalty"] = 0.00000086
        config["small_penalty"] = 0.00000142
        config["sparse_reward"] = 0.00000145

    init_epochs = m.config["epochs"]
    config_hash_to_checkpoint = dict()

    root = checkpoint_directory
    checkpoints = os.listdir(root)
    for chk in checkpoints:
        chk_path = os.path.join(root, chk)
        chk_file = os.listdir(chk_path)[0]
        config_hash_to_checkpoint[int(chk)] = os.path.join(chk_path, chk_file)

    best_n_by_eval = []
    best_n_by_effic = []

    # keep n best configurations around in a queue for display
    # and for comparing new configs against for saliency
    n_best = 8

    for cycle in range(num_generations):
        indices_to_remove = []
        progress = cycle / (num_generations - 1)

        pop_size = init_pop_size
        # pop_size = round(init_pop_size * math.pow(0.95, cycle))

        if len(config_population) > pop_size:
            config_population = config_population[0:pop_size]

        # each cycle we increase the number of timesteps, winnowing the population
        # size down to equalize the wall-clock time each generation.
        # This trades off exploration --> exploitation the more generations we go.
        n_epochs = round(init_pop_size / pop_size * init_epochs)

        print(f"Starting generation {cycle + 1}, num epochs per test: {n_epochs}, pop size: {pop_size}")

        itr = 0
        for i, arch_config in enumerate(config_population):
            config_hash = util.arch_hash(arch_config)

            try:
                if config_hash in config_hash_to_checkpoint:
                    m.load_checkpoint(config_hash_to_checkpoint[config_hash])
                else:
                    m.config["arch"] = arch_config
                    m.train_time = 0
                    m.epoch = 0
                    m.restore_architecture()
                    arch_config = m.config["arch"]
                    config_hash = util.arch_hash(arch_config)
                    m.set_device(configuration["device"])

                m.config["epochs"] = n_epochs

                print_status(itr + 1, pop_size, arch_config, config_hash, m)

                eval_loss = timeloss = 0
                dismissed = False
                for j in range(5):
                    train_loss, train_time = m.step()
                    eval_loss = m.evaluate()
                    timeloss = math.log(0.2 + max(0., eval_loss)) * math.log(2.718 + m.train_time * (1 - progress))

                    print(f"\t\t{m.epoch}\t{train_loss:0.2f}\t{eval_loss:0.2f}\t{train_time:2.2f}\t{timeloss}l×s")
                    # fade out effect of training time as we get further along; want to prioritize eval accuracy

                    nth_best_effic = 1e5 if len(best_n_by_effic) < n_best else best_n_by_effic[-1][0]

                    # stop early if our 'timeloss' is bad
                    if j == 0 and timeloss > nth_best_effic * 1.7:
                        indices_to_remove.append(itr)
                        dismissed = True

                    if timeloss > nth_best_effic * 1.3:
                        dismissed = True
                        break

                print_eval_status(eval_loss, timeloss, m)

                if not dismissed:
                    save_dir = os.path.join(checkpoint_directory, f"{config_hash}")
                    if not os.path.isdir(save_dir):
                        os.mkdir(save_dir)
                    directory = m.save_checkpoint(save_dir)
                    config_hash_to_checkpoint[config_hash] = directory

                arch_config["timeloss"] = timeloss
                arch_config["eval_loss"] = eval_loss

                best_n_by_effic = upsert_sorted(best_n_by_effic, timeloss, arch_config, config_hash, n_best)
                best_n_by_eval = upsert_sorted(best_n_by_eval, eval_loss, arch_config, config_hash, n_best)
                config_population[i] = arch_config
            except (RuntimeError, Exception) as r:
                print(f"Error with {arch_config}:\n", r, file=sys.stderr)
                indices_to_remove.append(itr)
            itr += 1

        if indices_to_remove:
            # delete these in reverse order otherwise the remaining
            # indices will become invalid after first deletion
            for index in sorted(indices_to_remove, reverse=True):
                print("Removing bad config: ", config_population[index])
                del config_population[index]
            indices_to_remove.clear()

        config_population = genetics.evolve_conv_config_population(window_size, config_population, param_budget,
                                                                   trials_before_bailing, pop_size,
                                                                   fraction_to_cull=culling_fraction,
                                                                   arch_temp=arch_temp,
                                                                   hyperparam_temp=hyperparam_temp)
        print_report_best_configs(best_n_by_effic, best_n_by_eval, n_best)

    print(f"\nBest eval {n_best} configurations")
    for conf in best_n_by_eval:
        print(f"{conf[1]['eval_loss']:2.3f}\t", conf)

    print(f"\nBest time-efficient {n_best} configurations")
    for conf in best_n_by_effic:
        print(f"{conf[1]['timeloss']:2.3f}\t", conf)

    print()


# breaking these out to help code readability

def print_status(index, pop_size, arch_config, config_hash, m: FilterbankModule):
    print(f"{index + 1} / {pop_size}"
          f"\tconv config\t{arch_config['conv_config']}\n"
          f"\t\tskip conns\t{arch_config['resid_cxns']}\n"
          f"\t\thash={config_hash}\t"
          f"params={m.get_num_model_params()}\t"
          f"lr={1e5 * arch_config['lr']:.2f}\t"
          f"large pen={1e5 * arch_config['large_penalty']:.2f}\t"
          f"small pen={1e5 * arch_config['small_penalty']:.2f}\t"
          f"sparse reward={1e5 * arch_config['sparse_reward']:.2f}")


def print_eval_status(eval_loss: float, timeloss: float, m: FilterbankModule):
    print(f"\n\t\teval loss={eval_loss:0.3f}\n\t\ttimeloss={timeloss:3.3f}\n\t\t"
          f"time={math.floor(m.train_time / 60)}m {round(m.train_time) % 60}s\n")


def print_report_best_configs(best_n_by_effic, best_n_by_eval, n_best):
    print(f"\nBest {n_best} timeloss configurations:")
    print("eval\tt-loss\tresid\tconv")

    for conf in best_n_by_effic:
        arch = conf[1]
        print(f"{arch['eval_loss']:2.3f}\t{arch['timeloss']:2.3f}\t"
              # f"{100000*arch['large_penalty']:.2f}\t"
              # f"{100000*arch['small_penalty']:.2f}\t"
              # f"{100000*arch['sparse_reward']:.2f}\t"
              # f"{100000*arch['lr']:.2f}\t",
              , arch["resid_cxns"], "\t"
              , arch["conv_config"]
              )
    print()

    print(f"\nBest {n_best} eval configurations:")
    print("eval\tt-loss\tresid\tconv")

    for conf in best_n_by_eval:
        arch = conf[1]
        print(f"{arch['eval_loss']:2.3f}\t{arch['timeloss']:2.3f}\t"
              # f"{100000*arch['large_penalty']:.2f}\t"
              # f"{100000*arch['small_penalty']:.2f}\t"
              # f"{100000*arch['sparse_reward']:.2f}\t"
              # f"{100000*arch['lr']:.2f}\t",
              , arch["resid_cxns"], "\t"
              , arch["conv_config"]
              )
    print()
