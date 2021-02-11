import copy
import math
from random import Random
from typing import Optional, List

from ConvBlock import ConvBlock
from ResidualBlock import ResidualBlock
import Utils as util


def get_decent_arch_config(window_size: int) -> dict:
    config_pop = evolve_conv_config_population(window_size, population=[], param_budget=2000000,
                                               trials_before_bailing=1000000, population_size=1)
    return config_pop[0]


def make_random_config_population(population_size: int,
                                  window_size: int,
                                  param_budget: int,
                                  trials_before_bailing: int):
    rand = Random()

    config_pop = []
    while len(config_pop) < population_size:
        num_layers = rand.randint(4, 11)
        rand_config = get_random_arch_config(window_size, param_budget, trials_before_bailing, num_layers)
        if rand_config["conv_config"] is not None:
            config_pop.append(rand_config)
    return config_pop


def evolve_conv_config_population(window_size: int,
                                  population: List,
                                  param_budget: int,
                                  trials_before_bailing: int,
                                  population_size: int,
                                  fraction_to_cull=0.1,
                                  make_arch_unique=True,
                                  arch_temp: float = 1.,
                                  hyperparam_temp: float = 1.) -> List:
    """Note:
        :param make_arch_unique:
        :param population_size:
        :param window_size:
        :param population:
        :param param_budget:
        :param trials_before_bailing:
        :param arch_temp:
        :param hyperparam_temp:
        :param fraction_to_cull:
    """

    rand = Random()

    # selection
    pop = copy.deepcopy(population)

    if fraction_to_cull > 0:
        pop = sorted(pop, key=lambda conf: conf["timeloss"])
        pop = pop[0:max(1, round(len(pop) * (1 - fraction_to_cull)))]

    # mutation
    for arch_config in pop:
        mutate_config(rand, window_size, arch_config, param_budget, arch_temp, hyperparam_temp)

    # breeding
    while len(pop) < population_size:
        parent_1 = round(rand.random() * rand.random() * len(pop) - 1)
        parent_2 = round(rand.random() * rand.random() * len(pop) - 1)

        child = crossover_config(rand, window_size, pop[parent_1], pop[parent_2], param_budget)
        pop.append(child)

    if make_arch_unique:
        hashes = set()
        uniq_pop = []
        for config in pop:
            hsh = util.arch_hash(config)
            if hsh not in hashes:
                uniq_pop.append(config)
            hashes.add(hsh)
        pop = uniq_pop

        print(f"Population has {len(uniq_pop)} unique configs out of {population_size}")

    while len(pop) < population_size:
        num_layers = rand.randint(4, 11)
        pop.append(get_random_arch_config(window_size, param_budget, trials_before_bailing, num_layers))

    return pop


def mutate_config(rand: Random, window_size: int, arch_config: dict, param_budget: int,
                  arch_temp: float = 1., hyperparam_temp: float = 1.) -> dict:
    cost = param_budget + 1
    viable = False
    new_conv_config = []
    tries = 0

    if arch_temp > 0:
        while cost > param_budget or not viable:
            new_conv_config = []
            last_layer = None

            for layer_config in arch_config["conv_config"]:
                # delete the block
                if rand.random() < 0.05 * arch_temp:
                    continue

                # dupe the block
                if last_layer is not None and rand.random() < 0.05 * arch_temp:
                    new_conv_config.append(last_layer)

                new_layer = copy.deepcopy(layer_config)
                last_layer = new_layer

                if rand.random() < 0.2 * arch_temp:
                    new_layer[0] = max(3, min(15, new_layer[0] + rand.randint(-1, 1)))
                if rand.random() < 0.2 * arch_temp:
                    new_layer[1] = max(1, min(24, new_layer[1] + rand.randint(-4, 4)))
                if rand.random() < 0.2 * arch_temp:
                    new_layer[2] = 2 ** rand.randint(0, 2)

                new_conv_config.append(new_layer)

            cost, viable = util.calc_config_param_cost_and_viability(window_size, new_conv_config)
            tries += 1

            if not viable and tries > 100:
                break

    if viable:
        arch_config["conv_config"] = new_conv_config
    else:
        new_conv_config = arch_config["conv_config"]

    new_resid_cxn = set()
    for idx_pair in arch_config["resid_cxns"]:
        src, dst = idx_pair
        # in case layer(s) deleted
        src = min(len(new_conv_config), src)

        # delete the connection
        if rand.random() < 0.1 * arch_temp:
            continue

        if rand.random() < 0.2 * arch_temp:
            src = max(0, min(len(new_conv_config) - 2, src + rand.randint(-1, 1)))
            dst = max(src + 2, min(len(new_conv_config), dst + rand.randint(-1, 1)))

        new_resid_cxn.add((src, dst))

    # add new residual connections
    if len(new_conv_config) > 2 and rand.random() < 0.05 * arch_temp:
        src = rand.randint(0, len(new_conv_config) - 2)
        dst = rand.randint(src + 2, len(new_conv_config))
        new_resid_cxn.add((src, dst))

    arch_config["resid_cxns"] = list(new_resid_cxn)

    if rand.random() < 0.2 * hyperparam_temp:
        arch_config["lr"] = math.exp(math.log(arch_config["lr"]) + 0.5 * hyperparam_temp * rand.gauss(0, 1))

    keys_to_mutate = ["sparse_reward", "large_penalty", "small_penalty"]
    for key in keys_to_mutate:
        if rand.random() < 0.2 * hyperparam_temp:
            arch_config[key] = math.exp(math.log(arch_config[key]) + 0.3 * hyperparam_temp * rand.gauss(0, 1))

    return arch_config


def crossover_config(rand: Random,
                     window_size: int,
                     parent_1: dict,
                     parent_2: dict,
                     param_budget: int) -> dict:
    viable = False
    new_conv_config = []
    cost = param_budget + 1
    tries = 0
    child_config = dict()
    p1_conv = parent_1["conv_config"]
    p2_conv = parent_2["conv_config"]

    while cost > param_budget or not viable:
        new_conv_config = []
        shorter = p1_conv if len(p1_conv) < len(p2_conv) else p2_conv
        longer = p1_conv if len(p1_conv) >= len(p2_conv) else p2_conv

        for i, _ in enumerate(shorter):
            source = p1_conv[i] if rand.random() < 0.5 else p2_conv[i]
            new_conv_config.append(copy.deepcopy(source))

        # make child have the average of the number of layers as the parents'
        for i in range((len(longer) - len(shorter)) // 2):
            new_conv_config.append(copy.deepcopy(longer[i]))

        cost, viable = util.calc_config_param_cost_and_viability(window_size, new_conv_config)

        if not viable and tries > 100:
            new_conv_config = copy.deepcopy(p1_conv)
            break

        tries += 1

    p1_resid = parent_1["resid_cxns"]
    p2_resid = parent_2["resid_cxns"]
    shorter = p1_resid if len(p1_resid) < len(p2_resid) else p2_resid
    longer = p1_resid if len(p1_resid) >= len(p2_resid) else p2_resid
    new_resid_cxns = set()

    for i, _ in enumerate(shorter):
        if p1_resid[i][1] > len(new_conv_config):
            source = p1_resid[i]
        elif p2_resid[i][1] > len(new_conv_config):
            source = p2_resid[i]
        else:
            source = p1_resid[i] if rand.random() < 0.5 else p2_resid[i]

        new_resid_cxns.add(source)

    for i in range((len(longer) - len(shorter)) // 2):
        if longer[i][1] <= len(new_conv_config):
            new_resid_cxns.add(longer[i])

    child_config["resid_cxns"] = list(new_resid_cxns)
    child_config["conv_config"] = new_conv_config

    keys_to_swap = ["lr", "sparse_reward", "large_penalty", "small_penalty"]
    for key in keys_to_swap:
        child_config[key] = parent_1[key] if rand.random() < 0.5 else parent_2[key]

    return child_config


def get_random_arch_config(window_size: int,
                           param_budget: int,
                           trials_before_bailing:
                           int, num_layers: int) -> dict:
    """    
    :param window_size: 
        the L dimension of the input tensor, used to be sure the 
        candidate architectures don't pool too much and end up with 
        < 1 length outputs 
    :param num_layers:
        the number of layers deep the convnet should be.
        This is not a randomly generated value in the arch
        search loop, because in this case the HP optimizer
        would get no signal to learn the expectation maximization
        unction. To give it that signal, we let ray tune choose
        the #layers. Since #layers is essentially the biggest
        feature of the arch, if at least that part can be learnable
        then the overall architecture search will be more tractible

    :param param_budget: 
        the allowable number of estimated parameters to use in the configuration. 
        Candidate configurations with more than this budget are ignored and retried. 
    :param trials_before_bailing:
        how many random configurations to try, from which to
        choose the best configuration.
        If no random configs are acceptable, bail out and we'll
        use the default
    :return:
        a dict with hyperparameters including lr, kernel penalties, 
        sparsity reward, and list of conv_config configurations, 
        each its own list with the format:
             [out_channel_log2, kernel_size, pool_stride] }

    """
    rand = Random()
    best_conv_configs = None
    resid_cxns = set()
    best_budget = 1e7
    num_trials = -1
    remaining_budget = param_budget

    # find the best (in terms of using most parameters up to the budget limit)
    # of 'trials_before_bailing' random architectures
    for s in range(trials_before_bailing):
        layer_configs = []

        remaining_budget = param_budget
        in_chans = 1
        comp_input_size = window_size

        for layer_num in range(num_layers):
            kernel_size = round(1.5 ** rand.randint(0, 10))
            out_channels_log2 = rand.randint(4, 11)
            out_chans = 2 ** out_channels_log2
            stride = util.kernel_size_to_stride(kernel_size, comp_input_size)

            # final layer needs to do 1x1 convolutions and no more reduction
            if layer_num == num_layers - 1:
                kernel_size = 1
                stride = 1
                out_chans = window_size
                out_channels_log2 = round(math.log(window_size, 2))

            # enforces the constraint of ending up with input_size=1 by last layer
            if layer_num == num_layers - 2:
                stride = 1
                pool_stride = comp_input_size
            else:
                pool_stride = 1 if comp_input_size == 1 else 2 ** rand.randint(0, 2)

            padding = math.ceil(kernel_size - stride) // 2

            total_ops = util.calc_conv_ops(comp_input_size, in_chans, out_chans, kernel_size, stride)
            remaining_budget -= total_ops

            temp_input_size = comp_input_size
            comp_input_size = util.calc_output_size(comp_input_size, kernel_size, stride, pool_stride, padding)

            if comp_input_size <= 0 or remaining_budget < 0:
                break

            layer = {"in_ch": in_chans, "out_ch_log2": out_channels_log2, "in_size": temp_input_size,
                     "out_size": comp_input_size, "kernel_size": kernel_size, "pool_stride": pool_stride}

            layer_configs.append(layer)
            in_chans = out_chans

        if comp_input_size <= 0:
            continue

        temp_resid_cxns = set()
        temp_resid_cxns.add((0, len(layer_configs)))  # can't go wrong with the big skip connection

        for src_idx in range(len(layer_configs) - 1):
            src_layer = layer_configs[src_idx]

            # proceed from the end backwards, as each connection reduces the
            # remaining budget and if we go over budget, at least we have
            # skip connections biased toward connecting the early to the late layers
            for dst_idx in range(len(layer_configs), src_idx + 1, -1):
                dst_layer = layer_configs[dst_idx - 1]
                # check ResidualBlock.__init__() to understand this logic
                src_coords = [src_layer["in_ch"], src_layer["in_size"]]
                dst_coords = [2 ** dst_layer["out_ch_log2"], dst_layer["out_size"]]
                src_t = src_coords[::-1]
                if math.dist(src_t, dst_coords) < math.dist(src_coords, dst_coords):
                    src_coords = src_t
                cost = 0
                if src_coords[0] != dst_coords[0]:  # means a conv is necessary
                    cost += src_coords[0] * dst_coords[0]
                if src_coords[1] != dst_coords[1]:  # means a pool is necessary
                    cost += max(src_coords[1], dst_coords[1])

                if remaining_budget >= cost and rand.random() < 0.1:
                    temp_resid_cxns.add((src_idx, dst_idx))

        if 0 < remaining_budget < best_budget:
            best_conv_configs = layer_configs
            resid_cxns = temp_resid_cxns
            best_budget = remaining_budget
            num_trials = s
            # print (f"Num layers: {num_layers}. "
            #        f"Best budget found in {num_trials} iterations. Budget was {(param_budget - best_budget)}, or "
            #        f"{100*(param_budget - best_budget) / param_budget:3.2f}% of optimal")

        # return early if it's this optimal
        if 0 < remaining_budget < param_budget * 0.1:
            break

    if best_conv_configs:
        print(f"Found random config which uses "
              f"{100 * (param_budget - best_budget) / param_budget:3.2f}% of budget in {num_trials} samples")
    else:
        print(f"Found no viable {num_layers}-layer configuration in {trials_before_bailing} trials, bailing...")

    conv_configs = []
    if best_conv_configs:
        for config in best_conv_configs:
            conv_configs.append([config["out_ch_log2"], config["kernel_size"], config["pool_stride"]])

    learning_rate = math.pow(10, rand.uniform(-4, -2))
    large_penalty = math.pow(10, rand.uniform(-6, -4))
    small_penalty = math.pow(10, rand.uniform(-5, -3))
    sparse_reward = math.pow(10, rand.uniform(-6, -4))

    arch_config = {"conv_config": conv_configs,
                   "resid_cxns": list(resid_cxns),
                   "lr": learning_rate,
                   "sparse_reward": sparse_reward,
                   "large_penalty": large_penalty,
                   "small_penalty": small_penalty,
                   "budget_used": param_budget - best_budget}

    return arch_config
