################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2022                                                             #
# Author(s): Florian Mies                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# TO RUN LOCALLY
import sys
sys.path.append("/home/florian/university/WS2021/ma/avalanche/fork/avalanche")
# REMOVE AFTER DEBUGGING

import argparse
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
import torch.optim.lr_scheduler
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- SCENARIO CREATION
    scenario = SplitMNIST(n_experiences=1, seed=1234)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=scenario.n_classes)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=100,
        train_epochs=4,
        eval_mb_size=100,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []

    import matplotlib.pyplot as plt
    import numpy as np
    iterations = 10
    f, axarr = plt.subplots(iterations, 10)
    k = 0
    for exp in range(iterations):
        print("Start of experience ",
              exp)
        # Reinit solver after each exp
        # cl_strategy.model = SimpleMLP(num_classes=scenario.n_classes)
        cl_strategy.train(scenario.train_stream[0])
        results.append(cl_strategy.eval(scenario.test_stream[0]))
        samples = cl_strategy.generator_strategy.model.generate(10)
        samples = samples.detach().cpu().numpy()
        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for experience " + str(k))
        np.vectorize(lambda ax: ax.axis('off'))(axarr)
        k += 1

    import pickle
    with open('results_mlvae_no_init_hid16_10iter.pkl', 'wb') as file:
        pickle.dump(results, file)
    print(results)
    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_mlp_no_init_hid16_10iter")
    plt.show()


""" import pickle
with open("results.pkl", 'rb') as f:
    res = pickle.load(f) """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)
