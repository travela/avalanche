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
import matplotlib.pyplot as plt
import numpy as np
from avalanche.benchmarks import SplitMNIST
from avalanche.models import MlpVAE
from avalanche.training.supervised import VAETraining
from avalanche.training.plugins import GenerativeReplayPlugin
from avalanche.logging import InteractiveLogger


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- SCENARIO CREATION
    scenario = SplitMNIST(n_experiences=10, seed=1234)
    # ---------

    # MODEL CREATION
    model = MlpVAE((1, 28, 28), nhid=2, device=device)

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = VAETraining(
        model,
        torch.optim.Adam(model.parameters(), lr=0.001),
        train_mb_size=100,
        train_epochs=4,
        device=device,
        plugins=[GenerativeReplayPlugin()],
        weighted_loss=True,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    f, axarr = plt.subplots(10, 10)
    k = 0
    for k in range(10):
        print("Start of experience ",
              k)
        cl_strategy.train(scenario.train_stream[0])
        print("Training completed")

        samples = model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for experience " + str(k))
        np.vectorize(lambda ax: ax.axis('off'))(axarr)
        k += 1

    f.subplots_adjust(hspace=1.2)
    plt.savefig("VAE_output_per_exp")
    plt.show()


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
