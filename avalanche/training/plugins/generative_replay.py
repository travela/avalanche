################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 05-03-2022                                                             #
# Author: Florian Mies                                                         #
# Website: https://github.com/travela                                          #
################################################################################

"""

All plugins related to Generative Replay.

"""

from copy import deepcopy
from avalanche.core import SupervisedPlugin
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
import torch


class GenerativeReplayPlugin(SupervisedPlugin):
    """
    Experience generative replay plugin.

    Updates the current mbatch of a strategy before training an experience
    by sampling a generator model and concatenating the replay data to the 
    current batch. 

    In this version of the plugin the number of replay samples is 
    increased with each new experience. Another way to implempent 
    the algorithm is by weighting the loss function and give more 
    importance to the replayed data as the number of experiences 
    increases. This will be implemented as an option for the user soon.

    :param generator_strategy: In case the plugin is applied to a non-generative
     model (e.g. a simple classifier), this should contain an Avalanche strategy 
     for a model that implements a 'generate' method 
     (see avalanche.models.generator.Generator). Defaults to None.
    :param untrained_solver: if True we assume this is the beginning of 
        a continual learning task and add replay data only from the second 
        experience onwards, otherwise we sample and add generative replay data
        before training the first experience. Default to True.
    :param replay_size: The user can specify the batch size of replays that 
        should be added to each data batch. By default each data batch will be 
        matched with replays of the same number.
    :param increasing_replay_size: If set to True, each experience this will 
        double the amount of replay data added to each data batch. The effect 
        will be that the older experiences will gradually increase in importance
        to the final loss.
    """

    def __init__(self, generator_strategy: "BaseTemplate" = None, 
                 untrained_solver: bool = True, replay_size: int = None,
                 increasing_replay_size: bool = False, 
                 start_replay_from_exp: int = None, 
                 GR_over_itself: bool = False):
        '''
        Init.
        '''
        super().__init__()
        self.generator_strategy = generator_strategy
        if self.generator_strategy:
            self.generator = generator_strategy.model
        else: 
            self.generator = None
        self.untrained_solver = untrained_solver
        self.model_is_generator = False
        self.replay_size = replay_size
        self.increasing_replay_size = increasing_replay_size
        self.GR_over_itself = GR_over_itself

        self.replay_statistics = []
        self.losses = []
        self.replay_samples = []
        self.replay_samples_lables = []
        self.start_replay_from_exp = start_replay_from_exp

    def before_training(self, strategy: "SupervisedTemplate", *args, **kwargs):
        """Checks whether we are using a user defined external generator 
        or we use the strategy's model as the generator. 
        If the generator is None after initialization 
        we assume that strategy.model is the generator.
        (e.g. this would be the case when training a VAE with 
        generative replay)"""
        if not self.generator_strategy:
            self.generator_strategy = strategy
            self.generator = strategy.model
            self.model_is_generator = True

    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """
        Make deep copies of generator and solver before training new experience.
        """
        # For weighted loss criterion: store the number of classes seen so far
        strategy.number_classes_until_now = len(
            set(strategy.experience.classes_seen_so_far))
        self.losses_exp = []
        # Once we see different classes than in the first experience, 
        # we start replaying data:
        if(self.start_replay_from_exp):
            if(strategy.experience.current_experience >= 
               self.start_replay_from_exp):
                self.untrained_solver = False
        # if (strategy.experience.classes_seen_so_far != 
        #        strategy.experience.classes_in_this_experience):
        #    self.untrained_solver = False
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        self.old_generator = deepcopy(self.generator)
        self.old_generator.eval()
        if not self.model_is_generator:
            self.old_model = deepcopy(strategy.model)
            self.old_model.eval()
        self.replay_statistics_exp = []

    def after_training_exp(self, strategy: "SupervisedTemplate",
                           num_workers: int = 0, shuffle: bool = True,
                           **kwargs):
        """
        Set untrained_solver boolean to False after (the first) experience,
        in order to start training with replay data from the second experience.
        """
        self.losses.append(self.losses_exp)
        if not self.untrained_solver:
            self.replay_statistics.append(self.replay_statistics_exp)
        # Generate some samples for each class:

        if not self.model_is_generator:
            replay = self.generator.generate(50).to(strategy.device)  
            with torch.no_grad():
                replay_output = strategy.model(replay).argmax(dim=-1)

                # Determine how many samples per class we would like to have
                expected_num_samples_per_class = 5
                # Check for each class if enough samples were generated
                for class_name in set(
                        strategy.experience.classes_seen_so_far):
                    # There should be an additional stopping criterion 
                    # (e.g. max expected_num_samples_per_class iterations)
                    balance_replay_iter = 0
                    while (sum(replay_output == class_name)
                            < expected_num_samples_per_class
                           ) and (balance_replay_iter
                                  < 10):
                        replay = torch.cat([replay, self.generator.generate(50)
                                            .to(strategy.device)])
                        replay_output = strategy.model(replay).argmax(dim=-1)
                        balance_replay_iter += 1
                # Keep only a fix amount of samples per class
                replay_samples_exp = []
                replay_samples_lables_exp = []
                for class_name in set(strategy.experience.classes_seen_so_far):
                    replay_samples_exp.extend(
                        replay[
                            replay_output == class_name]
                        [:expected_num_samples_per_class])
                    replay_samples_lables_exp.extend(
                        replay_output[replay_output == class_name]
                        [:expected_num_samples_per_class])
                self.replay_samples.append(replay_samples_exp)
                self.replay_samples_lables.append(replay_samples_lables_exp)

    def before_training_epoch(self, strategy: "SupervisedTemplate",
                              **kwargs):
        """
        Initializing empty list to stay losses of epoch.
        """
        self.losses_epoch = []

    def after_training_epoch(self, strategy: "SupervisedTemplate",
                             **kwargs):
        """
        Appending losses of epoch to list of losses of current experience.
        """
        self.losses_exp.append(self.losses_epoch)

    def before_training_iteration(self, strategy: "SupervisedTemplate",
                                  **kwargs):
        """
        Generating and appending replay data to current minibatch before 
        each training iteration.
        """
        if self.untrained_solver:
            # The solver needs to be trained before labelling generated data and
            # the generator needs to be trained before we can sample.
            return
        # determine how many replay data points to generate
        if self.replay_size:
            number_replays_to_generate = self.replay_size
        else:
            if self.increasing_replay_size:
                number_replays_to_generate = len(
                    strategy.mbatch[0]) * (
                        strategy.experience.current_experience)
            else:
                number_replays_to_generate = len(strategy.mbatch[0])
        # extend X with replay data
        replay = self.old_generator.generate(number_replays_to_generate
                                             ).to(strategy.device)
        if self.GR_over_itself:
            strategy.mbatch[0] = replay
        else:      
            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], replay], dim=0)
        # extend y with predicted labels (or mock labels if model==generator)
        if not self.model_is_generator:
            with torch.no_grad():
                replay_output = self.old_model(replay).argmax(dim=-1)
                self.replay_statistics_exp.extend(replay_output)
        else:
            # Mock labels:
            replay_output = torch.zeros(replay.shape[0])
        if self.GR_over_itself:
            strategy.mbatch[1] = replay_output.to(strategy.device)
            strategy.mbatch[-1] = torch.ones(
                replay.shape[0]).to(strategy.device) * strategy.mbatch[-1][0]
        else:
            strategy.mbatch[1] = torch.cat(
                [strategy.mbatch[1], replay_output.to(strategy.device)], dim=0)
            # extend task id batch (we implicitley assume a task-free case)
            strategy.mbatch[-1] = torch.cat([strategy.mbatch[-1], torch.ones(
                replay.shape[0]).to(strategy.device) * strategy.mbatch[-1][0]],
                dim=0)

    def after_training_iteration(self, strategy: "SupervisedTemplate",
                                 **kwargs):
        """
        Adding the loss of current iteration to th elist of losses of the epoch.
        """
        self.losses_epoch.append(strategy.loss.item())


class TrainGeneratorAfterExpPlugin(SupervisedPlugin):
    """
    TrainGeneratorAfterExpPlugin makes sure that after each experience of 
    training the solver of a scholar model, we also train the generator on the 
    data of the current experience.
    """

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        """
        The training method expects an Experience object 
        with a 'dataset' parameter.
        """
        for plugin in strategy.plugins:
            if type(plugin) is GenerativeReplayPlugin:
                print("Start training of Replay Generator.")
                plugin.generator_strategy.train(strategy.experience) 
                print("End training of Replay Generator.")
