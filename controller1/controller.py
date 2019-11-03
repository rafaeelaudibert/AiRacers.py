import controller_template as controller_template
import numpy as np
import random
import datetime
import time
import os
import json


class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")

        # Features hyperparamets
        self.vel_threshold = 120
        self.last_action = 5  # Nothing
        self.no_turn_times = 0

        self.data = []
    #######################################################################
    ##### METHODS YOU NEED TO IMPLEMENT ###################################
    #######################################################################

    def take_action(self, parameters: list) -> int:
        """
        :param parameters: Current weights/parameters of your controller
        :return: An integer corresponding to an action:
        1 - Right
        2 - Left
        3 - Accelerate
        4 - Brake
        5 - Nothing
        """

        features = self.compute_features(self.sensors)
        parameters = np.array(parameters).reshape(5, -1)

        computed_values = features * parameters
        summed_values = np.sum(computed_values, axis=1)

        return np.argmax(summed_values) + 1

    def compute_features(self, sensors):
        """
        :param sensors: Car sensors at the current state s_t of the race/game
        contains (in order):
            track_distance_left: 1-100
            track_distance_center: 1-100
            track_distance_right: 1-100
            on_track: 0 or 1
            checkpoint_distance: 0-???
            car_velocity: 10-200
            enemy_distance: -1 or 0-???
            position_angle: -180 to 180
            enemy_detected: 0 or 1
          (see the specification file/manual for more details)
        :return: A list containing the features you defined
        """

        # Fetch sensors
        track_distance_left, \
            track_distance_center, \
            track_distance_right, \
            on_track, \
            checkpoint_distance, \
            car_velocity, \
            enemy_distance, \
            position_angle, \
            enemy_detected = sensors

        # Compute features
        constant_f = 1.0
        approx_f = self.normalize(checkpoint_distance -
                                  self.old_next_checkpoint, -200, 200)
        left_f = self.normalize(track_distance_left, 0, 100)
        center_f = self.normalize(track_distance_center, 0, 100)
        right_f = self.normalize(track_distance_right, 0, 100)
        central_f = self.normalize(
            abs(track_distance_left - track_distance_right), 0, 100)
        ontrack_f = self.normalize(on_track, 0, 1)
        velocity_f = self.normalize(car_velocity, 0, 200)
        slow_f = self.normalize(int(car_velocity < self.vel_threshold), 0, 1)
        turn_f = self.normalize(max(self.no_turn_times, 10), 0, 10)
        last_action_f = -1 if self.last_action == 1 else 1 if self.last_action == 2 else 0

        # TODO FEATURES
        # Know how to leave grass
        # Remember last action (left, center, right)
        # Time without choosing to turn
        # Faster than threshold

        features = np.array([constant_f, approx_f, left_f, center_f,
                             right_f, central_f, ontrack_f, velocity_f,
                             slow_f, turn_f, last_action_f])

        # Update values
        self.old_next_checkpoint = checkpoint_distance

        return features

    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """

        iter = 0
        iter_unchanged = 0

        epsilon = 2
        multiple_epsilon_sum = 0

        new_order = True
        weight_numbers = []
        weight_index = 0

        neighbours = []
        best_weights = weights.copy()
        best_value = self.run_episode(best_weights.copy())

        # Learning process
        try:

            while True:
                print()
                print("Epsilon:", epsilon, " Iteration:", iter, " Unchanged Iterations:", iter_unchanged)

                old_weights = best_weights.copy()
                old_value = self.run_episode(old_weights.copy())

                #Generate new random weight order
                if(new_order):
                    weight_numbers.clear()
                    for i in range(len(best_weights)):
                        weight_numbers.append(i)
                    random.shuffle(weight_numbers)
                    weight_index = 0
                    new_order = False

                #Generate neighbours
                print("Generating neighbours using", multiple_epsilon_sum, "multiple weights...")
                neighbours.clear()
                for i in range(6):
                    new_neighbour = best_weights.copy()
                    for j in range(multiple_epsilon_sum + 1):
                        if weight_index + j >= len(weight_numbers):
                            new_neighbour[weight_numbers[weight_index - j]] += epsilon * 2 ^ i
                        else:
                            new_neighbour[weight_numbers[weight_index + j]] += epsilon * 2 ^ i
                    neighbours.append(new_neighbour.copy())
                for i in range(6):
                    new_neighbour = best_weights.copy()
                    for j in range(multiple_epsilon_sum + 1):
                        if weight_index + j >= len(weight_numbers):
                            new_neighbour[weight_numbers[weight_index - j]] -= epsilon * 2 ^ i
                        else:
                            new_neighbour[weight_numbers[weight_index + j]] -= epsilon * 2 ^ i
                    neighbours.append(new_neighbour.copy())
                print("", len(neighbours), "new neighbours generated.")

                #Compute Best Neighbours
                print("Computing", len(neighbours), "neighbours...")
                for i in range(len(neighbours)):
                    new_value = self.run_episode(neighbours[i].copy())
                    print(" Neighbour:", i, " Score:", new_value)
                    if new_value > best_value:
                        print("  New best value found:", new_value, ">", best_value)
                        best_value = new_value
                        best_weights = neighbours[i].copy()

                #Checking fitness
                if best_weights != old_weights:
                    if self.run_episode(best_weights.copy()) < old_value:
                        print("Best Score wrong:", best_value)
                        best_weights = old_weights.copy()
                        best_value = old_value
                    else:
                        print("Increased: +", best_value-old_value)

                #Iterations sums
                weight_index += 1
                if weight_index == len(best_weights):
                    new_order = True
                if np.sum(best_weights) == np.sum(old_weights):
                    iter_unchanged +=1
                else:
                    iter_unchanged = 0
                    multiple_epsilon_sum = 0
                    if iter_unchanged > len(best_weights):
                        epsilon = 2*random.random()
                    if iter_unchanged > len(best_weights)*(2+multiple_epsilon_sum):
                        multiple_epsilon_sum += 1
                        if multiple_epsilon_sum >= len(weight_numbers):
                            multiple_epsilon_sum = 0
                            new_order = True
                iter += 1

                # Save info
                self.data.append({
                    'epoch': iter,
                    'fitness': best_value,
                    'best_weight': list(best_weights),
                    'time': datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
                })

        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass

        # Save data to file
        with open('./data/common.json', 'w') as f:
            json.dump(self.data, f)

        # raise NotImplementedError("This Method Must Be Implemented")

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
