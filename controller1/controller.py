import controller_template as controller_template
import numpy as np
import random
import time


class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")

        # Features hyperparamets
        self.vel_threshold = 120
        self.last_action = 5  # Nothing
        self.no_turn_times = 0

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

        # Learning process
        k_inst = 5
        iter = 0
        epsilon = 0.5
        highest_values = []
        highest_weights = []
        random_weights = []
        epsilons = []
        iter_epsilon_unchanged = []

        random_weights.append(weights.copy())
        highest_values.append(-100000000)
        epsilons.append([epsilon for i in range(len(weights))])
        iter_epsilon_unchanged.append([0 for i in range(len(weights))])
        for j in range(1, k_inst):
            random_weights.append([random.uniform(-1, 1) for i in range(0, len(weights))])
            highest_values.append(-100000000)
            epsilons.append([epsilon for i in range(len(weights))])
            iter_epsilon_unchanged.append([0 for i in range(len(weights))])

        highest_weights.append(epsilons)
        highest_weights.append(iter_epsilon_unchanged)
        highest_weights.append(random_weights)
        highest_weights.append(random_weights)

        multiple_epsilon_sum = 0

        new_order = True
        weight_numbers = []
        weight_index = 0

        neighbours = []

        new_epsilon = []
        new_iter = []
        new_weight = []
        new_old = []

        try:
            while True:
                print()

                for i in range(len(highest_weights[2])):
                    highest_weights[3][i] = highest_weights[2][i].copy()

                # Generate new random weight order
                if (new_order):
                    weight_numbers.clear()
                    for i in range(len(weights)):
                        weight_numbers.append(i)
                    random.shuffle(weight_numbers)
                    weight_index = 0
                    new_order = False

                print(" Iteration:", iter, " Best Score:", highest_values)

                # Generate neighbours
                neighbours.clear()
                new_epsilon.clear()
                new_iter.clear()
                new_weight.clear()
                new_old.clear()
                for j in range(len(highest_weights[2])):
                    k = 0
                    for i in range(1):
                        new_epsilon.append(highest_weights[0][j].copy())
                        new_iter.append(highest_weights[1][j].copy())
                        new_weight.append(highest_weights[2][j].copy())
                        new_old.append(highest_weights[3][j].copy())
                        new_weight[k][weight_numbers[weight_index]] += new_epsilon[k][
                            weight_numbers[weight_index]]  # * 2 ** (float(i))
                        k =+ 1
                        new_epsilon.append(highest_weights[0][j].copy())
                        new_iter.append(highest_weights[1][j].copy())
                        new_weight.append(highest_weights[2][j].copy())
                        new_old.append(highest_weights[3][j].copy())
                        new_weight[k][weight_numbers[weight_index]] -= new_epsilon[k][
                            weight_numbers[weight_index]]  # * 2 ** (float(i))
                        k =+ 1

                neighbours.append(new_epsilon.copy())
                neighbours.append(new_iter.copy())
                neighbours.append(new_weight.copy())
                neighbours.append(new_old.copy())
                print("", len(neighbours[2]), "new neighbours generated.")

                # Compute Best Neighbours
                print("Computing", len(neighbours[2]), "neighbours...")
                for i in range(len(neighbours[2])):
                    new_value = self.run_episode(neighbours[2][i].copy())
                    #print(" Neighbour:", i, " Score:", new_value)
                    if new_value > highest_values[k_inst - 1]:
                        equal_value = 0
                        for b in range(len(highest_values)):
                            if new_value == highest_values[b]:
                                equal_value = 1
                                print("New value is equal!")
                        if equal_value == 0:
                            print("  New best value found:", new_value, ">", highest_values[k_inst - 1])
                            highest_values.pop(k_inst - 1)
                            highest_values.insert(k_inst - 1, new_value)

                            last_value = highest_values[k_inst - 1]
                            highest_values.sort(reverse=True)
                            for k in range(k_inst):
                                if highest_values[k] == last_value:
                                    highest_weights[0].insert(k, neighbours[0][i].copy())
                                    highest_weights[0].pop(k_inst)
                                    highest_weights[1].insert(k, neighbours[1][i].copy())
                                    highest_weights[1].pop(k_inst)
                                    highest_weights[2].insert(k, neighbours[2][i].copy())
                                    highest_weights[2].pop(k_inst)
                                    highest_weights[3].insert(k, neighbours[3][i].copy())
                                    highest_weights[3].pop(k_inst)
                                    break

                # Iterations count and epsilon variation
                for i in range(len(highest_weights[2])):
                    highest_weights[0][i][weight_numbers[weight_index]] *= 1.1
                    if np.sum(highest_weights[2][i]) == np.sum(highest_weights[3][i]):
                        highest_weights[1][i][weight_numbers[weight_index]] += 1
                        if highest_weights[1][i][weight_numbers[weight_index]] > 5 + 9:
                            highest_weights[0][i][weight_numbers[weight_index]] *= (2 ** 10)
                            highest_weights[1][i][weight_numbers[weight_index]] = 0
                        elif highest_weights[1][i][weight_numbers[weight_index]] > 5:
                            highest_weights[1][i][weight_numbers[weight_index]] /= 2
                        if highest_weights[0][i][weight_numbers[weight_index]] > 10:
                            highest_weights[0][i][weight_numbers[weight_index]] = random.random()
                    else:
                        highest_weights[1][i][weight_numbers[weight_index]] = 0
                        multiple_epsilon_sum = 0

                weight_index += 1
                if weight_index == len(weights):
                    new_order = True
                iter += 1


        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass



        best_weights = highest_weights[0].copy()
        print(self.run_episode(best_weights.copy()))
        # raise NotImplementedError("This Method Must Be Implemented")

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
