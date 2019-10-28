import controller_template as controller_template
import numpy as np
import random


class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")

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
        constant_f = 1
        approx_f = self.normalize(checkpoint_distance -
                                  self.old_next_checkpoint, -200, 200)
        left_f = self.normalize(track_distance_left, 0, 100)
        center_f = self.normalize(track_distance_center, 0, 100)
        right_f = self.normalize(track_distance_right, 0, 100)
        ontrack_f = self.normalize(on_track, 0, 1)
        velocity_f = self.normalize(car_velocity, 0, 200)

        # TODO FEATURES
        # Know how to leave grass
        # Remember last action (left, center, right)
        # Time without choosing to turn
        # Faster than threshold

        features = np.array([constant_f, approx_f, left_f, center_f,
                             right_f, ontrack_f, velocity_f])

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
        def generate_neighbours(best_weights, epsilon):
            neighbours = []
            for j in range(len(best_weights)):
                for i in range(len(best_weights[j])):
                    new_neighbour = best_weights[j].copy()
                    new_neighbour[i] += random.random()*epsilon
                    neighbours.append(new_neighbour.copy())
                for i in range(len(best_weights[j])):
                    new_neighbour = best_weights[j].copy()
                    new_neighbour[i] -= random.random() * epsilon
                    neighbours.append(new_neighbour.copy())
            print(len(neighbours), "novos vizinhos gerados.")
            return neighbours

        def compute_best_neighbours(neighbours, highest_values, highest_weights, k_inst):
            print()
            print("Computing ", len(neighbours), " neighbours.")
            for i in range(1,len(neighbours)):
                new_value = self.run_episode(neighbours[i])
                for j in range(k_inst):
                    if new_value > highest_values[j]:
                        print()
                        print("New best value", j, "found:", new_value, ">", highest_values[j])
                        new_weights = []
                        new_values = []
                        if j == 0:
                            new_weights.append(neighbours[i].copy())
                            new_weights.extend(highest_weights[1:k_inst].copy())

                            new_values.append(new_value)
                            new_values.extend(highest_values[1:k_inst])
                        else:
                            new_weights.extend(highest_weights[0:j])
                            new_weights.append(neighbours[i].copy())

                            new_values.extend(highest_values[0:j])
                            new_values.append(new_value)

                            if j+1 != k_inst:
                                new_weights.extend(highest_weights[j + 1:k_inst])
                                new_values.extend(highest_values[j + 1:k_inst])

                        highest_weights = new_weights.copy()
                        highest_values = new_values.copy()
                        print("Highest Values:", highest_values)
                        break
                print("Vizinho", i, "calculado como", new_value)
            return highest_values.copy(), highest_weights.copy()


        #best_value = self.run_episode(weights)
        #best_weights = np.array(weights).reshape(5, -1)
        # Learning process
        k_inst = 10
        iter = 0
        iter_unchanged = 0
        epsilon = 20

        highest_values = []
        highest_weights = []
        highest_weights.append(weights.copy())
        highest_values.append(self.run_episode(weights))
        for j in range(1, k_inst):
            highest_weights.append([random.uniform(-1, 1) for i in range(0, len(weights))])
            highest_values.append(self.run_episode(highest_weights[j]))



        try:
            while True:
                print()
                print("Iteration", iter, "after", iter_unchanged, "unchanged iterations. Epsilon actual value is", epsilon)
                print()

                old_weights = highest_weights.copy
                highest_values, highest_weights = compute_best_neighbours( generate_neighbours(highest_weights, epsilon),
                                                                              highest_values, highest_weights, k_inst)
                print()
                print("Highest Score: ", highest_values[0], highest_values[1], highest_values[2])
                print()
                if np.sum(old_weights) == np.sum(highest_weights):
                    iter_unchanged +=1
                else:
                    iter_unchanged = 0
                iter += 1
                if iter_unchanged > 3:
                    epsilon *= 0.5
                    if epsilon < 0.0001:
                        epsilon = 100


        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass

        best_weights = np.array(weights).reshape(5, -1)
        # raise NotImplementedError("This Method Must Be Implemented")

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
