import controller_template as controller_template
import numpy as np
import random


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
        def generate_neighbours(best_weights, epsilon, many):
            neighbours = []
            neighbours.append(best_weights.copy())
            for i in range(len(best_weights)): #Não faz mudancas no vizinho de maior pontuacao
                new_neighbour = best_weights.copy()
                new_neighbour[i] += random.choice((-1, 1)) * epsilon
                if many == 1:
                    for n in range(len(best_weights)):
                        if random.randint(0, 10) == 1 and i != n:
                            new_neighbour[n] += epsilon
                        elif random.randint(0, 10) == 1 and i != n:
                            new_neighbour[n] -= epsilon
                neighbours.append(new_neighbour.copy())
            print(len(neighbours), "novos vizinhos gerados.")
            return neighbours.copy()

        def compute_best_neighbour(neighbours):
            print("Computing ", len(neighbours), " neighbours.")
            best_value = self.run_episode(neighbours[0])
            old_value = best_value
            bestNeighbour = neighbours[0].copy()
            print("Best Score:", best_value)
            for i in range(1,len(neighbours)):
                new_value = self.run_episode(neighbours[i])
                print("Vizinho", i, "calculado como", new_value)
                if new_value > best_value:
                    print("New best value found:", new_value, ">", best_value)
                    best_value = new_value
                    bestNeighbour = neighbours[i].copy()
            return best_value, bestNeighbour.copy(), old_value

        #best_value = self.run_episode(weights)
        #best_weights = np.array(weights).reshape(5, -1)
        iter = 0
        iter_unchanged = 0
        epsilon = 1*random.random()
        many = 1
        best_weights = weights.copy()
        print(self.run_episode(weights))
        print(self.run_episode(weights))
        # Learning process
        try:

            while True:
                print("Iteration", iter, "after", iter_unchanged, "unchanged iterations. Epsilon actual value is", epsilon)

                best_value, best_weights, old_value = compute_best_neighbour(generate_neighbours(best_weights.copy(), epsilon, 1).copy())

                print("Best Score:  ", best_value)
                print("Old Score:   ", old_value)
                print()

                if best_value == old_value:
                    iter_unchanged +=1
                else:
                    iter_unchanged = 0
                iter += 1
                if iter_unchanged > 0:
                    epsilon = 1*random.random()
                    if iter_unchanged%3 == 0:
                        print("3")
                        many += 1
                        if many > 1:
                            many = 0
                        epsilon = 0.1*random.random()
                    if iter_unchanged%4 == 0:
                        print("4")
                        epsilon = 10*random.random()
                    if iter_unchanged%5 == 0:
                        print("5")
                        epsilon = 100*random.random()



        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass


        # raise NotImplementedError("This Method Must Be Implemented")

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
