import controller_template as controller_template
import numpy as np
import random


class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")
        
        # Hyperparameters
        self.neighbourhood_size = 30

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
        def generate_neighbours(weights, epsilon):
            neighbours = [weights.copy()] # Add the original weights in the array
            
            for i in range(self.neighbourhood_size):
                new_neighbour = [weight + np.random.uniform(-1, 1) * epsilon for weight in weights]
                neighbours.append(new_neighbour)
                
            print(len(neighbours), "novos vizinhos gerados.")
            return neighbours

        def compute_best_neighbour(neighbours):
            print("Computing ", len(neighbours), " neighbours.")
            best_value = self.run_episode(neighbours[0]) # First neighbour is the original one
            bestNeighbour = neighbours[0].copy()
            
            for i, weight in enumerate(neighbours[1:]):
                new_value = self.run_episode(weight)
                print("Vizinho", i, "calculado como", new_value)
                if new_value > best_value:
                    print("New best value found:", new_value, ">", best_value)
                    best_value = new_value
                    bestNeighbour = weight.copy()
            return best_value, bestNeighbour

        iter = 0
        iter_unchanged = 0
        epsilon = 0.05
        best_weights = weights.copy()
        best_value = self.run_episode(best_weights)
        print("Best Score: ", best_value)
        # Learning process
        try:
            while True:
                print("Iteration", iter, "after", iter_unchanged, "unchanged iterations. Epsilon actual value is", epsilon)

                old_value = best_value
                best_value, best_weights = compute_best_neighbour(generate_neighbours(best_weights.copy(), epsilon).copy())

                print("Best Score: ", best_value, end='\n\n')

                if best_value == old_value:
                    iter_unchanged +=1
                    epsilon *= 0.8
                else:
                    iter_unchanged = 0
                    
                iter += 1
                if iter_unchanged > 10:
                    epsilon = 10*random.random()


        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
