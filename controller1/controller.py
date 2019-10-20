import controller_template as controller_template
import numpy as np


class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # Quantity of neighbours to generate and to get
        self.neighbourhood_size = 50
        self.best_neighbourhood_size = 15

        # Iterate until mean moves less than this threshold
        self.epsilon = 1e-5

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")

        # Placeholder parameters for CMA-ES
        self.mean = None
        self.cov_matrix = None

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

        return np.argmax(summed_values)

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
        ontrack_f = self.normalize(on_track, 0, 1)
        velocity_f = self.normalize(car_velocity, 0, 200)

        # TODO FEATURES
        # Know how to leave grass
        # Remember last action (left, center, right)
        # Time without choosing to turn
        # Faster than threshold
        # Try to centralize better (left - right)

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

        # Make weights a np array
        weights = np.array(weights)

        # Initialize CMA-ES values
        self.mean = weights
        self.cov_matrix = np.identity(weights.shape[0])

        best_fitness = float("-inf")
        best_fitness_at_iter = 0
        best_weights = np.array(weights).reshape(5, -1)

        # Learning process
        try:
            loss = float("inf")
            iter = 1
            trials = 1  # Try to make backsearch
            while loss > self.epsilon:
                # Sample from a multivariate normal distribution
                params = np.random.multivariate_normal(
                    self.mean, self.cov_matrix, self.neighbourhood_size)

                # Evaluate with the generated parameters
                fitness = np.array([self.run_episode(param)
                                    for param in params])

                # Sort the parameters according to the evaluations,
                # taking the `self.best_neighbourhood_size` better ones
                best_params = params[fitness.argsort(
                )][::-1][:self.best_neighbourhood_size]

                # Compute the new mean
                new_mean = np.sum(best_params, axis=0) / \
                    self.best_neighbourhood_size

                # Compute new covariance matrix
                diff = best_params - self.mean
                matmul = np.matmul(diff[..., np.newaxis], np.transpose(
                    diff[..., np.newaxis], [0, 2, 1]))
                new_cov_matrix = np.sum(
                    matmul, axis=0) / self.best_neighbourhood_size

                # Compute loss
                loss = np.linalg.norm(self.mean - new_mean)
                print('Iter', iter, 'Loss:', loss, 'Max:', max(fitness))

                # Update mean and cov_matrix
                self.mean = new_mean
                self.cov_matrix = new_cov_matrix

                # Update best weight
                max_fitness = max(fitness)
                if max_fitness > best_fitness:
                    print('Updated best fitness to {}'.format(max_fitness))
                    best_fitness = max_fitness
                    best_fitness_at_iter = iter
                    best_weights = best_params[0]

                # Update iteration
                iter += 1

                # If can still backsearch, do it, to avoid local minimums
                if (iter - 5 > best_fitness_at_iter or loss < self.epsilon) and trials < 3:
                    trials += 1
                    self.mean = best_weights
                    self.cov_matrix = np.identity(best_weights.shape[0])
                    best_fitness_at_iter = iter - 1

        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
