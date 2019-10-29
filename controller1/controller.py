import controller_template as controller_template
import numpy as np
PARENTS = 10
MUTATION_RATIO = 0.05
MUTATION_EPSILON = 0.005

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)

        # This initialization value make it think we are moving forward in the 1st step
        self.old_next_checkpoint = float("-inf")

        # Initialize population values
        self.elements = None
        

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

        action_values = np.sum(features * parameters, axis=1)

        return np.argmax(action_values) + 1

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

        
        # Compute feature_lens
        
        ctrl_temp = Controller(chosen_track, bot_type=None, evaluate=False)
        fake_sensors = [53, 66, 100, 1, 172.1353274581511, 150, -1, 0, 0]
        features_len = len(ctrl_temp.compute_features(fake_sensors))

        elements = np.random.uniform(-1, 1, size=(PARENTS, features_len * 5))
        
        # Generations
        gen = 0

        fitness = np.array([self.run_episode(element) for element in elements])
        couples = np.zeros(PARENTS, 2)
        

        next_gen = np.zeros(*(PARENTS, features_len * 5)) # Auxiliar Array
        best_weights = elements[np.argmax(fitness)]
        best_fitness = max(fitness)

        # Learning process
        try:
            for i in range(PARENTS):
                fitness[i] = self.run_episode(elements[i])
        
            while True:
                
                # Resets the offspring count
                offspring = 0

                # Next Generation
                gen += 1
                
                min_fit = min(fitness)
                max_fit = max(fitness)
                fitness = [normalize(fit, min_fit, max_fit) for fit in fitness]

                # Couples from current generation
                couples = np.random.choice(list(range(PARENTS)), p=fitness, size=(PARENTS, 2))

                # Breeding
                for offspring, (p1, p2) in enumerate(couples):
                    for chromosome_it in range (features_len * 5): 
                        next_gen[offspring, chromosome_it] = \
                            np.random.choice([elements[p1, chromosome_it], elements[p2, chromosome_it]])

                elements = next_gen

                # Mutation
                for i in range(len(elements)):
                    for j in range(features_len * 5):
                        elements[i, j] += np.random.choice([0, np.random.uniform(-1, 1) * MUTATION_EPSILON], [1 - MUTATION_RATIO, MUTATION_RATIO])

                # Evaluate
                fitness = np.array([self.run_episode(element) for element in elements])
                
                print (*fitness)

                if max(fitness) > best_fitness:
                    best_fitness = max(fitness)
                    best_weights = elements[np.argmax(fitness)]

                pass

        except KeyboardInterrupt:  # To be able to use CTRL+C to stop learning
            pass

        # Return the weights learned at this point
        return best_weights

    @staticmethod
    def normalize(val, min, max):
        return 2 * ((val - min) / (max - min)) - 1
