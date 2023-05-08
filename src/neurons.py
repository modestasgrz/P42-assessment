import numpy as np


class SensorNeuron:
    # Activation function produces value from 0.0 to 1.0
    # in regards to the input and neuron's type
    #
    def __init__(self, type_id):
        self.type_id = type_id
        self.activation = {
            0: self.state,  # SN_0 - state sensory neuron
            1: self.state,  # SN_1 - state sensory neuron
            2: self.state,  # SN_2 - state sensory neuron
            3: self.state,  # SN_3 - state sensory neuron
            4: self.state,  # SN_4 - state sensory neuron
            5: self.reward,  # SN_6 - reward sensory neuron
            6: self.reward,  # SN_7 - reward sensory neuron
            7: self.reward,  # SN_8 - reward sensory neuron
            8: self.coins,  # SN_9 - coins sensory neuron
            9: self.coins,  # SN_10 - coins sensory neuron
            10: self.flag_get,  # SN_11 - flag get sensory neuron
            11: self.flag_get,  # SN_12 - flag get sensory neuron
            12: self.life,  # SN_13 - life sensory neuron
            13: self.score,  # SN_14 - score sensory neuron
            14: self.status,  # SN_16 - status sensory neuron
            15: self.time,  # SN_17 - time sensory neuron
            16: self.x_pos,  # SN_18 - x position sensory neuron
            17: self.y_pos,  # SN_19 - y position sensory neuron
        }.get(type_id, self.no_action)

    def __str__(self):
        return f"SN_{self.type_id}"

    def __call__(self, x, weight):
        # return self.activation(x) * weight
        return self.activation(x) * weight

    def state(self, x):
        return x  # If x - stadartized and normalized, will need to use softmax on state extraction nn

    def reward(self, x):
        # reward is clipped into range (-15; 15)
        # making reward <- (0; 1) (regarding algorithm implementation video
        # "sensory neurons provide value in a range (0; 1)")
        return (x + 15.0) / 30.0

    def coins(self, x):
        # According to some unverified sources on internet - hard limit on coins that Mario can hold = 9999
        # This number is used normalizing coins input, assuming that coin value can be in range (0; 9999)
        return x / 9999.0

    def flag_get(self, x):
        # Flag value is either True (= 1) or False (= 0)
        if x:
            return 1.0
        return 0.0

    def life(self, x):
        # Maximum number of available lifes is a member of {3; 2; 1}
        # Normalizing out of 3
        return x / 3.0

    def score(self, x):
        # According to https://www.youtube.com/watch?v=Pnu72O4PREE&t=355s
        # The highest score on Mario = 3,663,400 - should score value be standartized by this super high number?
        # My concern is, whether this activation has any significant influece on the output of the network,
        # when using this standartization
        return x / 3663400.0

    def status(self, x):
        # Mario's status, i.e., {'small', 'tall', 'fireball'}
        # 'tall' - better than small, fireball, I assume, as well
        if x == "small":
            return 0.0
        if x == "tall":
            return 1.0
        return 1.0

    def time(self, x):
        # Time starts at 400
        return x / 400.0

    def x_pos(self, x):
        # Without glitching through game mechanics, the maximum x could be 700 (very hardly of course)
        return x / 700.0

    def y_pos(self, x):
        # Without glitching through game mechanics, the maximum y could be 400 (very hardly of course)
        return x / 400.0

    def no_action(self, x):  # pylint: disable = unused-argument
        # No action selected for this neuron - action unindentified
        return 0.0


class InnerNeuron:
    # Simple standard neuron - perceptron
    def __init__(self, type_id):
        self.type_id = type_id

    def __str__(self):
        return f"IN_{self.type_id}"

    def __call__(self, x, weight):
        # Here x is considered to be a list of values
        return np.tanh(sum(x)) * weight


class ActionNeuron:
    # Action neuron should provide the number of an action they trying to invoke
    # This number at init matches neuron's type_id
    #
    # However, the dictionary of actions is presented here
    #
    # self.action = { # 4 bits are enough
    #       0 : ['NOOP'],
    #       1 : ['right'],
    #       2 : ['right', 'A'],
    #       3 : ['right', 'B'],
    #       4 : ['right', 'A', 'B'],
    #       5 : ['A'],
    #       6 : ['left'],
    #       7 : ['left', 'A'],
    #       8 : ['left', 'B'],
    #       9 : ['left', 'A', 'B'],
    #       10 : ['down'],
    #       11 : ['up'],
    #   }
    #
    # self.__call__() method calculates the outcome of inputs, but does not activate the action
    #
    # Action is activated, when particular neuron is selected in brain
    # by taking this neuron and calling self.activate() method on it
    # self.activate() simply provides the number of the selected action
    # which matches type_id of the neuron
    #
    # So, this neuron is a simple perceptron with additional, task-related, activate feature
    #
    def __init__(self, type_id):
        self.type_id = type_id

    def __str__(self):
        return f"AN_{self.type_id}"

    def __call__(self, x):
        return np.tanh(sum(x))

    def activate(self):
        return self.type_id
