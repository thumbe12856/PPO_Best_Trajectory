import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


# This function selects the probability distribution over actions
from baselines.common.distributions import make_pdtype

# Convolution layer
def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))


# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def categorical_sample(logits, d):
    # tf.multinomial = sample num_sample from logits
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    #return tf.one_hot(value, d)
    return value

"""
This object creates the PPO Network architecture
"""
class PPOPolicy(object):
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse = False):
        # This will use to initialize our kernels
        gain = np.sqrt(2)

        self.action_size = 13

        # Based on the action space, will select what probability distribution type
        # we will use to distribute action in our stochastic policy (in our case DiagGaussianPdType --> CategoricalPdType
        # aka Diagonal Gaussian, 3D normal distribution
        self.pdtype = make_pdtype(action_space)

        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        # Create the input placeholder
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        # Normalize the images
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        """
        Build the model
        3 CNN for spatial dependencies
        Temporal dependencies is handle by stacking frames
        (Something funny nobody use LSTM in OpenAI Retro contest)
        1 common FC
        1 FC for policy
        1 FC for value
        """
        with tf.variable_scope("model", reuse = reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)


            size = 256
            x = flatten(fc_common)
            x = tf.expand_dims(x, [0]) #[1, None, 512]
            #lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            lstm = tf.nn.rnn_cell.LSTMCell(size, name='basic_lstm_cell', state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(scaled_images)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
            self.state_in = [c_in, h_in]

            state_in = rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                                        lstm, x, initial_state=state_in, 
                                        sequence_length=step_size, time_major=False
                                        )
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])
            self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

            #self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)

            # [0, :] means pick action of first state from batch. Hardcoded b/c
            # batch=1 during rollout collection. Its not used during batch training.
            #self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
            #self.sample = categorical_sample(self.logits, ac_space)[0, :]
            #self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]
            # pi = logits
            # pd = dirstribution
            self.pi = linear(x, self.action_size, "action", normalized_columns_initializer(0.01))
            a0 = categorical_sample(self.pi, self.action_size)
            #a0 = categorical_sample(self.pi, self.action_size)[0, :]
            
            action_shape_list = a0.shape.as_list()
            logits_shape_list = self.pi.get_shape().as_list()[:-1]
            for xs, ls in zip(action_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)

            a_one_hot = tf.one_hot(a0, self.pi.get_shape().as_list()[-1])
            neglogp0 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pi, labels=a_one_hot)
        
            #self.pd = tf.nn.softmax(self.pi, dim=-1)[0, :]
            #neg = tf.fill(tf.shape(self.pi), -1.0)
            #neglogp0 = tf.multiply(self.pi, neg)
            
            # This build a fc connected layer that returns a probability distribution
            # over actions (self.pd) and our pi logits (self.pi).
            #--self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)
            #--print(self.pi)

            # Calculate the v(s)
            #--vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]
            vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
            #vf = fc_layer(x, 1, activation_fn=None)[:, 0]

        # Take an action in the action distribution (remember we are in a situation
        # of stochastic policy so we don't always take the action with the highest probability
        # for instance if we have 2 actions 0.7 and 0.3 we have 30% chance to take the second)
        #--a0 = self.pd.sample()

        # Calculate the neg log of our probability
        #--neglogp0 = self.pd.neglogp(a0)

        # Function use to take a step returns action to take and V(s)
        #def step(state_in, *_args, **_kwargs):
        def step(ob, c, h):

            # return a0, vf, neglogp0
            #return sess.run([a0, vf, neglogp0], {inputs_: state_in})
            return sess.run([a0, vf, neglogp0] + self.state_out, {inputs_: ob, c_in: c, h_in: h})

        # Function that calculates only the V(s)
        #def value(state_in, *_args, **_kwargs):
        def value(ob, c, h):
            #return sess.run(vf, {inputs_: state_in})
            return sess.run(vf, {inputs_: ob, c_in: c, h_in: h})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            #return sess.run(a0, {inputs_: state_in})
            return sess.run(a0, {c_in: c, h_in: h})

        def get_initial_features():
            # Call this function to get reseted lstm memory cells
            #return sess.run(self.state_init)
            return self.state_init

        def entropy():
            a0 = self.pi - tf.reduce_max(self.pi, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)


        self.inputs_ = inputs_
        self.c_in = c_in
        self.h_in = h_in
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action
        self.get_initial_features = get_initial_features
        self.entropy = entropy
