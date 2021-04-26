'''
Selector-predictor implementations
'''
from collections import defaultdict
import numpy as np

from methods import *
from selector_predictor import *

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

EPS = np.finfo(tf.float32.as_numpy_dtype).tiny

def sample_concrete(params, training=None):
    ''' Concrete distribution sampler '''
    loga, temp = params
    stretch_eps = 0.1

    shape = tf.shape(loga)
    noise = tf.random.uniform(shape, EPS, 1.0 - EPS)
    concrete = tf.nn.sigmoid((tf.math.log(noise) - tf.math.log(1 - noise) + loga) / (EPS + temp))

    concrete = concrete * (1 + 2 * stretch_eps) - stretch_eps
    return tf.clip_by_value(concrete, 0, 1)


def sample_bin(p):
    ''' Bernoulli distribution sampler '''
    noise = tf.random.uniform(tf.shape(p), 0, 1)
    return tf.cast(noise <= p, tf.float32)


def make_generator(func, params):
    ''' For tf.keras.Model.fit() '''
    while True:
        yield func(**params)


def sample_data(bs = 128, mu = [], label = [], std = 0.125):
    n_data = bs // mu.shape[0] + 1
    offset = np.random.randint(n_data * mu.shape[0] - bs)
    X = tf.tile(tf.convert_to_tensor(mu, 'float32'), (n_data, 1))[offset:bs+offset]
    X = X + tf.random.normal(X.shape, 0., std)
    Y = tf.tile(tf.convert_to_tensor(label, 'float32'), (n_data,))[offset:bs+offset]
    return X, Y


def binarise_solution_from_array(s_hat):
    d = s_hat.shape[-1]
    mult = 1 << np.arange(0, d)
    return np.sum(s_hat * mult, axis=-1, dtype='int')


class L2X_Concrete(tf.keras.layers.Layer):
    ''' Adapted from the original code:
    https://github.com/Jianbo-Lab/L2X/blob/master/synthetic/explain.py '''
    def __init__(self, t, k, **kwargs):
        self.t = tf.constant(t, 'float32')  # temperature
        self.k = k  # selected variables
        super(L2X_Concrete, self).__init__(**kwargs)

    def call(self, logits):
        ''' logits: tf.tensor (BATCH_SIZE, d) '''
        assert len(logits.shape) == 2
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]

        uniform = tf.random.uniform(shape =(batch_size, self.k, d),   # self
                minval = EPS,
                maxval = 1.0 - EPS )

        gumbel = - tf.math.log(-tf.math.log(uniform))
        noisy_logits = (gumbel + tf.expand_dims(logits, 1)) / self.t
        samples = tf.math.softmax(noisy_logits)
        samples = tf.math.reduce_max(samples, axis = 1)  # (BATCH_SIZE, d)

        return samples

    def compute_output_shape(self, input_shape):
        return input_shape



class SelectionStability(tf.keras.callbacks.Callback):
    ''' A modified tf.EarlyStopping for the estimated selection solution. '''
    def __init__(self, x, eval_func, patience = 3, after = 0, **kwargs):
        super(SelectionStability, self).__init__(**kwargs)
        self.last_sol = 0
        self.first_iteration = True
        self.x = x                   # probably mu of our problem
        self.eval_func = eval_func   # get selection solution
        self.wait = 0
        self.patience = patience
        self.after = after

    def on_epoch_end(self, epoch, logs=None):
        if self.first_iteration:
            self.last_sol = self.eval_func(self.x)
            self.first_iteration = False
        new_sol = self.eval_func(self.x)

        if self.after > 0:
            self.after -= 1
        else:
            if tf.reduce_all(tf.equal(new_sol, self.last_sol)):
                self.wait += 1
                if self.wait >= self.patience:
                    self.model.stop_training = True
            else:
                self.wait = 0

        self.last_sol = new_sol


class L2X(Explainer):
    def __init__(self, params):
        super().__init__('L2X')
        self.params = params
        self.callbacks = []

    def create_model(self):
        ''' Almost same as original paper. Our synthetic problem are quite simple,
        thus we do not need to crank up F that much, for that we simply check that
        the overall prediction accuracy is not abnormally low during training.
        As for selection, quite oppositely, augmenting F degrades the selection
        quality (more degenerate solutions appear). '''
        F = self.params['F']
        INPUTDIM = self.params['input_dim']

        # Selector
        x = Input(INPUTDIM)
        fs = Dense(F, activation='selu')(x)
        fs = Dense(F, activation='selu')(fs)
        fs = Dense(INPUTDIM)(fs)
        mask = L2X_Concrete(self.params['t'],
                            min(self.params['input_dim'], self.params['k']) )(fs)

        # Predictor
        restricted_x = Lambda(lambda l: l[0] * l[1])([x, mask])
        z = restricted_x
        z = Dense(F, activation='selu')(z)
        z = BatchNormalization()(z)
        z = Dense(F, activation='selu')(z)
        z = BatchNormalization()(z)
        z = Dense(1, activation='sigmoid')(z)
        y = z

        self.get_mask = tf.keras.backend.function([x], fs)
        self.m = Model(inputs=[x], outputs=[y])
        self.m.compile(loss='binary_crossentropy', optimizer=Adam(1e-3), metrics=['acc'])

    def add_callback(self, mu, patience=3, after=10):
        self.callbacks = [SelectionStability(mu, self.get_selection_tf,
                                    patience=patience, after=after)]

    def train(self, data_gen, n_epochs = 100, verbose=0):
        ''' Train with an early stopping callback if the selection stabilises. '''
        self.m.fit(data_gen, steps_per_epoch=100, epochs=n_epochs, verbose=verbose,
                    callbacks=[self.callbacks])

    def explain(self, x):
        ''' Returns selection logits (after training there is no sampling). '''
        return self.get_mask(x)

    def get_selection_tf(self, x):
        ''' tf.tensors for our custom callback. '''
        logit = self.get_mask(x)
        s_hat = tf.greater_equal(logit,
                (tf.reduce_max(logit, axis=-1, keepdims=True) * self.params['threshold']) )
        return s_hat  # boolean

    def get_selection(self, x):
        ''' Selection solution as array. '''
        logit = self.get_mask(x)
        s_hat = 1.0 * (logit >= (np.max(logit, axis=-1, keepdims=True) * self.params['threshold']) )
        return s_hat



class Invase_Model(tf.keras.Model):
    ''' Reimplementation of INVASE Minus.
        * mode: "l0" is the implementation that should be derived from the paper
        "l1" is how they actually do it on their repository.
    '''
    def __init__(self, input_dim, F=64, reg=0.1, mode='l0', threshold=0.8):
        self.mode = mode
        self.threshold = threshold

        super(Invase_Model, self).__init__()
        self.reg_multiplier = tf.constant(reg)

        x = Input(input_dim)
        actor = Dense(F, activation='selu')(x)
        actor = Dense(F, activation='selu')(actor)
        actor = Dense(input_dim, activation='sigmoid')(actor)
        mask = Lambda(sample_bin, name='sampler')(actor)
        self.m_actor = Model(inputs=[x], outputs=[actor, mask])

        x2 = Input(input_dim)
        critic = Dense(F, activation='selu')(x2)
        critic = BatchNormalization()(critic)
        critic = Dense(F, activation='selu')(critic)
        critic = BatchNormalization()(critic)
        critic = Dense(1, activation='sigmoid')(critic)
        self.m_critic = Model(inputs=[x2], outputs=[critic])

    def compile(self):
        super(Invase_Model, self).compile()
        self.opt = Adam(1e-4)
        self.train_loss1 = tf.keras.metrics.Mean(name='critic_loss')
        self.train_loss2 = tf.keras.metrics.Mean(name='actor_loss')
        self.train_acc = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.test_acc = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        self.m_actor.compile(optimizer=self.opt)
        self.m_critic.compile(optimizer=self.opt)

    def call(self, x):
        actor, mask = self.m_actor(x)
        return self.m_critic(x * mask)

    @staticmethod
    def safe_log(x):
        return tf.math.log((1 - 2 * EPS) * x + EPS)

    def train_step(self, data):
        x, y = data
        base_actor, base_mask = self.m_actor(x, training=True)
        base_p = self.m_critic(x * base_mask, training=True)
        reward = -tf.keras.losses.binary_crossentropy(y, base_p[:,0]) # max when y = base_p
        if self.mode == 'l0':
            reward = reward - self.reg_multiplier * tf.reduce_mean(base_mask, axis=1)

        # critic training
        with tf.GradientTape() as tape:
            xr = x * base_mask
            p = self.m_critic(xr, training=True)
            loss_value1 = tf.keras.losses.binary_crossentropy(y, p[:,0])
        grads = tape.gradient(loss_value1, self.m_critic.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.m_critic.trainable_weights))

        # actor training
        with tf.GradientTape() as tape:
            actor, mask = self.m_actor(x, training=True)
            loss_value2 = reward * tf.reduce_sum(
                                base_mask * self.safe_log(actor)
                                + (1. - base_mask) * self.safe_log(1. - actor), axis=1)
            if self.mode == 'l1':
                loss_value2 = loss_value2 - self.reg_multiplier * tf.reduce_mean(actor, axis=1)
            loss_value2 = -tf.reduce_mean(loss_value2)

        grads = tape.gradient(loss_value2, self.m_actor.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.m_actor.trainable_weights))

        self.train_loss1.update_state(loss_value1)
        self.train_loss2.update_state(loss_value2)
        self.train_acc.update_state(y, p[:,0])
        return {'c_loss': self.train_loss1.result(),
                'a_loss': self.train_loss2.result(),
                'acc': self.train_acc.result() }

    def test_step(self, data):
        x, y = data
        logit = self.m_actor(x, training=False)[0]
        mask = tf.cast(tf.greater_equal(logit,
                (tf.reduce_max(logit, axis=-1, keepdims=True) * self.threshold) ), tf.float32)

        p = self.m_critic(x * mask, training=False)
        self.test_acc.update_state(y, p[:,0])
        return { 'acc': self.test_acc.result() }



class INVASE(L2X):
    def __init__(self, params):
        super(INVASE, self).__init__(params)
        self.name = 'INVASE'
        self.params = params

    def create_model(self):
        self.m = Invase_Model(self.params['input_dim'],
                            F = self.params['F'],
                            reg = self.params['reg'],
                            mode = self.params['mode'])
        self.m.compile()

    def get_mask(self, x):
        return self.m.m_actor(x)[0]
