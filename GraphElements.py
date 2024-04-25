import tkinter
import time
import AmeCommunication
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import random
from collections import deque
import itertools
import pickle
file_name = "data_file.data"
file = open(file_name, "w")
seed_value = 0

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


file_name = "data_file.data"
file = open(file_name, "w")

class ChartData(object):
    def __init__(self, size_max=0):
        self.x = []
        self.y = []
        self.last_x = 0
        self.size_max = size_max

    def add_point(self, x, y):
        if (x - self.last_x) ** 2 < 0.0001:
            return
        self.last_x = x
        self.x.append(1.0 * x)
        self.y.append(1.0 * y)
        if 0 < self.size_max < len(self.x):
            self.x.pop(0)
            self.y.pop(0)

    @staticmethod
    def get_scaled_vector(vector, data_range, canvas_width):
        scaled = []
        for val in vector:
            scaled.append((val - data_range[0]) * canvas_width / (data_range[1] - data_range[0]))
        return scaled

    def get_line(self, horizontal_range, canvas_width, vertical_range, canvas_height):
        x_scaled = self.get_scaled_vector(self.x, horizontal_range, canvas_width)
        y_scaled = self.get_scaled_vector(self.y, vertical_range, canvas_height)
        line = []
        for i, j in zip(x_scaled, y_scaled):
            if 0 <= i < canvas_width and 0 <= j < canvas_height:
                line.append(i)
                line.append(canvas_height - 1 - j)
        return line


class Chart(object):
    def __init__(self, canvas, horizontal_range, vertical_range, nb_max_points=5000):
        self.canvas = canvas
        self.data = ChartData(nb_max_points)
        self.x_range = horizontal_range
        self.y_range = vertical_range
        self.line = None

    def ensure_visible(self, x, y):
        width = 1.0 * self.canvas.winfo_width()
        height = 1.0 * self.canvas.winfo_height()
        scaled_width = 1.0 * (self.x_range[1] - self.x_range[0])
        scaled_height = 1.0 * (self.y_range[1] - self.y_range[0])
        x_step = scaled_width / width
        y_step = scaled_height / height
        x_max_visible = self.x_range[0] + (width - 1.0) * x_step
        y_max_visible = self.y_range[0] + (height - 1.0) * y_step
        if x > x_max_visible:
            self.x_range = (self.x_range[0] + x - x_max_visible, self.x_range[1] + x - x_max_visible)
        if x < self.x_range[0]:
            self.x_range = (x, x + self.x_range[1] - self.x_range[0])
        if y >= y_max_visible:
            self.y_range = (self.y_range[0], y + 0.1 * abs(y) + 1e-6)
        if y <= self.y_range[0]:
            self.y_range = (y - 0.1 * abs(y) - 1e-6, self.y_range[1])

    def add_point(self, x, y):
        self.ensure_visible(x, y)
        self.data.add_point(x, y)

    def update(self):
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        l = self.data.get_line(self.x_range, width, self.y_range, height)
        if len(l) > 2:
            new_line = self.canvas.create_line(l)
            if self.line:
                self.canvas.delete(self.line)
            self.line = new_line

    def clean(self):
        self.data.x = []
        self.data.y = []
        self.data.last_x = 0

class Chrono(object):
    def __init__(self):
        self.time_start = time.time()

    def start(self):
        self.time_start = time.time()

    def restart(self):
        ret = time.time() - self.time_start
        self.time_start = time.time()
        return ret

    def get_time(self):
        return time.time() - self.time_start

    def set_time(self, time_origin):
        self.time_start = time.time() - time_origin
####################################################

class PIDControllerOuter(object):
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, tau=0.1, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau = tau
        self.last_time = None
        self.last_output = None
        self.last_error = 0.0
        self.integral = 0.0
        self.output_limits = output_limits
        self.dummy_state = 0.0
        
    def set_parameters(self,kp,ki,kd):
        self.kp = kp
        self.ki = ki 
        self.kd = kd


    def calculate(self, error, current_time):
        if self.last_time is None:
            self.last_time = current_time
            return 0

        dt = current_time - self.last_time
        if dt <= 0:
            return 0

        # Update integral and derivative state with anti-windup via clamping
        self.integral += error * dt
        if self.last_output is not None:
            # Check if we have upper limit and if we do, check if the error and output are greater than this limit
            if self.output_limits[1] is not None and error > 0 and self.last_output >= self.output_limits[1]:
                self.integral -= error * dt  # Anti-windup for upper limit
            # Check if we have lower limit and if we do, check if the error and output are less than this limit
            elif self.output_limits[0] is not None and error < 0 and self.last_output <= self.output_limits[0]:
                self.integral -= error * dt  # Anti-windup for lower limit


        # Derivative state with filter
        d_error = error - self.last_error
        self.dummy_state += dt * (d_error - self.dummy_state) / self.tau
        derivative = self.dummy_state / self.tau

        # Compute output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_output = output
        self.last_error = error
        self.last_time = current_time

        # Apply output limits
        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        return output
    

class Critic(object):
    def __init__(self, state_size, action_size,
                learning_rate,tau=0.1):
        
        # Store parameters
        self._tau = tau
        self._learning_rate = learning_rate

        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.model = self._generate_model(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(self._learning_rate)
        self.target_model = self._generate_model(state_size, action_size)

    def load_models(self, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def get_action_gradients(self, states, actions):
        # Convert actions to TensorFlow tensor if it's not already
        actions = tf.convert_to_tensor(actions)
        states = tf.convert_to_tensor(states)

        # Ensure both states and actions have at least one dimension
        if len(actions.shape) == 0:
            actions = tf.expand_dims(actions, axis=0)
        if len(states.shape) == 0:
            states = tf.expand_dims(states, axis=0)

        # Compute action gradients
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions], training=True)
            q_value_gradients = tape.gradient(q_values, actions)

        return q_value_gradients

    def train(self, states, actions, q_targets):
        # Ensure both states and actions are numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        q_targets = np.array(q_targets)

        # Check and align batch sizes
        if states.shape[0] != actions.shape[0]:
            raise ValueError("Mismatch in batch size of states and actions")

        # Train the model
        with tf.GradientTape() as tape:
            q_values = self.model([states, actions], training=True) 
            loss = self.loss_function(q_targets, q_values)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss
    
    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        target_weights = [self._tau * main_weight + (1 - self._tau) *
                          target_weight for main_weight, target_weight in
                          zip(main_weights, target_weights)]
        self.target_model.set_weights(target_weights)

    def update_target_model(self, target_model, tau):
        # Soft update the target model weights
        new_weights = [tau * w + (1 - tau) * tw for w, tw in zip(self.model.weights, target_model.weights)]
        target_model.set_weights(new_weights)


    def _generate_model(self, state_size, action_size):
        state_input = layers.Input(shape=(state_size,))
        action_input = layers.Input(shape=(action_size,))
        concat = layers.Concatenate()([state_input, action_input])
        
        net = layers.Dense(32, activation='relu')(concat)
        outputs = layers.Dense(1)(net)
        
        model = tf.keras.Model([state_input, action_input], outputs)
        return model



class Actor(object):
    def __init__(self, state_size, action_size,learning_rate, tau=0.1):

        self._tau = tau
        self._learning_rate = learning_rate
        self._state_size = state_size
        self._action_size = action_size

        self.model = self._generate_model(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self._target_model = self._generate_model(state_size, action_size)


    def load_models(self, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def _compute_loss(self, predictions, gradients):
        return tf.reduce_mean(tf.square(predictions - gradients))

    def train(self, states, gradients):
        states = np.array(states)
        if len(states.shape) == 1:
            states = np.expand_dims(states, axis=0)
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = self._compute_loss(predictions, gradients)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self._target_model.get_weights()
        target_weights = [self._tau * main_weight + (1 - self._tau) *
                          target_weight for main_weight, target_weight in
                          zip(main_weights, target_weights)]
        self._target_model.set_weights(target_weights)

    def _generate_model(self, state_size, action_size):
        inputs = layers.Input(shape=(state_size,))
        net = layers.Dense(16, activation='relu')(inputs)
        outputs = layers.Dense(action_size, activation='tanh')(net)
        scaled_outputs = outputs * [3500, 5000, 2] 
        model = tf.keras.Model(inputs, scaled_outputs)
        return model

class DDPG(object):
    def __init__(self, state_size, action_size, actor_learning_rate=0.0001,
                 critic_learning_rate=0.001, batch_size=32, discount=0.99,
                 memory_size=1000, tau=0.1, not_prev_train = True):

        self._discount = discount
        self._batch_size = batch_size
        self._memory_size = memory_size
        self.noise_std_dev = 5
        self.not_prev_train = not_prev_train
        self._actor = Actor(state_size=state_size, action_size=action_size,
                            learning_rate=actor_learning_rate, tau=tau)

        self._critic = Critic(state_size=state_size, action_size=action_size,
                            learning_rate=critic_learning_rate, tau=tau)

        self._memory = deque()

    def add_gaussian_noise(self,parameters, reward):
        self.update_noise_level(reward)
        noise = np.random.normal(0, self.noise_std_dev, size=len(parameters))
        return parameters + noise
    
    def update_noise_level(self, performance_metric, some_threshold = 0,min_noise_std = 0.01,max_noise_std =20 ):
        if performance_metric < some_threshold:
            # Decrease noise if performance is good
            self.noise_std_dev *= 1.5  # Adjust this factor as needed
        else:
            # Increase noise if performance is not improving
            self.noise_std_dev *= 0.9  # Adjust this factor as needed
        self.noise_std_dev = np.clip(self.noise_std_dev, min_noise_std, max_noise_std)

    def get_action(self, state,reward):
        state_array = np.array([state]).reshape(1, -1)
        raw_actions = self._actor.model.predict(state_array).flatten()
        return raw_actions


    def train(self):
        print("roperoperoperoperoperoperopereroperoperoperoperop")
        if len(self._memory) < self._batch_size:
            self._batch_size = self._memory
        else:
            self._batch_size = 32
        self._train()

    def _train(self):
        states, actions, rewards, done, next_states = self._get_sample()
        self._train_critic(states, actions, next_states, done, rewards)
        self._train_actor(states)
        self._update_target_models()

    def _get_sample(self):

        sample = random.sample(self._memory, self._batch_size)
        states, actions, rewards, done, next_states = zip(*sample)

        states = np.reshape(np.array(states), (self._batch_size, -1))
        actions = np.reshape(np.array(actions), (self._batch_size, -1))
        rewards = np.reshape(np.array(rewards), (self._batch_size, 1))
        next_states = np.reshape(np.array(next_states), (self._batch_size, -1))
        done = np.array(done).astype(np.float32)
        return states, actions, rewards, done, next_states


    def _train_critic(self, states, actions, next_states, done, rewards):
        q_targets = self._get_q_targets(next_states, done, rewards)
        self._critic.train(states, actions, q_targets)

    def _get_q_targets(self, next_states, done, rewards):
        next_actions = self._actor._target_model.predict(next_states)
        next_q_values = self._critic.target_model.predict([next_states,
                                                           next_actions])
        q_targets = [reward if this_done else reward + self._discount *
                                                       (next_q_value)
                     for (reward, next_q_value, this_done)
                     in zip(rewards, next_q_values, done)]
        return q_targets

    def _train_actor(self, states):
        action_for_gradients = self._actor.model.predict(states)
        gradients = self._critic.get_action_gradients(states, action_for_gradients)
        self._actor.train(states, gradients)
 
    def _get_gradients(self, states):
        action_for_gradients = self._actor.model.predict(states)
        gradients = self._critic.get_action_gradients(states, action_for_gradients)
        return gradients
    
    def _update_target_models(self):
        self._critic.train_target_model()
        self._actor.train_target_model()

    def _remember(self, state, action, reward, next_state, done):
        self._memory.append((state, action, reward, done, next_state))
        if len(self._memory) > self._memory_size:
            self._memory.popleft()





####################################################
class MainWindow(object):
    def __init__(self):
        self.is_running = False
        self.rett = None
        self.first_time = [0,0,0]
        self.rmse=0
        self.root = tkinter.Tk()
        self.left_frame = tkinter.Frame(self.root)
        self.left_frame.pack(side='left')
        self.button = tkinter.Button(self.left_frame, text='launch/stop', command=self.launch_callback)
        self.button.pack()
        self.shm_entry = tkinter.Entry(self.left_frame, width=30)
        self.shm_entry.insert(tkinter.END, 'shm_0')
        self.target_bucket=[]
        self.actual_bucket=[]
        self.shm_entry.pack()

        self.time_val = tkinter.StringVar()
        self.time_val.set('time_val')
        self.output_val = tkinter.StringVar()
        self.output_val.set('output')
        self.label = tkinter.Label(self.left_frame, textvariable=self.time_val)
        self.label.pack()
        self.time_label = tkinter.Label(self.left_frame, textvariable=self.output_val)
        self.time_label.pack()
        self.frame = tkinter.Frame(self.root)
        self.frame.pack()
        self.canvas = tkinter.Canvas(self.frame)
        self.canvas.pack()
        self.chart = Chart(self.canvas, (0.0, 5.0), (1e-6,-1e-6))
        self.shm = AmeCommunication.AmeSharedmem()
        self.chrono = Chrono()
        self.last_refresh_time = 0
        #####################################################
        self.kp = [10,10]
        self.ki = [0,0]
        self.kd = [0,0]    

        self.cnt = 0
        self.val = 8
        self.const = 0 
        self.pid_controller_outer = PIDControllerOuter(self.kp[0], self.ki[0], self.kd[0])
        self.pid_controller_outer_1 = PIDControllerOuter(self.kp[1], self.ki[1], self.kd[1]) 
        tkinter.mainloop()

    def launch_callback(self):
        if self.is_running:
            self.is_running = False
            self.chart.clean()
            self.canvas.after(100, self.shm.close)
        else:
            try:
                self.shm.init(False, str(self.shm_entry.get()), 4, 5)
                ret = self.shm.exchange([0.0, 0.0, 0.0, 0.0, 0.0])
                self.is_running = True
            except RuntimeError:
                return
            self.chrono.set_time(ret[1])
            self.dis_output = ret[2]
            self.dis_output_target = ret[3]
            self.ve_output = ret[4] 
            self.agent = DDPG(state_size=1,action_size=3)
            self.add_points()

    def add_points(self):     
        while self.is_running:
            t = self.chrono.get_time()
            try:
                self.pid_controller_outer.set_parameters(self.kp[0],self.ki[0],self.kd[0])
                self.pid_controller_outer_1.set_parameters(self.kp[1],self.ki[1],self.kd[1])
                #####################################################
                error = self.dis_output_target - self.dis_output
                output_val_dis = self.pid_controller_outer.calculate(error, t)

                error_velocity = output_val_dis - self.ve_output
                output_val= self.pid_controller_outer_1.calculate(error_velocity, t)
                ret = self.shm.exchange([0.0, t, output_val])
                self.rett = ret   
                self.dis_output = ret[2]
                self.dis_output_target = ret[3]
                self.ve_output = ret[4] 
                self.target_bucket.append(self.dis_output_target)
                self.actual_bucket.append(self.dis_output) 
                #####################################################
                next_error = self.dis_output_target - self.dis_output
                self.reward= self.calculate_reward(error,next_error)
                self.action= [self.kp[0],self.ki[0],self.kd[0]]
                self.agent._remember(error, self.action, self.reward, next_error, done=False)
                    
                # Train the agent with the sampled minibatch from the memory
                if self.cnt == 32:
                    #self.cnt = 0
                    self.agent.train()
                self.kp[0], self.ki[0], self.kd[0] = self.agent.get_action(error,self.reward)
                print(self.kp[0],self.ki[0], self.kd[0], " PID parameters")
                self.cnt += 1
                #new_point = (rett[1], rett[2], rett[3], rett[4], rett[5], rett[6], rett[7], rett[8], rett[9])
                #file.write(f"{new_point[0]} {new_point[1]} {new_point[2]} {new_point[3]} {new_point[4]}  {new_point[5]} {new_point[6]} {new_point[7]} {new_point[8]}\n")

            except RuntimeError:
                return
            print(ret[1], ret[2])
            self.chart.add_point(ret[1], ret[2])
            t = self.chrono.get_time()
            if t - self.last_refresh_time > 0.1:
                print("Warning: exchange rate goes too fast to ensure a smooth display")
                print("Try with a smaller sample time")
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set('time: ' + str(self.chrono.get_time()))
                self.output_val.set('val: ' + str(self.dis_output))
                self.chart.canvas.after(1, self.add_points)
                break
            elif ret[1] - t > 0.001:
                self.last_refresh_time = t
                self.chart.update()
                self.time_val.set('time: ' + str(self.chrono.get_time()))
                self.output_val.set('val: ' + str(self.dis_output))
                self.chart.canvas.after(1 + int((ret[1] - t) * 1000), self.add_points)
                break
        self.rmse=self.calc_rmse()
        print(self.rmse, "rmsersmersmermsemrse")

        
    def calculate_reward(self, current_error, next_error):
        # Define a reward scaling factor
        scale_factor = 1000  # Adjust based on trial and error to find a suitable scale

        # Define thresholds
        high_precision_threshold = 0.001  # High precision threshold
        acceptable_error_threshold = 0.01  # Acceptable error threshold, adjust based on your error range

        # Calculate the absolute error difference
        error_diff = abs(next_error - current_error)

        if abs(next_error) < high_precision_threshold:
            # High precision: large positive reward for very small errors
            reward = scale_factor * (1 - abs(next_error) / high_precision_threshold)
        elif abs(next_error) <= acceptable_error_threshold:
            # Acceptable precision: smaller positive reward for acceptable errors
            reward = scale_factor * (1 - abs(next_error) / acceptable_error_threshold)
        else:
            # Large errors: negative exponential penalty for errors exceeding acceptable threshold
            reward = -scale_factor * (abs(next_error) / acceptable_error_threshold) ** 2

        return reward
    
    def calc_rmse(self):
        rmse = np.sqrt(np.mean((np.array(self.target_bucket) - np.array(self.actual_bucket)) ** 2))
        return rmse
    
    def finalize(self):
        file.close()
        self.shm.close()

if __name__ == '__main__':

    main_window = MainWindow()

    main_window.add_points()  # Start collecting data

    # Finally, close the file and clean up
    main_window.finalize()
