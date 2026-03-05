
# from https://github.com/YyzHarry/SV-RL/blob/master/sv_rl/dqn_model.py








import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=18):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class DQN_RAM(nn.Module):

    def __init__(self, in_features=4, num_actions=18):
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Dueling_DQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=18):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x













# ------------------------------------------------------









class QNetwork(nn.Module):
    """Q-network with configurable hidden dimensions for scaling studies"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class SACAgent:
    """
    Soft Actor-Critic implementation with predictable scaling support.
    
    This follows the "Value-Based Deep RL Scales Predictably" methodology:
    1. UTD ratio controls the tradeoff between data and compute
    2. Hyperparameters scale predictably with UTD to maintain stability
    3. Supports model size scaling alongside UTD scaling
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 scaling_laws: ScalingLaws,
                 hidden_dims: list = [256, 256],
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,  # Temperature for entropy regularization
                 device: str = 'cpu',
                 automatic_entropy_tuning: bool = True,
                 target_entropy: Optional[float] = None):
        
        self.device = device
        self.scaling = scaling_laws
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Get scaled hyperparameters based on current UTD
        hparams = scaling_laws.compute_hyperparameters()
        self.batch_size = hparams['batch_size']
        self.lr = hparams['learning_rate']
        self.target_update_freq = hparams['target_update_freq']
        self.utd = hparams['utd_ratio']
        
        # Networks
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic1_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic2_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Policy network (Gaussian policy)
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(device)
        
        # Optimizers with scaled learning rate
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr)
        
        # Entropy temperature
        if self.automatic_entropy_tuning:
            if target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
            
        self.update_counter = 0
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer: ReplayBuffer):
        """
        Perform UTD ratio number of updates per call.
        
        This is the key to the scaling behavior: we can vary the amount of compute
        per environment step while maintaining performance through hyperparameter
        scaling.
        """
        if len(replay_buffer) < self.batch_size:
            return {}
        
        metrics = {
            'critic_loss': 0,
            'policy_loss': 0,
            'alpha_loss': 0,
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        }
        
        # Perform UTD updates
        for _ in range(int(self.utd)):
            # Sample batch
            state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
            
            with torch.no_grad():
                # Target Q-value
                next_action, next_log_prob, _ = self.policy.sample(next_state)
                target_q1 = self.critic1_target(next_state, next_action)
                target_q2 = self.critic2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q = reward + (1 - done) * self.gamma * target_q
            
            # Current Q-values
            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            self.critic2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            
            metrics['critic_loss'] += critic_loss.item()
            
            # Policy update (less frequent for computational efficiency)
            if self.update_counter % 2 == 0:
                pi, log_pi, _ = self.policy.sample(state)
                q1_pi = self.critic1(state, pi)
                q2_pi = self.critic2(state, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                
                policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                metrics['policy_loss'] += policy_loss.item()
                
                # Update temperature
                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    
                    self.alpha = self.log_alpha.exp()
                    metrics['alpha'] = self.alpha.item()
                    metrics['alpha_loss'] += alpha_loss.item()
            
            # Soft update target networks (frequency scaled by UTD)
            if self.update_counter % self.target_update_freq == 0:
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)
            
            self.update_counter += 1
        
        # Average losses over UTD updates
        for key in metrics:
            if 'loss' in key:
                metrics[key] /= self.utd
                
        return metrics
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)








# ----------------------------------------------------




# from 





# Value/Reward Function implementing NAS over the MPC Optimization Space




import logging

import numpy as np
import tensorflow as tf

from Utils.child_network import ChildCNN
from Utils.cifar10_processor import get_tf_datasets_from_numpy
from Utils.config import child_network_params, controller_params

logger = logging.getLogger(__name__)

def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    
    Args:
        values (list): A list of rewards 
    Returns:
        (float) The last value of the exponential moving average
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]

class Controller(object):

    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.num_cell_outputs = controller_params['components_per_layer'] * controller_params['max_layers']
        self.reward_history = []
        self.architecture_history = []
        self.divison_rate = 100
        with self.graph.as_default():
            self.build_controller()

    def network_generator(self, nas_cell_hidden_state):
        # number of output units we expect from a NAS cell
        with tf.name_scope('network_generator'):
            nas = tf.contrib.rnn.NASCell(self.num_cell_outputs)
            network_architecture, nas_cell_hidden_state = tf.nn.dynamic_rnn(nas, tf.expand_dims(
                nas_cell_hidden_state, -1), dtype=tf.float32)
            bias_variable = tf.Variable([0.01] * self.num_cell_outputs)
            network_architecture = tf.nn.bias_add(network_architecture, bias_variable)
            return network_architecture[:, -1:, :]

    def generate_child_network(self, child_network_architecture):
        with self.graph.as_default():
            return self.sess.run(self.cnn_dna_output, {self.child_network_architectures: child_network_architecture})

    def build_controller(self):
        logger.info('Building controller network')
        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):
            # Input to the NASCell
            self.child_network_architectures = tf.placeholder(tf.float32, [None, self.num_cell_outputs], 
                                                              name='controller_input')
            # Discounted rewards
            self.discounted_rewards = tf.placeholder(tf.float32, (None, ), name='discounted_rewards')

        # Build controller
        with tf.name_scope('network_generation'):
            with tf.variable_scope('controller'):
                self.controller_output = tf.identity(self.network_generator(self.child_network_architectures), 
                                                     name='policy_scores')
                self.cnn_dna_output = tf.cast(tf.scalar_mul(self.divison_rate, self.controller_output), tf.int32,
                                              name='controller_prediction')

        # Set up optimizer
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(0.99, self.global_step, 500, 0.96, staircase=True)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        # Gradient and loss computation
        with tf.name_scope('gradient_and_loss'):
            # Define policy gradient loss for the controller
            self.policy_gradient_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.controller_output[:, -1, :],
                labels=self.child_network_architectures))
            # L2 weight decay for Controller weights
            self.l2_loss = tf.reduce_sum(tf.add_n([tf.nn.l2_loss(v) for v in
                                                   tf.trainable_variables(scope="controller")]))
            # Add the above two losses to define total loss
            self.total_loss = self.policy_gradient_loss + self.l2_loss * controller_params["beta"]
            # Compute the gradients
            self.gradients = self.optimizer.compute_gradients(self.total_loss)

            # Gradients calculated using REINFORCE
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

        with tf.name_scope('train_controller'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

        logger.info('Successfully built controller')


    def train_child_network(self, cnn_dna, child_id):
        """
        Trains a child network and returns reward, or the validation accuracy
        Args:
            cnn_dna (list): List of tuples representing the child network's DNA
            child_id (str): Name of child network
        Returns:
            (float) validation accuracy
        """
        logger.info("Training with dna: {}".format(cnn_dna))
        child_graph = tf.Graph()
        with child_graph.as_default():
            sess = tf.Session()

            child_network = ChildCNN(cnn_dna=cnn_dna, child_id=child_id, **child_network_params)

            # Create input pipeline
            train_dataset, valid_dataset, test_dataset, num_train_batches, num_valid_batches, num_test_batches = \
                get_tf_datasets_from_numpy(batch_size=child_network_params["batch_size"])

            # Generic iterator
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            next_tensor_batch = iterator.get_next()

            # Separate train and validation set init ops
            train_init_ops = iterator.make_initializer(train_dataset)
            valid_init_ops = iterator.make_initializer(valid_dataset)

            # Build the graph
            input_tensor, labels = next_tensor_batch

            # Build the child network, which returns the pre-softmax logits of the child network
            logits = child_network.build(input_tensor)
            
            # Define the loss function for the child network
            loss_ops = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name="loss")

            # Define the training operation for the child network
            train_ops = tf.train.AdamOptimizer(learning_rate=child_network_params["learning_rate"]).minimize(loss_ops)

            # The following operations are for calculating the accuracy of the child network
            pred_ops = tf.nn.softmax(logits, name="preds")
            correct = tf.equal(tf.argmax(pred_ops, 1), tf.argmax(labels, 1), name="correct")
            accuracy_ops = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

            initializer = tf.global_variables_initializer()

            # Training
            sess.run(initializer)
            sess.run(train_init_ops)

            logger.info("Training child CNN {} for {} epochs".format(child_id, child_network_params["max_epochs"]))
            for epoch_idx in range(child_network_params["max_epochs"]):
                avg_loss, avg_acc = [], []

                for batch_idx in range(num_train_batches):
                    loss, _, accuracy = sess.run([loss_ops, train_ops, accuracy_ops])
                    avg_loss.append(loss)
                    avg_acc.append(accuracy)

                logger.info("\tEpoch {}:\tloss - {:.6f}\taccuracy - {:.3f}".format(epoch_idx,
                                                                                   np.mean(avg_loss), np.mean(avg_acc)))

            # Validate and return reward
            logger.info("Finished training, now calculating validation accuracy")
            sess.run(valid_init_ops)
            avg_val_loss, avg_val_acc = [], []
            for batch_idx in range(num_valid_batches):
                valid_loss, valid_accuracy = sess.run([loss_ops, accuracy_ops])
                avg_val_loss.append(valid_loss)
                avg_val_acc.append(valid_accuracy)
            logger.info("Valid loss - {:.6f}\tValid accuracy - {:.3f}".format(np.mean(avg_val_loss),
                                                                              np.mean(avg_val_acc)))

        return np.mean(avg_val_acc)

    def train_controller(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        step = 0
        total_rewards = 0
        child_network_architecture = np.array([[10.0, 128.0, 1.0, 1.0] *
                                               controller_params['max_layers']], dtype=np.float32)

        for episode in range(controller_params['max_episodes']):
            logger.info('=============> Episode {} for Controller'.format(episode))
            step += 1
            episode_reward_buffer = []

            for sub_child in range(controller_params["num_children_per_episode"]):
                # Generate a child network architecture
                child_network_architecture = self.generate_child_network(child_network_architecture)[0]

                if np.any(np.less_equal(child_network_architecture, 0.0)):
                    reward = -1.0
                else:
                    reward = self.train_child_network(cnn_dna=child_network_architecture,
                                                      child_id='child/{}'.format("{}_{}".format(episode, sub_child)))
                episode_reward_buffer.append(reward)

            mean_reward = np.mean(episode_reward_buffer)

            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network_architecture)
            total_rewards += mean_reward

            child_network_architecture = np.array(self.architecture_history[-step:]).ravel() / self.divison_rate
            child_network_architecture = child_network_architecture.reshape((-1, self.num_cell_outputs))
            baseline = ema(self.reward_history)
            last_reward = self.reward_history[-1]
            rewards = [last_reward - baseline]
            logger.info("Buffers before loss calculation")
            logger.info("States: {}".format(child_network_architecture))
            logger.info("Rewards: {}".format(rewards))

            with self.graph.as_default():
                _, loss = self.sess.run([self.train_op, self.total_loss],
                                        {self.child_network_architectures: child_network_architecture,
                                         self.discounted_rewards: rewards})

            logger.info('Episode: {} | Loss: {} | DNA: {} | Reward : {}'.format(
                episode, loss, child_network_architecture.ravel(), mean_reward))