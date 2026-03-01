

# GRPO (?)










# This implementation is based on the paper: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf
#
# pip install torch transformers
# python grpo_demo.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# GRPO Configuration parameters as per formula
G = 4  # Number of samples in the group (G in the formula)
epsilon = 0.15  # ε in the formula - Clipping limit
beta = 0.0005  # β in the formula - KL penalty weight
learning_rate = 0.001

# Example data - Simulating a single q from distribution P(Q)
question = "What is the capital of Brazil?"
possible_answers = ["Brasília", "Rio de Janeiro", "São Paulo", "Fortaleza"]
correct_answer_idx = 0  # Brasília

# Preprocessing with BERT to get input representations
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")


def get_embedding(text):
    """Convert text to embedding using BERT"""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()


question_embedding = get_embedding(question)














# Policy Model (π_θ in the formula)
class PolicyModel(nn.Module):
    def __init__(self, num_actions, embedding_dim=768):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_actions)

    def forward(self, x):
        # Returns π_θ(o|q) - action probabilities given the state
        return torch.softmax(self.fc(x), dim=-1)









# Initialize π_θ (current policy) and π_θ_old (old policy)
num_actions = len(possible_answers)
policy = PolicyModel(num_actions)  # π_θ in the formula
old_policy = PolicyModel(num_actions)  # π_θ_old in the formula
old_policy.load_state_dict(policy.state_dict())


def train_step():
    """
    Implements one optimization step of GRPO according to the formula:
    J_GRPO(θ) = E[...] 1/G ∑(min(π_θ/π_θ_old * A_i, clip(π_θ/π_θ_old, 1-ε, 1+ε) * A_i))
    """
    # 1. Sample G outputs from old policy π_θ_old
    with torch.no_grad():
        probs_old = old_policy(question_embedding)
        sampled_actions = torch.multinomial(probs_old.squeeze(), G, replacement=True)

    # 2. Calculate probabilities from new policy π_θ
    probs_new = policy(question_embedding)

    # 3. Calculate ratio π_θ/π_θ_old
    ratios = probs_new[0, sampled_actions] / probs_old[0, sampled_actions]

    # 4. Calculate rewards and advantages (A_i in the formula)
    rewards = torch.tensor(
        [1.0 if idx == correct_answer_idx else -0.1 for idx in sampled_actions]
    )
    # A_i = (r_i - mean({r_1,...,r_G})) / std({r_1,...,r_G})
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 5. Implement clipping as per formula
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    # 6. Calculate loss according to min(.) in formula
    loss_policy = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    # 7. Calculate KL divergence as per formula (2)
    # D_KL(π_θ||π_ref) = π_ref(o_i|q)/π_θ(o_i|q) - log(π_ref(o_i|q)/π_θ(o_i|q)) - 1
    ratio_kl = probs_old.detach() / probs_new
    kl_penalty = (ratio_kl - torch.log(ratio_kl) - 1).mean()

    # 8. Total loss with KL penalty
    total_loss = loss_policy + beta * kl_penalty

    # 9. Update policy
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss, loss_policy, kl_penalty


# Train the model
print("Starting training...")
for epoch in range(100):
    loss, policy_loss, kl = train_step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Policy Loss: {policy_loss.item():.4f}")
        print(f"  KL Divergence: {kl.item():.4f}")

# Test the trained policy
with torch.no_grad():
    probs_final = policy(question_embedding)
    predicted_answer_idx = torch.argmax(probs_final).item()
    probabilities = probs_final[0].numpy()

print("\nFinal Results:")
print(f"Predicted answer: '{possible_answers[predicted_answer_idx]}'")
print("\nProbabilities for each answer:")
for answer, prob in zip(possible_answers, probabilities):
    print(f"{answer}: {prob:.4f}")



















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