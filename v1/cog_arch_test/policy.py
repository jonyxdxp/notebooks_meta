

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