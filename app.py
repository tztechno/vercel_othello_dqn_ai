# main.py
from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import os
import io
import requests
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

# Rest of the OthelloGame class and DQN class remain the same...
[Previous OthelloGame and DQN classes remain exactly as before]

class DQN_Agent:
    def __init__(self, learning_rate=0.001, gamma=0.99):
        self.device = torch.device("cpu")  # Force CPU for Vercel
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.gamma = gamma
        self.epsilon = 0.01  # Set to minimum for deployment
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state, valid_moves, training=False):
        if not valid_moves:
            return None
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        valid_q_values = [(move, q_values[0][move[0] * 8 + move[1]].item()) 
                         for move in valid_moves]
        return max(valid_q_values, key=lambda x: x[1])[0]

class OthelloAI:
    def __init__(self):
        self.game = OthelloGame()
        self.ai = DQN_Agent()
        self.trained = False
        self.load_model()
    
    def load_model(self):
        try:
            # Download model from HuggingFace
            model_path = hf_hub_download(
                repo_id="stpete2/dqn_othello_20250216",
                filename="model.pth"
            )
            
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Load the model state
            self.ai.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.ai.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.ai.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.ai.epsilon = checkpoint['epsilon']
            self.trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.trained = False
            return False

# Initialize game and AI
game_instance = OthelloGame()
ai_instance = OthelloAI()

[Rest of the Flask routes remain exactly as before]
