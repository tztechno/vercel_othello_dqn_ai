# main.py
from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import os
import io

app = Flask(__name__)

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class OthelloGame:
    def __init__(self):
        self.board_size = 8
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        center = self.board_size // 2
        self.board[center-1:center+1, center-1:center+1] = [[-1, 1], [1, -1]]
        self.current_player = 1
        return self.get_state()
    
    def get_valid_moves(self):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_valid_move(self, row, col):
        if self.board[row, col] != 0:
            return False
            
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        for dr, dc in directions:
            if self._would_flip(row, col, dr, dc):
                return True
        return False
    
    def _would_flip(self, row, col, dr, dc):
        r, c = row + dr, col + dc
        to_flip = []
        while 0 <= r < self.board_size and 0 <= c < self.board_size:
            if self.board[r, c] == 0:
                return False
            if self.board[r, c] == self.current_player:
                return len(to_flip) > 0
            to_flip.append((r, c))
            r, c = r + dr, c + dc
        return False
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        pieces_flipped = 0
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            to_flip = []
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if self.board[r, c] == 0:
                    break
                if self.board[r, c] == self.current_player:
                    for flip_r, flip_c in to_flip:
                        self.board[flip_r, flip_c] = self.current_player
                        pieces_flipped += 1
                    break
                to_flip.append((r, c))
                r, c = r + dr, c + dc
        
        self.current_player *= -1
        if not self.get_valid_moves():
            self.current_player *= -1
        
        return True
    
    def get_state(self):
        return self.board.tolist()
    
    def get_winner(self):
        if self.get_valid_moves():
            return None
        
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)
        
        if black_count > white_count:
            return 1
        elif white_count > black_count:
            return -1
        else:
            return 0

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 64)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
            checkpoint = torch.load("model.pth", map_location='cpu')
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset_game():
    game_instance.reset()
    if request.json.get('player_color') == "White (Second)":
        valid_moves = game_instance.get_valid_moves()
        if valid_moves:
            action = ai_instance.ai.get_action(
                game_instance.get_state(),
                valid_moves,
                training=False
            )
            if action:
                game_instance.make_move(*action)
    
    return jsonify({
        'board': game_instance.get_state(),
        'currentPlayer': game_instance.current_player,
        'validMoves': game_instance.get_valid_moves(),
        'winner': game_instance.get_winner()
    })

@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json
    row, col = data['row'], data['col']
    
    if game_instance.make_move(row, col):
        # AI's turn
        valid_moves = game_instance.get_valid_moves()
        if valid_moves:
            action = ai_instance.ai.get_action(
                game_instance.get_state(),
                valid_moves,
                training=False
            )
            if action:
                game_instance.make_move(*action)
    
    return jsonify({
        'board': game_instance.get_state(),
        'currentPlayer': game_instance.current_player,
        'validMoves': game_instance.get_valid_moves(),
        'winner': game_instance.get_winner()
    })

if __name__ == '__main__':
    app.run(debug=True)
