import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import numpy as np

class TicTacToeEnv(gym.Env):
    
    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([-1]*9), high=np.array([1]*9), dtype=np.int)
        self.action_space = spaces.Discrete(9)
        #empty pos :0, agent mark pos: 1, opponent mark pos: -1
        self.marks = { 0:' ', -1:'O', 1:'X'}
        #opponent mark : O
        #default opponent
        self.opponent = 'random'
        self.available = [0,1,2,3,4,5,6,7,8]
        self.status = 'play'
    
    #can pass arbitrary oppponent policy as well, for now not supported
    def set_opponent(self,opponent, **kwargs):
        if opponent == 'any':
            toss = np.random.rand()
            if toss<0.5:
                self.opponent = 'random'
            else:
                self.opponent = 'safe'
        elif opponent == 'custom':
            self.opponent = opponent
            self.opponent_policy = kwargs['policy']
        elif opponent != 'random' and opponent != 'safe':
            raise ValueError('Arbitrary opponent not supported. Supported opponent : Random, Safe')
        else:
            self.opponent = opponent
    
    def is_valid_move(self,pos):
        if pos in self.available:
            return True
        else:
            return False
    
    #if remaining in row of pos  = mark return true
    def check_row(self,pos,mark):
        rows = [[0,1,2],[3,4,5],[6,7,8]]
        for row in rows:
            if pos in row:
                row.remove(pos)
                for p in row:
                    if self.cur_state[p]!=mark:
                        return False
                return True
    #if remaining in col of pos  = mark return true        
    def check_col(self,pos,mark):
        cols = [[0,3,6],[1,4,7],[2,5,8]]
        for col in cols:
            if pos in col:
                col.remove(pos)
                for p in col:
                    if self.cur_state[p]!=mark:
                        return False
                return True
    
    #if remaining in diagonals of pos  = mark return true 
    def check_diag(self,pos,mark):
        diags = [[0,4,8],[2,4,6]]
        if pos != 4:
            for diag in diags:
                if pos in diag:
                    diag.remove(pos)
                    for p in diag:
                        if self.cur_state[p]!=mark:
                            return False
                    return True
        else:
            if (self.cur_state[0]==mark and self.cur_state[8]==mark) or (self.cur_state[2]==mark and self.cur_state[6]==mark):
                return True
            else:
                return False
    
    def is_winning_move(self,pos,mark):
        return self.check_row(pos,mark) or self.check_col(pos,mark) or self.check_diag(pos,mark)
     
    def is_blocking_move(self,pos,mark):
        return self.check_row(pos,-1*mark) or self.check_col(pos,-1*mark) or self.check_diag(pos,-1*mark)
        
    def is_draw_move(self,pos,mark):
        if len(self.available)==1 and not self.is_winning_move(pos,mark):
            return True
        else:
            return False
    
    def get_winnig_moves(self,mark):
        #check rows, cols, diagonals
        moves = []
        for pos in self.available:
            if self.is_winning_move(pos,mark):
                moves.append(pos)
        return moves
        
    def get_blocking_moves(self,mark):
        #check rows, cols, diagonals
        moves = []
        for pos in self.available:
            if self.is_blocking_move(pos,mark):
                moves.append(pos)
        return moves

    def play_opponent_move(self):
        if self.opponent == 'random':
            #sample random pos among available
            action = random.sample(self.available,1)[0]
            if self.is_winning_move(action,-1):
                    self.status = 'lose'
            elif self.is_draw_move(action,-1):
                    self.status = 'draw'
            self.cur_state[action] = -1
            self.available.remove(action)
        elif self.opponent == 'safe':
            moves = self.get_winnig_moves(-1)
            if len(moves)==0: #no winning moves
                moves = self.get_blocking_moves(-1)
                if len(moves) == 0:
                    moves = self.available
            else:
                self.status = 'lose' #agent loose, opponent wins as winning move exist
                
            action = random.sample(moves,1)[0]
            if self.status == 'play' and self.is_draw_move(action,-1):
                self.status = 'draw'
            self.cur_state[action] = -1
            self.available.remove(action)
        elif self.opponent == 'custom':
            action = np.argmax(self.opponent_policy(tuple(self.cur_state)))
            if self.is_valid_move(action):
                if self.is_winning_move(action,-1):
                    self.status = 'lose'
                elif self.is_draw_move(action,-1):
                    self.status = 'draw'
                self.cur_state[action] = -1
                self.available.remove(action)
            else:
                self.status = 'win'
        else:
            raise ValueError('Arbitrary opponent not supported. Supported opponent : Random, Safe')
    
    #act method
    def step(self,action):
        done = False
        #agent turn then opponent turn
        if not self.is_valid_move(action):
            done = True
            reward = -5
            self.status = 'lose'
        else:
            if self.is_winning_move(action,1):
                self.status = 'win'
                self.cur_state[action] = 1
                self.available.remove(action)
                reward = 2
                done = True
            elif self.is_draw_move(action,1):
                self.status = 'draw'
                self.cur_state[action] = 1
                self.available.remove(action)
                reward = 1
                done = True
            elif self.is_blocking_move(action,1):
                self.status = 'play'
                self.cur_state[action] = 1
                self.available.remove(action)
                reward = 1
            else:
                self.status = 'play'
                self.cur_state[action] = 1
                self.available.remove(action)
                reward = 0
            #opponent turn
            if self.status == 'play':
                self.play_opponent_move()
                if self.status == 'lose':
                    reward +=-2 #or +=-2
                    done = True
                elif self.status == 'draw':
                    reward += 1
                    done = True
                elif self.status == 'win':
                    done = True #no reward bcz opponent lost bcz of his wrong move not because of moves of agent
        
        return tuple(self.cur_state),reward,done,{'available_pos':self.available, 'game_status':self.status}
    
    def reset(self):
        self.opponent = 'random'
        self.available = [0,1,2,3,4,5,6,7,8]
        self.status = 'play'
        
        toss = random.random()
        if toss<0.5:
            #agent turn
            self.cur_state = [0]*9
        else:
            #opponent turn
            action = random.sample(self.available,1)[0]
            state = [0]*9
            state[action] = -1
            self.cur_state = state
            self.available.remove(action)
    
    #print method
    def render(self,mode ='console'):
        if mode!='console':
            raise NotImplementedError()
        else:            
            row = '{}|{}|{}'
            breaker = '-----'
            print(row.format(self.marks[self.cur_state[0]],self.marks[self.cur_state[1]],self.marks[self.cur_state[2]]))
            print(breaker)
            print(row.format(self.marks[self.cur_state[3]],self.marks[self.cur_state[4]],self.marks[self.cur_state[5]]))
            print(breaker)
            print(row.format(self.marks[self.cur_state[6]],self.marks[self.cur_state[7]],self.marks[self.cur_state[8]]))
    
    def close(self):
        pass
    
            
        
    

            
            