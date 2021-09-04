import gym
from learning import qLearning   
import argparse
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import collections
from matplotlib import pyplot as plt

def play(env,policy,opponent, episodes, opponent_policy=None):
    print('Playing..') 
    test_rewards = []
    result = {'win':0,'draw':0,'lose':0}
    for e in range(episodes):
        print('\nNew episode {} starting..\n'.format(e))
        state = env.reset()
        #set opponent type
        env.set_opponent(opponent,policy = opponent_policy)
        done = False
        env.render()
        total_reward = 0
#         i=0
        while not done: #and i<max_steps:
            action = policy(state)
            print('\nmarking X at pos:{}'.format(action))
            state, r, done, info = env.step(action)
            total_reward += r
            if done:
                print('\nEpisode finished') 
                status = info['game_status']
                print('\nGame Status: {}'.format(status))
                if status == 'win':
                    result['win']+=1
                elif status == 'lose':
                    result['lose']+=1
                elif status == 'draw':
                    result['draw']+=1
            else:
                env.render()
                print('\nCurrent Game Status: {}'.format(info['game_status']))
                print('\nAvailable Positions:{}'.format(info['available_pos']))
#             i += 1
        test_rewards.append(total_reward)
    return result, test_rewards

def main():
    
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_episodes', '-eps1', nargs = 1, type = int, default = [10000], help = 'Number of episodes for training')
    parser.add_argument('-test_episodes', '-eps2', nargs = 1, type = int, default = [1000], help = 'Number of episodes for testing')
    parser.add_argument('-opponent','-o', nargs = 1, required = True, help = 'Opponent type for training(random,safe,any(to select randomly in every game))')
    parser.add_argument('-learning_rate', '-lr', nargs = 1, type = float, default = [0.4], help = 'Learning rate')
    parser.add_argument('-gamma', '-gm', nargs = 1, type = float, default = [0.99], help = 'Discount factor')
    parser.add_argument('-epsilon', '-epn', nargs = 1, type = float, default = [0.9], help = 'epsilon')
    parser.add_argument('-improve', '-imp', nargs = 1, type = int, default = [1000], help = 'Number of improvement episodes')
    args = parser.parse_args()
    
    env = gym.make('tictactoe:tictactoe-v0')
    
    gamma = args.gamma[0]
    alpha = args.learning_rate[0]
    epsilon = args.epsilon[0]
    train_episodes = args.train_episodes[0]
    test_episodes = args.test_episodes[0]
#     max_steps = args.max_steps[0]
    opponent = args.opponent[0]
    
    Q = collections.defaultdict(lambda : np.zeros(env.action_space.n))

    
    #training
    print('\nTraining against {} opponent..'.format(opponent))
    train_rewards = [] 
    progress = []
    for eps in [200]*int(train_episodes/200):
        policy, rewards = qLearning(Q,env,gamma, alpha, epsilon, eps,opponent)
        train_rewards += rewards
        result, _ = play(env,policy,opponent,100)
        progress.append(result['win'])
    
    
    #testing against opponents
    print('\nTesting against random opponent..')
    result_random , test_rewards_random = play(env,policy,'random',test_episodes)
    print('\nTesting against safe opponent..')
    result_safe , test_rewards_safe = play(env,policy,'safe',test_episodes)
    print('\nTesting against self...')
    result_self, test_rewards_self = play(env,policy,'custom',test_episodes,policy)
   

    #evaluation
    print('\n***Test results after training({} episodes) against {} opponent***'.format(train_episodes,opponent))
    print('\nAverage Test Rewards playing against random agent:',end='')
    print(np.mean(test_rewards_random))
    print('\nTest Results(match count) playing {} episodes against random agent:'.format(test_episodes),end='')
    print(result_random)
    print('\nAverage Test Rewards playing against safe agent:',end='')
    print(np.mean(test_rewards_safe))
    print('\nTest Results(match count) playing {} episodes against safe agent:'.format(test_episodes),end='')
    print(result_safe)
    print('\nAverage Test Rewards playing against self:',end='')
    print(np.mean(test_rewards_self))
    print('\nTest Results(match count) playing {} episodes against self :'.format(test_episodes),end='')
    print(result_self)
    
    # Progress during traininig
    #save plots images
    plt.plot(range(200,train_episodes+1,200),progress)
    plt.xlabel('Episodes')
#     plt.xticks(range(200,train_episodes+1,200))
#     plt.yticks(range(-10,50,2))
    plt.ylabel('Number of wins(out of 100 test games)')
    plt.title('Progress(after every 200 train games against {} opponent)'.format(opponent))
#     plt.legend()
    plt.savefig('training_progress_against_{}_opponent.png'.format(opponent))
    
    
    #further improvement by training against self for 1000 games
    improve_episodes = args.improve[0]
    print('Further training against self for {} games...'.format(improve_episodes))
    policy, _ = qLearning(Q,env,gamma, alpha, epsilon, improve_episodes,'custom',policy)
#     for eps in [200]*int(train_episodes/200):
#         policy, _ = qLearning(Q,env,gamma, alpha, epsilon, eps,'custom',policy)
#         train_rewards += rewards
#         result, _ = play(env,policy,opponent,100)
#         progress.append(result['win'])
    

    #testing against opponents
    print('\nTesting against random opponent..')
    result_random , test_rewards_random = play(env,policy,'random',test_episodes)
    print('\nTesting against safe opponent..')
    result_safe , test_rewards_safe = play(env,policy,'safe',test_episodes)
    print('\nTesting against self...')
    result_self, test_rewards_self = play(env,policy,'custom',test_episodes,policy)
    
    #evaluation
    print('***Test results after further training({} episodes) against self***'.format(improve_episodes))
    print('\nAverage Test Rewards playing against random agent:',end='')
    print(np.mean(test_rewards_random))
    print('\nTest Results(match count) playing {} episodes against random agent:'.format(test_episodes),end='')
    print(result_random)
    print('\nAverage Test Rewards playing against safe agent:',end='')
    print(np.mean(test_rewards_safe))
    print('\nTest Results(match count) playing {} episodes against safe agent:'.format(test_episodes),end='')
    print(result_safe)
    print('\nAverage Test Rewards playing against self:',end='')
    print(np.mean(test_rewards_self))
    print('\nTest Results(match count) playing {} episodes against self :'.format(test_episodes),end='')
    print(result_self)
    
    
if __name__=="__main__": 
    main()
    