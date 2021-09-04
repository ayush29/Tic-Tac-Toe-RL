Files & Folders
1. tictactoe: Folder contains my implementation of OpenAI gym compatible tictactoe environment
2. learning.py: Implementation of QLearning Algorithm
3. test.py: Python script to train, play and evaluate tictactoe agent.
4. Output: Output files and plots
5. TicTacToe.pdf : Explanation of Assumptions and Evaluation Results


Required packages
1. python3
2. gym
3. Numpy
4. Install my implementation of pacman environment using "pip install -e tictactoe"

How to run?
Run "python test.py -h" to know about command line arguments required then run with suitable values. 

To Reproduce the reported results run :
python test.py -o random > train_against_random.txt
python test.py -o safe > train_against_safe.txt
python test.py -o any > train_against_any.txt


