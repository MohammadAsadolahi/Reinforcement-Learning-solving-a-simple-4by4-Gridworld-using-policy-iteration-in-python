# Reinforcement-Learning-solving-a-simple-4*4-Gridworld-using-policy-iteration-method
solving a simple 4\*4 Gridworld almost similar to openAI gym frozenlake using policy iteration method Reinforcement Learning  
WRITTEN BY MOHAMMAD ASADOLAHI  
Mohammad.E.Asadolahi@gmail.com  
https://github.com/mohammadAsadolahi  
this program is using Reinfrocement learning to solve a 4*4 gridworld like frozen lake enviroment in open ai gym  
the method used is policy iteration whitch is one of fundamental manners of Dynamic Programing  

     | S | O | O | O |  
     | O | O | O | * |  
     | O | * | O | O |  
     | O | * | O | T |  

  
  S= start cell  
  O= normal cells  
  *= penalized cells  
  T= terminate cell  
  
our agent goal is to find policy to go from S(start) cell to T(goal) cell with maximum reward(or minimum negative reward)  
valid actions are storend in GridWorld actions array.  
positive and negative rewards in each cell is stored in Gridworld  "Rewards" dictionary and can be modified by user .the current rewards for *(hole) cells ant T(goal) cell has been set to:  
self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -10}  
for example reward to go in (3,3) in enviroment witch is the goal will be +5 so agent gets +5 reward whenever go to cell (3,3)  
the size of Gridworld can be changed in GridWorld calss by adding space actions  
***************************
Algorithm Flow
***************************
  first we initialize a random policy that indicate prefered moves in every cell:  
  
     | D |  | L |  | D |  | L |  
     | R |  | R |  | R |  | U |   
     | R |  | L |  | D |  | D |   
     | U |  | R |  | R |   
U = going up  
D = going down  
L = going left  
R = going right  
  
and we initialize value table like:  

     | 0 |  | 0 |  | 0 |  | 0 |   
     | 0 |  | 0 |  | 0 |  | 0 |   
     | 0 |  | 0 |  | 0 |  | 0 |  
     | 0 |  | 0 |  | 0 |  | 5 |   

  
then we start using value iteration to update this policy value and as well as policy till convergence of policy witch will be optimal policy  
  sample outputs:
  
  --------------------------------  
 step:0   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 0.0005500000000000001 |  | 5.5000000000000016e-05 |  | -1.945 |   
 | 5.5000000000000016e-06 |  | 5.5000000000000016e-05 |  | -1.945 |  | 0.55 |   
 | 5.500000000000001e-07 |  | 5.5000000000000016e-06 |  | 0.55 |  | 5.5 |   
 | -9.945 |  | 0.55 |  | 5.5 |  | 5 |   
 
    Policy:  
 | R |  | L |  | L |  | D |   
 | U |  | U |  | D |  | D |   
 | U |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:1   
   --------------------------------  
     Value Table:  
 | 5.500000000000001e-07 |  | 5.5000000000000016e-08 |  | -0.1945 |  | -1.945 |   
 | 5.5000000000000016e-08 |  | 5.500000000000002e-09 |  | 5.500000000000003e-10 |  | 0.55 |   
 | 5.500000000000002e-09 |  | -9.945 |  | 0.55 |  | 5.5 |   
 | -9.945 |  | -0.9945 |  | 5.5 |  | 5 |   
   
    Policy:  
 | D |  | L |  | L |  | D |   
 | U |  | U |  | D |  | D |   
 | U |  | R |  | D |  | D |   
 | U |  | R |  | R |   
  
  --------------------------------  
 step:2   
   --------------------------------  
     Value Table:  
 | 5.500000000000001e-07 |  | -0.019450000000000002 |  | 0.005500000000000001 |  | -1.945 |   
 | 5.5000000000000016e-08 |  | -1.9945 |  | 0.05500000000000001 |  | 0.55 |   
 | -1.9945 |  | -9.945 |  | 0.005500000000000001 |  | 0.0005500000000000001 |   
 | -9.945 |  | 0.55 |  | 5.5 |  | 5 |   
  
    Policy:  
 | D |  | R |  | D |  | D |   
 | U |  | R |  | R |  | L |   
 | U |  | D |  | D |  | D |   
 | R |  | R |  | R |   
  
  --------------------------------  
 step:3   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 5.5000000000000016e-06 |  | 0.005500000000000001 |  | -1.945 |   
 | 5.5000000000000016e-06 |  | 0.005500000000000001 |  | 0.05500000000000001 |  | 0.55 |   
 | 5.500000000000001e-07 |  | 0.05500000000000001 |  | 0.55 |  | 5.5 |   

 | 5.5000000000000016e-08 |  | -1.9945 |  | -10.19945 |  | 5 |   

    Policy:    
 | D |  | D |  | D |  | D |   
 | R |  | D |  | D |  | D |   
 | R |  | R |  | R |  | D |   
 | U |  | U |  | R |   
 
  --------------------------------  
 step:4   
   --------------------------------  
     Value Table:  
 | 5.500000000000002e-09 |  | -0.19945000000000002 |  | 0.005500000000000001 |  | -1.945 |   
 | 5.500000000000003e-10 |  | 0.005500000000000001 |  | 0.05500000000000001 |  | 0.55 |   
 | 5.500000000000003e-11 |  | 0.05500000000000001 |  | 0.55 |  | 5.5 |   
 | 5.500000000000004e-12 |  | 0.55 |  | 5.5 |  | 5 |   

    Policy:  
 | D |  | D |  | D |  | D |   
 | R |  | D |  | D |  | D |   
 | R |  | D |  | D |  | D |   
 | R |  | R |  | R |   
 
  --------------------------------  
 step:5   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 0.0005500000000000001 |  | 5.5000000000000016e-05 |  | -1.945 |   
 | 5.5000000000000016e-06 |  | 5.500000000000001e-07 |  | 0.05500000000000001 |  | 0.55 |   
 | -1.9945 |  | -9.945 |  | 0.55 |  | 5.5 |   
 | -9.945 |  | 0.55 |  | 5.5 |  | 5 |   

    Policy:  
 | R |  | L |  | D |  | D |   
 | U |  | R |  | D |  | D |   
 | U |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:6   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 0.0005500000000000001 |  | 5.5000000000000016e-05 |  | -1.945 |   
 | 5.5000000000000016e-06 |  | 5.500000000000001e-07 |  | -1.945 |  | 0.55 |   
 | 5.500000000000001e-07 |  | -9.945 |  | 0.55 |  | 5.5 |   
 | -9.945 |  | 0.55 |  | 5.5 |  | 5 |   

    Policy:  
 | R |  | L |  | L |  | D |   
 | U |  | U |  | D |  | D |   
 | U |  | D |  | D |  | D |   
 | R |  | R |  | R |   
 
  --------------------------------  
 step:7   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 0.0005500000000000001 |  | 0.005500000000000001 |  | 0.0005500000000000001 |   
 | 0.0005500000000000001 |  | 0.005500000000000001 |  | 0.05500000000000001 |  | 0.55 |   
 | -2.9945 |  | -9.945 |  | 0.55 |  | -1.945 |   
 | -0.29945 |  | -2.9945 |  | 5.5 |  | 5 |  
  
    Policy:  
 | D |  | D |  | D |  | D |   
 | R |  | R |  | D |  | L |   
 | U |  | R |  | D |  | D |   
 | U |  | R |  | R |   
 
  --------------------------------  
 step:8   
   --------------------------------  
     Value Table:  
 | 5.500000000000002e-09 |  | 5.500000000000003e-10 |  | 0.005500000000000001 |  | -1.99945 |   
 | 5.500000000000003e-10 |  | -1.9945 |  | 0.05500000000000001 |  | 0.55 |   
 | -0.019945000000000004 |  | 0.05500000000000001 |  | 0.005500000000000001 |  | -1.945 |   
 | -9.945 |  | 0.55 |  | 5.5 |  | 5 |   

    Policy:  
 | D |  | R |  | D |  | D |    
 | U |  | D |  | R |  | L |   
 | R |  | D |  | D |  | D |   
 | R |  | R |  | R |   
 
  --------------------------------  
 step:9   
   --------------------------------  
     Value Table:  
 | 5.5000000000000016e-05 |  | 5.5000000000000016e-06 |  | 0.005500000000000001 |  | -1.99945 |   
 | 5.5000000000000016e-06 |  | 0.005500000000000001 |  | 0.05500000000000001 |  | 0.005500000000000001 |   
 | 5.500000000000001e-07 |  | 0.0005500000000000001 |  | 0.55 |  | 5.5 |   
 | 5.5000000000000016e-08 |  | 0.55 |  | 5.5 |  | 5 |   

    Policy:  
 | D |  | D |  | D |  | L |   
 | R |  | R |  | D |  | D |   
 | R |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  
exploited:1177  explored:323  
  
  
  
  
  
  
---------------------------------------------------------------------------------------------------  
and according to the step 9 results (value table and computed policy) its obvious that computed optimal policy 
----------------------------    
  
