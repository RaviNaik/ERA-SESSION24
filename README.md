# ERA-SESSION24 - Reinforcement Learning

## Grid World Experiments:

### Tasks
Setup Berkely Ai RL course experiments. 
1. :heavy_check_mark: Share a screenshot of you running the whole code on your computer. 
2. :heavy_check_mark: You would need to find code for some files to run it successfully.
3. :heavy_check_mark: Once done, understand the code you copied and write a pseudo-code to explain what is happening. You need to explain:
```python
 1. __ init__
 2. getQValue
 3. computeValueFromQValue
 4. computeActionFromQValues
 5. getAction
 6. update
```

### Experiment Screenshot
![image](https://github.com/RaviNaik/ERA-SESSION24/assets/23289802/cfb61485-f991-4768-a1f2-e12064259ff5)


## Car Game Project: 
Perform Reinforcement training for a car object moving through the streets of Tokyo city. 

### Tasks
1. :heavy_check_mark: Create a new map of some other city for the code shared above
2. :heavy_check_mark: Add a DNN with 1 more FC layer. 
3. :heavy_check_mark: Your map must have 3 targets A1>A2>A3 and your car/robot/object must target these alternatively.
   
### Model Architecture
```python
class Network(nn.Module):
   
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30, nb_action)
   
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```

### Training Curve
![training_curve](https://github.com/RaviNaik/ERA-SESSION24/assets/23289802/b6a3d7b1-a754-4ea9-9afd-481565277e3f)

### Snapshot of Training
![image](https://github.com/RaviNaik/ERA-SESSION24/assets/23289802/79ec18b7-3cbd-4eaa-866b-41d744a69d5f)
