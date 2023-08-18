# %%
# Gradient Descent Video Follow Along Project

# We have two types of machine learning problems
#1)regression problems - predicting price of a stock coming up with predictions on past data
# Classification problems- will a user like or dislike a movie?
# Can you predict a user will like or dislike based on two variables of movie time and IMDB ratings

# The first thing we want to do is create a model. 
# Want to find the line which will result in least error

# Gradient Descent will help us find the best line
# But HOW does gradient descent do this?

# The regression in scikit learn won't give you the best estimation??

## sklearn.fit(x,y) Why won't this give you the best approximation
# Behind sklearn.fit, gradient descent will work behind the scenes

# Gradient Descent is basically just the computer conducting trial and error!
# First, we start with a random line. That is the first step.
# We take any value for m and b and then draw a line.
# For this given line we predict all the errors for all points.

# We continusouly reduce m and b so that the error reduces.

# How do we adjust the line? We derive it.

# Gradient descent is an optimization alogrithm which can be applied to any problem in general.



# %%
!pwd
%cd /Users/kylenunn/Desktop/Machine-Learning
!pwd

# %%
pip install matplotlib

# %%
pip install pandas


# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# %%
data = pd.read_csv('/Users/kylenunn/Desktop/Machine-Learning/Student_Scores.csv', encoding= 'unicode_escape')
data

data.columns = ['Hours','Scores']
data

# %%
x = data['Hours']
y = data['Scores']
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Hours vs. Scores")
plt.plot(x,y,'o')
plt.show()

# Now we are going to draw a line closest to the points without scikit learn
# Instead we are using gradient descent

# %%
# Helper function use when needed

# %%
# We are going to implement gradient descent algorithm one step at a time
# We are only doing this, in theory, to our training data set.
# The parameters are m and b namely if we change them then the line changes
def gradient_desc(all_x,all_y,m,b):
    total_error = 0 
    
    for x,y_actual in zip(all_x,all_y):
        y_pred = m*x + b
                          
        error = y_pred - y_actual
        total_error += error
        
        
        delta_m = error * x
        delta_b = error
        
        # When we changed the above delta_m and delta_b from 1's to error * x. and error, the graph went wild
        
        
        m = m - delta_m * .001
        b = b - delta_b * .001
    
    return m,b,total_error

# The learning rate is the amount we are stepping each time.. 
# If learning rate is too high and too small then we won't find anything out
# If the learning rate is too low the gradient descent will take forever.. forever.. forever..


# %%
x = data['Hours']
y = data['Scores']
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title("Hours vs. Scores")
plt.plot(x,y,'o')

# %%
m,b

# %%
# We need to adjust delta m and b so that the error reduces.
# I want to plot this on a graph (plot regression line function)
# So we can see it on the graph
# Adjust m and b means adjust the line
# The quantity we are trying to reduce is called the cost function
# We are trying to reduce the cost function
# Cost function can be denoted with Jtheta(m,b)
# This isn't math this is notation
# Need to know the difference between the mathematics and notation
