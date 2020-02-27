%matplotlib inline
connections = skeleton['edges'] - 1 # matlab numbers from 1, so subtract one
#connections.shape is 25 x 2, it gives the connections between all the joints
fig = plt.figure(figsize=(8,8))
plt.xlim(0,140)
plt.ylim(140,0) #switch the bounds so that the fly is facing downwards as in the movie
plt.axis('scaled')
#plots the connections between joints
for c in connections:
    start_indx = c[0] 
    end_indx = c[1] 
    plt.plot([xpos[start_indx], xpos[end_indx]], [ypos[start_indx], ypos[end_indx]], 'ro-')
    
#this is the vectorized, single-line pythonic version
#[plt.plot([xpos[c[0]],xpos[c[1]]],[ypos[c[0]],ypos[c[1]]]) for c in connections];

    
# can use this to visualize a single joint --> figure out joint = # corresponds 
# to what joint. --> the connections to the joint will light up 
    if c[0] == 5: 
        plt.plot([xpos[start_indx], xpos[end_indx]], [ypos[start_indx], ypos[end_indx]], 'bo-')
