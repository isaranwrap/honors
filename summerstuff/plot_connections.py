#Returns none, plots to current working axis
def plot_connections(xpos, ypos,num_points = 25):
    plt.cla()
    plt.xlim(0,148)
    plt.ylim(148,0)
    [plt.plot([xpos[c[0]],xpos[c[1]]],[ypos[c[0]],ypos[c[1]]]) for c in connections[:num_points]];
    
  
