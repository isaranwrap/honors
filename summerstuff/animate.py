#creates animation of skeleton frame over video of fly 
def animate(i):
    #plt.figure()
    xpos = preds[:,0,i] #this takes out the x positions of all the joints in the ith frame
    ypos = preds[:,1,i] #same, for y -- shape is (32, )
    plot_connections(xpos, ypos)
    plt.scatter(xpos, ypos, s=10)
    
    frame = box[i,:,:]
    rotated = ndimage.rotate(frame, 270)
    img = np.fliplr(rotated)
    plt.imshow(img)
    #return plot_connections(xpos, ypos), plt.scatter(xpos, ypos, s=10), plt.imshow(img)
