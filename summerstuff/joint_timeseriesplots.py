%matplotlib inline

preds = mat['preds'][0,0][2] # shape is a 32 x 2 x 751 --> joints x dimensions x #_frames

joint = 0 # joint from 0 to 31, represents one of the specified points. In this case, 0 is the head pos.

t = np.linspace(0, 750, 751) # creating a time point for every frame --> [0, 1, 2, ... , 749, 750]

x = preds[joint,0,:] # this takes out the x positions of a single joint over time --> time series of single joint
y = preds[joint,1,:] # same w/ y --> time series of joint pos... shape is (751, ) 

xpos = preds[:, 0, 0] # this takes out the x positions of all the joints in the first frame 
ypos = preds[:, 1, 0] # this takes out the y positions ... shape is (32, )

filter_window = 5

mf_x = medfilt(x, kernel_size=filter_window) #applying the median filter to the joint time series.
mf_y = medfilt(y, kernel_size=filter_window) #shape is still (751, ) but each val is median of sliding window 

plt.plot(t, x, label='regular, x')
#plt.plot(t[::filter_window], x[::filter_window], label='every nth, x')
plt.plot(t, mf_x, label='median filtered, x')

#plt.plot(t, y, label='regular, y')
#plt.plot(t[::filter_window], y[::filter_window], label='every nth, y')
#plt.plot(t, mf_y, label='median filtered, y')
plt.title('median filtered vs regular time series of x pos of single joint')
plt.legend()

plt.figure()
plt.plot(t, y, label='regular, y')
plt.plot(t, mf_y, label='medial filtered, y')
plt.title('median filtered vs regular time series of y pos of single joint')
plt.legend()

#to visualize fly 

#plt.figure()
#plt.scatter(xpos, ypos) ## this is a completely different figure, not a time series but a scatter plot of all 
# the joint pos for a given frame
