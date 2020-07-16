from scipy.signal import medfilt    
import pandas as pd
from moviepy.editor import VideoFileClip
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip
import string
from tqdm import tqdm

def loadh5(fname = None, ratenterframe = 2000):
    """
    Load a DLC tracking file from HFSP Dataset. 
    
    Input:
    fname : Path to h5 file. Leave as None to load default file. 
    ratenterframe : Remove data from h5 file before rat has entered. 
    
    Return:
    (t, bparts, 3) shape array file containing x,y, cmapvalue in the last dimension.
    
    """
    if fname is None: 
        h5 = pd.read_hdf('/mnt/HFSP_Data/iterations_HFSP_Data_SfN_2019/R19001_0115_session_fullvid_0DeepCut_resnet152_OutlineJul26shuffle1_1030000.h5').values
        h5 = np.reshape(h5, (h5.shape[0], int(h5.shape[1]/3.0), 3))
        ratenterframe = 300
        h5 = h5[ratenterframe:]
        return h5
    else:
        h5 = pd.read_hdf(fname)
        h5 = np.reshape(h5, (h5.shape[0], int(h5.shape[1]/3.0), 3))
        h5 = h5[ratenterframe:]
        return h5


def loadclip(fname=None, ratenterframe = 2000):
    """
    Load a video file form HFSP Dataest. 
    Input:
    fname : Path to video file. Leave as None to load default file. 
    ratenterframe : Clip the video file to remove frames before this frame #.
    
    Return:
    VideoFileClip object of moviepy package. 
    """
    if fname is None: 
        clip = VideoFileClip('/mnt/HFSP_Data/iterations_HFSP_Data_SfN_2019/R19001_0115_session_fullvid_0.avi')
        ratenterframe = 300
        clip = clip.cutout(0, ratenterframe/clip.fps)
        return clip
    else:
        clip = VideoFileClip('/mnt/HFSP_Data/iterations_HFSP_Data_SfN_2019/R19001_0115_session_fullvid_0.avi')
        clip.cutout(0, ratenterframe/clip.fps)
        return clip

def getdsetnames():
    """
    Return: (list1, list2)
    list1 contains path to h5 files and list2 contains path to avi files from HFSP Datset. 
    elements in list1 and list2 correspond to each other. 
    """
    testh5sfnames = ['/mnt/HFSP_Data/iterations_HFSP_Data_SfN_2019/'+i for i in os.listdir('/mnt/HFSP_Data/iterations_HFSP_Data_SfN_2019/') if '.h5' in i]
    for i in os.listdir('/mnt/HFSP_Data/Outline-Kanishk-2019-07-26/precmap'):
        if 'R19001_' in i and '.h5' in i and 'fullvideocut' not in i:
            testh5sfnames.append('/mnt/HFSP_Data/Outline-Kanishk-2019-07-26/precmap/'+i)
        if 'R19002_' in i and '.h5' in i and 'fullvideocut' not in i:
            testh5sfnames.append('/mnt/HFSP_Data/Outline-Kanishk-2019-07-26/precmap/'+i)
    testh5sfnames.sort()
    vidnames = ['/mnt/HFSP_Data/Aman_Data_2018/190117/R19001_0117_session_fullvid_0.avi',
            '/mnt/HFSP_Data/Aman_Data_2018/190118/R19001_0118_session_fullvid_0.avi',
            '/mnt/HFSP_Data/Aman_Data_2018/190117/R19002_0117_session_fullvid_0.avi',
            '/mnt/HFSP_Data/Aman_Data_2018/190118/R19002_0118_session_fullvid_0.avi']
    vidnames = vidnames + ['/mnt/HFSP_Data/HSFP_Data_SfN_2019/'+h5sfname.split('/')[-1][:-49]+'.avi' for h5sfname in testh5sfnames[4:]]
    return testh5sfnames, vidnames

def ratenterframes():
    """
    Return : list
    List containing frame# when rat enters the arena. Numbers corresopond to elements in getdsetnames() return.
    """
    return [860,970,1350,1120,
               250,0,0,430,
               1310,620,290,1150,
               1280,1450,1320,760]
    
def loadh5s(indlist = None):
    """
    Load all h5s from HFSP dattaset if indlist is None, else load indices in indlist, correspond to elements in 
    getdsetnames() return. 
    """
    if indlist is None:
        h5s = [pd.read_hdf(dset).values[enterframe:] for dset,enterframe in zip(getdsetnames()[0], ratenterframes())]
        h5s = [np.reshape(h5, (h5.shape[0], int(h5.shape[1]/3.0), 3)) for h5 in h5s]
        return h5s
    else:
        dsetnames = np.array(getdsetnames()[0])
        ratenterlist = np.array(ratenterframes())
        h5s = [pd.read_hdf(dset).values[enterframe:] for dset,enterframe in zip(dsetnames[indlist], ratenterlist[indlist])]
        h5s = [np.reshape(h5, (h5.shape[0], int(h5.shape[1]/3.0), 3)) for h5 in h5s]
        return h5s
    
def loadclips(indlist = None):
    """
    Load all videoclips from HFSP dattaset if indlist is None, else load indices in indlist corresponding to elements
    in getdsetnames() return. 
    """
    if indlist is None:
        clips = [VideoFileClip(fname) for fname in getdsetnames()[1]]
        clips = [clip.cutout(0, ratenterframe/clip.fps) for clip, ratenterframe in zip(clips, ratenterframes())]
        return clips
    
    else:
        dsetnames = np.array(getdsetnames()[1])[indlist]
        ratenterlist = np.array(ratenterframes())[indlist]
        clips = [VideoFileClip(fname) for fname in dsetnames]
        clips = [clip.cutout(0, ratenterframe/clip.fps) for clip, ratenterframe in zip(clips, ratenterlist)]
        return clips


def getvelocity(h5, median_filt_len=9, bpart = 5):
    """
    Input:
    h5 : 3d numpy array of shape (t, bparts, coords)
    median_filt_len : Odd integer for median filtering window. 
    bpart : (int or string) Bpart index in h5 file to use to calculate velocities. (run bpartref() to check)
            if 'all', use median filtered COM trajectory to calculate velocity.
    Return animal velocity in a numpy array shaped (t,1).
    """
    if bpart != 'all':
        return medfilt(np.linalg.norm(np.diff(h5[:,bpart,:2], axis=0), axis=1), median_filt_len)
    else:
        return medfilt(np.linalg.norm(np.diff(np.mean(h5[:,:,:2],axis=1),axis=0), axis=1), median_filt_len)
#*****************************************************************************        


def egoh5(h5, bindcenter=8, align=True, b1=2, b2=11):
    """
    Return h5 in the center of frame of body part index. If align is true, rotate data so that vector from b2 to b1
    always points East.
    """

    ginds = np.setdiff1d(np.arange(h5.shape[1]), bindcenter)
    egoh5 = h5[:,:,:2] - h5[:,[bindcenter for i in range(h5.shape[1])],:2]
    egoh5 = egoh5[:,ginds]
    if not align:
        return egoh5
    dir_arr = egoh5[:, b1] - egoh5[:, b2-1]
    dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis]
    for t in tqdm(range(egoh5.shape[0])):
        rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
        egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    return egoh5


def nnData(h5, lag=10, overlap=False, egocentered = False, ratbodyonly=True):
    """
    Create RNN training samples using lag value. h5 array is first egocenterd. 
    Inputs: h5 data to use
    lag : number of timepoints in each sample in input to RNN. 
    overlap : If false, input values across samples do not overlap.
    egoocentered : if True, skip egocentering.
    ratbodyonly : Skip tail points when egocentering in vivo.
    
    Return 
    (input samples, output samples) - equal in size along axis 0.
    Input samples have shape (nsamples, lag, features)
    Output Samples have shape (nsamples, features)
    """
    if not egocentered:
        eh5 = egoh5(h5)
        if ratbodyonly:
            eh5 = eh5[:,:13,:]
        eh5 = np.reshape(eh5, (eh5.shape[0], eh5.shape[1]*2))
    else:
        if len(h5.shape) > 2:
            h5 = np.reshape(h5, (h5.shape[0], h5.shape[1]*h5.shape[2]))
        eh5 = h5
    if not overlap:
        outputs = eh5[lag::lag]
        inputs = []
        a = np.concatenate(([0],np.arange(lag,eh5.shape[0],lag)))
        for i in range(1, a.shape[0]):
            inputs.append(eh5[a[i-1]:a[i]])
        return np.array(inputs), outputs
    else:
        outputs = eh5[lag:]
        inputs = [eh5[i-lag:i] for i in range(lag,eh5.shape[0])]
        return np.array(inputs), outputs

def nnDatas(h5s, lag=10, n=100000, egocentered=False, ratbodyonly=True):
    n -= n%len(h5s)
    perh5 = int(n/len(h5s))
    allgoodinds = [np.random.choice(np.arange(lag,h5.shape[0]),perh5, replace=False) for h5 in h5s]
    def expandginds(goodinds):
        outginds = np.zeros((goodinds.shape[0]*(lag+2)))
        goodinds = np.sort(goodinds)
        for i,gind in enumerate(goodinds):
            outginds[i*(lag+2):(i+1)*(lag+2)] = np.arange(gind-lag, gind+2)
        return outginds.astype('int')

    allgoodinds = [expandginds(goodinds) for goodinds in allgoodinds]
    return nnDatag(h5s, allgoodinds, lag, n, egocentered, ratbodyonly)


def nnDatag(h5s, allgoodinds, lag, n = 100000, egocentered=False, ratbodyonly=True):
    """
    Create RNN sample from a list of h5s given inds to choose from (list allgoodinds).

    Input
    h5s : list of h5s
    allgoodinds : list of good inds, corresponding to h5s list. 
    lag : number of timepoints in each sample in input to RNN. 
    n : int or str (eg 'all') Number of samples to generate. 
    egoocentered : if True, skip egocentering.
    ratbodyonly : Skip tail points when egocentering in vivo.
    
    Return 
    (input samples, output samples) - equal in size along axis 0.
    Input samples have shape (nsamples, lag, features)
    Output Samples have shape (nsamples, features)
    """
    lagindslist = [laginds(goodinds, lag) for goodinds in allgoodinds]
    # print('Sanity CHeck *******')
    # print(lagindslist[0])
    # print('*********')
    if n == 'all':
        outX = []
        outy = []
        for h5,(Xlist, ylist) in zip(h5s, lagindslist):
            if not egocentered:
                eh5 = egoh5(h5)
                if ratbodyonly:
                    eh5 = eh5[:,:13,:]
                eh5 = np.reshape(eh5, (eh5.shape[0], eh5.shape[1]*2))
                outX += [eh5[X] for X in Xlist]
                outy += [eh5[y] for y in ylist]
            else:
                if len(h5.shape) > 2:
                    h5 = np.reshape(h5, (h5.shape[0], h5.shape[1]*h5.shape[2]))
                outX += [h5[X] for X in Xlist]
                outy += [h5[y] for y in ylist]
        return np.array(outX), np.array(outy)

    else:
        lenlist = [lagind[1].shape[0] for lagind in lagindslist]
        indstochoose = np.zeros(np.sum(lenlist))
        indstochoose[np.random.choice(np.arange(indstochoose.shape[0]), n, replace=False)] = 1
        indstochoose = np.split(indstochoose.astype('bool'), np.cumsum(lenlist)[:-1])
        
        lagindslist = [(Xlist[i2c], ylist[i2c]) for i2c, (Xlist, ylist) in zip(indstochoose, lagindslist)]
        outX = np.zeros((n, lag, 26))
        outy = np.zeros((n, lag, 26))
        count = 0
        for h5,(Xlist, ylist) in zip(h5s, lagindslist):
            if not egocentered:
                eh5 = egoh5(h5)
                if ratbodyonly:
                    eh5 = eh5[:,:13,:]
                eh5 = np.reshape(eh5, (eh5.shape[0], eh5.shape[1]*2))
                for X, y in zip(Xlist, ylist):
                    outX[count] = eh5[X]
                    outy[count] = eh5[y]
                    count += 1
            else:
                if ratbodyonly:
                    h5 = h5[:,:13,:]
                if len(h5.shape) > 2:
                    h5 = np.reshape(h5, (h5.shape[0], h5.shape[1]*h5.shape[2]))
                for X, y in zip(Xlist, ylist):
                    outX[count] = h5[X]
                    outy[count] = h5[y]
                    count += 1
        return outX, outy


def laginds(goodinds, lag):
    """
    Given an array of goodinds, pick inds based on lag value.
    """
    goodindssplit = np.split(goodinds, np.where(np.ediff1d(goodinds) > 1)[0] + 1)
    goodindssplit = [goodinds for goodinds in goodindssplit if len(goodinds) > lag + 1]
    Xlist = []
    ylist = []
    for indlist in goodindssplit:
        if len(indlist) % lag != 0:
            Xlist += np.split(indlist[:-(len(indlist) % lag)], len(indlist) // lag)
            if len(indlist)%lag != 1:
                ylist += np.split(indlist[1:-(len(indlist) % lag)+1], len(indlist) // lag)
            else:
                ylist += np.split(indlist[1:], len(indlist) // lag)
        else:
            Xlist += np.split(indlist[:-lag], (len(indlist) // lag) - 1)
            ylist += np.split(indlist[1:-lag+1], (len(indlist) // lag) - 1)
        # ylist += indlist[lag::lag].tolist()

    return np.array(Xlist), np.array(ylist)

def laginds_old(goodinds, lag):
    """
    Given an array of goodinds, pick inds based on lag value.
    """
    goodindssplit = np.split(goodinds, np.where(np.ediff1d(goodinds)>1)[0]+1)
    goodindssplit = [goodinds for goodinds in goodindssplit if len(goodinds)>lag+1]
    Xlist = []
    ylist = []
    for indlist in goodindssplit:
        if len(indlist)%lag != 0:
            Xlist += np.split(indlist[:-(len(indlist)%lag)], len(indlist)//lag)
        else:
            Xlist += np.split(indlist[:-lag], (len(indlist)//lag)-1)
        ylist += indlist[lag::lag].tolist()
        
    return np.array(Xlist), np.array(ylist)
    
def offsetinds(allgoodinds):
    """offset goodinds list based on ratenterframe values."""
    return [goodinds-ratenterframe for goodinds,ratenterframe in zip(allgoodinds, ratenterframes())]
    
import os
def setGPU(GPUid = 0):
    """
    Set GPU for keras and tensorflow. Run before importing any modules.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUid)
    return

def plotloss(history, norm_factor = 1):
    """
    Plot the loss curves from keras history object.
    Input :
    norm_factor : normalization factor for loss values. 
    
    """
    fig, axes = plt.subplots(2,1,figsize=(12,12))
    ax = axes[0]
    ax.plot(history.epoch, np.sqrt(np.array(history.history['loss'])*norm_factor), 
            label='Training Loss')
    try:
        ax.plot(history.epoch, np.sqrt(np.array(history.history['val_loss'])*norm_factor), 
            label='Validation Loss')
    except:
        pass
    ax.set_ylabel('Mean Pixel Loss')
    ax.set_xlabel('Epochs')
    ax.grid('on')
    ax = axes[1]
    ax.semilogy(history.epoch, np.sqrt(np.array(history.history['loss'])*norm_factor), 
            label='Training Loss')
    try:
        ax.semilogy(history.epoch, np.sqrt(np.array(history.history['val_loss'])*norm_factor), 
            label='Validation Loss')
    except:
        pass
    ax.set_ylabel('Mean Loss )in Pixels)')
    ax.set_xlabel('Epochs')
    fig.suptitle(str(history.params))
    ax.grid('on')

    return fig, ax 

def bpartref(h5, t = 1000):
    """
    Plots a frame and DLC tracked points for reference.
    """
    fig, ax = plt.subplots()
    for i in range(h5.shape[1]):
        ax.scatter(h5[t, i, 0], h5[t, i, 1], marker='$'+str(i)+'$', s=50)
    ax.set_xlim([np.min(h5[t,:,0])-50, np.max(h5[t,:,0])+50])
    ax.set_ylim([np.min(h5[t,:,1])-50, np.max(h5[t,:,1])+50])
    return fig, ax

def plotframe(t, h5, clip):
    fig, ax = plt.subplots(figsize=(7,7))
    fig.subplots_adjust(0,0,1,1,0,0)
    ax.imshow(clip.get_frame(t/clip.fps))
    sc = ax.scatter(h5[t,:,0], h5[t,:,1], s=12, c = 'darkred')
    ax.axis('off')
    return fig, ax

def _getinds(outpshape, val_frac = 0.2):
    valinds = np.unique(np.random.randint(0,outpshape, int(outpshape*val_frac)))
    traininds = np.setdiff1d(np.arange(outpshape), valinds)
    return traininds, valinds


def create_NN_comparison(clip, h5, nnh5, inds2plot = np.arange(100), pad=400, outfname=None):
    """
    Give movipy.videofile, dlc tracked h5 and RNN h5, plot indices in inds2plot and save as outfname.
    """
    tlag = h5.shape[0]-nnh5.shape[0]
    h5 = h5[tlag:]
    center = medfilt2d(h5[:,8,:2], [7,1])
    h5 = egoh5(h5)
    connections = [[0,2,5,10,12],[0,1,4,7,9,12],[0,3,6,8,11,12],[1,2,3],[4,5,6],[7,8],[9,10,11]]
    fig, ax = plt.subplots(figsize=(10,10))    
    def make_frame(i):
        t = inds2plot[int(i*clip.fps)]
        ax.clear()
        ax.imshow(clip.get_frame((t+tlag)/clip.fps))
        for l in connections:
            ax.plot(h5[t,l,0]+center[t,0], h5[t,l,1]+center[t,1], color='royalblue', alpha=0.5, lw=3)[0]
            ax.plot(nnh5[t,l,0]+center[t,0], nnh5[t,l,1]+center[t,1], color='orangered', alpha=0.5, lw=3)[0]        
        
        ax.set_xlim([center[t,0]-pad,center[t,0]+pad])
        ax.set_ylim([center[t,1]-pad,center[t,1]+pad])
        ax.axis('off')
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=inds2plot.shape[0]/clip.fps)
#     animation.write_gif('matplotlib.gif', fps=clip.fps)
    if outfname is None:
        outfname = 'videos/vid_'+''.join([np.random.choice(list(string.ascii_letters)) for _ in range(20)])+'.mp4'
        animation.write_videofile(outfname, fps=clip.fps, audio=False, threads=12)
        print('Video saved as ', outfname)
    else:
        if outfname[-4:] != '.mp4':
            outfname += '.mp4'
            outfname = 'videos/'+outfname
        animation.write_videofile(outfname, fps=clip.fps, audio=False, threads=12)
        print('Video saved as ', outfname)
        
def measure_RNN_timescale(model, ts):
    """
    model : Keras model
    ts : 2D (or 3D) numpy array to drive RNN with before measuring timescales, shape (t, n_features).
        shape (n_trials, t, nfeatures) if averaging over multiple trials. 
    """
    counts = []
    if len(ts.shape) < 2:
        raise ValueError('Time series input cannot have dimension less than 2!')
    if len(ts.shape) == 2:
        ts = np.rollaxis(np.atleast_3d(ts),2)
    for t in ts:
        for s in t:
            s = np.swapaxes(np.atleast_3d(s), 1,2)
            y_ = model.predict(s)
        count = 0
        y = np.zeros_like(y_)
        diff = []
        while count < 4000:#not np.array_equal(y, y_):
            y = np.swapaxes(np.atleast_3d(y_), 1,2)
            y_ = model.predict(y)
            count += 1
            diff.append(np.mean(y-y_))
        counts.append(count)
    for i,t in enumerate(ts):
        print('Time to converge after driving for %i timesteps : %i'%(t.shape[0], counts[i]))
    return counts, diff


        
        
        
        
    