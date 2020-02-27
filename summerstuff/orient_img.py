#rotates image so fly is oriented vertically facedown
def orient_img(frame):
    frame = box[i,:,:]
    rotated = ndimage.rotate(frame, 270)
    img = np.fliplr(rotated)
    return img
