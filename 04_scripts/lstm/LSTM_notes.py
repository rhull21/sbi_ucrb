from sklearn.utils import shuffle

# sliding windows
def _sliding_windows(data, seq_length, future=1):
    """This takes in a dataset and sequence length, and uses this
    to dynamically create (multiple) sliding windows of data to fit.
    This works for only one feature (as written)
    """
    x = []
    y = []

    for i in range(len(data)-seq_length-future):
        tx = data[i:(i+seq_length)]
        ty = data[i+seq_length+future-1]
        x.append(tx)
        y.append(ty)

    return np.array(x),np.array(y)

# break up into Predictors and Predictands
def sliding_windows(data_window, seq_length, fut_length=1):
    """This uses the basic function _sliding_windows to create sliding
    sliding windows for datasets with large feature spaces where
    data_window.columns = [Feature1, Feature2..., Featuren, Y]
    (i.e., the final column is the 'target' value to be predicted)
    """
    it = 0
    cols = data_window.columns

    if len(cols) > 1:
        for idx in cols:
            x_0, y = _sliding_windows(data[idx].to_numpy(), seq_length, fut_length)
            if idx == cols[0]:
                x = np.zeros((x_0.shape[0],x_0.shape[1],len(cols)))
            x[:,:,it] = x_0
        it = it+1
    else:
        x, y = _sliding_windows(data.to_numpy(), seq_length, fut_length)
    return x, y

# assemble master dataset
for idx in range(len(name_ens_l)):
    # ensemble member name and K
    member_name, K = name_ens_l[idx], K_ens_l[idx]
    # extract data
    data_window = data[labelist[:-1]]
    # append parameter data
    data_window['K'], data_window['Y'] = K, data[member_name]
    # create sliding windows
    x_temp, y_temp = sliding_windows(data_window, seq_length, fut_length)
    # create x and y to pass on to next step
    if idx == 0:
        x, y = x_temp, y_temp
    else:
        x, y = np.append(x,x_temp, axis=0), np.append(y,y_temp, axis=0)
    # clean up
    del x_temp, y_temp, member_name, K, data_window

# shuffle?
x, y = shuffle(x, y)

# Save Dataset for later

# Train Model
