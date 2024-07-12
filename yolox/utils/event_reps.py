from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.lib import recfunctions as rfn
import torch


def count_measure(t):
    return 1


def timesurface_measure(t_events, t_target, tau, decay='exp'):
    # todo: add the event number limits per pixel (see paper HOTS)
    # todo: add normalized timesurface measurement function (see paper ASTMNet)
    if decay == 'exp':
        return np.exp((t_events - t_target) / tau)
    elif decay == 'tanh':
        return 1 - np.tanh((t_target - t_events) / tau)
    elif decay == 'lin':
        return (t_events - t_target) / tau
    else:
        raise NotImplementedError(f"Decay function {decay} not implemented.")


import numpy as np


# Code adapted from Tonic
def to_voxel_grid_numpy(events, sensor_size, n_time_bins=10):
    """Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    Implements the event volume from Zhu et al. 2019, Unsupervised event-based learning of optical
    flow, depth, and egomotion.

    Parameters:
        events: ndarray of shape [num_events, num_event_channels]
        sensor_size: size of the sensor that was used [W,H].
        n_time_bins: number of bins in the temporal axis of the voxel grid.

    Returns:
        numpy array of n event volumes (n,w,h,t)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    if len(events) == 0:
        return np.zeros(((n_time_bins, 1, sensor_size[1], sensor_size[0])), float)

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()

    # normalize the event timestamps so that they lie between 0 and n_time_bins
    ts = (
            n_time_bins
            * (events["t"].astype(float) - events["t"][0])
            / (events["t"][-1] - events["t"][0])
    )
    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    pols = events["p"]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid


def to_voxel_cube_numpy(events, sensor_size, num_slices, tbins=2):
    """ Representation that creates voxel cube in paper "Object detection with spiking neural networks on automotive \
        event data[C]2022 International Joint Conference on Neural Networks (IJCNN)"
        Parameters:
            events: ndarray of shape [num_events, num_event_channels]
            sensor_size: size of the sensor that was [W, H].
            num_slices: n slices of the voxel cube.
            tbins: number of micro bins in a slice
        Returns:
            numpy array of voxel cube (n,2*tbin,h,w)
    """
    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2
    if len(events) == 0:
        # logger.info('Warning: representation without events')
        return np.zeros(shape=[num_slices, 2 * tbins, sensor_size[0], sensor_size[1]])
    events['t'] -= events['t'][0]
    times = events['t']
    time_window = (times[-1] - times[0]) // num_slices
    events = events[events['t'] < time_window * num_slices]
    # feats = torch.nn.functional.one_hot(torch.from_numpy(events['p']).to(torch.long),
    #                                     2 * tbins)  # 2*tbin 通道维度

    coords = torch.from_numpy(
        structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

    # Bin the events on T timesteps
    coords = torch.floor(coords / torch.tensor([time_window, 1, 1]))

    # TBIN computations
    tbin_size = time_window / tbins

    # get for each ts the corresponding tbin index
    tbin_coords = (events['t'] % time_window) // tbin_size
    # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
    tbin_feats = ((events['p'] + 1) * (tbin_coords + 1)) - 1

    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * tbins)

    sparse_tensor = torch.sparse_coo_tensor(
        coords.t().to(torch.int32),
        feats,
        (num_slices, sensor_size[1], sensor_size[0], 2 * tbins)
    )

    voxel_cube = sparse_tensor.coalesce().to(float).to_dense().permute(0, 3, 1, 2)  # torch.Tensor [n, 2*tbin, H, W]
    return voxel_cube.numpy()


def to_timesurface_numpy(slices, sensor_size, dt, tau, overlap=0):
    assert dt >= 0, print("Parameter delta_t cannot be negative.")
    if slices[0] is None:
        return np.zeros(shape=[len(slices), 2, sensor_size[1], sensor_size[0]])
    all_surfaces = []
    memory = np.zeros((sensor_size[::-1]), dtype=int)  # p y x
    x_index = slices[0].dtype.names.index("x")
    y_index = slices[0].dtype.names.index("y")
    p_index = slices[0].dtype.names.index("p")
    t_index = slices[0].dtype.names.index("t")
    start_t = slices[0][0][t_index]
    for i, slices in enumerate(slices):
        slice = structured_to_unstructured(slices, dtype=int)
        indices = slice[:, [p_index, y_index, x_index]].T
        timestamps = slice[:, t_index]
        memory[tuple(indices)] = timestamps
        diff = -((i + 1) * dt + start_t - memory)
        surf = np.exp(diff / tau)
        all_surfaces.append(surf)  # [n,p, H, W]
    return np.stack(all_surfaces, axis=0)
