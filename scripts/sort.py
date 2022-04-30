"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
from typing import Any
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix) -> np.ndarray:
  """Function solves the optimal assignment problem
  using Jonker-Volgenant algorithm for dense (LAPJV)
  matrices

  Args:
      cost_matrix (np.ndarray): (N,M) matrix where
      c_ij refers to cost b/w ith row and jth column
      elements

  Returns:
      np.ndarray : (P,2) where P = number of valid
      assignments which are optimal based on input
      cost matrix. eg : if cost matrix shape = (7,6),
      there are 7 rows for which 6 columns are 
      to be assigned.

      Possible output = [[0,1]
                         [1,2]
                         [2,3]
                         [3,4]
                         [4,5]]
      means there are 5 valid assignments, 2nd column
      to 1st row, 3rd column to 2nd row and 4th column
      to 3rd row and so on
  """
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))



def iou_batch(bb_test : np.ndarray, bb_gt : np.ndarray) -> np.ndarray:
  """ Computes IOU between two bboxes given in [x1,y1,x2,y2] form 

  Args:
      bb_test (np.ndarray): (M,5) where M is number of bounding
      boxes in bb_test. 5 columns indicate [x1,y1,x2,y2,confidence].

      bb_gt (np.ndarray): (N,5) where N is number of bounding
      boxes in bb_gt.

  Returns:
      iou (np.ndarray) : (M,N) array containing intersection
      over union for each combination of bounding boxes in 
      bb_test and bb_gt

  Example
    bb_test = (7,5) and bb_gt = (6,5), then return value
    will have shape = (7,6)
  """
  # assume bb_test shape = (7,5) and bb_gt shape = (6,5)
  
  # insert axis at 1st axis, bb_gt shape = (1,6,5)
  bb_gt = np.expand_dims(bb_gt, 0)

  # insert axis at 2nd axis, bb_gt shape = (7,1,5)
  bb_test = np.expand_dims(bb_test, 1)
  
  # xx1, yy1 denote the x,y coordinates of left top 
  # corner of intersection between the bounding boxes
  # xx1 shape = yy1 shape = (7,6)
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])

  # xx2, yy2 denote the x,y coordinates of right bottom 
  # corner of intersection between the bounding boxes
  # xx2 shape = yy2 shape = (7,6)
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  
  # calculate the intersection area using width * height
  # wh  = (7,6)
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h

  # iou = intersection / union where 
  # union = area(bb_test) + are(bb_gt) - intersection
  # iou shape = (7,6)
  iou = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(iou)  


def convert_bbox_to_z(bbox : np.ndarray) -> np.ndarray:
  """ Converts a bounding box in the form [x1,y1,x2,y2, confidence]
   to measurement z in the form [x,y,s,r] where x,y is the centre of 
   the box and s is the scale/area and r is the aspect ratio

  Args:
      bbox (np.ndarray): (5,) vector representing bounding box
      top left and bottom right x,y coordinates and confidence

  Returns:
      np.ndarray : (4,1) measurement vector equivalent to input
      bounding box
  """
  # w,h,x,y are scalar doubles
  w = bbox[2] - bbox[0]    # width of bbox
  h = bbox[3] - bbox[1]    # height of bbox
  x = bbox[0] + w/2.       # x-coordinate of centre of bbox
  y = bbox[1] + h/2.       # y-coordinate of centre of bbox
  s = w * h                # s = bbox area
  r = w / float(h)         # w = width to height aspect ratio
  return np.array([x, y, s, r]).reshape((4, 1))



def convert_x_to_bbox(x : np.ndarray, score : Any=None) -> np.ndarray:
  """ Converts a bounding box in the centre form [x,y,s,r] to
    [x1,y1,x2,y2] form where x1,y1 is the top left and 
    x2,y2 is the bottom right coordinates

  Args:
      x (np.ndarray): (4,) vector representing bounding box
      centre x,y coordinates and area and aspect ratio

      score (Any, optional): confidence of bbox. Defaults to None.
      If given, the value is given as such in output 

  Returns:
      np.ndarray: (1,N) vector representing bounding box
      top left and bottom right x,y coordinates
      N = 4 if score is None
      N = 5 if score is not None
  """
  # x[2] = (w*h), x[3] = (w/h)
  # w and h are scalar doubles
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w

  # convert coordinates from centre to top left and bottom right
  if(score==None):
    return np.array([x[0]-w/2., x[1]-h/2., 
                     x[0]+w/2., x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2., x[1]-h/2.,
                     x[0]+w/2., x[1]+h/2.,
                     score]).reshape((1,5))


class KalmanBoxTracker(object):
  count = 0
  def __init__(self,bbox : np.ndarray) -> None:
    """ This class represents the internal state of individual
    tracked objects observed as bbox.  Initialises a tracker 
    using initial bounding box. State vector = 
    [u,v,s,r,u_dot,v_dot,s_dot] as per SORT paper 
    where u,v is centre of bbox, s = area of bbox, 
    r = aspect ratio of bbox.

    Motion model is a constant velocity model and measurement 
    model assumes [u,v,s,r] states are observed

    Args:
        bbox (np.ndarray): (5,) vector representing bounding box
        top left and bottom right x,y coordinates and confidence
    """
    # define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)

    # transition matrix shape = (7,7)
    self.kf.F = np.array([
                          [1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]
                        ])

    # measurement matrix shape = (4,7)
    self.kf.H = np.array([
                          [1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]
                        ])

    # measurement noise shape = (4,4)
    self.kf.R[2:,2:] *= 10.

    # state covariance shape = (7,7)
    # give high uncertainty to the unobservable initial velocities
    self.kf.P[4:,4:] *= 1000. 
    self.kf.P *= 10.

    # process noise shape = (7,7)
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    # state is initialized to using input bbox 
    self.kf.x[:4] = convert_bbox_to_z(bbox)

    # counter for incrementing track number
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1

    # track management variables
    self.time_since_update = 0
    self.history = []

    # hits = total number of detections associated with this track
    # hit_streak = number of consecutive frames where track is associated
    self.hits = 0
    self.hit_streak = 0
    self.age = 0


  def update(self,bbox : np.ndarray) -> None:
    """ Performs Kalman filter update using observed bbox
    measurement to get posterior state and covariance
    Also resets tracker variables accordindly

    Args:
        bbox (np.ndarray): (5,) vector representing bounding box
        top left and bottom right x,y coordinates and confidence
    """
    # reset history related varibles
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    
    # convert bbox to [u,v,s,r] format and
    # perform kalman filter update
    self.kf.update(convert_bbox_to_z(bbox))



  def predict(self) -> np.ndarray:
    """ Predicts state using constant velocity model 
    and returns predicted state

    Returns:
        np.ndarray: predicted state vector
    """
    # if the old area + in delta_area equals 0
    # let delta_area be zero
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0

    # perform prediction
    self.kf.predict()
    self.age += 1

    # if no state
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  # pattern = path to folder containing detection text files for the 
  # scenario
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  print(pattern)
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
