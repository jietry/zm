""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
import tf_sampling
import tf_grouping
import tf_interpolate
from sklearn.neighbors import KDTree


def knn_kdtree(nsample, xyz, new_xyz):
    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample), dtype=np.int32)
    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=30)
        _, indices[batch_idx] = kdt.query(q_X, k = nsample)

    return indices
def knn_kdtree_rm_boundary(nsample, xyz, new_xyz, boundary_label):
    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample), dtype=np.int32)
    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
#        X = X[boundary_label[batch_idx] == 0]
        X[boundary_label[batch_idx] == 1] += 1000
        kdt = KDTree(X, leaf_size=30)
        _, indices[batch_idx] = kdt.query(q_X, k = nsample)

    return indices

def sampling_with_boundary_label(npoint, xyz, labels):

      labels = tf.tile(tf.expand_dims(labels, -1), [1, 1, 3])
      idx = tf_sampling.farthest_point_sample(npoint, xyz)
      new_xyz = tf_sampling.gather_point(xyz, idx)
      sub_labels = tf_sampling.gather_point(labels, idx)
      return new_xyz, sub_labels[:, :, 0]

def grouping(points, K, src_xyz, q_xyz, use_xyz = True, boundary_label = None):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    point_indices = tf.py_func(knn_kdtree, [K, src_xyz, q_xyz], tf.int32)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis = 3)
    idx.set_shape([batch_size, npoint, K, 2])

    grouped_xyz = tf.gather_nd(src_xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1,1,K,1]) # translation normalization

    if points is not None:
       if boundary_label is None:
            grouped_points = group_point(points, idx)
            grouped_boundary_label = None
       else:
            points = tf.concat([points, tf.expand_dims(boundary_label, [-1])], 2)
            grouped_points =tf.gather_nd(points, idx)
            grouped_boundary_label = grouped_points[:, :, :, -1]
            grouped_points = grouped_points[:, :, :, :-1]
       if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
       else:
            new_points = grouped_points
    else:
     new_points = grouped_xyz
    return grouped_xyz, new_points, idx, grouped_boundary_label


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else: 
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius,K,mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', use_xyz=True, use_nchw=False, boundary_label=None):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all: 
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)   
        else:
           if boundary_label is not None:
               new_xyz, sub_boundary_label =sampling_with_boundary_label(npoint, xyz, boundary_label)
               #print('sub_boundary_label', sub_boundary_label.shape)

               tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label, [-1]), [1, 1, points.shape[2]])
               #print('tmp_boundary_label', tmp_boundary_label.shape)
               #print('points', points.shape)
               points = points * tmp_boundary_label
               #print('points', points.shape)

           new_points, idx, grouped_xyz,grouped_boundary_label= grouping(points,K, xyz, new_xyz, boundary_label=boundary_label)
           #print('grouped_boundary_label', grouped_boundary_label.shape)
       # Point Feature Embedding

        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
       
        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
            for i, num_out_channel in enumerate(mlp2):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,)
            if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points,sub_boundary_label,idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

 
def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True, boundary_label=None):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10) 
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        if boundary_label is None:
            pass
        else:
            tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label, [-1]), [1, 1, interpolated_points.shape[2]])
            #print('tmp_boundary_label', tmp_boundary_label.shape)
            interpolated_points = interpolated_points * tmp_boundary_label
        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1




def pointnet_upsample(xyz1, xyz2, points2,scope,boundary_label=None):
    """ PointNet Feature Propogation (FP) Module
            Input:
                xyz1: (batch_size, ndataset1, 3) TF tensor
                xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
                points2: (batch_size, ndataset2, nchannel2) TF tensor
            Return:
                new_points: (batch_size, ndataset1, nchannel2) TF tensor
    """
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)  # B x ndataset1 x nchannel2
        if boundary_label is None:
            pass
        else:
            tmp_boundary_label = tf.tile(tf.expand_dims(boundary_label, [-1]), [1, 1, interpolated_points.shape[2]])
            #print('tmp_boundary_label', tmp_boundary_label.shape)
            interpolated_points = interpolated_points * tmp_boundary_label
        return interpolated_points
