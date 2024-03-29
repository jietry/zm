import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
#import util
from PointConv import feature_encoding_layer, feature_decoding_layer,feature_decoding_layer_depthwise
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_upsample
import pointconv_util
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl
def get_boundary_model_loss(labels_pl, point_cloud, is_training, num_class, sigma, bn_decay=None, weight_decay = None):
    # generate boundary 0 if boundary else 1
    num_neighbor = 30
    ratio = 0.6
    g_xyz, g_labels, g_idx, _ = pointconv_util.grouping(tf.cast(tf.expand_dims(labels_pl, [-1]), tf.float32), num_neighbor, point_cloud[:, :, :3], point_cloud[:, :, :3], use_xyz = False)
    g_labels = tf.squeeze(g_labels, [-1])
    self_labels = tf.cast(tf.tile(tf.expand_dims(labels_pl, [-1]), [1, 1, num_neighbor]), tf.float32)
    same_label_num = tf.reduce_sum(tf.cast(tf.equal(g_labels, self_labels), tf.float32), axis=2)

    boundary_points = tf.cast(tf.greater_equal(same_label_num, (num_neighbor * ratio)), tf.int32)
    boundary_points = tf.cast(boundary_points, tf.float32)

    target_boundary_label = tf.stop_gradient(boundary_points)

    # ========================================

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = point_cloud

    #difference extraction
    _, tmp_grouped_feature, _, _ = pointconv_util.grouping(l0_points, 8, l0_xyz, l0_xyz, boundary_label=None, use_xyz=False)
    mean, var = tf.nn.moments(tmp_grouped_feature, axes=2)
    print(var.shape)
    # tmp_average_feature = tf.reduce_mean(tmp_grouped_feature, axis=2)
    l0_points = var

    net = tf_util.conv1d(l0_points, 32, 1, padding='VALID', bn=True, is_training=is_training, scope='boundary_fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='boundary_fc2', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=is_training, scope='boundary_fc3', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='boundary_fc4', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 1, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='boundary_fc5')
    net = tf.squeeze(net,[2])

    boundary_weight = 2 - 1 * target_boundary_label
    boundary_loss = tf.losses.sigmoid_cross_entropy(tf.cast(target_boundary_label, tf.int32), net, weights=boundary_weight)

    return net, boundary_loss

def get_model(boundary_label, point_cloud, is_training, num_class, sigma, bn_decay=None, weight_decay = None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    boundary_label = tf.sigmoid(boundary_label)
    boundary_label = tf.stop_gradient(boundary_label)

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:,:,:3]
    l0_points = point_cloud
    # Feature encoding layers
    k=30
    l1_xyz, l1_points, sub_boundary1, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1,K=32, mlp=[32, 32, 64], mlp2=None, group_all=False,is_training=is_training, bn_decay=bn_decay, scope='layer1', boundary_label=boundary_label)
   
    adj_matrix_l1 = tf_util.pairwise_distance(l1_points)
    nn_idx = tf_util.knn(adj_matrix_l1, k=k)
    net = tf.concat([l1_xyz, l1_points], axis=-1)
    edge_feature_l1 = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)
    edge_feature_l1 = tf.reduce_max(edge_feature_l1, axis = -2, keep_dims = False)
    l1_points = edge_feature_l1

    l2_xyz, l2_points, sub_boundary2 = feature_encoding_layer(l1_xyz, l1_points, npoint=256, radius = 0.2, sigma = 2 * sigma, K=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer2', boundary_label=sub_boundary1)
   
    adj_matrix_l2 = tf_util.pairwise_distance(l2_points)
    nn_idx = tf_util.knn(adj_matrix_l2, k=k)
    net1 = tf.concat([l2_xyz, l2_points], axis=-1)
    edge_feature_l2 = tf_util.get_edge_feature(net1, nn_idx=nn_idx, k=k)
    edge_feature_l2 = tf.reduce_max(edge_feature_l2, axis = -2, keep_dims = False)
    l2_points = edge_feature_l2

    l3_xyz, l3_points,sub_boundary3 = feature_encoding_layer(l2_xyz, l2_points, npoint=64, radius = 0.4, sigma = 4 * sigma, K=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer3', boundary_label=sub_boundary2)

    adj_matrix_l3 = tf_util.pairwise_distance(l3_points)
    nn_idx = tf_util.knn(adj_matrix_l3, k=k)
    net2 = tf.concat([l3_xyz, l3_points], axis=-1)
    edge_feature_l3 = tf_util.get_edge_feature(net2, nn_idx=nn_idx, k=k)
    edge_feature_l3 = tf.reduce_max(edge_feature_l3, axis = -2, keep_dims = False)
    l3_points = edge_feature_l3

    l4_xyz, l4_points ,_= feature_encoding_layer(l3_xyz, l3_points, npoint=36, radius = 0.8, sigma = 8* sigma, K=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay, scope='layer4')
  
    # Feature decoding layers
    l3_points = feature_decoding_layer_depthwise(l3_xyz, l4_xyz, l3_points, l4_points, 0.8, 8* sigma, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = feature_decoding_layer_depthwise(l2_xyz, l3_xyz, l2_points, l3_points, 0.4, 4* sigma, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2', boundary_label=sub_boundary2)
    l1_points = feature_decoding_layer_depthwise(l1_xyz, l2_xyz, l1_points, l2_points, 0.2, 2* sigma, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3', boundary_label=sub_boundary1)
    l0_points =pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay, scope='fa_layer4', boundary_label=boundary_label)
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    net = tf.concat([net,point_cloud],axis=2)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')
    return net, end_points

def get_loss(pred, label, smpw, boundary_loss):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + weight_reg + boundary_loss
    tf.summary.scalar('classify loss', classify_loss)
    tf.summary.scalar('total loss', total_loss)
    return total_loss

if __name__=='__main__':
    import pdb
    pdb.set_trace()

    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)
