import tensorflow as tf
import utils.shapenet_provider as sp
import utils.matplot_viewer as mpv
import cfgs.pointnet_config as pn_cfg
import models.pointnet_cls as model
import os
import numpy as np

# Download shapenet data
sp.download_data()
# Get pointnet config
cfg = pn_cfg.get_pointnet_config()


def show_pc():
    for idx in range(len(cfg.train_files)):
        temp_data = sp.load_h5(cfg.train_files[idx])
        temp_labels = temp_data[1]
        # print(temp_labels)
        for j in range(3, len(temp_data[0])):
            mpv.show_pointcloud_fromarray(temp_data[0][j], cfg.label_names[temp_labels[j][0]])


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(cfg.gpu_idx)):
            pc_pl, labels_pl = model.get_inputs_pl(cfg.batch_size, cfg.point_num)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            pred, end_points = model.get_model(pc_pl, is_training_pl, bn_decay=bn_decay)
            loss = model.get_loss(pred, labels_pl, end_points)
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(cfg.batch_size)
            learning_rate = get_learning_rate(batch)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})
        ops = {'pc_pl': pc_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'step': batch}
        for epoch in range(cfg.max_epoch):
            print('**** EPOCH %03d ****' % epoch)

            train_one_epoch(sess, ops)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(cfg.model_dir, "model%03d.ckpt" % epoch))
                print("Model saved in file: %s" % save_path)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        cfg.base_lr,  # Base learning rate.
                        batch * cfg.batch_size,  # Current index into the dataset.
                        cfg.decay_step,          # Decay step.
                        cfg.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      0.5,
                      batch*cfg.batch_size,
                      float(cfg.decay_step),
                      0.5,
                      staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def train_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(cfg.train_files))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(cfg.train_files)):
        print('----' + str(fn) + '-----')
        current_data, current_label = sp.loadDataFile(cfg.train_files[train_file_idxs[fn]])
        current_data = current_data[:, 0:cfg.point_num, :]
        current_data, current_label, _ = sp.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // cfg.batch_size

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * cfg.batch_size
            end_idx = (batch_idx + 1) * cfg.batch_size

            # Augment batched point clouds by rotation and jittering
            rotated_data = sp.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = sp.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pc_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training, }
            step, _, loss_val, pred_val = sess.run([ops['step'],
                                                    ops['train_op'],
                                                    ops['loss'],
                                                    ops['pred']],
                                                   feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cfg.batch_size
            loss_sum += loss_val

        print('mean loss: %f' % (loss_sum / float(num_batches)))
        print('accuracy: %f' % (total_correct / float(total_seen)))


train()
