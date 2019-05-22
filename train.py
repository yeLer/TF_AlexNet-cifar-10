# coding=utf-8
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time

from alexnet import model
from data import get_data_set

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("save_path", default="./snapshot/cifar-10/", help="model save path")
tf.flags.DEFINE_string("board_path", default="./snapshot/board/", help="board save path")
tf.flags.DEFINE_integer("image_size", default=32, help="image size")
tf.flags.DEFINE_integer("num_channels", default=3, help="input images channels")
tf.flags.DEFINE_integer("class_size", default=10, help="the classification class")
tf.flags.DEFINE_integer("iter_times", default=5000, help="the iter times")
tf.flags.DEFINE_integer("batch_size", default=128, help="the batch_size for training/testing")
tf.flags.DEFINE_bool("fine_tune", default=False, help="fine tune on latest model")
tf.flags.DEFINE_float("learning_rate", default=1e-3, help="the train learning rate")


def main(argv=None):
    train_x, train_y, tain_l, _ = get_data_set("train")
    x, y, output, global_step, y_pred_cls = model()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar("Accyracy/train", accuracy)
    tf.summary.histogram('histogram', accuracy)
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(FLAGS.board_path, sess.graph)
        sess.run(tf.global_variables_initializer())
        if FLAGS.fine_tune:
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("Trying to restore last checkpoint..... ")
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored...")
        for i in range(FLAGS.iter_times):
            randidx = np.random.randint(len(train_x), size=FLAGS.batch_size)  # 此处返回的是小于冷（train）的离散均匀分布，总共有128个
            batch_xs = train_x[randidx]
            batch_ys = train_y[randidx]

            start_time = time()
            i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
            duration = time() - start_time

            if (i_global % 10 == 0) or (i == FLAGS.iter_times - 1):
                _loss, batch_acc, result_merged = sess.run([loss, accuracy, merged], feed_dict={x: batch_xs, y: batch_ys})
                msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, " \
                      "{4:.2f} sec/batch)"
                print(msg.format(i_global, batch_acc, _loss, FLAGS.batch_size / duration, duration))
                train_writer.add_summary(result_merged, i_global)

            if (i_global % 500 == 0) or (i == FLAGS.iter_times - 1):
                acc = predict_test(sess, x, y, y_pred_cls)
                # print('test accuracy is:{}'.format(acc))
                saver.save(sess, save_path=FLAGS.save_path+"/AlexNet-cifar10", global_step=global_step)
                print("Saved checkpoint:{0}-{1}".format("AlexNet-cifar10", i_global))


def predict_test(sess, x, y, y_pred_cls, show_confusion_matrix=False):
    test_x, test_y, test_l, _ = get_data_set("test")
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)  # 返回一个新的数组，用零填充
    while i < len(test_x):
        j = min(i + FLAGS.batch_size, len(test_x))
        batch_xs = test_x[i:j, :]
        # batch_xs是128*3072的大小，最后一个是16*3072
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100

    correct_numbers = correct.sum()

    print("test_x的长度：{3}, Accuracy on Test-Set:{0:.2f}%({1}/{2})".format(acc, correct_numbers, len(test_x), len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(FLAGS.class_size):
            class_name = "({}){}".format(i, test_l[i])
            print(cm[i:], class_name)
        class_numbers = ["({0})".format(i) for i in range(FLAGS.class_size)]
        print("".join(class_numbers))

    return acc


if __name__ =="__main__":
    tf.app.run()
