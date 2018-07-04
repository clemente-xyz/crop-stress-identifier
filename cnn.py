import tensorflow as tf


class Convolutional_Neural_Network:
    def __init__(
        self,
        classes,
        batch_size,  # number of images that enters to the model every iteration
        img_dimension,
    ):
        self.classes = classes
        self.batch_size = batch_size
        self.img_dimension = img_dimension
        self.x = tf.placeholder(
            "float", [None, img_dimension]
        )  # input vars, where we specify the dimension of the img (28*28=784 px)
        self.y = tf.placeholder("float")  # output var

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")

    def maxpool2d(self, x):
        #                        size of window         movement of window
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"
        )

    def model(self, x):
        # define the layers structure
        weights = {
            "w_conv1": tf.Variable(
                tf.random_normal(
                    [5, 5, 1, 32]
                )  # 5 by 5 px convolution that takes 1 input and generates 32 features
            ),
            "w_conv2": tf.Variable(
                tf.random_normal(
                    [5, 5, 32, 64]
                )  # 5 by 5 px convolution that takes 32 inputs and generates 64 features
            ),
            "w_fc": tf.Variable(
                tf.random_normal(
                    [7 * 7 * 64, 1024]
                )  # 7*7 img times 64 features (reduced from 28*28 img over previous convolutions) with 1024 neurons
            ),
            "out": tf.Variable(tf.random_normal([1024, self.classes])),
        }
        biases = {
            "b_conv1": tf.Variable(
                tf.random_normal(
                    [32]
                )  # 5 by 5 px convolution that takes 1 input and generates 32 features
            ),
            "b_conv2": tf.Variable(
                tf.random_normal(
                    [64]
                )  # 5 by 5 px convolution that takes 32 inputs and generates 64 features
            ),
            "b_fc": tf.Variable(
                tf.random_normal(
                    [1024]
                )  # 7*7 img times 64 features (reduced from 28*28 img over previous convolutions) with 1024 neurons
            ),
            "out": tf.Variable(tf.random_normal([self.classes])),
        }

        x = tf.reshape(x, shape=[-1, 28, 28, 1])  # reshape inputs from 784 to 28*28 px.

        conv1 = tf.nn.relu(self.conv2d(x, weights["w_conv1"]) + biases["b_conv1"])
        conv1 = self.maxpool2d(conv1)

        conv2 = tf.nn.relu(self.conv2d(conv1, weights["w_conv2"]) + biases["b_conv2"])
        conv2 = self.maxpool2d(conv2)

        fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights["w_fc"]) + biases["b_fc"])

        output = tf.matmul(fc, weights["out"]) + biases["out"]

        return output

    def train_me(self, data_set):  # takes x: input data
        prediction = self.model(self.x)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=self.y)
        )  # we calculate de difference between the prediction and the optimus value
        optimization = tf.train.AdamOptimizer().minimize(
            cost
        )  # using AdamOptimizer model we minimize the cost (considering learning_rate=0.001)

        no_epochs = 3  # number of cycles feed foward + backprop

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            for epoch in range(no_epochs):
                epoch_loss = 0
                for i in range(int(data_set.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = data_set.train.next_batch(self.batch_size)
                    i, c = session.run(
                        [optimization, cost],
                        feed_dict={self.x: epoch_x, self.y: epoch_y},
                    )
                    epoch_loss = epoch_loss + c

                print(
                    "Epoch: ",
                    epoch,
                    "/ completed out of: ",
                    no_epochs,
                    "/ loss: ",
                    epoch_loss,
                )

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))

            print(
                "Accuracy: ",
                accuracy.eval(
                    {self.x: data_set.test.images, self.y: data_set.test.labels}
                ),
            )

