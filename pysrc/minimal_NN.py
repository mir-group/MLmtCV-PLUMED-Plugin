import tensorflow as tf
import numpy as np
from tensorflow.python.ops.parallel_for.gradients import jacobian
import tensorflow.keras.backend as K


def neural_net(x):
    '''Define the neural network'''
    w1 = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="bias")
    x1 = x**2
    pred = K.dot(x1, w1)
    return tf.identity(pred, name="nn_return")


def grad(x):
    ''' Define the gradients '''
    J = jacobian(Y, X)
    return tf.reshape(J, [-1], name="grad_return")

# prepare input
X = tf.placeholder("float", [1.0, 2.0], name="x")
x0 = np.array([3.0, 4.0]).reshape([1, 2])
feed_dict = {X: x0}

# define an operation
Y = neural_net(X)
dy_dx = grad(X)

sess = tf.keras.backend.get_session()
sess.run(tf.global_variables_initializer())

print(sess.run(Y, feed_dict))
print(sess.run(dy_dx, feed_dict))

export_dir = "models/linear_simple"
inputs = {"x": X}
outputs = {"y": Y, "dy_dx": dy_dx}
# # Option 1
# tf.saved_model.simple_save(sess, export_dir, inputs=inputs, outputs=outputs)

from  tensorflow.python import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

builder = saved_model.builder.SavedModelBuilder(export_dir)
signature = predict_signature_def(inputs=inputs, outputs=outputs)
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=["serve"],
                                     signature_def_map={'predict': signature})
builder.save()


# # the following doesn't work...
# export_dir = "models/linear"
# builder = tf.saved_model.Builder(export_dir)
#
# # Build the signature_def_map
# tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
# tensor_info_y = tf.saved_model.utils.build_tensor_info(Y)
# tensor_info_dy = tf.saved_model.utils.build_tensor_info(dy_dx)
#
# signature = (
#     tf.saved_model.signature_def_utils.build_signature_def(
#         inputs=inputs,
#         outputs=outputs,
#         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
#
# builder.add_meta_graph_and_variables(
#        sess, [tf.saved_model.tag_constants.SERVING],
#        signature_def_map={
#            'predict': signature,
#        },
#        main_op=tf.tables_initializer(),
#        strip_default_attrs=True)
#
# builder.save()
