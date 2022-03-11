import tensorflow as tf
import numpy as np

export_dir = "models/linear_simple"

sess=tf.Session()
tf.saved_model.loader.load(sess, ['serve'], export_dir)

fout=open("graph.info", "w+")
graph = tf.get_default_graph()
for op in graph.get_operations():
     print(op, file=fout)
X = graph.get_tensor_by_name("x:0")
w1 = graph.get_tensor_by_name("bias:0")
print("w1", sess.run(w1))

# Now, access the op that you want to run.
Y = graph.get_tensor_by_name("nn_return:0")
dy_dx = graph.get_tensor_by_name("grad_return:0")

#test
x0 = np.array([[-1.393487, 1.796201]])
feed_dict = {X: x0}
print("x0", x0)
print("Y", sess.run(Y,feed_dict))
dy_dx_0 = sess.run(dy_dx,feed_dict)
dy_dx_1 = np.array([2*x0[0, 0], 6*x0[0, 1], 4*x0[0, 0], 8*x0[0, 1]])
print("dy_dx", dy_dx_0)
print("analytical", dy_dx_1)
print("delta", dy_dx_0 - dy_dx_1)

#test
x0 = np.array([[-2.0, 1.0]])
feed_dict = {X: x0}
print("x0", x0)
print("Y", sess.run(Y,feed_dict))
print("dy_dx", sess.run(dy_dx,feed_dict))

