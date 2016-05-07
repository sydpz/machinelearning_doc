tensorflow-0.8 的一大特性为可以部署在分布式的集群上，本文的内容由Tensorflow的分布式部署手册翻译而来，该手册链接为[TensorFlow分布式部署手册](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/how_tos/distributed/index.md)

---

##分布式TensorFlow
本文介绍了如何搭建一个TensorFlow服务器的集群，并将一个计算图部署在该分布式集群上。以下操作建立在你对 [TensorFlow的基础操作](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/basic_usage.md)已经熟练掌握的基础之上。
###Hello world的分布式实例的编写
以下是一个简单的TensorFlow分布式程序的编写实例
```
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```
[tf.train.Server.create_local_server()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/train.md#Server.create_local_server) 会在本地创建一个单进程集群，该集群中的服务默认为启动状态。
###创建集群(cluster)
TensorFlow中的集群(cluster)指的是一系列能够对TensorFlow中的图(graph)进行分布式计算的任务(task)。每个任务是同服务(server)相关联的。TensorFlow中的服务会包含一个用于创建session的主节点和一个用于图运算的工作节点。另外, TensorFlow中的集群可以拆分成一个或多个作业(job), 每个作业可以包含一个或多个任务。下图为作者对集群内关系的理解。
创建集群的必要条件是为每个任务启动一个服务。这些任务可以运行在不同的机器上，但你也可以在同一台机器上启动多个任务(比如说在本地多个不同的GPU上运行)。每个任务会做如下的两步工作：

 1. 创建一个  `tf.train.ClusterSpec` 用于对集群中的所有任务进行描述，该描述内容对于所有任务应该是相同的。
 2. 创建一个`tf.train.Server` 并将`tf.train.ClusterSpec`  中的参数传入构造函数，并将作业的名称和当前任务的编号写入本地任务中。
###创建`tf.train.ClusterSpec` 的具体方法

`tf.train.ClusterSpec` 的传入参数是作业和任务之间的关系映射，该映射关系中的任务是通过ip地址和端口号表示的。具体映射关系如下表所示：
<table>
  <tr><th><code>tf.train.ClusterSpec</code> construction</th><th>Available tasks</th>
  <tr>
    <td><pre>
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
</pre></td>
<td><code>/job:local/task:0 local<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><pre>
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222", 
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
</pre></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

###为每一个任务创建`tf.train.Server` 的实例
每一个[tf.train.Server](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/train.md#Server) 对象都包含一个本地设备的集合, 一个向其他任务的连接集合，以及一个可以利用以上资源进行分布式计算的“会话目标”("[session target](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/client.md#Session)")。每一个服务程序都是一个指定作业的一员，其在作业中拥有自己独立的任务号。每一个服务程序都可以和集群中的其他任何服务程序进行通信。
以下两个代码片段讲述了如何在本地的2222和2223两个端口上配置不同的任务。
```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```
```python
# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```
**注** ：当前手动配置任务节点还是一个比较初级的做法，尤其是在遇到较大的集群管理的情况下。tensorflow团队正在开发一个自动程序化配置任务的节点的工具。例如：集群管理工具[Kubernetes](http://kubernetes.io/)。如果你希望tensorflow支持某个特定的管理工具，可以将该请求发到[GitHub issue](https://github.com/tensorflow/tensorflow/issues) 里。
### 为模型指定分布式的设备
为了将某些操作运行在特定的进程上，可以使用[tf.device()](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/framework.md#device) 函数来指定代码运行在CPU或GPU上。例如：
```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```
在上面的例子中，