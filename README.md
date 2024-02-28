# sa_votenet 
Paper title: USING 3D SHUFFLE ATTENTION IN DEEP HOUGH VOTING TO IMPROVE 3D POINT CLOUDS OBJECT DETECTION 
A shuffle attention based deep hough voting for 3D object detection using the sunrgbd dataset showing promising improvements


ABSTRACT

Recently, attention mechanisms have developed into an important tool for performance improvement of deep neural networks. In computer vision, attention mechanisms are generally divided into two main branches: spatial and channel attention to seize the pixel pairwise correlation and channel dependency respectively. The fusion of both attention mechanisms for better operation certainly results in the increase of the computational operating cost. This paper proposes a new lighter but efficient deep Hough voting mechanism combined with a piecewise shifted n-sigmoid shuffle attention (SA) structure to effectively predict bounding box parameters directly from 3D scenes and detect objects more accurately. In essence, the authors propose an end-to-end 3D object detection network based on a symbiosis of Hough voting, the SA mechanism and deep point set networks. Specifically, SA first clusters channel dimensions into numerous sub-features and finally processes them. This allows the model to improve accuracy by selectively attending to more relevant features of the input data. To achieve this, all the features are combined. Moreover, a channel shuffle operator is adopted that uses a special piecewise shifted sigmoid activation function allows data communication. This activation function endeavours to enhance the learning and generalization capacity of the 3D neural networks while reducing the vanishing gradient dilemma. The proposed model outperformed state-of-the-art 3D detection methods when validated based on the sizable SUNRGBD dataset, by compacted model size and higher efficiency. Experiments conducted using the n-sigmoid SA Deep Hough Voting model showed an increase of 12.02 mean accuracy precision (mAP) when compared to the VoteNet. It also got 9.92 mAP higher compared to the MLVCNet, and 10.32 mAP higher than the Point Transformer. The proposed model not only decreases the vanishing gradient but also brings out valuable features by fusing channel-wise and spatial information while improving accuracy results in 3D object detection.



Installation
Install Pytorch and Tensorflow (for TensorBoard). It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Ubuntu 18.04, Pytorch v1.1, TensorFlow v1.14, CUDA 10.0 and cuDNN v7.4. Note: After a code update on 2/6/2020, the code is now also compatible with Pytorch v1.2+

Compile the CUDA layers for PointNet++, which we used in the backbone network:

cd pointnet2
python setup.py install


Install the following Python dependencies (with pip install):

matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'



Train and test on SUN RGB-D
To train a new sa-voteNet model on SUN RGB-D data (depth images):

CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd


To test the trained model with its checkpoint:

python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

A properly trained SA-VoteNet should have around 69.72 mAP@0.25 and 54 mAP@0.5.

