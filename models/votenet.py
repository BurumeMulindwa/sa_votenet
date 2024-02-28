# My original Votenet.py module
import torch
import torch.nn as nn
import numpy
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'votenet-main'))
sys.path.append(os.path.join(ROOT_DIR, 'votenet-main/models/'))

from models.backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss

from torch.nn.parameter import Parameter

from scipy.integrate import quad
import torch.nn.functional as F



'''This will be added in the model in pytorch-cifar-main as rp_sigmoid-SENet ans shall 
be made active instead of the now present SENet18'''

class nsigmoid(nn.Module):
    def __init__(self, x):
        '''init method'''
        super().__init__()
        self.x = x

    def forward(self, x):
        '''
        Forward pass of the function.
        '''
        
        x = torch.tensor(x , requires_grad=True)
        #print(x)
        
        '''
        f = torch.sigmoid(x)
        f = max_score + torch.log(torch.sum(torch.exp(max_score - x)))
        '''
        #max_score = torch.max(x).cuda()
        #print(max_score)
        '''
        f = 3.125*((50*torch.log(torch.exp((3*x)/50 + 3/1000) + 1))/3 - (50*torch.log(torch.exp((3*x)/50 - 3/1000) + 1))/3 - (50*torch.log(torch.exp((3*x)/50 - 3/5000) + 1))/3 + (50*torch.log(torch.exp((3*x)/50 + 3/5000) + 1))/3 - (50*torch.log(torch.exp((3*x)/50 - 9/5000) + 1))/3 + (50*torch.log(torch.exp((3*x)/50 + 9/5000) + 1))/3 - (50*torch.log(torch.exp((3*x)/50 - 21/5000) + 1))/3 + (50*torch.log(torch.exp((3*x)/50 + 21/5000) + 1))/3)
        '''
        f = 3.15*((100*torch.log(torch.exp((3*x)/100 + 3/2000) + 1))/3 - (100*torch.log(torch.exp((3*x)/100 - 3/2000) + 1))/3 - (100*torch.log(torch.exp((3*x)/100 - 3/10000) + 1))/3 + (100*torch.log(torch.exp((3*x)/100 + 3/10000) + 1))/3 - (100*torch.log(torch.exp((3*x)/100 - 9/10000) + 1))/3 + (100*torch.log(torch.exp((3*x)/100 + 9/10000) + 1))/3 - (100*torch.log(torch.exp((3*x)/100 - 21/10000) + 1))/3 + (100*torch.log(torch.exp((3*x)/100 + 21/10000) + 1))/3)
        
        #print(f)
        return f



import torch
import torch.nn as nn

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cweight = nn.Parameter(torch.zeros(1, (groups * 2), 1))
        self.cbias = nn.Parameter(torch.ones(1, (groups * 2), 1))
        self.sweight = nn.Parameter(torch.zeros(1, (groups * 2), 1))
        self.sbias = nn.Parameter(torch.ones(1, (groups * 2), 1))

        self.nsigmoid = nsigmoid(channel)
        self.gn = nn.GroupNorm(channel // (2 * groups), channel)
    '''
    @staticmethod
    def channel_shuffle_squeeze(x, groups):
        #print("x_channel_shuffle shape:", x.shape)
        x = torch.squeeze(x, dim=1)
        #print("new x_channel_shuffle shape:", x.shape)
        b, h, w = x.shape

        x = x.view(b, h, w) #manipulate the shape of a tensor
        x = x.permute(1, 2, 0) #rearrange the dimensions of a tensor according to a given order

        # flatten
        x = x.reshape(b, -1, w) #change the shape of a tensor while keeping its elements intact

        return x
    '''
    def forward(self, x):
        #print("x_original shape:", x.shape)
        
        b, h, w = x.shape
        
        # channel split
        x_0, x_1 = x.chunk(2, dim=1)
        #print("x_0 shape:", x_0.shape)
        #print("x_1 shape:", x_1.shape)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.nsigmoid(xn)

        
        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.nsigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        #out = self.channel_shuffle_squeeze(out, 2)
        return out



class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
        
        # Add Shuffle Attention module right after the Backbone
        self.sa_module = sa_layer(128, groups=64)  # You may need to adjust 'groups' as needed.

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formatted as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        
        # ----------Shuffle Attention------ 
        end_points['fp2_features'] = self.sa_module(end_points['fp2_features'])
                
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        print('xyz:', xyz.shape)
        print('features:', features.shape)
        
        
        features = torch.squeeze(features, 1)
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features

        end_points = self.pnet(xyz, features, end_points)

        return end_points



if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')

