import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from torchstat import stat
import torchvision.models as models

import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import torch
import torchvision
from thop import profile
try:
    from utils import *
except (ImportError, ModuleNotFoundError):
    # utils may not be available when importing model submodules directly
    pass
from torch.autograd import Variable
try:
    from IPython.display import clear_output
except ImportError:
    # IPython may not be available
    pass

from .FLDCF_multiModal.FLDCF_multiModal import CONFIGS as CONFIGS_ViT_seg
# from .FLDCF_multiModal.FLDCF_multiModal import CONFIGS as CONFIGS_ViT_seg

# /media/lscsc/nas/mading/fakedetect/src/model/FLDCF_multiModal/FLDCF_multiModal.py
# /media/lscsc/nas/mading/fakedetect/src/model/FLDCF_multiModal/model/vitcross_seg_modeling.py

def get_model(name,args):
    if name == 'crnet':
        from .crnet_small.crnet import CDnetV1_MODEL as M
    if name == 'CDnetV2':
        from .CDnetV2.CDnetv2_model import CDnetV2_MODEL as M
    if name == 'face':
        from .faceforensics.face import TransferModel as M
    if name == 'deepfake':
        from .deepfake.deepfake import Deepfake
        return Deepfake()
    if(name =='patch'):
        from .patchfake.patch import create_model
        return create_model() 
    if(name =='capsule'):
        from .capsule.capsule import CapsuleNet as M
    if(name =='scunet'):
        from .scunet.scunet import SCSEUnet as M
    if(name =='dfcn'):
        from .dfcn.dfcn import normal_denseFCN as M
    if(name =='my2'): # FLDCF
        from .my.my import Restore2 as M
    if(name =='mylocal'):
        from .my.my import Restorelocal as M
    if(name =='restore'):
        from .my.my import Restoretrain as M
    if(name =='FLDCF2'): # not FLDCF2
        from .FLDCF2.FLDCF2 import FLDCF2 as M
    if(name=='mvss'):
        from .mvss.mvssnet import get_mvss

        print('==> Building model..')
        model = get_mvss(args, num_classes=2).cuda()
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(model, inputs=(dummy_input,))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        return get_mvss(backbone='resnet50',
                             pretrained_base=True,
                             nclass=1,
                             sobel=True,
                             constrain=True,
                             n_input=3).cuda()
    if(name =='movenet'): # SE-Network
        from .movenet.movenet import Movenet as M
    if name == 'segformer':
        # Lightweight SegFormer-like model for compatibility
        from .segformer.segformer import create_segformer as M_create
        print('==> Building SegFormer...')
        try:
            model = M_create(args=None, num_classes=1).cuda()
        except Exception:
            model = M_create(args=None, num_classes=1)

        # compute FLOPs / Params / Memory / Speed for input 1x3x256x256
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        try:
            model = model.to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f}G')
            print(f'Parameters: {params_m:.2f}M')
        except Exception as e:
            print(f'Warning: Could not compute FLOPs/Params: {e}')
            flops_g, params_m = 0, 0

        # Memory
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)
                torch.cuda.empty_cache()
                model.eval()
                with torch.no_grad():
                    _ = model(dummy_input)
                memory_allocated = torch.cuda.max_memory_allocated(device=device)
                memory_mb = memory_allocated / (1024 ** 2)
                print(f'Memory: {memory_mb:.2f}MB')
            else:
                memory_mb = 0
        except Exception as e:
            print(f'Warning: Could not compute memory: {e}')
            memory_mb = 0

        # Speed (FPS)
        try:
            model.eval()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            num_iterations = 10
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = num_iterations / elapsed_time if elapsed_time > 0 else 0
            print(f'Speed: {fps:.2f}FPS')
        except Exception as e:
            print(f'Warning: Could not compute speed: {e}')
            fps = 0

        print('=' * 60)
        print('SegFormer Model Summary:')
        print(f'  Input Size: 1x3x256x256')
        print('-' * 60)
        print(f'  FLOPs: {flops_g:.2f}G')
        print(f'  Parameters: {params_m:.2f}M')
        print(f'  Memory: {memory_mb:.2f}MB')
        print(f'  Speed: {fps:.2f}FPS')
        print('=' * 60)

        return model
    if(name =='FLDCF_multiModal'): # FLDCF2
        from .FLDCF_multiModal.FLDCF_multiModal import FLDCF_multiModal as M
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))

        print('==> Building model..')
        print('M-FLDCF')
        # model = M(args, config_vit, num_classes=2).cuda()
        # dummy_input = torch.randn(1, 3, 256, 256).cuda()
        # dummy_input1 = torch.randn(1, 1, 256, 256).cuda()
        # flops, params = profile(model, inputs=(dummy_input,dummy_input1))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        model_instance = M(args, config_vit)
        model_instance.load_from(weights=np.load(config_vit.pretrained_path))
        return model_instance, config_vit

        # return M(args, config_vit), config_vit

    if(name =='FLDCF_multiModal_TransUNet'): # FLDCF2_multiModal_TransUNet
        from .FLDCF_multiModal_TransUNet.FLDCF_multiModal_TransUNet import FLDCF_multiModal_TransUNet as M
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))

        print('==> Building model..')
        # model = M(args, config_vit, num_classes=2).cuda()
        # dummy_input = torch.randn(1, 3, 256, 256).cuda()
        # dummy_input1 = torch.randn(1, 1, 256, 256).cuda()
        # # flops, params = profile(model, inputs=(dummy_input,dummy_input1))
        # flops, params = profile(model, inputs=(dummy_input,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

        model_instance = M(args, config_vit)
        model_instance.load_from(weights=np.load(config_vit.pretrained_path))
        return model_instance, config_vit

        # return M(args, config_vit), config_vit
    
    if(name == 'ClipViTL14'): # CLIP Vision Transformer Large 14
        from .ClipViTL14.ClipViTL14 import ClipViTL14 as M
        from .ClipViTL14.ClipViTL14 import get_model_performance_metrics
        
        print('==> Building ClipViTL14 model...')
        model = M(args, num_classes=2, pretrained=True).cuda()
        
        # Compute and print performance metrics
        print('==> Computing ClipViTL14 performance metrics...')
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f}G')
            print(f'Parameters: {params_m:.2f}M')
        except Exception as e:
            print(f'Warning: Could not compute FLOPs and Parameters: {e}')
            flops_g, params_m = 0, 0
        
        # Compute memory usage
        try:
            torch.cuda.reset_peak_memory_stats(device='cuda')
            torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input)
            
            memory_allocated = torch.cuda.max_memory_allocated(device='cuda')
            memory_mb = memory_allocated / (1024 ** 2)
            print(f'Memory: {memory_mb:.2f}MB')
        except Exception as e:
            print(f'Warning: Could not compute memory: {e}')
            memory_mb = 0
        
        # Compute speed (FPS)
        try:
            model.eval()
            torch.cuda.synchronize(device='cuda')
            
            # Warm up
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize(device='cuda')
            
            # Timing
            num_iterations = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize(device='cuda')
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            fps = num_iterations / elapsed_time if elapsed_time > 0 else 0
            print(f'Speed: {fps:.2f}FPS')
        except Exception as e:
            print(f'Warning: Could not compute speed: {e}')
            fps = 0
        
        # Print summary
        print('=' * 60)
        print('ClipViTL14 Model Summary:')
        print(f'  Input Size: 1x3x256x256')
        print('-' * 60)
        print(f'  FLOPs: {flops_g:.2f}G')
        print(f'  Parameters: {params_m:.2f}M')
        print(f'  Memory: {memory_mb:.2f}MB')
        print(f'  Speed: {fps:.2f}FPS')
        print('=' * 60)
        
        return model, None
    
    if(name == 'EfficientNet'): # EfficientNet model for fake detection
        from .EfficientNet.EfficientNet import EfficientNet as M
        
        print('==> Building EfficientNet model...')
        model = M(args, num_classes=2, pretrained=True).cuda()
        
        # Compute and print performance metrics
        print('==> Computing EfficientNet performance metrics...')
        dummy_input = torch.randn(1, 3, 256, 256).cuda()
        
        try:
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f}G')
            print(f'Parameters: {params_m:.2f}M')
        except Exception as e:
            print(f'Warning: Could not compute FLOPs/Params: {e}')
            flops_g = 0
            params_m = 0
        
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            memory_allocated = torch.cuda.max_memory_allocated()
            memory_mb = memory_allocated / (1024 ** 2)
            print(f'Memory: {memory_mb:.2f}MB')
        except Exception as e:
            print(f'Warning: Could not compute Memory: {e}')
            memory_mb = 0
        
        try:
            num_iterations = 100
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input)
            
            torch.cuda.synchronize(device='cuda')
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            fps = num_iterations / elapsed_time if elapsed_time > 0 else 0
            print(f'Speed: {fps:.2f}FPS')
        except Exception as e:
            print(f'Warning: Could not compute speed: {e}')
            fps = 0
        
        # Print summary
        print('=' * 60)
        print('EfficientNet Model Summary:')
        print(f'  Input Size: 1x3x256x256')
        print('-' * 60)
        print(f'  FLOPs: {flops_g:.2f}G')
        print(f'  Parameters: {params_m:.2f}M')
        print(f'  Memory: {memory_mb:.2f}MB')
        print(f'  Speed: {fps:.2f}FPS')
        print('=' * 60)
        
        return model, None
    
    # if name == 'crnet':
        # from .crnet_small.crnet import CDnetV1_MODEL as M
    # NPR-DeepfakeDetection
    if name == 'NPR-DeepfakeDetection':
        from .NPR_DeepfakeDetection.trainer import Trainer1 as M
        return M(args).cuda()
        # return Trainer()
    if name == 'GenD':
        # Use full GenD implementation with framework compatibility
        from .GenD.gend_full import GenD
        return GenD(args, num_classes=2, backbone='clip', pretrained=False).cuda()
    
    if name == 'ForensicsSAM':
        # ForensicsSAM: Forensic-focused Segment Anything Model
        from .ForensicsSAM import ForensicsSAM
        try:
            from segment_anything import sam_model_registry
        except ImportError:
            print("Warning: segment_anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            raise
        
        # 构建基础SAM模型
        model_type = getattr(args, 'sam_type', 'vit_b')  # 默认使用ViT-B
        checkpoint_path = getattr(args, 'sam_checkpoint', None)
        
        print(f'==> Building ForensicsSAM with {model_type}...')
        if checkpoint_path:
            sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        else:
            sam_model = sam_model_registry[model_type]()
        image_embedding_size = 64  # SAM ViT-B/L/H的标准输出大小
        
        # 包装为ForensicsSAM
        r = getattr(args, 'lora_r', 4)  # LoRA秩
        lora_layer = getattr(args, 'lora_layer', None)
        with_detector = getattr(args, 'with_detector', True)
        
        model = ForensicsSAM(
            sam_model=sam_model,
            r=r,
            lora_layer=lora_layer,
            with_detector=with_detector
        ).cuda()
        
        # ========== 计算模型的完整性能指标 ==========
        print('==> Computing model complexity and performance metrics...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        
        # 1. 计算FLOPs和Parameters
        try:
            flops, params = profile(model, inputs=(dummy_input, False), verbose=False)
            flops_g = flops / 1e9
            params_m = params / 1e6
            print(f'FLOPs: {flops_g:.2f}G')
            print(f'Parameters: {params_m:.2f}M')
        except Exception as e:
            print(f"Warning: Could not compute FLOPs and Parameters: {e}")
            flops_g, params_m = 0, 0
        
        # 2. 计算内存占用 (MB)
        try:
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                _ = model(dummy_input, activate_adv=False)
            
            memory_allocated = torch.cuda.max_memory_allocated(device=device)
            memory_mb = memory_allocated / (1024 ** 2)
            print(f'Memory: {memory_mb:.2f}MB')
        except Exception as e:
            print(f"Warning: Could not compute memory: {e}")
            memory_mb = 0
        
        # 3. 计算推理速度 (FPS)
        try:
            model.eval()
            torch.cuda.synchronize(device=device)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy_input, activate_adv=False)
            
            torch.cuda.synchronize(device=device)
            
            # 计时
            num_iterations = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(dummy_input, activate_adv=False)
            
            torch.cuda.synchronize(device=device)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            fps = num_iterations / elapsed_time
            print(f'Speed: {fps:.2f}FPS')
        except Exception as e:
            print(f"Warning: Could not compute speed: {e}")
            fps = 0
        
        # 4. 打印完整的模型信息摘要
        print('=' * 60)
        print('ForensicsSAM Model Summary:')
        print(f'  Model Type: {model_type}')
        print(f'  LoRA Rank: {r}')
        print(f'  With Detector: {with_detector}')
        print(f'  Input Size: 1x3x256x256')
        print('-' * 60)
        print(f'  FLOPs: {flops_g:.2f}G')
        print(f'  Parameters: {params_m:.2f}M')
        print(f'  Memory: {memory_mb:.2f}MB')
        print(f'  Speed: {fps:.2f}FPS')
        print('=' * 60)
        
        return model

    print('==> Building model..')
    # model = M(args, num_classes=2).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = M(args).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # flops, params = profile(model, inputs=(dummy_input,))
    # flops, params = profile(model, inputs=dummy_input)
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    return M(args), None




class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making '+args.model+'...')

        self.args = args
        self.scale = int(args.scale)
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        #choose model
        # self.model, config_vit = get_model(args.model,args).to(self.device)
        # self.model, config_vit = get_model(args.model,args) #2025.3.11
        # self.model = get_model(args.model,args).to(self.device) ##
        model_result = get_model(args.model,args)
        
        # Handle both tuple and single model returns
        if isinstance(model_result, tuple):
            self.model, _ = model_result  # Unpack tuple
        else:
            self.model = model_result  # Single model return
        # print(type(self.model))
        # print(1)
        # print(self.model)
        self.model = self.model.to(self.device) #2025.3.24
        # if (args.model == 'FLDCF_multiModal' or args.model == 'FLDCF_multiModal_TransUNet'):
        #     self.model.load_from(weights=np.load(config_vit.pretrained_path))
        # print('Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))

        # stat(self.model, (3, 224, 224), (1, 224, 224)) ###



        # # 查看模型在GPU上的内存占用情况
        # allocated_memory = torch.cuda.memory_allocated(device=self.device)
        # cached_memory = torch.cuda.memory_cached(device=self.device)
        # print(f"Allocated memory: {allocated_memory/1024/1024}")
        # print(f"Cached memory: {cached_memory/1024/1024}")




        if args.precision == 'half': self.model.half()
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)

    # def forward(self, x, y):
    def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            # return self.forward_chop(x, y)  #test的正常情况
            return self.forward_chop(x)  #test的正常情况
        else:
            # # if (self.args.model == "FLDCF_multiModal" or self.args.model == "FLDCF_multiModal_TransUNet"):
            if (self.args.model == "FLDCF_multiModal" ):
                # return self.model(x, y)  #train的正常情况
                return self.model(x)  #train的正常情况
            else: return self.model(x)  #train的正常情况

    def forward(self, x, y):
    # def forward(self, x):
        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x, y)  #test的正常情况
            # return self.forward_chop(x)  #test的正常情况
        else:
            # # if (self.args.model == "FLDCF_multiModal" or self.args.model == "FLDCF_multiModal_TransUNet"):
            if (self.args.model == "FLDCF_multiModal" ):
                return self.model(x, y)  #train的正常情况
                # return self.model(x)  #train的正常情况
            else: return self.model(x)  #train的正常情况

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        # Always use map_location to ensure model loads correctly regardless of device
        device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
        kwargs = {'map_location': device}

        if resume == -1:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                )
        else:
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )

    def forward_chop(self, x, shave=10, min_size=120000):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size +=4-h_size%4
        w_size +=8-w_size%8
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half  #这里默认输入输出不是一个格式，而是倍数的格式，不太好改，所以暂时关掉chop
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output

