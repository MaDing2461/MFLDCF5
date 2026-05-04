# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #在卡2调试
import time
import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from thop import profile, clever_format

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args) 


# from torchstat import stat
# import torchvision.models as models
# model = models.resnet152()
# stat(model, (3, 224, 224))


def get_model_memory(model1):
    # 获取模型中参数的总大小（字节数）
    total_params = sum(p.numel() for p in model1.parameters())
    # 每个参数默认占用4字节（float32），故计算内存占用（MB）
    return total_params * 4 / (1024 ** 2)

def get_activation_memory(model_obj, input_size, device=None):
    if device is None:
        device = next(model_obj.parameters()).device

    # 构造输入数据
    input_tensor = torch.randn(input_size).to(device)
    # 记录中间层激活的内存
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.nelement() * output.element_size())

    hooks = []
    for layer in model_obj.children():
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # 运行一次前向传播
    model_obj(input_tensor)

    # 解除挂接
    for hook in hooks:
        hook.remove()
    
    return sum(activations) / (1024 ** 2)  # 返回结果（MB）


def _format_metric(value, unit, precision=3):
    if value is None:
        return 'N/A'
    return f'{value:.{precision}f}{unit}'


def _count_parameter_memory(model_obj):
    param_memory = sum(p.numel() * p.element_size() for p in model_obj.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model_obj.buffers())
    return (param_memory + buffer_memory) / (1024 ** 2)


def _normalize_model_name(model_name):
    model_name = str(model_name or '').lower()
    if model_name in ('fldcf', 'my2'):
        return 'FLDCF'
    if model_name in ('mfldcf', 'm-fldcf', 'fldcf_multimodal', 'fldcf_multi_modal', 'FLDCF_multiModal'):
        return 'MFLDCF'
    return model_name


def _build_complexity_inputs(model_name, batch_size, image_size, device, dtype):
    if isinstance(image_size, int):
        height, width = image_size, image_size
    else:
        height, width = image_size

    rgb = torch.randn(batch_size, 3, height, width, device=device, dtype=dtype)
    normalized_name = _normalize_model_name(model_name)

    if normalized_name == 'MFLDCF':
        dsm = torch.randn(batch_size, 1, height, width, device=device, dtype=dtype)
        return (rgb, dsm), f'{batch_size}x3x{height}x{width} + {batch_size}x1x{height}x{width}'

    return (rgb,), f'{batch_size}x3x{height}x{width}'


def _snapshot_forward_hooks(model_obj):
    return {module: module._forward_hooks.copy() for module in model_obj.modules()}


def _restore_forward_hooks_and_clear_thop(model_obj, hook_snapshots):
    for module in model_obj.modules():
        if module in hook_snapshots:
            module._forward_hooks.clear()
            module._forward_hooks.update(hook_snapshots[module])

        for attr_name in ('total_ops', 'total_params'):
            if hasattr(module, attr_name):
                delattr(module, attr_name)


def _profile_once(model_obj, inputs):
    hook_snapshots = _snapshot_forward_hooks(model_obj)
    try:
        return profile(model_obj, inputs=inputs, verbose=False)
    except TypeError:
        # Some wrappers require a second argument even when the wrapped FLDCF
        # model only uses RGB input.
        if len(inputs) == 1:
            return profile(model_obj, inputs=(inputs[0], None), verbose=False)
        raise
    finally:
        _restore_forward_hooks_and_clear_thop(model_obj, hook_snapshots)


def evaluate_model_complexity(
        model_obj,
        model_name=None,
        batch_size=1,
        image_size=256,
        warmup=10,
        repeat=100,
        device=None):
    """
    统计 FLDCF / MFLDCF 的 FLOPs、显存、参数量和 FPS。

    Args:
        model_obj: 可以传 model.Model(args, checkpoint)，也可以传实际的 nn.Module。
        model_name: 'my2'/'FLDCF' 或 'FLDCF_multiModal'/'MFLDCF'。
        batch_size: 测 FPS 时的 batch size，默认 1。
        image_size: 输入图像大小，默认 256；也可以传 (height, width)。
        warmup: FPS 预热次数。
        repeat: FPS 正式计时次数。
        device: 默认自动使用模型所在设备；没有参数时使用 cuda/cpu。

    Returns:
        dict: 包含 flops, params, memory_mb, fps 等原始数值。
    """
    if model_name is None and hasattr(model_obj, 'args'):
        model_name = getattr(model_obj.args, 'model', None)

    target_model = model_obj.get_model() if hasattr(model_obj, 'get_model') else model_obj
    target_model.eval()

    if device is None:
        try:
            device = next(target_model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    target_model = target_model.to(device)
    dtype = next(target_model.parameters()).dtype
    inputs, input_desc = _build_complexity_inputs(model_name, batch_size, image_size, device, dtype)

    with torch.no_grad():
        flops, params_from_thop = _profile_once(target_model, inputs)

    params = sum(p.numel() for p in target_model.parameters())
    param_memory_mb = _count_parameter_memory(target_model)

    peak_memory_mb = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = target_model(*inputs)
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    with torch.no_grad():
        for _ in range(warmup):
            _ = target_model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        time_start = time.time()
        for _ in range(repeat):
            _ = target_model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        time_end = time.time()

    elapsed = time_end - time_start
    fps = repeat * batch_size / elapsed if elapsed > 0 else 0

    flops_fmt, params_fmt = clever_format([flops, params], '%.3f')
    normalized_name = _normalize_model_name(model_name)

    print('=' * 60)
    print(f'{normalized_name} Complexity Summary')
    print(f'Input: {input_desc}')
    print('-' * 60)
    print(f'FLOPs: {flops_fmt} ({flops / 1e9:.3f} GFLOPs)')
    print(f'Parameters: {params_fmt} ({params / 1e6:.3f} M)')
    print(f'Parameter Memory: {_format_metric(param_memory_mb, " MB")}')
    print(f'Peak Inference Memory: {_format_metric(peak_memory_mb, " MB")}')
    print(f'FPS: {fps:.3f}')
    print('=' * 60)

    return {
        'model_name': normalized_name,
        'input': input_desc,
        'flops': flops,
        'flops_g': flops / 1e9,
        'params': params,
        'params_m': params / 1e6,
        'params_from_thop': params_from_thop,
        'parameter_memory_mb': param_memory_mb,
        'peak_inference_memory_mb': peak_memory_mb,
        'fps': fps,
    }


def main():
    if checkpoint.ok:                  
        loader = data.Data(args)
        _model = model.Model(args, checkpoint) ###

        # 查看模型在GPU上的内存占用情况
        # allocated_memory = torch.cuda.memory_allocated(device=device)
        # allocated_memory = torch.cuda.max_memory_allocated(device=device)
        # torch.cuda.max_memory_allocated()
        # cached_memory = torch.cuda.memory_cached(device=device)
        # print(f"Allocated memory: {allocated_memory/1024/1024}")
        # print(f"Cached memory: {cached_memory/1024/1024}")

        # 只想检测 FLOPs / Memory / Parameters / FPS 时，取消下面两行注释。
        evaluate_model_complexity(_model, model_name=args.model, batch_size=1, image_size=256)
        # return





        # # input = torch.randn(1, 3, 224, 224)
        # # input1 = torch.randn(1, 1, 224, 224)
        # input = torch.randn(1, 3, 256, 256)
        # input1 = torch.randn(1, 1, 256, 256)
        # flops, params = profile(_model, inputs=(input, input1))
        # print(flops, params) # 1819066368.0 11689512.0
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params) # 1.819G 11.690M

        # model_memory = get_model_memory(model)
        # print(f"Model Parameters Memory Usage: {model_memory:.2f} MB")
        # activation_memory = get_activation_memory(model, (1, 784))
        # print(f"Activation Memory Usage: {activation_memory:.2f} MB")
        # total_memory = model_memory + activation_memory
        # print(f"Total Memory Usage: {total_memory:.2f} MB")


        # model1 = models.resnet152()
        # stat(model1, (3, 224, 224))
        # stat(_model, (3, 224, 224))

        # for name, parms in _model.named_parameters():
        #     if 'layers.0.residual_group.blocks.0.attn.relative_position_bias_table' in name:
        #         print(parms)
        #     print('%-50s' % name, '%-30s' % str(parms.shape), '%-10s' % str(parms.nelement()))
        # print('Total params: %.2fM' % (sum(p.numel() for p in _model.parameters())/1000000.0))
        # input = torch.randn(16, 3, 32, 32).cuda()
        # flops, params = profile(_model, inputs=(input, ))
        # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # predict = _model(input)  #影响speed的值
        # torch.cuda.synchronize()
        # time_start = time.time()
        # predict = _model(input)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # print('Speed: %.3f FPS\n' % (1/(time_end-time_start)))
        # flops, params = profile(_model, inputs=(input, ))
        # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
        # optimizer = optim.SGD(_model.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
        # for _ in range(1000):
        #     optimizer.zero_grad()
        #     _model(input)

        _loss = loss.Loss_fake() if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.testtrain(False)
        checkpoint.done()

if __name__ == '__main__':
    main()

# cd /media/lscsc/nas/mading/fakedetect
# python src/test.py --model your_model_name --save my10
# python src/test.py --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save my10
# python src/test.py --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/my12/model/model_latest.pt' --save my13
# python src/test.py --model crnet --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/crnetVDSM1/model/model_latest.pt' --save crnetVDSM_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_DSM.pt' --save my2VDSM_test
# python src/test.py --model FLDCF2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF2_V_DSM.pt' --save FLDCF2VDSM_test
# python src/test.py --model FLDCF2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF2_V_DSM_2.pt' --save FLDCF2VDSM_test
# python src/test.py --model CDnetV2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_CDnetV2_V_DSM.pt' --save CDnetV2VDSM_test
# python src/test.py --model CDnetV2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_CDnetV2_V_DSM_contentBasedPrior.pt' --save CDnetV2VDSM_test

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save my2VRGB_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_forgedDSM.pt' --save FLDCF_multiModal_V_NIRRG_forgedDSM_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM2.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM2_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_LaMa.pt' --save FLDCF_LaMa_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_Repaint.pt' --save FLDCF_Repaint_test
# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_V_All.pt' --save FLDCF_All_test

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_Repaint1.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_Repaint_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_All.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_All_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM3_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM3_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM4_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM4_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_LaMa_test

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_LaMa_test2

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder_NIRRG_withGTDSM_withDeep_withShallow_all/model/model_25.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_All_test

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_All.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_All_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM5_Repaint.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM5_Repaint_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM6_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM6_LaMa_test
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa_test

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_V_CycleGAN.pt' --save FaceForensics_V_CycleGAN_test


# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_all
# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_repaint
# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_scuV.pt' --save SCSE-Unet_V_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_mvssV.pt' --save MVSSNet_V_repaint

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_moveV.pt' --save SE-Network_V_repaint

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_repaint

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa.pt' --save FLDCF2_V_repaint

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_2_LaMa.pt' --save FLDCF2_V_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_mvssV.pt' --save MVSSNet_V_lama

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_moveV.pt' --save SE-Network_V_lama

# python src/test.py --model capsule --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_Capsule_Vaihingen_all.pt' --save Capsule_Vaihingen__repaint

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_Vaihingen_all.pt' --save FaceForensics_Vaihingen__repaint

# python src/test.py --model deepfake --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_DeepFake-Detection_Vaihingen_all.pt' --save DeepFake-Detection_Vaihingen__all

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_all

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt' --save FLDCF2_V_all

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutPyramidFeatureExtractionModule.pt' --save FLDCF2_V_repaint_withoutPyramidFeatureExtractionModule

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutDSM.pt' --save FLDCF2_V_all_withoutDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutGlobalFeatureBlock.pt' --save FLDCF2_V_repaint_withoutGlobalFeatureBlock

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All_withoutLocalFeatureBlock.pt' --save FLDCF2_V_repaint_withoutLocalFeatureBlock

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_V_NIRRG_withoutGTDSM7_All.pt' --save FLDCF2_TransUNet_V_lama_withoutDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder7_NIRRG_withFakeDSM_withDeep_withShallow_all/model/model_34.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All2.pt' --save FLDCF2_V_lama_fakeDSM

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All.pt' --save FLDCF2_V_lama

# python src/test.py --model my2 --data_train 'Vaihingen' --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_lama

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_multiModalV_modifiedEncoder7_NIRRG_withGTDSM_withDeep_withShallow_all/model/model_32.pt' --save FLDCF2_V_lama

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF2_V_lama

# MFFLDCF == FLDCF2
# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_Potsdam_NIRRG_GTDSM7_All.pt' --save MFFLDCF_Potsdam_all

# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_SCSEUnet_Potsdam.pt' --save SCSE-Unet_Potsdam_lama

# python src/test.py --model mvss --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_MVSSNet_Potsdam.pt' --save MVSSNet_Potsdam_lama

# python src/test.py --model movenet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_SE-Network_Potsdam1.pt' --save SE-Network_Potsdam_lama

# python src/test.py --model scunet --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/SCSE-Unet_Potsdam_all/model/model_20.pt' --save SCSE-Unet_Potsdam_lama

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_Potsdam_all.pt' --save FLDCF_Potsdam_lama

# python src/test.py --model my2 --pre_train '/media/lscsc/nas/mading/fakedetect/experiment/FLDCF_Potsdam_all/model/model_20.pt' --save FLDCF_Potsdam_lama

# python src/test.py --model capsule --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_Capsule_Potsdam_all.pt' --save Capsule_Potsdam__lama

# python src/test.py --model face --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FaceForensics_Potsdam_all1.pt' --save FaceForensics_Potsdam__lama

# python src/test.py --model deepfake --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_DeepFake-Detection_Potsdam_all.pt' --save DeepFake-Detection_Potsdam__lama

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_2

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_3

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_4

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_5

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_6

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow_7


# python src/test.py --model my2 --data_train_dir 'fakeV' --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_myV.pt' --save FLDCF_V_gt

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF2_V_gt

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_V_NIRRG_FakeDSM7_All2.pt' --save FLDCF2_V_gt_fakeDSM


# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_modifiedEncoder7_NIRRG_withoutGTDSM_withDeep_withShallow

# python src/test.py --model FLDCF_multiModal_TransUNet --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_TransUNet_WHUGCD_RGB_withoutGTDSM7_All6types.pt' --save FLDCF_multiModal_TransUNet_WHU-GCD_withoutGTDSM_x_3

# python src/test.py --model FLDCF_multiModal --pre_train '/media/lscsc/nas/mading/fakedetect/model/model_FLDCF_multiModal_Potsdam_NIRRG_GTDSM7_All.pt' --save MFLDCF_train_Potsdam_test_Vaihingen_gtDSM




# CUDA_VISIBLE_DEVICES=0 nohup python -u src/main.py --optimizer ADAM --dsm_option False --data_train Vaihingen --model NPR-DeepfakeDetection --save NPR-DeepfakeDetection__Fake_Vaihingen__All3Methods__NIRRG_withoutGTDSM > NPR-DeepfakeDetection__Fake_Vaihingen__All3Methods__NIRRG_withoutGTDSM 2>&1 &
# python src/test.py --optimizer ADAM --data_train_dir fakeV --dsm_option False --data_train Vaihingen --model NPR-DeepfakeDetection --pre_train '/media/lscsc/nas2/mading/fakedetect/model/NPR-DeepfakeDetection_Vaihingen_Repaint_LaMa.pt' --save NPR-DeepfakeDetection__Fake_Vaihingen__All2Methods__Test
# /media/lscsc/nas2/mading/fakedetect/model/NPR-DeepfakeDetection_Vaihingen_Repaint_LaMa.pt

# python src/test.py --optimizer ADAM --data_train_dir fake_Potsdam --dsm_option False --data_train Vaihingen --model NPR-DeepfakeDetection --pre_train '/media/lscsc/nas2/mading/fakedetect/model/NPR-DeepfakeDetection_Potsdam_Repaint_LaMa.pt' --save NPR-DeepfakeDetection__Fake_Potsdam__All2Methods__Test
# /media/lscsc/nas2/mading/fakedetect/model/NPR-DeepfakeDetection_Potsdam_Repaint_LaMa.pt

# python src/test.py --data_train_dir fakeV --dsm_option False --data_train Vaihingen --model GenD --pre_train '/media/lscsc/nas2/mading/fakedetect/model/GenD_Vaihingen_Repaint_LaMa.pt' --save GenD__Fake_Vaihingen__All2Methods__Test
# /media/lscsc/nas2/mading/fakedetect/model/GenD_Vaihingen_Repaint_LaMa.pt

# python src/test.py --data_train_dir fake_Potsdam --dsm_option False --data_train Vaihingen --model GenD --pre_train '/media/lscsc/nas2/mading/fakedetect/model/GenD_Potsdam_Repaint_LaMa.pt' --save GenD__Fake_Potsdam__All2Methods__Test
# python src/test.py --model segformer --data_train_dir 'fakeV' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/Segformerb3__Fake_Vaihingen__All2Methods__Test/model/model_0.pt' --save segformer_V_all
# python src/test.py --model ClipViTL14 --data_train_dir 'fakeV' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/ClipViTL14__Fake_Vaihingen__All2Methods__Test/model/model_0.pt' --save ClipViTL14_Vaihingen_all_test
# python src/test.py --model EfficientNet --data_train_dir 'fakeV' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/EfficientNet__Fake_Vaihingen__All2Methods__Test/model/model_0.pt' --save EfficientNet_Vaihingen_all_test
# python src/test.py --model segformer --data_train_dir 'fake_Potsdam' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/Segformerb3__Fake_Potsdam__All2Methods__Test/model/model_0.pt' --save segformer_Potsdam_all_test
# python src/test.py --model ClipViTL14 --data_train_dir 'fake_Potsdam' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/ClipViTL14__Fake_Potsdam__All2Methods__Test/model/model_0.pt' --save ClipViTL14_Potsdam_all_test
# python src/test.py --model EfficientNet --data_train_dir 'fake_Potsdam' --data_train 'Vaihingen' --pre_train '/media/lscsc/nas2/mading/fakedetect/experiment/EfficientNet__Fake_Potsdam__All2Methods__Test/model/model_0.pt' --save EfficientNet_Potsdam_all_test







# cd /media/lscsc/nas2/mading/fakedetect







# python src/test.py --dsm_option True --data_train VaihingenWithDSM --data_train_dir fakeV --model FLDCF_multiModal --pre_train '/media/lscsc/nas2/mading/MFLDCF2/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa_test_2026422_1519





# cd /media/lscsc/nas2/mading/MFLDCF3
# python src/test.py --dsm_option True --data_train VaihingenWithDSM --data_train_dir fakeV --model FLDCF_multiModal --pre_train '/media/lscsc/nas2/mading/MFLDCF2/model/model_FLDCF_multiModal_V_NIRRG_GTDSM7_All2.pt' --save FLDCF_multiModal_V_NIRRG_GTDSM7_LaMa_test_2026422_1519

# python src/test.py --model FLDCF_multiModal --dsm_option True --data_train_dir 'fakeV' --data_train 'VaihingenWithDSM' --pre_train '/media/lscsc/nas2/mading/MFLDCF2/model/20260422_model_MFLDCF_Vaihingen_GTDSM_Pretrained_All2Methods.pt' --save MFLDCF_Vaihingen_GTDSM_Pretrained_All2Methods_test_20260422_1559

# python src/test.py --model FLDCF_multiModal --dsm_option True --data_train_dir 'fakeV' --data_train 'VaihingenWithDSM' --pre_train '/media/lscsc/nas2/mading/MFLDCF2/model/model_MFLDCF__FakeVaihingenMM_NIRRG_GTDSM_20260422_2015_pretrain.pt' --save MFLDCF_Vaihingen_GTDSM_Pretrained_All2Methods_test_20260427_1231
