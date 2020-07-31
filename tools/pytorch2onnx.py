import argparse
import io

import mmcv
import onnx
import torch
import numpy as np
# from mmcv.ops import RoIAlign, RoIPool
from mmcv.runner import load_checkpoint
from onnx import optimizer
import onnxruntime
from torch.onnx import OperatorExportTypes

from mmdet3d.models import build_detector
from tools.remove_initializer import remove_initializer_from_input

from mmdet3d.ops.voxel.voxelize import voxelization




def export_onnx_model(model, inputs, passes):
    """Trace and export a model to onnx format. Modified from
    https://github.com/facebookresearch/detectron2/

    Args:
        model (nn.Module):
        inputs (tuple[args]): the model will be called by `model(*inputs)`
        passes (None or list[str]): the optimization passed for ONNX model

    Returns:
        an onnx model
    """
    assert isinstance(model, torch.nn.Module)

    # make sure all modules are in eval mode, onnx may change the training
    # state of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)


    # Export the model to ONNX
    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                input_names=['input'],
                output_names=['output'],
                operator_export_type=OperatorExportTypes.ONNX,
                # keep_initializers_as_inputs=True,
                # verbose=True,  # NOTE: uncomment this for debugging
                # export_params=True,
                dynamic_axes = {'input': {0 : 'point_num'}},
                do_constant_folding=True,
                opset_version=12,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    # Apply ONNX's Optimization
    if passes is not None:
        all_passes = optimizer.get_available_passes()
        assert all(p in all_passes for p in passes), \
            f'Only {all_passes} are supported'
    # onnx_model = optimizer.optimize(onnx_model, passes)
    return onnx_model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def check_onnx_model(model_file, dummy, torch_result):


    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(model_file)

    # compute ONNX Runtime output prediction
    ort_inputs = {inputs.name: to_numpy(dummy[i]) for i, inputs in enumerate(ort_session.get_inputs())}

    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_result[0]), ort_outs[0], rtol=1e-03, atol=1e-07)

    # with open("test_result", "wb") as f:
    #     (torch_result[0]).detach().numpy().tofile(f)
    # f.close()

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")



def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet pytorch model conversion to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out', type=str, required=True, help='output ONNX filename')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[18279,4],
        help='input image size')
    parser.add_argument(
        '--passes', type=str, nargs='+', help='ONNX optimization passes')
    args = parser.parse_args()
    return args


def register_custom_op():

    from torch.onnx.symbolic_helper import parse_args
    @parse_args('v', 'v', 'v', 'v','v', 'v', 'v', 'v', 'v')
    def regis_hard_voxelize(g,
                      points,
                      voxels,
                      coors,
                      num_points_per_voxel,
                      voxel_size,
                      coors_range,
                      max_points,
                      max_voxels,
                      NDim):
        return g.op("my_onnx::hard_voxelize", points,
                    voxels,
                    coors,
                    num_points_per_voxel,
                    voxel_size,
                    coors_range,
                    max_points,
                    max_voxels,
                    NDim)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic("voxelization::hard_voxelize", regis_hard_voxelize, 12)

def main():
    args = parse_args()

    if not args.out.endswith('.onnx'):
        raise ValueError('The output file must be a onnx file.')


    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    register_custom_op()

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # Only support CPU mode for now
    model.cpu().eval()

    # TODO: a better way to override forward function
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'ONNX conversion is currently not currently supported with '
            f'{model.__class__.__name__}')

    # input_data = torch.zeros(args.shape,dtype=next(model.parameters()).dtype,
    #                      device=next(model.parameters()).device)
    # input_data.fill_(0.5)

    points = np.fromfile("/home/liyuqi/Downloads/save_result/points_2", dtype=np.float32)
    points=points.reshape([-1,4])
    points=torch.from_numpy(points)

    onnx_model = export_onnx_model(model, (points,), args.passes)
    onnx_model = remove_initializer_from_input(onnx_model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    print(f'saving model in {args.out}')
    onnx.save(onnx_model, args.out)

    torch_result = model(points)

    check_onnx_model(args.out, [points],torch_result)



if __name__ == '__main__':

    main()