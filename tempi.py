from mmcv_custom.checkpoint import _load_checkpoint
import torch
x = _load_checkpoint('cascade_mask_rcnn_swin_base_patch4_window7.pth')
state_dict_backbone = {}
state_dict_neck = {}
state_dict_rpn = {}
state_dict_roihead = {}
for k,v in x['state_dict'].items():
    if k.startswith('backbone'):
        state_dict_backbone[k.replace('backbone.','')] = v
    elif k.startswith('neck'):
        state_dict_neck[k.replace('neck.','')] = v
    elif k.startswith('rpn_head'):
        state_dict_rpn[k.replace('rpn_head.','')] = v
    elif k.startswith('roi_head'):
        state_dict_roihead[k.replace('roi_head.','')] = v
    else:
        assert False
x['state_dict'] = state_dict_backbone
torch.save(x,"cascade_mask_rcnn_swin_base_patch4_window7_backbone.pth")
x['state_dict'] = state_dict_neck
torch.save(x,"cascade_mask_rcnn_swin_base_patch4_window7_neck.pth")
x['state_dict'] = state_dict_rpn
torch.save(x,"cascade_mask_rcnn_swin_base_patch4_window7_rpn_head.pth")
x['state_dict'] = state_dict_roihead
torch.save(x,"cascade_mask_rcnn_swin_base_patch4_window7_roi_head.pth")
y=4