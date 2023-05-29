import torch
from model.model import ModelWrapper
# from model import ModelWrapper
from model.Sign2Text.utils.load_model_from_checkpoint import load_model_from_checkpoint
import cv2, numpy as np
from model.utils import Logger as LOG


class SignToText(ModelWrapper):
    def __init__(self, device='cpu'):
        self.model = load_model_from_checkpoint('model/Sign2Text_test.pt', train=False, device=device)
        self.langauges = {
            "US": "en",
            "UK": "en",
            "DE": "de",
            "DK": "da"}
    def inference(self, rawvideo_input, input_length):
        rawvideo_input = rawvideo_input.unsqueeze(0) 
        dbg_str = "SLT Inference"
        start_time = LOG.log_dbg(dbg_str, mode=1)
        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(
        #         wait=2,
        #         warmup=2,
        #         active=6,
        #         repeat=1),
        #     on_trace_ready=tensorboard_trace_handler,
        #     with_stack=True
        # ) as profiler:
        output = self.model.predict(rawvideo_input,input_length)
        LOG.log_dbg(dbg_str, mode=2, prev_start_time=start_time)
        return output


# def main():
#     import torch
#     from torch import from_numpy as np2tensor, uint8, Tensor
#     from cv2 import imwrite, cvtColor, COLOR_RGB2BGR
#     import torchvision.transforms.functional as TF, torch.nn as nn, torch.nn.functional as F
#     class VideoPipeline:
#         def __init__(self, W_in = 640, H_in = 480, WH_out = 224):
#             self.W_in = W_in
#             self.H_in = H_in
#             self.W_out = self.H_out = WH_out
#             self.spatial_upsample = nn.Upsample(size=(self.W_in, self.W_in), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)

#         def __call__(self, rawvideo, to_file=False):
#             rawvideo = rawvideo.float()
#             rawvideo = self.reshape(rawvideo,[0,3,1,2]) # rearrange order of dimensions to match torch's transformations API
#             rawvideo = self.crop(rawvideo)
#             rawvideo = self.downsample(rawvideo)
#             if to_file: # if we are writing the output to an mp4 file via cv2
#                 rawvideo = rawvideo.type(uint8)
#                 dim_order = [0,2,3,1]
#             else: # if we are doing SLT inference
#                 rawvideo = self.normalize(rawvideo) 
#                 dim_order = [1,0,2,3] # ! ensure order of dimensions is correct
#             rawvideo = self.reshape(rawvideo,dim_order) # rearrange order of dimensions back to match our model
#             return rawvideo
        
#         """ Rearrange order of dimensions of video tensor """
#         def reshape(self, rawvideo, order):
#             assert len(order) == 4
#             return rawvideo.permute(*order).contiguous()
        
#         """ Bilinear downsampling (interpolation) for (down-) resizing frames """
#         def downsample(self, rawvideo):
#             return F.interpolate(rawvideo, size=(self.H_out, self.W_out), scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)

#         """ Center crop to minimal dimension of height and width """
#         def crop(self, rawvideo):
#             min_spatial_dim = min(self.H_in, self.W_in)
#             return TF.center_crop(rawvideo, (min_spatial_dim, min_spatial_dim))

#         """ Bilinear upsampling (interpolation) for (up-) resizing frames """
#         def upsample(self, rawvideo):
#             return self.spatial_upsample(rawvideo)

#         """ Max normalization (uint8) to scalar in [0;1] """
#         def normalize(self, rawvideo):
#             return rawvideo.div(255)

#     def load_mp4video_from_file(FILE_PATH='output.mp4'):
#         cap = cv2.VideoCapture(FILE_PATH)
#         frames = []
#         while cap.isOpened():
#             ret,frame = cap.read()
#             if not ret:
#                 break
#             frame = cvtColor(frame,COLOR_RGB2BGR)
#             frames.append(frame)
#         cap.release()
#         return np2tensor(np.array(frames))
#     class LOG:
#         @staticmethod
#         def log_dbg(*input, mode=None):
#             pass
#     s2t = SignToText()
#     pipe = VideoPipeline(W_in=224, H_in=224)
#     processed_vid = load_mp4video_from_file(FILE_PATH='processed_output.mp4')
#     num_frames = torch.Tensor(processed_vid.shape[0])
#     processed_vid = pipe(processed_vid)
#     y = s2t(processed_vid, num_frames)
#     print(y)
#     pass