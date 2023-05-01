import subprocess
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR

if __name__ != '__main__':
    from model.utils import Logger as LOG

def write_video_tensor_to_mp4(ipt_tns, w=640, h=480, fps=30, OUT_FILE_PATH='output.mp4'):
    dbg_str = f"Writing video Tensor to {OUT_FILE_PATH}"
    LOG.log_dbg(dbg_str,mode=1)
    codec = 'libx264'
    command = ['ffmpeg',
            '-y', # overwrite output file if it exists
            '-f', 'rawvideo',
            '-s', f"{w}x{h}",
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-vcodec', codec,
            OUT_FILE_PATH]
    
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for frame in ipt_tns.detach().numpy():
            proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        if proc.stderr is not None:
            proc.stderr.close()
        proc.wait()
    
    LOG.log_dbg(dbg_str,mode=2)

def write_single_frame_to_png(video_array, frame_idx=0):
    img_path = "test.png"
    dbg_str = f"Writing first frame from video array to {img_path}"
    LOG.log_dbg(dbg_str,mode=1)
    single_frame = video_array[frame_idx]
    imwrite(img_path, cvtColor(single_frame,COLOR_RGB2BGR))
    LOG.log_dbg(dbg_str,mode=2)


def webm_bytes_to_tensor(webm_bytes):
    # Use ffmpeg to convert WEBM video into rawvideo format (one pixel, one byte (per color channel))
    # Then load this into a torch tensor
    dbg_str = "Loading WEBM byte stream into torch Tensor"
    LOG.log_dbg(dbg_str, mode=1)
    command = ['ffmpeg', 
        '-f', 'webm',          # ? webm because this is what Flutter Web uses
        # '-pix_fmt', 'yuv420p', # ? webm uses this pixel format
        '-i', 'pipe:0', 
        '-f', 'rawvideo', 
        '-pix_fmt', 'rgb24',
        '-']
    with subprocess.Popen(command, 
                          stdin=subprocess.PIPE, 
                          stdout=subprocess.PIPE) as process:
        stdout, _ = process.communicate(input=webm_bytes)
        process.stdin.close()
        process.stdout.close()
        process.wait()
        video_tensor = torch.frombuffer(stdout, dtype=torch.uint8)

        LOG.log_dbg("Length of buffer: ", len(stdout)%(10**6), "·10^6")
        LOG.log_dbg("video_tensor shape (bytes->uint8): ", video_tensor.shape[0]%(10**6), "·10^6")
        
        # Reshape the video array to the correct shape
        height = 480
        width = 640
        channels = 3
        num_frames = video_tensor.shape[0] // (height * width * channels)
        video_tensor = video_tensor.reshape((num_frames, height, width, channels))

        LOG.log_dbg("Final video_tensor shape (uint8 frame array):",video_tensor.shape)
        LOG.log_dbg(dbg_str,mode=2)

        return video_tensor

class VideoPipeline:
    def __init__(self):
        self.W_in = 640
        self.H_in = 480
        self.W_out = self.H_out = 224
        self.spatial_upsample = nn.Upsample(size=(self.W_in, self.W_in), scale_factor=None, mode='bilinear', align_corners=None, recompute_scale_factor=None)

    def __call__(self, rawvideo, to_file=False):
        rawvideo = rawvideo.double()
        rawvideo = self.reshape(rawvideo,[0,3,1,2]) # rearrange order of dimensions to match torch's transformations API
        rawvideo = self.crop(rawvideo)
        rawvideo = self.downsample(rawvideo)
        if to_file: # if we are writing the output to an mp4 file via cv2
            rawvideo = rawvideo.type(torch.uint8)
            dim_order = [0,2,3,1]
        else: # if we are doing SLT inference
            rawvideo = self.normalize(rawvideo) 
            dim_order = [1,0,2,3] # ! ensure order of dimensions is correct
        rawvideo = self.reshape(rawvideo,dim_order) # rearrange order of dimensions back to match our model
        return rawvideo
    
    """ Rearrange order of dimensions of video tensor """
    def reshape(self, rawvideo, order):
        assert len(order) == 4
        return rawvideo.permute(*order).contiguous()
    
    """ Bilinear downsampling (interpolation) for (down-) resizing frames """
    def downsample(self, rawvideo):
        return F.interpolate(rawvideo, size=(self.H_out, self.W_out), scale_factor=None, mode='bilinear', align_corners=False, recompute_scale_factor=None)

    """ Center crop to minimal dimension of height and width """
    def crop(self, rawvideo):
        min_spatial_dim = min(self.H_in, self.W_in)
        return TF.center_crop(rawvideo, (min_spatial_dim, min_spatial_dim))

    """ Bilinear upsampling (interpolation) for (up-) resizing frames """
    def upsample(self, rawvideo):
        return self.spatial_upsample(rawvideo)

    """ Max normalization (uint8) to scalar in [0;1] """
    def normalize(self, rawvideo):
        return rawvideo.div(255)


def load_mp4video_from_file(FILE_PATH='output.mp4'):
    cap = cv2.VideoCapture(FILE_PATH)
    frames = []
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        frame = cvtColor(frame,COLOR_RGB2BGR)
        frames.append(frame)
    cap.release()
    return torch.from_numpy(np.array(frames))

if __name__ == '__main__':
    class LOG:
        @staticmethod
        def log_dbg(*input, mode=None):
            pass
    import cv2, numpy as np
    pipe = VideoPipeline()
    vid = load_mp4video_from_file()
    processed_vid = pipe(vid, to_file=True)
    write_video_tensor_to_mp4(processed_vid, w=224, h=224, fps=30, OUT_FILE_PATH='processed_output.mp4')