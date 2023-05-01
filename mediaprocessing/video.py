import subprocess
import torch, numpy as np
from cv2 import imwrite, cvtColor, COLOR_RGB2BGR

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