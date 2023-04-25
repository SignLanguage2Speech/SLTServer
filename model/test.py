import subprocess
import torch, cv2, numpy as np, webm
import torchvision.transforms.functional as TF

# def test_write_streamed_video(ipt_tns, num_frames, h=240, w=320, c=3, fps=29.92, OUT_FILE_PATH='output.webm'):
def test_write_video_tensor_to_mp4(ipt_tns, w=640, h=480, fps=30, OUT_FILE_PATH='output.mp4'):
    dbg_str = f"Writing video Tensor to {OUT_FILE_PATH}"
    log_dbg(dbg_str,mode=1)
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
    
    del ipt_tns
    log_dbg(dbg_str,mode=2)

def test_write_single_frame(video_array, frame_idx=0):
    img_path = "test.png"
    dbg_str = f"Writing first frame from video array to {img_path}"
    log_dbg(dbg_str,mode=1)
    single_frame = video_array[frame_idx]
    cv2.imwrite(img_path, cv2.cvtColor(single_frame,cv2.COLOR_RGB2BGR))
    log_dbg(dbg_str,mode=2)


def test_convert_webm_bytes_to_tensor(webm_bytes):
    # Use ffmpeg to convert WEBM video into rawvideo format (one pixel, one byte (per color channel))
    # Then load this into a torch tensor
    dbg_str = "Loading WEBM byte stream into torch Tensor"
    log_dbg(dbg_str,mode=1)
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

        log_dbg("Length of buffer: ", len(stdout)%(10**6), "·10^6")
        log_dbg("video_tensor shape (bytes->uint8): ", video_tensor.shape[0]%(10**6), "·10^6")
        

        # Reshape the video array to the correct shape
        height = 480
        width = 640
        channels = 3
        num_frames = video_tensor.shape[0] // (height * width * channels)
        video_tensor = video_tensor.reshape((num_frames, height, width, channels))

        log_dbg("Final video_tensor shape (uint8 frame array):",video_tensor.shape)
        log_dbg(dbg_str,mode=2)
        test_write_streamed_video(video_tensor)

_mode2flag = {
    0: "INFO",
    1: "STARTING",
    2: "FINISHED",
    -1:"ERROR",
}
def log_dbg(*args, mode=0):
    print(f"DEBUG @ {_mode2flag[mode]} ::", *args)

def test_load_webm_from_bytes(webm_bytes):
    # Use ffmpeg to decode the video to numpy array
    process = subprocess.Popen(['ffmpeg', 
                                '-f', 'webm',          # webm because this is what Flutter Web uses
                                '-i', 'pipe:0', 
                                '-pix_fmt', 'yuv420p', # webm uses this pixel format
                                '-f', 'rawvideo', 
                                '-pix_fmt', 'rgb24',   # bgr for testing with OpenCV, otherwise rgb
                                '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate(input=webm_bytes)
    video_array = np.frombuffer(stdout, dtype=np.uint8)

    # Reshape the video array to the correct shape
    height = 480
    width = 640
    channels = 3
    num_frames = video_array.size // (height * width * channels)
    video_array = video_array.reshape((num_frames, height, width, channels))
    
    frames = []
    for frame_array in video_array:
        frame = torch.from_numpy(frame_array).detach().clone() # video_array is non-writeable
        frames.append(frame)

    video_tensor = torch.stack(frames)
    del frames

    print(video_tensor.shape)
    test_write_streamed_video(video_tensor)