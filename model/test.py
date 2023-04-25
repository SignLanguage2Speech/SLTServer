import subprocess
import torch, cv2, numpy as np, webm
import torchvision.transforms.functional as TF

# def test_write_streamed_video(ipt_tns, num_frames, h=240, w=320, c=3, fps=29.92, OUT_FILE_PATH='output.webm'):
def test_write_streamed_video(ipt_tns, h=640, w=480, fps=30, OUT_FILE_PATH='output.mp4'):
    ipt_np = ipt_tns.detach().numpy()

    """
    # fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    # video_writer = cv2.VideoWriter(OUT_FILE_PATH, fourcc, fps, (h, w))

    # # random_ipt = np.random.randn(*ipt_np.shape)

    # # for frame in random_ipt:
    # for frame in ipt_np:
    #     print("writing frame")
    #     video_writer.write(frame)
    # video_writer.release()
    """

    # output file name and video settings
    codec = 'libx264'
    # codec = 'mp4v'

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
        for frame in ipt_np:
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.stderr.close()
    proc.wait()

def write_single_frame(video_array, frame_idx=0):
    single_frame = video_array[frame_idx]
    cv2.imwrite("test.png", cv2.cvtColor(single_frame,cv2.COLOR_RGB2BGR))


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

    print(video_tensor.shape)
    test_write_streamed_video(video_tensor, num_frames)