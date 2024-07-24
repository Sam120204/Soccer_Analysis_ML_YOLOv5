from utils import read_video, save_video

def main():
    #read videos
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    
    #save videos
    save_video(video_frames, 'output_videos/08fd33_4.avi')
    
if __name__ == "__main__":
    main()