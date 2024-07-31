from utils import read_video, save_video
from trackers import Tracker
import cv2
def main():
    #read videos
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pk1')
    
    # save cropped images of a player
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]
        
        # crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        # save the cropped image
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break
    
    # Draw Output
    ## Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    
    
    #save videos
    save_video(output_video_frames, 'output_videos/08fd33_4.avi')
    
if __name__ == "__main__":
    main()