from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break
        return detections
        
    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],  # {0:{"bbox":[0,0,0,0]}, 1:...}
            "referees": [],
            "ball": []
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # {0:person, 1:goalkeeper..}
            cls_names_inverse = {v: k for k, v in cls_names.items()}  # make it to {person:0, ....}
            
            # convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inverse["player"]
            
            # track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    
                if cls_id == cls_names_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
            
        return tracks
