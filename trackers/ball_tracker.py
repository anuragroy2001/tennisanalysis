from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path, trajectory_length=10):

        self.model = YOLO(model_path)
        self.trajectory_points = []
        self.trajectory_length = trajectory_length
    
    def interpolate_ball_positions(self, ball_detections):
        ball_positions = [x.get(1,[]) for x in ball_detections]

        #convert the ilist into a pandas dataframe

        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #interpolate the missing values

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions =[{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
               ball_detections = pickle.load(f)
            return ball_detections
        


        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)


        return ball_detections
    
    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1,len(df_ball_positions)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] >0 and df_ball_positions['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] <0 and df_ball_positions['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    df_ball_positions['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits


    def detect_frame(self, frame):
        results = self.model.track(frame,conf=0.15 )[0]

        ball_dict  = {}
        for box in results.boxes:
    
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict


    # def draw_bboxes(self, video_frames, player_detections):
    #     output_video_frames = []
    #     for frame, ball_dict in zip(video_frames, player_detections):
    #         #Draw bounding boxes
    #         for track_id, bbox in ball_dict.items():
    #             x1, y1, x2, y2 = map(int, bbox)
    #             cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            
    #         # Draw the ball trajectory with fading effect
    #         ball_positions = [x.get(1, []) for x in player_detections]
    #         for i in range(max(0, len(ball_positions) - 10), len(ball_positions) - 1):
    #             x1, y1, x2, y2 = map(int, ball_positions[i])
    #             x3, y3, x4, y4 = map(int, ball_positions[i + 1])
    #             alpha = (len(ball_positions) - 1 - i) / 10.0  # Fading effect
    #             color = (0, int(255 * alpha), int(255 * alpha))
    #             cv2.line(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (int((x3 + x4) / 2), int((y3 + y4) / 2)), color, 2)
            
            
               
    #         output_video_frames.append(frame)
            
    #     return output_video_frames

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Store the center of the ball for trajectory
                self.trajectory_points.append((center_x, center_y))

                # Limit the trajectory length to self.trajectory_length points
                if len(self.trajectory_points) > self.trajectory_length:
                    self.trajectory_points = self.trajectory_points[-self.trajectory_length:]

                # Draw bounding box and ID
                cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw the trajectory line connecting the centers of the ball
            for i in range(1, len(self.trajectory_points)):
                if self.trajectory_points[i - 1] is None or self.trajectory_points[i] is None:
                    continue
                cv2.line(frame, self.trajectory_points[i - 1], self.trajectory_points[i], (0, 255, 0), 2)

            output_video_frames.append(frame)

        return output_video_frames
    