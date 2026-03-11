import sys
print(sys.executable)

import cv2
import time

from video.stream_reader import VideoStreamReader
from detection.person_detector import PersonDetector
from detection.tracker import CentroidTracker
from analytics.presence import PresenceMonitor
from analytics.inactivity import InactivityMonitor
from analytics.behavior_classifier import BehaviorClassifier
from analytics.behavior_engine import BehaviorEngine
from analytics.trajectory_tracker import TrajectoryTracker
from alerts.alert_manager import AlertManager
from face.face_detector import FaceDetector
from face.face_encoder import FaceEncoder
from face.face_recognizer import FaceRecognizer


POST_AREA = (100,100,400,400)

MIN_PERSON_HEIGHT = 120
MIN_PERSON_WIDTH = 40


def iou(boxA, boxB):

    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])

    interW=max(0,xB-xA)
    interH=max(0,yB-yA)

    interArea=interW*interH

    if interArea==0:
        return 0.0

    boxAArea=(boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea=(boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return interArea/float(boxAArea+boxBArea-interArea)


def main():

    stream = VideoStreamReader(source=0)

    detector = PersonDetector(conf_threshold=0.5)

    tracker = CentroidTracker(
        max_disappeared=300,
        max_distance=300
    )

    presence_monitor = PresenceMonitor(
        absence_threshold=10,
        confirm_time=3
    )

    inactivity_monitor = InactivityMonitor(
        inactivity_threshold=30,
        position_threshold=40,
        window_time=5
    )

    behavior_classifier = BehaviorClassifier(device="cpu")
    behavior_engine = BehaviorEngine()

    trajectory_tracker = TrajectoryTracker(roi=POST_AREA, grid_size=2)

    alert_manager = AlertManager()

    face_detector = FaceDetector()
    face_encoder = FaceEncoder()
    face_recognizer = FaceRecognizer(tolerance=0.55)

    frame_count = 0

    behavior_cache = {}
    identity_memory = {}

    presence_buffer = 0

    warmup_frames = 120

    face_boxes = []
    identities = []

    last_alert_time = {}
    ALERT_COOLDOWN = 15
    last_path = {}

    while True:
        frame_count+=1
        frame = stream.read_frame()
        if frame is None:
            break

        if frame_count < warmup_frames:
            cv2.imshow("Guard Monitoring",frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        detections = detector.detect(frame)
        person_boxes = []
        valid_centroids = []

        for (x1,y1,x2,y2,conf) in detections:
            if (x2-x1)<MIN_PERSON_WIDTH or (y2-y1)<MIN_PERSON_HEIGHT:
                continue

            box=(x1,y1,x2,y2)
            person_boxes.append(box)
            if iou(box,POST_AREA) > 0.25:
                cx=int((x1+x2)/2)
                cy=int((y1+y2)/2)
                valid_centroids.append((cx,cy))

        if len(valid_centroids)>0:
            presence_buffer=20
        else:
            presence_buffer=max(0,presence_buffer-1)

        buffered_centroids = valid_centroids if presence_buffer>0 else []
        post_status = presence_monitor.update(buffered_centroids)

        if post_status=="PRESENT":
            activity_status = inactivity_monitor.update(valid_centroids)
        else:
            activity_status = "NO_PERSON"

        tracked_objects = tracker.update(person_boxes)
        trajectory_tracker.update(tracked_objects)

        if frame_count % 15 == 0:
            face_boxes = face_detector.detect(frame,list(tracked_objects.values()))
            face_encodings = face_encoder.encode(frame,face_boxes)
            identities = face_recognizer.recognize(face_encodings)

        for track_id,person_box in tracked_objects.items():
            best_iou = 0.0
            recognized_id = "UNKNOWN"

            for i,fb in enumerate(face_boxes):
                overlap = iou(person_box,fb[:4])
                if overlap > best_iou:
                    best_iou = overlap
                    recognized_id = identities[i] if i < len(identities) else "UNKNOWN"

            if recognized_id != "UNKNOWN":
                identity_memory[track_id] = recognized_id

            guard_id = identity_memory.get(track_id,"UNKNOWN")

            if track_id not in behavior_cache:
                behavior_cache[track_id] = {
                    "label":"ANALYZING",
                    "last_update":0
                }

            if frame_count % 20 == 0 and post_status=="PRESENT":
                label,confidence = behavior_classifier.predict(frame,person_box)
                behavior_cache[track_id]["label"] = label
                behavior_cache[track_id]["last_update"] = frame_count

            label = behavior_cache[track_id]["label"]

            if post_status=="PRESENT":
                final_state = behavior_engine.update(track_id,label)
            else:
                final_state = "ANALYZING"

            x1,y1,x2,y2 = person_box
            color=(0,255,0)

            if "CONFIRMED" in final_state:
                color=(0,0,255)
            elif "POSSIBLE" in final_state:
                color=(0,165,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.putText(
                frame,
                f"{guard_id} | {final_state}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            zone = trajectory_tracker.get_current_zone(track_id)
            path = trajectory_tracker.get_path(track_id)
            if path and last_path.get(track_id) != path:
                print("Zone:", path)
                last_path[track_id] = path

            cv2.putText(
                frame,
                f"Zone: {zone}",
                (x1, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Path: {path}",
                (x1, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0,255,255),
                2
            )

            current_time = time.time()

            def send_alert(key,msg):
                last = last_alert_time.get(key,0)
                if current_time - last > ALERT_COOLDOWN:
                    alert_manager.send_alert(msg,guard_id)
                    last_alert_time[key]=current_time


            if guard_id!="UNKNOWN":
                if post_status=="ABSENT":
                    send_alert("missing","Guard Missing")
                elif "CONFIRMED_SLEEPING" in final_state:
                    send_alert("sleep","Guard Sleeping")
                elif "CONFIRMED_PHONE_USE" in final_state:
                    send_alert("phone","Phone Usage")
                elif "CONFIRMED_DISTRACTED" in final_state:
                    send_alert("distracted","Guard Distracted")
                elif activity_status=="INACTIVE":
                    send_alert("idle","Guard Idle")

        rx1,ry1,rx2,ry2 = POST_AREA
        cv2.rectangle(frame,(rx1,ry1),(rx2,ry2),(255,0,0),2)

        post_color=(0,255,0) if post_status=="PRESENT" else (0,255,255) if post_status=="TEMPORARILY_EMPTY" else (0,0,255)

        activity_color=(0,255,0) if activity_status=="ACTIVE" else (0,0,255) if activity_status=="INACTIVE" else (255,255,255)

        cv2.putText(frame,f"Post Status: {post_status}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,post_color,2)

        cv2.putText(frame,f"Activity Status: {activity_status}",(20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,activity_color,2)

        cv2.imshow("Guard Monitoring",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()