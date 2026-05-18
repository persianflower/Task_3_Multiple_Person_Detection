import cv2
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from ultralytics import YOLO

#logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger=logging.getLogger("MultiPersonDetection")

#threshold config

@dataclass
class DetectorConfig:
    detection_confi:float = 0.70
    pose_confi: float = 0.70
    person_limit: int = 1
    violation_frame_num: int = 6 #how many times more than one person in frame before treated as violation
    max_violations: int = 3
    min_face_area: float = 0.003 #if lesser than this it is a poster
    edge_exclusion: int = 0.05 #to exclude reflections at edges
    risk_weight_per_frame: float = 1.0
    detect_model_path: str = "yolo26n.pt"
    pose_model_path: str = "yolo26n-pose.pt"
    use_haar: bool = True #for face count
    haar_path: str = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

#curr session

@dataclass
class Session:
    session_risk_score: float = 0.0
    violation_frequency: int = 0 #total violations
    violation_frames: int = 0 #violation consequently
    total_frames: int = 0
    terminated: bool = False
    termination_reason: Optional[str] = None
    violation_log: list = field(default_factory=list)


#frame result

@dataclass
class FrameRes:
    person_count: int
    face_count: int
    violation: bool
    violation_reason: Optional[str]
    session_risk_score: float
    violation_freq: int
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def to_api(self)-> dict:
        return asdict(self)
    
#detector

class MultiPersonDetection:
    def __init__(self, config: DetectorConfig):
        self.cofig=config
        self.state=Session()
        logger.info(f"Loading detection model: %s {config.detect_model_path}")
        self.detect_model=YOLO(config.detect_model_path)
        logger.info(f"Loading pose model: %s {config.pose_model_path}")
        self.pose_model=YOLO(config.pose_model_path)
        self.face_cascade=None

        if config.use_haar:
            self.face_cascade=cv2.CascadeClassifier(config.haar_path)
            if self.face_cascade.empty():
                logger.warning("Haar cascade not loaded!")
                self.face_cascade=None
    
    def process_frame(self,frame) -> FrameRes:
        self.state.total_frames+=1
        h,w=frame.shape[:2]
        frame_area=h*w

        person_count=self._count_persons(frame,w)
        face_count=self._count_faces(frame,frame_area)

        actual_count=max(person_count,face_count)

        violation,reason=self._evaluate_violation(actual_count)
        self._update_risk(violation)

        res=FrameRes(
            person_count=actual_count,
            face_count=face_count,
            violation=violation,
            violation_reason=reason,
            session_risk_score=round(self.state.session_risk_score,2),
            violation_freq=self.state.violation_frequency,
            timestamp=time.time()
        )

        if violation:
            self.state.violation_log.append(
                {"frame": self.state.total_frames, "count": actual_count, "reason: ": reason, "time: ": res.timestamp}
            )
        return res
    
    @property 
    def should_terminate(self) -> bool:
        return self.state.terminated
    
    def summary(self) -> dict: 
        return {
            "total_frames": self.state.total_frames,
            "session_risk_score": round(self.state.session_risk_score, 2),
            "violation_frequency":self.state.violation_frequency,
            "terminated":self.state.terminated,
            "termination_reason":self.state.termination_reason,
            "violation_log":self.state.violation_log
        }
    
    def _count_persons(self,frame,frame_width:int) -> int:
        res=self.detect_model.track(
            frame, verbose=False, conf=self.cofig.detection_confi, imgsz=640, classes=[0]
        )
        count=0
        edge_margin=int(frame_width*self.cofig.edge_exclusion)

        for r in res:
            for box in r.boxes:
                if int(box.cls[0])!=0:
                    continue
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cx=(x1+x2)//2
                if cx<edge_margin or cx>frame_width-edge_margin:
                    logger.debug("Edge filtered box at cx=%d",cx)
                count+=1
        return count
    
    def _count_faces(self,frame,frame_area:int) -> int:
        if self.face_cascade is None:
            return 0
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        grey=cv2.equalizeHist(grey)

        faces=self.face_cascade.detectMultiScale(
            grey, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE
        )

        count=0
        for (x,y,fw,fh) in faces if len(faces) else []:
            face_area=fw*fh
            ratio=face_area/frame_area
            if ratio>=self.cofig.min_face_area:
                count+=1
            else:
                logger.debug(
                    "Filtered smaller faces (area ratio %.4f < %.4f)", 
                    ratio, self.cofig.min_face_area
                )
        
        return count
    
    def _evaluate_violation(self, actual_count:int):
        if actual_count <= self.cofig.person_limit:
            self.state.violation_frames = 0
            return False,None
        self.state.violation_frames+=1
        if self.state.violation_frames<self.cofig.violation_frame_num:
            return False, None
        if self.state.violation_frames==self.cofig.violation_frame_num:
            self.state.violation_frequency+=1
            logger.warning(
                "Violation #%d - %d personse detect in frame", self.state.violation_frequency,actual_count
            )
            if self.state.violation_frames>=self.cofig.max_violations:
                self.state.terminated=True
                self.state.termination_reason=(
                    f"Exceeded max violations ({self.cofig.max_violations})"
                )
                logger.error("Session Terminated: %s",self.state.termination_reason)
        reason=f"{actual_count} persons detected (limit={self.cofig.person_limit})"
        return True,reason
    
    def _update_risk(self,violation: bool):
        if violation:
            self.state.session_risk_score+=self.cofig.risk_weight_per_frame/30
    
def draw(frame, res:FrameRes, terminated: bool):
    h,w=frame.shape[:2]
    status_color=(0,0,255) if res.violation else (0,255,100)
    lines=[
        f"Person: {res.person_count} Face: {res.face_count}",
        f"Violation: {res.violation}",
        f"Risk Score: {res.session_risk_score:.1f}",
        f"Events: {res.violation_freq}"
    ]

    for i,text in enumerate(lines):
        cv2.putText(frame,text,(16,h-100+i*24),cv2.FONT_ITALIC,0.6,status_color,2)
        
    if terminated:
        cv2.putText(frame, "Session Terminated",(w//4,h//2), cv2.FONT_HERSHEY_COMPLEX, 1.4, (0,0,255), 3)
        
    if res.violation_reason:
        cv2.putText(frame,res.violation_reason, (16,30), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,100,255), 2)
        
    #maine
def main():
    config=DetectorConfig()
    detector=MultiPersonDetection(config)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcom couldnt open")
        return
        
    logger.info("Starting, Press [q] to exit")

    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                logger.warning("Frame read failed!")
                break
            res=detector.process_frame(frame)
            if detector.state.total_frames % 30 ==0:
                print(res.to_json())
            draw(frame,res,detector.should_terminate)
            cv2.imshow("AI Interview - multi person detector", frame)
            if detector.should_terminate:
                cv2.waitKey(2000)
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary=detector.summary()
        logger.info("Session Summary")
        print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()






