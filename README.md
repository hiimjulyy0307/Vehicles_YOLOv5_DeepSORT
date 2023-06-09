# Vehicle detection by YOLOv5 and tracking by DeepSORT and counting

## Introduction

The detections algorithm generated by [YOLOv5 ](https://github.com/ultralytics/yolov5)
The tracking algorithm generated by [Deep Sort ](https://github.com/ZQPei/deep_sort_pytorch)
Counting vehicles in area using pointPolygonTest

## Setting before run tracker:

1. Create new folder:
2. Run Virtual Environment:
3. Git clone:
`git clone https://github.com/hiimjulyy0307/yolov5_deepsort_counting_vehicles.git`
4. Install package:
`pip install -r requirements.txt`

## if u have nvidia gpu install this:

1. CUDA Deep Neural Network (cuDNN)
2. Setting cuDNN global folder
3. Install pytorch versions [PyTorch](https://pytorch.org/get-started/previous-versions/)
Im using cudnn 11.6 so 
`pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116`

## Setting file track!
Make sure that you Run Virtual Environment before you run :D

1. File Track.py detect + tracking and not counting 
2. File Track2.py detect + tracking and counting vehicles using YOLOv5l.pt (you can changes version of YOLOv5 n s m l x)
3. File Track3.py detect + tracking and counting vehicles using Custom training model (vietnamese version - base on yolov5s.pt)

4. Can changes variable about yolo_model, deep_sort_model, source, ... from line 320 and up.
```bash
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5l.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos\hihi.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') 
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
```
## Run 
```bash
$ python track.py 
         track2.py
         track3.py                 

```

# Thanks
