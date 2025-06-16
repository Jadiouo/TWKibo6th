from ultralytics import YOLO
if __name__ == "__main__":
    print("This is a YOLOv8 training script.")
    yolo_path = r'E:\gitrepo\yolo-V8-main\yolov8n.pt'
    datayaml_path = r'E:\gitrepo\yolo-V8-main\kiborpc\kiborpcdata.yaml'
    cfgyaml_path = r'E:\gitrepo\yolo-V8-main\kiborpc\cfg.yaml'

    model = YOLO(yolo_path)  # Load a pretrained YOLOv8n model

    # Train model - mapping your YOLOv7 parameters to YOLOv8
    results = model.train(
        data= datayaml_path,  # Path to the data YAML file
        cfg= cfgyaml_path,  # Path to the configuration YAML file
    )
    
