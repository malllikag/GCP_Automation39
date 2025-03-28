from roboflow import Roboflow
from ultralytics import YOLO 


#Obtain dataset from Roboflow using an API key 
def main():
    
    #Training the model to recognize the new numbered markers
    #rf = Roboflow(api_key="FO5qGN6i5rUl6JXX2o7j")
    #project = rf.workspace("yolov10-pztcv").project("tr-bvsoh")
    #version = project.version(1)
    #dataset = version.download("yolov9")

    #Full dataset with new markers 
    #rf = Roboflow(api_key="FO5qGN6i5rUl6JXX2o7j")
    #project = rf.workspace("yolov10-pztcv").project("tr-bvsoh")
    #version = project.version(2)
    #dataset = version.download("yolov9")                


#Model training - training from scratch using pretrained weights 
    
    model = YOLO(r"C:\Users\mgummuluri\Work\Tasks\YOLOv10\runs\detect\train2\weights\best.pt")
    #model.train(data=r"C:\Users\mgummuluri\Work\Tasks\YOLOv10\tr-2\data.yaml", epochs=200, imgsz=2048, batch=4, lr0=0.0003, mosaic=0.5, mixup=0.1, flipud=0.5, scale=0.5)
    
    # After training, perform validation
    #print("Running validation on the validation set:")
    #val_results = model.val(data=r"C:\Users\mgummuluri\Work\Tasks\YOLOv10\tr-1\data.yaml", split='val', imgsz=2048, batch=4)
    #print(val_results)
    
    #Test set
    #print("Running evaluation on the test set:")
    #test_results = model.val(data=r"C:\Users\mgummuluri\Work\Tasks\YOLOv10\tr-1\data.yaml", split='test', imgsz=2048, batch=4)
    #print(test_results)  

    #Predict on unseen data
    model.predict(source=r"C:\Users\mgummuluri\Work\Data(11-6-24)\240523_McCoy_CACTF\240523_McCoy_CACTF_Skydio\McCoy_CACTF_GRID__2024-05-23T17-37-18.901308+00-00", imgsz=2048, save=True)
     
    
if __name__ == "__main__":
    main()
                