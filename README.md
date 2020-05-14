# ECG
Project on writing a library for ECG processing oriented on lab anymals: dogs, rabbits and mice.

```from ecg_detection import ecg```

To use it on your own dataset, change the path : ```ecg.path ='my/path'```

Note that uploading files implies the following structure of the files:
- path
  * animal_type
    * animal_item
        * recordings for this animal

For example:
- my/path
  * dogs
    * dog_1
        * dog_1_recording.txt
        
###Sample Results
Detection for a rabbit
 <img src="./images/rabbit_example.png" width="500px"/> 
 * Note that there are may be problems with peaks detection if your recording has big baseline drift and r-peaks are down-directed. In this case, revert the recording manually and try agaim. We are currently working on total baseline removing without changing the ECG recording (buttherworth may shift J point).