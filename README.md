# Face recognition and tracking app

The version of the program compiled for Linux x64 is located in the "compiled" folder.

To use it, you need to run the "face_rec" file.

The program accepts the following arguments:


    -mode (Mandatory parameter)
	    training
	    processing
    -dir (Mandatory parameter)
	    Path to the directory with faces for creating a descriptor file or path to the directory with videos for processing
    -metrics
	    If this parameter is passed, quality metrics will be additionally calculated (only for the processing mode). In order for the metric calculation to be possible, txt files with bounding boxes should also be present in the directory with videos.

Examples for test launch:


    ./face_rec -mode training -dir ./faces
    ./face_rec -mode processing -dir ./faces -metrics
