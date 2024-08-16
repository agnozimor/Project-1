from roboflow import Roboflow
rf = Roboflow(api_key="9FoTabUd595p59eYeo8C")
project = rf.workspace("college-74jj5").project("freshness-fruits-and-vegetables")
version = project.version(7)
dataset = version.download("yolov9")