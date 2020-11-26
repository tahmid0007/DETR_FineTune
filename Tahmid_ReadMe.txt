1. To use finetune- load skeleton model from one of the fb provided checkpoints
	- edit numClass as required
	- CoCo demands that the coco_path = root has a train, val and annotation folder with the json files
	- while building dataset just provide the coco api this root path and the rest it will do
	- no. of classes = numClass = max ID + 1
	- e.g., chess dataset has 0-12 ids so numClass = 13, json file contains the category ids e.g., white-bishop = 7
2. To Train from scratch
	- Just load skeleton model as before
	- do not load weights from state_dict