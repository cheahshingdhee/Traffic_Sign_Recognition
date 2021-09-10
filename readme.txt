Python Interpreter - Python 3.7

Instruction to run the program:
Step 1: Make sure you have Python 3.7 installed in your system.

Step 2: Create a virtual environment and activate the environment for this project.
Open Command Prompt with current directory, and key in several command below:
python -m venv .venv
.venv\Scripts\activate.bat

Step 3: Install required packages with following command
pip install -r requirements.txt

Step 4: Please place all test data/image into 'test_data' folder and key in the file name into inputSignNames.txt.

Step 5: Test the classifier with model by running
python traffic_sign_classifier.py



Extra 1: Train the model with CNN
python train_cnn_model.py

Extra 2:
cd detection_model_and_svm
cd SVM_Training
python Image_segementation.py
python SVM_training.py
cd ..
cd Traffic_Detection_related_module
python Traffic_sign_detection_Classification_final.py
python Traffic_Sign_Detection_Final.py

Extra 3: Test the detection classification in real time
python traffic_sign_detection_classification_real_time_webcam.py

Information:
Graph images will save in results folder.
Model trained will save in model folder.
