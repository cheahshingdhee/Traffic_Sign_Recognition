Python Interpreter - Python 3.7

Instruction to run the program:
Step 1: Make sure you have Python 3.7 installed in your system.

Step 2: Create a virtual environment for this project.
Open Command Prompt with current directory, and key in sevaral command below:
python -m venv .venv
.venv\Scripts\activate.bat

Step 3: Install required packages with following command
pip install -r requirements.txt

Step 4: Train the model by running
python train_cnn_model.py

Step 5: Test the detector with model by running
python traffic_sign_detector.py

Results images will save in results folder.
Model trained will save in model folder.

