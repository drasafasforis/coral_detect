#coral_detect
import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
from PIL import Image

def get_file_names(folder_path):
    try:
        # Get a list of files in the specified folder
        file_names = os.listdir(folder_path)

        # Filter out subdirectories and keep only file names
        file_names = [file for file in file_names if os.path.isfile(os.path.join(folder_path, file))]

        return file_names
    
    except OSError as e:
        print(f"An error occurred: {e}")
        return []

file_names = get_file_names('/home/mendel/project_files/coral_yolo_files/processed_images')
for filename in file_names:
    # Specify the TensorFlow model, labels, and image
    script_dir = pathlib.Path(__file__).parent.absolute()
    model_file = os.path.join(script_dir, 'best-int8_edgetpu.tflite')
    label_file = os.path.join(script_dir, 'data.txt')
    image_file = os.path.join(script_dir, '/home/mendel/project_files/coral_yolo_files/processed_images/'+filename)

    # Initialize the TF interpreter
    interpreter = edgetpu.make_interpreter(model_file)
    interpreter.allocate_tensors()

    # Resize the image
    size = common.input_size(interpreter)
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

    # Run an inference
    common.set_input(interpreter, image)
    interpreter.invoke()

    detections = detect.get_objects(interpreter, score_threshold=0.25)

    # Print the results with label and probability
    labels = dataset.read_label_file(label_file)
    for detection in detections:
        label = labels.get(detection.id, detection.id)
        probability = detection.score
        print(f'Label: {label}, Probability: {probability}')
    