import os
import glob
import shutil
import pathlib
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from VideoToImage import VideoToImage

class PinDetector:
    def __init__(self, videos_folder='pin-detector/videos', target_dataset_folder='pin-detector/pin_detector_model/images/'):

        self.target_dataset_folder = target_dataset_folder

        ###################################
        #### Step1 >> Image Generation ####
        ###################################
        """ Generate n images from each video and move them 80% in train dir and 20% in val dir"""

        self.dataset_generator = VideoToImage(
            videos_folder, target_dataset_folder)

        ###################################
        #### Step2 >> Image Labeling ####
        ###################################
        """ This step is done manually through imagelbl app, by which we generate an xml file representing the border positions around the pin"""

        ###################################
        #### Step3 >> XML to CSV ####
        ###################################
        """ Generate csv file containing the border positions of each pin"""

        if os.path.exists(os.path.join(self.dataset_generator.dataset_dir, 'csv_data', 'train_labels.csv')):
            print("csv dataset already exist...Please double check csv_data folder")
        else:
            self.generate_csv()

        ###################################
        ### Step4 >> Create Label Map ###
        ###################################
        """ label map is the separate source of record for class annotations (the "answer key" for each image) """

        self.generate_label_map()

        ###################################
        ### Step5 >> Generate TF_Record ###
        ###################################
        """ TFRecord format is a simple format for storing a sequence of binary records, it has More efficient storage, Fast I/O and other advatages"""
        self.generate_tf_records()

        ###################################
        ##### Step6 >> Choose a Model #####
        ###################################
        """ There are different tf2 models in Detection Model Zoo, so we gonna download one of them and extract it in pre_trained_models folder"""
        """ https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md """
        """ For this task I used (ssd_resnet50_v1_fpn_640x640_coco17) , its speed and accuracy is not bad """

        ###################################
        ## Step7 >> Configuring Pipeline ##
        ###################################
        """
        The TensorFlow Object Detection API uses protobuf files to configure the training and evaluation process.
         The schema for the training pipeline can be found in object_detection/protos/pipeline.proto
        """
        """ the config file is split into 5 parts """
        """ under pin_detector_model folder create my_models folder and add a configuration file for our downloaded model """

        """
        configuration file is split into 5 parts:

        1- The model configuration. This defines what type of model will be trained (ie. meta-architecture, feature extractor).
        2- The train_config, which decides what parameters should be used to train model parameters (ie. SGD parameters, input preprocessing and feature extractor initialization values).
        3- The eval_config, which determines what set of metrics will be reported for evaluation.
        4- The train_input_config, which defines what dataset the model should be trained on.
        5- The eval_input_config, which defines what dataset the model will be evaluated on. Typically this should be different than the training input dataset.
        """

        """ create file : pin_detector_model/my_models/ssd_resnet50_v1_fpn_640x640_coco17/pipeline.config"""

        ###################################
        ## Step8 >> Training Model ##
        ###################################


        """ copy this file  TensorFlow/models/research/object_detection/model_main_tf2.py into pin-detector-model"""

        """ in model_main_tf2.py , edit those variables:
        model_dir= my_models/ssd_resnet50_v1_fpn_640x640_coco17
        pipeline_config_path= my_models/ssd_resnet50_v1_fpn_640x640_coco17/pipeline.config
        """
        
        """
        run model_main_tf2.py
        """
        
        ###################################
        ## Step9 >> Evaluating Model ##
        ###################################

        # 1- export the trained model checkpoints to a single frozen inference graph.
        """
        The term inference refers to the process of executing a TensorFlow Lite 
        model on-device in order to make predictions based on input data.

        (Copy the TensorFlow/models/research/object_detection/exporter_main_v2.py script and paste it in pin_detector_model )

        then change some directories in that file
        """
        # 2- run this (convert checkpoint to saved_model format)
        """
        python exporter_main_v2.py 
        """

        # 3- load model >> there are different methods the first one using the down below load_model_method1 method or other
        # methods in pin_detector_model_loader.py file


    def xml_to_df(self, path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height',
                       'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)

        return xml_df

    def generate_csv(self):

        csv_folder_path = os.path.join(
            os. getcwd(), 'pin-detector/pin_detector_model/annotations')

        if not os.path.exists(csv_folder_path):
            try:
                os.makedirs(csv_folder_path)
            except OSError:
                print('Error: Creating directory of data')
        else:

            print("Annotations folder already exist")
            return
        for directory in ['train', 'val']:
            image_path = os.path.join(
                self.dataset_generator.dataset_dir, directory)
            xml_df = self.xml_to_df(image_path)
            xml_df.to_csv(
                f'{csv_folder_path}/{directory}_labels.csv', index=None)

        print('annotations has been generated successfully')


    def generate_label_map(self):

        maplabel_path = os.path.join(
            os. getcwd(), 'pin-detector/pin_detector_model/annotations/', 'labelmap.pbtxt')
        if not os.path.exists(maplabel_path):
            with open(f'{maplabel_path}', 'w') as f:
                # we add just one type now , it gonna be edited in the future to be more dynamic for more pin types
                f.writelines("""
                    item { name: "Pin_A",
                    id: 1,
                    display_name: "Pin_A"}
                    """)
            f.close()
            print("Labelmap has been created successfully")
        else:
            print("Labelmap already exist")

    def generate_tf_records(self):

        tf_records_train_output = os.path.join(os. getcwd(), 'pin-detector/pin_detector_model/annotations/train.record')
        tf_records_val_output = os.path.join(os. getcwd(), 'pin-detector/pin_detector_model/annotations/val.record')

        if os.path.exists(tf_records_train_output) and os.path.exists(tf_records_val_output):
            print("TF_Records have already been generated")
        else:
            from TFRecordGenerator import TFRecordGenerator

            # generate tf records for training images
            train_csv_folder_path = os.path.join(os. getcwd(), 'pin-detector/pin_detector_model/annotations/train_labels.csv')
            train_images_dir = os.path.join(os. getcwd(), 'pin-detector/pin_detector_model/images/train/')
            train_tf_record_generator = TFRecordGenerator(train_csv_folder_path, tf_records_train_output, train_images_dir).main()

            print("Train tf_records have been created")

            # generate tf records for validation images
            val_csv_folder_path = os.path.join( os. getcwd(), 'pin-detector/pin_detector_model/annotations/val_labels.csv')
            val_images_dir = os.path.join( os. getcwd(), 'pin-detector/pin_detector_model/images/val/')
            val_tf_record_generator = TFRecordGenerator(val_csv_folder_path, tf_records_val_output, val_images_dir).main()
            print("Val tf_records have been created")
    
    
    def load_model_method1(self):
        """  This is the first method that could be used to load model from saved_model foramt"""
        loaded_model = tf.saved_model.load('./exported-models/my_model/saved_model/')
        return loaded_model





if __name__ == "__main__":
    # a = PinDetector()
    print(os.getcwd())
