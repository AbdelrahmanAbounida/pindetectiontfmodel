import cv2
import os
import glob
import shutil


class VideoToImage:
    """ This class is similar to VideoToImage in pin-classification pkg with small differences"""

    def __init__(self, videos_folder = 'pin-detector/pin_detector_model/videos',target_dataset_folder = 'pin-detector/pin_detector_model/images/', class_name="pin_A"):
        """
        target_dataset_folder: folder in which you want to generate the images
        videos_folder: folder in which the videos exist
        class_name: is the name of the object that we wanna detect, in the future it will be a list of pins with different types A,B,...
        """
        self.videos_dir = os.path.join(os. getcwd(), videos_folder)
        self.list_of_videos = glob.glob(f"{self.videos_dir}/*.mp4")
        self.dataset_dir = os.path.join(os. getcwd(), target_dataset_folder)
        self.class_name = class_name

        self.generate_images_from_videos()
        self.create_train_val_dataset()
        

    def generate_images_from_videos(self, force_generate=False, num_of_generated_images_per_video=100):
        """ This method is similar to class VideoToImage in pin-classification pkg
            force_generate: if True >> this method will regenerate images even if they exist
            num_of_generated_images_per_video: number of images to generate per each video
        """

        # if there are folders or images in the images folder
        if len(glob.glob(f"{self.dataset_dir}/*")):
            print("There are some folders/files in the dataset directory")

            if not force_generate:
                return False

        currentframe = 0
        for index, video in enumerate(self.list_of_videos):

            cam = cv2.VideoCapture(video)
            i = 0

            while (i < num_of_generated_images_per_video):
                # reading from frame
                ret, frame = cam.read()
                if ret:
                    name = f'{self.dataset_dir}/{self.class_name}_' + \
                        str(currentframe) + '.jpg'
                    print('Creating...' + name)
                    cv2.imwrite(name, frame)
                    currentframe += 1
                else:
                    break

                i += 1

            cam.release()
            cv2.destroyAllWindows()
    

    def create_train_val_dataset(self):
        
        images = glob.glob(self.dataset_dir + '/*.jpg')
        train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

        if os.path.exists(os.path.join(self.dataset_dir, 'train')):
            if os.path.exists(os.path.join(self.dataset_dir, 'val')):
                print("Dataset already generated and extist in train and val folder")
                self.total_train , self.total_val =  len(glob.glob(os.path.join(self.dataset_dir, 'train/*'))), len(glob.glob(os.path.join(self.dataset_dir, 'val/*')))
            else:
                raise TypeError("val direction should exist beside train direction")
        
        for t in train:
            if not os.path.exists(os.path.join(self.dataset_dir, 'train')):
                os.makedirs(os.path.join(self.dataset_dir, 'train'))
            shutil.move(t, os.path.join(self.dataset_dir, 'train'))

        for v in val:
            if not os.path.exists(os.path.join(self.dataset_dir, 'val')):
                os.makedirs(os.path.join(self.dataset_dir, 'val'))
            shutil.move(v, os.path.join(self.dataset_dir, 'val'))

        return 
