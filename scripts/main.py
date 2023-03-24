from PinDetector import PinDetector
import os
import glob


if __name__ == "__main__":
    d = PinDetector()
    print(d.dataset_generator.total_train)