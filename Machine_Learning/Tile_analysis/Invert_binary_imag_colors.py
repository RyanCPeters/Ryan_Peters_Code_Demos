import os
import skimage
from skimage import io
from skimage import util


class BinaryInverter:

    def main(self):
        path_string = \
            "C:\\Users\\Peter\\GitHub_Remotes\\Internship_folders_summer_2018\\Ryan_deep_learning_data\\backup\\label"\
                .replace("\\", "/")
        dest_folder = os.path.join(path_string, "/inverted_labels")
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        walk_data = os.walk(path_string).__next__()
        for file in walk_data[2]:
            img = skimage.io.imread(os.path.join(path_string,file))
            skimage.io.imsave(os.path.join(dest_folder,file), skimage.util.invert(img))


if __name__ == "__main__":
    BinaryInverter().main()

