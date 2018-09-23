import datetime
from typing import TextIO

import numpy as np
import os
import json

from skimage import io


class ImageLabelZipper(object):

    def sum_pxl_values(self, tile: np.ndarray) -> int:
        cumulative_sum = 0
        for row in tile:
            for pxl in row:
                cumulative_sum += pxl
        return cumulative_sum

    def zipper_img_label_to_json(self) -> None:
        # os_walk_data = os.walk('C:/Users/Peter/Documents/GitHub_Remotes/Internship_folders_summer_2018/Ryan_deep_learning_data/training_backup/').__next__()
        # C:\Users\Peter\GitHub_remotes\Internship_folders_summer_2018\Ryan_deep_learning_data\backup
        path_to_training_data = 'C:\\Users\\Peter\\GitHub_Remotes\\Internship_folders_summer_2018\\Ryan_deep_learning_data\\backup'\
            .replace("\\", "/")
        os_walk_data = os.walk(path_to_training_data).__next__()
        img_list = []
        lbl_list = []
        print("\n\nos_walk_data's elements:", '\nroot:\t', os_walk_data[0], '\ndirs:\t', os_walk_data[1], '\nfiles:\t',
              os_walk_data[2])
        for root, dirs, imgfiles in os.walk(os.path.join(os_walk_data[0], os_walk_data[1][0])):
            for img in imgfiles:
                img_list.append(img)
        for root, dirs, lblfiles in os.walk(os.path.join(os_walk_data[0], os_walk_data[1][1])):
            for lbl in lblfiles:
                lbl_list.append(lbl)
        ordinal_img_label_mapping = dict()
        start_time = datetime.datetime.now().strftime("%m-%d_%Hhours_%Mmins")
        ordinal_img_label_mapping[start_time] = {}
        num_of_files = len(img_list)
        interval_scalar = 1
        for ordinal_pos in range(num_of_files):
            if (num_of_files / (ordinal_pos + 1)) % 10 == 0:
                print("%d%% complete" % (10 * interval_scalar))
                interval_scalar += 1

            image = io.imread(os.path.join(os_walk_data[0], os_walk_data[1][0], img_list[ordinal_pos]))
            label = io.imread(os.path.join(os_walk_data[0], os_walk_data[1][1], lbl_list[ordinal_pos]))
            ordinal_img_label_mapping[start_time][ordinal_pos] = \
                {

                    img_list[ordinal_pos]:
                        {
                            'file_size':
                                os.stat(
                                    os.path.join(os_walk_data[0], os_walk_data[1][0], img_list[ordinal_pos])).st_size,
                            'tile_shape': image.shape,
                            'tile_cumulative_pxl_value': np.asscalar(self.sum_pxl_values(image))
                        },
                    lbl_list[ordinal_pos]:
                        {
                            'file_size':
                                os.stat(
                                    os.path.join(os_walk_data[0], os_walk_data[1][1], lbl_list[ordinal_pos])).st_size,
                            'tile_shape': label.shape,
                            'tile_cumulative_pxl_value': np.asscalar(self.sum_pxl_values(label))
                        },
                    "delta":
                        {
                            'file_size':
                                np.asscalar(np.abs(os.stat(
                                    os.path.join(os_walk_data[0], os_walk_data[1][0], img_list[ordinal_pos])).st_size -
                                                   os.stat(
                                                       os.path.join(os_walk_data[0], os_walk_data[1][1],
                                                                    lbl_list[ordinal_pos])).st_size)),
                            'tile_shape': (np.asscalar(np.abs(image.shape[0] - label.shape[0])),
                                           np.asscalar(np.abs(image.shape[1] - label.shape[1]))),
                            'tile_cumulative_pxl_value': np.asscalar(
                                np.abs(self.sum_pxl_values(image) - self.sum_pxl_values(label)))
                        }

                }
        output_path_string = os.path.join(os_walk_data[0], "mapped_image_label_details", )
        if not os.path.exists(output_path_string):
            os.makedirs(output_path_string)

        with open(os.path.join(output_path_string, "mapped_dic_8_28_18.txt"), "a+") as output_txt:
            output_txt.write(ordinal_img_label_mapping.__str__())

        with open(os.path.join(output_path_string, "mapped_dic_8_28_18.json"), "a+") as output_file:
            json.dump(ordinal_img_label_mapping, output_file, indent=2)

    def main(self):
        self.zipper_img_label_to_json()


if __name__ == '__main__':
    ImageLabelZipper().main()
