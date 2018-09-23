import json
import os
from matplotlib import pyplot as mymplt

class Analyze_Image_Label_Pairs:

    def figure_batching(self, label:str,fig_num:int,):
        pass

    def main(self):
        path_to_training_data = os.path.join('C:\\Users\\Peter\\GitHub_Remotes\\Internship_folders_summer_2018\\DeepLearningData_RCP\\Image_analysis_ML\\Adapted_Unet\\UnetLogicFiles\\mapped_image_label_details').replace("\\","/")
        # path_to_training_data = os.path.join('C:\\Users\\Peter\\GitHub_remotes\\Internship_folders_summer_2018\\DeepLearningData_RCP\\Image_analysis_ML\\Adapted_Unet\\UnetLogicFiles\\mapped_image_label_details').replace("\\", "/")
        target_file = '/mapped_dic_8_28_18.json'
        with open(path_to_training_data+target_file, 'r') as input_file:
            json_dict = json.load(input_file)
        keys = list(json_dict)
        time_dict = json_dict[keys[0]]
        img_size_list = []
        lbl_size_list = []
        delta_size = []
        img_cumsum_list = []
        lbl_cumsum_list = []
        delta_cumsum = []
        # remove this comment :P
        for ordinal in time_dict:
            name_ordinal = "{:0>4}".format(ordinal)
            img_size_list.append(time_dict[ordinal]["images.%s.png"%name_ordinal]["file_size"])
            img_cumsum_list.append(time_dict[ordinal]["images.%s.png"%name_ordinal]["tile_cumulative_pxl_value"])
            lbl_size_list.append(time_dict[ordinal]["binary.%s.png"%name_ordinal]["file_size"])
            lbl_cumsum_list.append(time_dict[ordinal]["binary.%s.png"%name_ordinal]["tile_cumulative_pxl_value"])
            delta_size.append(time_dict[ordinal]["delta"]["file_size"])
            delta_cumsum.append(time_dict[ordinal]["delta"]["tile_cumulative_pxl_value"])

        img_size_max = max(img_size_list)
        img_cumsum_max = max(img_cumsum_list)
        lbl_size_max = max(lbl_size_list)
        lbl_cumsum_max = max(lbl_cumsum_list)
        delta_size_max = max(delta_size)
        delta_cumsum_max = max(delta_cumsum)
        img_size_y = img_size_list
        lbl_size_y = lbl_size_list
        delta_size_y = delta_size
        img_cumsum_y = []
        lbl_cumsum_y = []  # lbl_cumsum_list
        delta_cumsum_y = delta_cumsum
        # each sublist is [dest,src,max]
        full_combine = [
            # [img_size_y,img_size_list,img_size_max],
            # [lbl_size_y,lbl_size_list,lbl_size_max],
            # [delta_size_y,delta_size,delta_size_max],
            [img_cumsum_y,img_cumsum_list,img_cumsum_max],
            [lbl_cumsum_y,lbl_cumsum_list,lbl_cumsum_max]
            # [delta_cumsum_y,delta_cumsum,delta_cumsum_max]
        ]

        for sub in full_combine:
            dest,src,list_max = sub
            for val in src:
                dest.append((-val)+list_max)

        tile_ordinal_x = range(len(img_size_y))
        fig_number = 0
        # mymplt.figure(fig_number)
        fig, ax1 = mymplt.subplots()

        # mymplt.subplot(1, 1, 1)
        ax1.plot(tile_ordinal_x, img_size_y, 'ro-', label="Image FileSize")
        ax1.set_ylabel('Image Size')
        ax1.tick_params('y', colors='r')
        ax2 = ax1.twinx()
        ax2.plot(tile_ordinal_x, img_cumsum_y, 'bx-', label="Image CumSum")
        ax2.set_ylabel('Image CumSum')
        ax2.tick_params('y',colors='b')
        fig.tight_layout()
        mymplt.title(
                "Image File size and CumSum")

        mymplt.xlabel("Ordinal index position")
        mymplt.legend()
        # mymplt.plot()
        # mymplt.show()
        mymplt.savefig(os.path.join(path_to_training_data,'image_pair_analysis_plot_%d.png'%fig_number))

        fig_number += 1
        fig, ax1 = mymplt.subplots()

        # mymplt.subplot(1, 1, 1)
        ax1.plot(tile_ordinal_x, lbl_size_y, 'ro-', label="Label FileSize")
        ax1.set_ylabel('Label Size')
        ax1.tick_params('y', colors='r')
        ax2 = ax1.twinx()
        ax2.plot(tile_ordinal_x, lbl_cumsum_y, 'bx-', label="Label CumSum")
        ax2.set_ylabel('Label CumSum')
        ax2.tick_params('y',colors='b')
        fig.tight_layout()
        mymplt.title(
                "Label File size and CumSum")

        mymplt.xlabel("Ordinal index position")
        mymplt.legend()
        # mymplt.plot()
        # mymplt.show()
        mymplt.savefig(os.path.join(path_to_training_data,'image_pair_analysis_plot_%d.png'%fig_number))

        fig_number += 1
        # mymplt.figure(fig_number)
        fig, ax1 = mymplt.subplots()

        ax1.plot(tile_ordinal_x, delta_cumsum_y, 'gx-', label="Difference in cumulative pixel value sum")
        ax1.set_ylabel('CumSum Deltas')
        ax1.tick_params('y',colors='g')
        ax2 = ax1.twinx()
        ax2.plot(tile_ordinal_x, delta_size_y, 'b.--', label="Difference in file size")
        ax2.set_ylabel("FileSize Deltas")
        ax2.tick_params('y',colors='b')
        fig.tight_layout()
        mymplt.title(
                "Size and CumSum Deltas")

        mymplt.xlabel("Ordinal index position")

        mymplt.legend()
        mymplt.plot()
        mymplt.show()
        mymplt.savefig(os.path.join(path_to_training_data,'image_pair_analysis_plot_%d.png'%fig_number))

if __name__ == "__main__":
    Analyze_Image_Label_Pairs().main()
