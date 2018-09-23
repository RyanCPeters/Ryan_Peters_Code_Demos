import collections
import time
import unittest
import logging
# from UnetLogicFiles.unetLogic import *
from DeepLearningData_RCP.Image_analysis_ML.Adapted_Unet.UnetLogicFiles.unetLogic import *

success_sig = "\t\t\x1b[90;42m"+"[+]"+"\x1b[m"+"\x1b[32m"+" Success "+"\x1b[m"
failure_sig = "\t\t\x1b[90;101m"+"[-]"+"\x1b[m"+"\x1b[31m"+" Failure "+"\x1b[m"
test_dec_sig = "\n\n\x1b[96m"+"[ ] "
RESET = "\x1b[m"

class TestUnet(unittest.TestCase):
    """
    Unit-testing module for testing the various functions from unetLogic.py
    """
    
    def setUp(self):
        """
        sets up the class scoped local_path variable
        :return:
        """
        self.local_path = UnetMain.data_path_updater()
        return
    
    def test_UnetModel_unet(self):
        
        print(test_dec_sig+"testing output for unit-test on".upper() + " UnetModel.unet()"+RESET)
        logger = logging.getLogger(name="UnetModel_unet_logger")
        try:
            model = UnetModel.unet()
            self.assertIsNotNone(model)
        except Exception as ex:
            template = failure_sig + "An exception of type {0} occurred. {1}{2}{3}{4}"
            ex_type_name = type(ex).__name__
            ex_args = "\n\t\t\tArguments: " + str(ex.args)
            ex_context = "\n\t\t\tContext: " + str(ex.__context__)
            ex_cause = "\n\t\t\tCause: " + str(ex.__cause__)
            ex_tb = "\n\t\t\tTraceback: " + str(ex.with_traceback(ex.__traceback__))
            logger.exception(template.format(ex_type_name, ex_args, ex_context, ex_cause, ex_tb))
            # print(logger.__str__())
            print(logger)
            time.sleep(.05)
        else:
            print(success_sig+"UnetModel.unet() has passed")
            time.sleep(.05)
        return
    
    def test_UnetData_trainGenerator(self):
        print(test_dec_sig+"testing output for unit-test on".upper() + " UnetData.raw2mask_tuple_generator(...)"+RESET)
        logger = logging.getLogger(name="UnetData_trainGenerator_logger")
        data_path = self.local_path
        myGen = UnetData.raw2mask_tuple_generator(batch_size=100,
                                                  # train_path='../data/membrane/train',
                                                  train_path=data_path+data_path_dict["training_folder"],
                                                  image_folder='image',
                                                  mask_folder='label',
                                                  save_to_dir=None)
        try:
            self.assertIsNotNone(myGen)
            self.assertIsNotNone(myGen.__next__(),"is the training generator working?")
        except Exception as ex:
            template = failure_sig + "An exception of type {0} occurred. {1}{2}{3}{4}{5}{6}"
            ex_type_name = type(ex).__name__
            ex_args = "\n\t\t\tArguments: " + str(ex.args)
            ex_context = "\n\t\t\tContext: " + str(ex.__context__)
            ex_cause = "\n\t\t\tCause: " + str(ex.__cause__)
            ex_path = "\n\t\t\tDir Path: " + data_path+training_folder
            ex_globe_path_dict = "\n\t\t\tDict of Paths: " + str(data_path_dict.values())
            ex_tb = "\n\t\t\tTraceback: " + str(ex.with_traceback(ex.__traceback__))
            logger.exception(template.format(ex_type_name, ex_args, ex_context, ex_cause, ex_path, ex_globe_path_dict, ex_tb))
            print(logger.__str__())
            time.sleep(.05)
        else:
            print(success_sig+"UnetData.raw2mask_tuple_generator(...) has passed")
    
    def test_UnetData_testGenerator(self):
        print(test_dec_sig+"testing output for unit-test on".upper() + " UnetData.testGenerator(...)"+RESET)
        logger = logging.getLogger(name="UnetData_testGenerator_logger")
        data_path = self.local_path
        
        try:
            testGen = UnetData.testGenerator(data_path+test_folder)
            self.assertIsNotNone(testGen, data_path+test_folder)
            img_count = 0
            count_int = 3 # testGen.__sizeof__()
            while img_count < count_int and testGen.__next__() is not None:
                img_count += 1
            # self.assertEquals(img_count, count_int)
            print("number of images is ", str(img_count))
        except FileNotFoundError as fe:
            print(failure_sig+"Still getting the FileNotFoundError from testGenerator function")
            print(fe)
        except StopIteration as ex:
            template = failure_sig + "A StopIteration error occurred. {1}{2}{3}{4}{5}"
            ex_type_name = type(ex).__name__
            ex_args = "\n\t\t\tArguments: " + str(ex.args)
            ex_context = "\n\t\t\tContext: " + str(ex.__context__)
            ex_cause = "\n\t\t\tCause: " + str(ex.__cause__)
            ex_path = "\n\t\t\tDir Path: " + data_path+test_folder
            ex_tb = "\n\t\t\tTraceback: " + str(ex.with_traceback(ex.__traceback__))
            logger.exception(template.format(ex_type_name, ex_args, ex_context, ex_cause, ex_path, ex_tb))
            print(logger.__str__())
            time.sleep(.05)
        except AssertionError as ex:
            template = failure_sig+"An AssertionError occurred. {1}{2}{3}{4}"
            ex_type_name = type(ex).__name__
            ex_args = "\n\t\t\tArguments: " + str(ex.args)
            ex_context = "\n\t\t\tContext: " + str(ex.__context__)
            ex_cause = "\n\t\t\tCause: " + str(ex.__cause__)
            ex_tb = "\n\t\t\tTraceback: " + str(ex.with_traceback(ex.__traceback__))
            logger.exception(template.format(ex_type_name, ex_args,ex_context, ex_cause, ex_tb))
            print(logger.__str__())
            time.sleep(.05)
        except Exception as ex:
            template = failure_sig + "An exception of type {0} occurred. {1}{2}{3}{4}{5}"
            ex_type_name = type(ex).__name__
            ex_args = "\n\t\t\tArguments: " + str(ex.args)
            ex_context = "\n\t\t\tContext: " + str(ex.__context__)
            ex_cause = "\n\t\t\tCause: " + str(ex.__cause__)
            ex_path = "\n\t\t\tDir Path: " + data_path+training_folder
            ex_tb = "\n\t\t\tTraceback: " + str(ex.with_traceback(ex.__traceback__))
            logger.exception(template.format(ex_type_name, ex_args, ex_context, ex_cause, ex_path, ex_tb))
            print(logger.__str__())
            time.sleep(.05)
        else:
            print(success_sig+"Yay! no more errors from testGenerator")
    
    def test_UnetValidationCallback(self):
        printable_ratio = 0.25
        printable_epoch_count = 10
        print(test_dec_sig+"testing output for unit-test on".upper()+(f" UnetValidationCallback(path_dict,{printable_ratio},{printable_epoch_count})")+RESET)
        logger = logging.getLogger(name="UnetValidationCallback_logger")
        
        with open("../UnetLogicFiles/training_path_dict_file.json", "r") as path_data_file:
            path_dict:dict() = json.load(path_data_file)
        self.assertIsNotNone(path_dict)
        validation = UnetValidationCallback(path_dict,printable_ratio,printable_epoch_count)
        self.assertIsNotNone(validation)
        print("validation.__sizeof__() = ",validation.__sizeof__())

if __name__ == '__main__':
    unittest.main()
