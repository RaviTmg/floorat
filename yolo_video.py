import sys
import argparse
import cv2
import math
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            t_image = np.array(r_image)
            height, width = t_image.shape[:2]
            grayscale = cv2.cvtColor(t_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(grayscale, 75, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 250 , 1)

            for line in lines:
                rho, theta= line[0]
                # Stores the value of cos(theta) in a 
                a = np.cos(theta) 
                # Stores the value of sin(theta) in b 
                b = np.sin(theta) 
                # x0 stores the value rcos(theta) 
                x0 = a*rho
                # y0 stores the value rsin(theta) 
                y0 = b*rho
                # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
                x1 = int(x0 + width*(-b)) 
                # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
                y1 = (y0 + width*(a)) 
                # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
                x2 = int(x0 - width*(-b)) 
                # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
                y2 = (y0 - width*(a)) 
                # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
                # (0,255,0) denotes the colour of the line to be  
                #drawn. In this case, it is green.  
                cv2.line(t_image,(x1,int(y1)), (x2,int(y2)), (0,255,0),2)  
            cv2.imshow("Image", t_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
