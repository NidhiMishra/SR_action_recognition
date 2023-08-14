from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys
import numpy as np
from PIL import Image

import tensorflow as tf

#For vgg
import test_lib
from vgg19 import Vgg19

#For Skeleton CNN
import pickle
import os
import SkelData_Helper
import SkelNet_train

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def nparray_to_image(Body):
    Body.astype(np.uint8)
    data = np.zeros((Body.shape[0], Body.shape[1], 3), dtype=np.uint8)
    data[:, :, 0] = Body
    data[:, :, 1] = Body
    data[:, :, 2] = Body
    img = Image.fromarray(data, mode="RGB")
    img = img.resize((224, 224), Image.ANTIALIAS)
    imrgb = np.asarray(img)
    return imrgb


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def run(self):
        #body, frame parameters
        frame_num = 0
        #frame_interval = 50
        frame_interval = 100
        jointcount = 25
        bodyavailable = [False, False, False, False, False, False]
        Positional_Reference = [4, 8, 12, 16]
        Feature_Share_dir = os.path.abspath("./Feature_Share")
        if not os.path.exists(Feature_Share_dir):
            os.makedirs(Feature_Share_dir)
        ##################################################################
        # setup_vgg19()
        model = np.load(test_lib.model_vgg19, allow_pickle=True, encoding='latin1').item()
        print("The VGG model is loaded.")

        # Design the graph.
        graph = tf.Graph()
        with graph.as_default():
            nn = Vgg19(model=model)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Run the graph in the session.
        with tf.Session(graph=graph, config=config) as sess:
            tf.initialize_all_variables().run()
            print("Tensorflow initialized all variables.")
            ##################################################################
            # -------- Main Program Loop -----------
            while not self._done:
                # --- Main event loop
                for event in pygame.event.get(): # User did something
                    if event.type == pygame.QUIT: # If user clicked close
                        self._done = True # Flag that we are done so we exit this loop

                    elif event.type == pygame.VIDEORESIZE: # window resized
                        self._screen = pygame.display.set_mode(event.dict['size'],
                                                   pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

                # --- Game logic should go here -- Action Recognition
                if frame_num % frame_interval == 0 and frame_num != 0:
                    print("frame", frame_num)
                    print(bodyavailable)

                    #Convert to Cylindrical coordinates
                    rho, theta = cart2pol(Skel_X, Skel_Y)

                    for b in range(0,self._kinect.max_body_count):

                        if bodyavailable[b]:
                            p_iter = 0
                            #Store images for each positional reference
                            imgs = []
                            #Store feature maps for training
                            Feature_map = np.zeros((7168, 12), dtype=float)
                            for p_r in Positional_Reference:

                                conv_joints_chain = [3, 2, 20, 1, 0, 4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24, 12, 13, 14, 15, 16, 17, 18, 19]
                                conv_joints_chain.remove(p_r)
                                Body_rho = rho[:, conv_joints_chain, p_iter, b]
                                Body_theta = theta[:, conv_joints_chain, p_iter, b]
                                Body_Z = Skel_Z[:, conv_joints_chain, p_iter, b]
                                p_iter = p_iter + 1
                                Body_rho = 255* ((Body_rho - np.amin(Body_rho)) / (np.amax(Body_rho) - np.amin(Body_rho)).astype(np.float32))
                                Body_theta = 255 * ((Body_theta - np.amin(Body_theta)) / (np.amax(Body_theta) - np.amin(Body_theta)).astype(np.float32))
                                Body_Z = 255 * ((Body_Z - np.amin(Body_Z)) / (np.amax(Body_Z) - np.amin(Body_Z)).astype(np.float32))

                                Image_rho = nparray_to_image(Body_rho)
                                Image_theta = nparray_to_image(Body_theta)
                                Image_Z = nparray_to_image(Body_Z)

                                imgs.append(Image_rho)
                                imgs.append(Image_theta)
                                imgs.append(Image_Z)
                            #Run the predictions for vgg19 model to obtain conv5_1 layer ouptut
                            preds = sess.run(nn.conv5_1,
                                            feed_dict={
                                                 nn.inputRGB: imgs
                                             })

                            place_feature = 0
                            for pred in enumerate(preds):
                                #print(pred[1].shape)
                                map3D = pred[1]
                                map3D[map3D<0] = 0
                                #print(map3D)
                                feature = np.sum(map3D,axis=0)
                                feature = np.divide(feature,14)
                                #print(feature.shape)
                                #print(feature)

                                Feature_map[:, place_feature] = np.reshape(feature,(7168))
                                place_feature = place_feature + 1

                            print(Feature_map)
                            np.save(os.path.join(Feature_Share_dir, 'Check_' + str(b) + 'Skel.npy'), Feature_map)

                    #Create a file to start action prediction
                    if True in bodyavailable:
                        file = open(os.path.join(Feature_Share_dir, 'Action_prediction.txt'), "w")
                        file.close()

                    print("Waiting for completion")
                    while os.path.exists(os.path.join(Feature_Share_dir, 'Action_prediction.txt')):
                        dummy = 0
                        #print("Waiting for completion")

                    print("Completed")
                    Skel_X = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count), dtype=float)
                    Skel_Y = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count), dtype=float)
                    Skel_Z = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count), dtype=float)
                    bodyavailable = [False, False, False, False, False, False]

                if frame_num == 0:
                    Skel_X = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count),
                                      dtype=float)
                    Skel_Y = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count),
                                      dtype=float)
                    Skel_Z = np.zeros((frame_interval, jointcount, len(Positional_Reference), self._kinect.max_body_count),
                                      dtype=float)
                # --- Getting frames and drawing
                # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
                if self._kinect.has_new_color_frame():
                    frame_num += 1
                    frame = self._kinect.get_last_color_frame()
                    self.draw_color_frame(frame, self._frame_surface)
                    frame = None

                # --- Cool! We have a body frame, so can get skeletons
                if self._kinect.has_new_body_frame():
                    self._bodies = self._kinect.get_last_body_frame()

                # --- draw skeletons to _frame_surface
                if self._bodies is not None:
                    for i in range(0, self._kinect.max_body_count):
                        body = self._bodies.bodies[i]
                        if not body.is_tracked:
                            continue

                        if not bodyavailable[i]:
                            bodyavailable[i] = True

                        joints = body.joints
                        Joint_X = np.zeros(jointcount, dtype=float)
                        Joint_Y = np.zeros(jointcount, dtype=float)
                        Joint_Z = np.zeros(jointcount, dtype=float)
                        #Jointcout 25 - [0 , 24]
                        for joint_iter in range(jointcount):
                            #print(joint_iter, joints[joint_iter].Position.x, joints[joint_iter].Position.y, joints[joint_iter].Position.z)
                            Joint_X[joint_iter] = joints[joint_iter].Position.x
                            Joint_Y[joint_iter] = joints[joint_iter].Position.y
                            Joint_Z[joint_iter] = joints[joint_iter].Position.z

                        p_iter = 0
                        frame_no = frame_num % frame_interval
                        for p_r in Positional_Reference:
                            Skel_X[frame_no, :, p_iter, i] = Joint_X - Joint_X[p_r]
                            Skel_Y[frame_no, :, p_iter, i] = Joint_Y - Joint_Y[p_r]
                            Skel_Z[frame_no, :, p_iter, i] = Joint_Z - Joint_Z[p_r]
                            p_iter = p_iter + 1
                        # convert joint coordinates to color space
                        joint_points = self._kinect.body_joints_to_color_space(joints)
                        self.draw_body(joints, joint_points, SKELETON_COLORS[i])

                # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
                # --- (screen size may be different from Kinect's color frame size)
                h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
                target_height = int(h_to_w * self._screen.get_width())
                surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
                self._screen.blit(surface_to_draw, (0,0))
                surface_to_draw = None
                pygame.display.update()

                # --- Go ahead and update the screen with what we've drawn.
                pygame.display.flip()

                # --- Limit to 60 frames per second
                self._clock.tick(60)

            # Close our Kinect sensor, close the window and quit.
            self._kinect.close()
            pygame.quit()

            #Delete the files and folder used for sharing folders
            for file in os.listdir(Feature_Share_dir):
                file_path = os.path.join(Feature_Share_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            file = open(os.path.join(Feature_Share_dir, 'Action_end.txt'), "w")
            file.close()

__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();

