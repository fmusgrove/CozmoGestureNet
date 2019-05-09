import argparse
import datetime
from multiprocessing import Queue, Pool
from threading import Thread
from collections import Counter
from time import sleep, time

import cv2
import numpy as np
import tensorflow as tf

from Common.cozmo_controller import cozmo_controller, run_cozmo_controller
from HandDetector.utils import detector_utils as detector_utils
from ModelBuilder.data_preprocess import LABEL_NAMES, process_image_for_model

frame_processed = 0
score_thresh = 0.4
image_burst_max_saved = 64


# Create a detect_worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

def detect_worker(input_queue, output_queue, cap_params, frame_processed):
    """
    Function endpoint for
    :param input_queue:
    :param output_queue:
    :param cap_params:
    :param frame_processed:
    :return:
    """
    print('>> Loading frozen model for detection worker')
    detection_graph, sess = detector_utils.load_detect_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        # print("> ===== in detect_worker loop, frame ", frame_processed)
        frame = input_queue.get()
        if frame is not None:
            # Run hand segmentation detection
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # Get bounding boxes for the hands
            roi_bounds = detector_utils.get_segmentation_bounds(cap_params['num_hands_detect'],
                                                                cap_params["score_thresh"], scores, boxes,
                                                                cap_params['im_width'], cap_params['im_height'])

            # Isolate the first hand
            if len(roi_bounds) > 0:
                p1, p2, hand_score = roi_bounds[0]
                hand: np.ndarray = frame[p1[1]:p2[1], p1[0]:p2[0]].copy()
                if not all([arr_size != 0 for arr_size in hand.shape]):
                    hand = None
            else:
                hand = None
                hand_score = 0

            # Draw bounding boxes
            detector_utils.draw_box_on_image(frame, roi_bounds)

            # Add frame annotated with bounding box to queue
            output_queue.put((frame, hand, hand_score))
            frame_processed += 1
        else:
            output_queue.put((frame, None, 0))
    sess.close()


def classification_worker(input_queue, output_queue):
    print('>> Loading frozen model for classification worker')
    classification_model: tf.keras.models.Model = tf.keras.models.load_model(
        'Models/hand_class_graph/optimized_classification_graph_v2.model')

    while True:
        img_frame = input_queue.get()
        if img_frame is not None:
            processed_image = process_image_for_model(img_frame)
            prediction = classification_model.predict([processed_image])
            # print(prediction)
            confidence_score = np.amax(prediction[0])
            if confidence_score > 0.92:
                label_index = int(np.argmax(prediction[0]))
                # Add frame with classification to the output queue
                output_queue.put((np.squeeze(processed_image), LABEL_NAMES[label_index]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=240,
        help='Height of the frames in the video stream')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    parser.add_argument(
        '-dc',
        '--data-collect',
        dest='data_collect_mode',
        type=bool,
        default=False,
        help='Data collection mode for adding classes to the recognition model'
    )
    parser.add_argument(
        '-cn',
        '--class-name',
        dest='class_name',
        type=str,
        default='NullClass',
        help='Name of the class to add to the recognition model'
    )
    parser.add_argument(
        '-sn',
        '--starting-number',
        dest='class_starting_number',
        type=int,
        default=0,
        help='Starting number for the images to be saved (to avoid overwriting images already collected)'
    )
    args = parser.parse_args()

    detector_input_q = Queue(maxsize=args.queue_size)
    detector_output_q = Queue(maxsize=args.queue_size)

    classification_input_q = Queue(maxsize=args.queue_size)
    classification_output_q = Queue(maxsize=args.queue_size)

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = (320, 240)
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up parallelized detection workers
    detect_pool = Pool(args.num_workers, detect_worker,
                       (detector_input_q, detector_output_q, cap_params, frame_processed))

    if args.data_collect_mode:
        cv2.namedWindow('Isolated Hand', cv2.WINDOW_AUTOSIZE)
        classification_pool = None
    else:
        cv2.namedWindow('Classification Model View', cv2.WINDOW_NORMAL)
        classification_pool = Pool(args.num_workers, classification_worker,
                                   (classification_input_q, classification_output_q))

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    # Start the Cozmo management thread
    cozmo_thread = Thread(target=run_cozmo_controller, daemon=True).start()

    output_frame: np.ndarray
    isolated_hand: np.ndarray
    image_burst_count = 0
    image_burst_total = args.class_starting_number
    should_save_images = False
    top_classes_received = []
    num_classes_expected_time = 1
    last_time_val = time()
    x = 0
    print('Waiting for image')
    while True:
        # Grab the latest image off Cozmo's camera
        frame = cozmo_controller.latest_image

        if frame is not None:
            # Convert the frame from a PIL image to an OpenCV image format
            frame = np.array(frame)

            # Convert from RGB to BGR
            frame = frame[:, :, ::-1].copy()

            # Flip the frame across the vertical axis for more natural perspective
            frame = cv2.flip(frame, 1)

            index += 1

            detector_input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            output_frame, isolated_hand, hand_score = detector_output_q.get()

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time
            # print("frame ",  index, num_frames, elapsed_time, fps)

            if output_frame is not None:
                if isolated_hand is not None:
                    try:
                        # Convert to greyscale and add to the classification queue
                        isolated_hand = cv2.cvtColor(isolated_hand, cv2.COLOR_RGB2GRAY)
                        if not args.data_collect_mode:
                            classification_input_q.put(isolated_hand)
                    except Exception as e:
                        print('Error converting to grayscale:', e)
                if should_save_images:
                    if not (image_burst_count % image_burst_max_saved == 0) or image_burst_count == 0:
                        # Save the image to the dataset folder
                        cv2.imwrite(f'res/TrainingData/raw/{args.class_name}/{image_burst_total}.png',
                                    isolated_hand)
                        image_burst_count += 1
                        image_burst_total += 1
                    else:
                        # Stop collecting images if the maximum has been reached
                        should_save_images = False
                        image_burst_count = 0
                        print(f'{image_burst_total} total images saved')
                if args.display > 0:
                    if args.fps > 0:
                        detector_utils.draw_fps_on_image(f'FPS: {int(fps)}', output_frame)
                    if args.data_collect_mode:
                        detector_utils.draw_score_on_image(f'Score: {round(float(hand_score), 2)}', output_frame)
                        if isolated_hand is not None:
                            cv2.imshow('Isolated Hand', isolated_hand)
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                    if args.data_collect_mode and cv2.waitKey(1) & 0xFF == ord('s'):
                        should_save_images = True
                        print(f'Now saving {image_burst_max_saved} images')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if num_frames == 400:
                        num_frames = 0
                        start_time = datetime.datetime.now()
                    else:
                        print(f'frames processed: {index} elapsed time: {elapsed_time} FPS: {int(fps)}')
            else:
                break

        if not args.data_collect_mode:
            if time() - last_time_val > 3:
                # Expect to receive as many classifications as the current fps in one second to surpass the
                # threshold and trigger the action on Cozmo
                num_classes_expected_time = max(1, int(fps * 0.8))
                last_time_val = time()
                top_classes_received.clear()
                print(f'Cleared, expecting at least {num_classes_expected_time} triggers in 3 second')

            if not classification_output_q.empty():
                processed_class_frame, predicted_class = classification_output_q.get()
                top_classes_received.append(predicted_class)

                # Count the top classes received and only react to the most prevalent classification to
                # smooth the output (linked to the fps to scale with performance)
                if len(top_classes_received) > num_classes_expected_time:
                    top_class = Counter(top_classes_received).most_common(1)[0][0]
                    top_classes_received.clear()

                    # Pass command on to Cozmo (only if he isn't already reacting to a command
                    if cozmo_controller.command_q.empty() and not cozmo_controller.command_run_lock.locked():
                        cozmo_controller.command_q.put(top_class)

                if args.display > 0:
                    cv2.imshow('Classification Model View', processed_class_frame)

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print('Ending FPS:', fps)
    detect_pool.terminate()
    if classification_pool is not None:
        classification_pool.terminate()
    cv2.destroyAllWindows()
