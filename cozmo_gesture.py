import argparse
import datetime
from multiprocessing import Queue, Pool
from threading import Thread

import cv2
import numpy
import tensorflow as tf

from HandDetector import image_grab
from HandDetector.utils import detector_utils as detector_utils

frame_processed = 0
score_thresh = 0.15
image_burst_max_saved = 25


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


def worker(input_queue, output_queue, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        frame = input_queue.get()
        if frame is not None:
            # Actual detection. Variable boxes contains the bounding box coordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Get bounding boxes for the hands
            roi_bounds = detector_utils.get_segmentation_bounds(cap_params['num_hands_detect'],
                                                                cap_params["score_thresh"], scores, boxes,
                                                                cap_params['im_width'], cap_params['im_height'])

            # Isolate the first hand
            if len(roi_bounds) > 0:
                p1, p2, hand_score = roi_bounds[0]
                hand = frame[p1[1]:p2[1], p1[0]:p2[0]].copy()
            else:
                hand = None
                hand_score = 0

            # draw bounding boxes
            detector_utils.draw_box_on_image(frame, roi_bounds)

            # add frame annotated with bounding box to queue
            output_queue.put((frame, hand, hand_score))
            frame_processed += 1
        else:
            output_queue.put((frame, None, 0))
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
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
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=240,
        help='Height of the frames in the video stream.')
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
        help='Data collection mode for adding classes to the recognition model.'
    )
    parser.add_argument(
        '-cn',
        '--class-name',
        dest='class_name',
        type=str,
        default='NullClass',
        help='Name of the class to add to the recognition model.'
    )
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = (320, 240)
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to parallelize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Isolated Hand', cv2.WINDOW_AUTOSIZE)

    cozmo_thread = Thread(target=image_grab.run_cozmo_photostream, daemon=True).start()

    print('Waiting for image')
    image_burst_count = 0
    image_burst_total = 0
    should_save_images = False
    x = 0
    while True:
        # Wait for a new image to be available from Cozmo
        # wait_for_image = image_grab.photo_stream.image_available.wait()

        # Grab the image and clear the event
        frame = image_grab.photo_stream.latest_image
        # image_grab.photo_stream.image_available.clear()

        if frame is not None:
            # Convert the frame from a PIL image to an OpenCV image format
            frame = numpy.array(frame)

            # Convert from RGB to BGR
            frame = frame[:, :, ::-1].copy()

            frame = cv2.flip(frame, 1)
            index += 1

            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output_frame, isolated_hand, hand_score = output_q.get()

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time
            # print("frame ",  index, num_frames, elapsed_time, fps)

            if output_frame is not None:
                if isolated_hand is not None:
                    # isolated_hand = cv2.cvtColor(isolated_hand, cv2.COLOR_RGB2BGR)
                    # Convert to greyscale
                    isolated_hand = cv2.cvtColor(isolated_hand, cv2.COLOR_RGB2GRAY)
                    if should_save_images:
                        if not (image_burst_count % image_burst_max_saved == 0) or image_burst_count == 0:
                            # Save the image to the dataset folder
                            cv2.imwrite(f'res/TrainingData/{args.class_name}/{image_burst_total}.png', isolated_hand)
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
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                    if isolated_hand is not None:
                        cv2.imshow('Isolated Hand', isolated_hand)
                    if args.data_collect_mode and cv2.waitKey(1) & 0xFF == ord('s'):
                        should_save_images = True
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if num_frames == 400:
                        num_frames = 0
                        start_time = datetime.datetime.now()
                    else:
                        print("frames processed: ", index, "elapsed time: ",
                              elapsed_time, "fps: ", str(int(fps)))
            else:
                # print("video end")
                break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print('Ending FPS:', fps)
    pool.terminate()
    cv2.destroyAllWindows()
